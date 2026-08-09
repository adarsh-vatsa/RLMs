import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "anthropic" not in sys.modules:
    anthropic_stub = types.ModuleType("anthropic")

    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.messages = None

    anthropic_stub.Anthropic = FakeAnthropic
    sys.modules["anthropic"] = anthropic_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

import numpy as np

import semantic_cache_system as scs


class FakeEmbedder:
    def encode_query(self, query):
        return np.array([1.0, 0.0], dtype="float32")

    def encode_single(self, text):
        return np.array([1.0, 0.0], dtype="float32")

    def encode_documents(self, documents):
        return np.array([[1.0, 0.0] for _ in documents], dtype="float32")


class FakeSearchIndex:
    def __init__(self, results=None):
        self.results = results or []
        self.added = []
        self.loaded = False

    @property
    def total(self):
        return len(self.results) or len(self.added)

    def search(self, query_embedding, top_k=20):
        return self.results[:top_k]

    def add(self, embeddings, metadata):
        self.added.extend(metadata)

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def load(self, path):
        self.loaded = True


class FakeReranker:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def rerank(self, query, documents, top_k=5):
        self.calls.append({"query": query, "documents": documents, "top_k": top_k})
        return self.results[:top_k]


def make_controller():
    return scs.SemanticCacheController(
        metrics=scs.ExecutionMetrics(),
        embedder=FakeEmbedder(),
        reranker=None,
        corpus_id="test",
    )


def make_entry(query, result, scope=None):
    entry = {
        "query": query,
        "result": result,
        "embedding": np.array([1.0, 0.0], dtype="float32"),
        "source_context": result,
        "grounding_info": {},
    }
    if scope is not None:
        entry["data_scope_hash"] = scope
    return entry


class DataScopedSearchCacheTests(unittest.TestCase):
    def setUp(self):
        self._iterative_batch_patch = patch.object(scs, "ITERATIVE_BATCH_MAX_CHUNKS", 1)
        self._iterative_batch_patch.start()
        self.addCleanup(self._iterative_batch_patch.stop)

    def test_embedding_query_uses_configured_instruction(self):
        engine = scs.EmbeddingEngine.__new__(scs.EmbeddingEngine)
        seen = {}

        def fake_encode(texts, instruction=""):
            seen["texts"] = texts
            seen["instruction"] = instruction
            return np.array([[1.0, 0.0]], dtype="float32")

        engine.encode = fake_encode

        with patch.object(scs, "EMBEDDING_QUERY_INSTRUCTION", "custom evidence instruction"):
            embedding = engine.encode_query("Which option is supported?")

        self.assertEqual(seen["texts"], ["Which option is supported?"])
        self.assertEqual(seen["instruction"], "custom evidence instruction")
        self.assertEqual(embedding.shape, (1, 2))

    def test_embedding_device_cuda_requires_available_cuda(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        fake_torch = types.SimpleNamespace(cuda=FakeCuda)

        with patch.object(scs, "EMBEDDING_DEVICE", "cuda"):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                scs._resolve_embedding_device(fake_torch)

    def test_embedding_device_auto_uses_cuda_when_available(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return True

        fake_torch = types.SimpleNamespace(cuda=FakeCuda)

        with patch.object(scs, "EMBEDDING_DEVICE", "auto"):
            self.assertEqual(scs._resolve_embedding_device(fake_torch), "cuda")

    def test_embedding_dtype_auto_prefers_bfloat16_on_cuda(self):
        class FakeCuda:
            @staticmethod
            def is_bf16_supported():
                return True

        fake_torch = types.SimpleNamespace(
            cuda=FakeCuda,
            float32="float32",
            float16="float16",
            bfloat16="bfloat16",
        )

        with patch.object(scs, "EMBEDDING_DTYPE", "auto"):
            dtype, name = scs._resolve_embedding_dtype(fake_torch, "cuda")

        self.assertEqual(dtype, "bfloat16")
        self.assertEqual(name, "bfloat16")

    def test_exact_hits_are_limited_to_active_data_scope(self):
        query = "same question"
        controller = make_controller()
        controller.cache = {
            "chunk-a": [make_entry(query, "answer from scope a", "scope-a")],
            "chunk-b": [make_entry(query, "answer from scope b", "scope-b")],
        }

        controller.data_scope_hash = "scope-a"
        self.assertEqual(controller.search(query)["answer"], "answer from scope a")

        controller.data_scope_hash = "scope-b"
        self.assertEqual(controller.search(query)["answer"], "answer from scope b")

        controller.data_scope_hash = "scope-c"
        result = controller.search(query)
        self.assertFalse(result["from_cache"])
        self.assertEqual(result["answer"], "No relevant documents found.")

    def test_legacy_unscoped_entries_are_skipped_when_scope_is_active(self):
        query = "same question"
        controller = make_controller()
        controller.cache = {"legacy": [make_entry(query, "legacy answer")]}
        controller.data_scope_hash = "current-scope"

        result = controller.search(query)
        self.assertFalse(result["from_cache"])
        self.assertEqual(result["answer"], "No relevant documents found.")

    def test_semantic_lookup_filters_candidates_to_active_scope(self):
        controller = make_controller()
        controller.cache = {
            "chunk-a": [make_entry("question a", "answer a", "scope-a")],
            "chunk-b": [make_entry("question b", "answer b", "scope-b")],
        }
        controller._cache_index = FakeSearchIndex(
            [
                (0.99, {"cache_idx": 0}),
                (0.98, {"cache_idx": 1}),
            ]
        )
        controller.activate_data_scope("scope-b")

        def fake_sniper(query, candidates):
            self.assertEqual([entry["result"] for entry, _, _ in candidates], ["answer b"])
            return {"hit": True, "id": 0}

        with patch.object(controller, "_llm_sniper_evaluate", side_effect=fake_sniper):
            result = controller.lookup_cached_result("paraphrased question")

        self.assertTrue(result["from_cache"])
        self.assertEqual(result["cache_type"], "semantic")
        self.assertEqual(result["answer"], "answer b")

    def test_knowledge_hits_are_limited_to_active_data_scope(self):
        controller = make_controller()
        controller.cache = {
            "chunk-a": [make_entry("source question", "wrong scope answer", "scope-a")],
            "chunk-b": [make_entry("source question", "right scope answer", "scope-b")],
        }
        controller.knowledge = [
            {"subject": "Scott Derrickson", "relation": "document", "object": "1544120", "source_cache_idx": 0},
        ]
        controller.knowledge_index = FakeSearchIndex([(0.99, {"fact_idx": 0})])
        controller.data_scope_hash = "scope-b"

        result = controller.search("Which document mentions Scott Derrickson?")
        self.assertFalse(result["from_cache"])
        self.assertEqual(result["answer"], "No relevant documents found.")

    def test_scoped_entry_persistence_and_legacy_load_behavior(self):
        query = "same question"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            controller = make_controller()
            controller.data_scope_hash = "scope-a"
            controller.cache = {"chunk-a": [make_entry(query, "scoped answer", "scope-a")]}
            controller.save(path)

            with patch.object(scs, "FAISSIndex", FakeSearchIndex):
                loaded = make_controller()
                self.assertTrue(loaded.load(path))
                self.assertEqual(
                    loaded.cache["chunk-a"][0]["data_scope_hash"],
                    "scope-a",
                )

            legacy_path = path / "legacy"
            legacy_path.mkdir()
            (legacy_path / "corpus_config.json").write_text(
                '{"corpus_id": "test", "data_scope_hash": "scope-a"}',
                encoding="utf-8",
            )
            (legacy_path / "cache_entries.json").write_text(
                '{"legacy": [{"query": "same question", "result": "legacy answer", '
                '"embedding": [1.0, 0.0], "source_context": "legacy"}]}',
                encoding="utf-8",
            )
            (legacy_path / "knowledge.json").write_text("[]", encoding="utf-8")

            with patch.object(scs, "FAISSIndex", FakeSearchIndex):
                legacy = make_controller()
                self.assertFalse(legacy.load(legacy_path))

    def test_compact_mcq_store_omits_context_and_knowledge(self):
        controller = make_controller()
        controller.activate_data_scope("scope-a")

        with patch.object(scs, "FAISSIndex", FakeSearchIndex):
            entry = controller.store_compact_mcq(
                "Which option?",
                "A",
                model_used="executor",
                source_id="source-a",
                route="direct_fit",
                provenance={"final_rendered_input_tokens": 123},
            )

        self.assertNotIn("source_context", entry)
        self.assertEqual(entry["facts"], [])
        self.assertEqual(controller.knowledge, [])
        self.assertEqual(entry["data_scope_hash"], "scope-a")
        self.assertEqual(entry["route"], "direct_fit")

    def test_compact_mcq_store_rejects_invalid_answer(self):
        controller = make_controller()
        controller.activate_data_scope("scope-a")

        with self.assertRaisesRegex(ValueError, "exactly one"):
            controller.store_compact_mcq(
                "Which option?",
                "Answer: A",
                model_used="executor",
                source_id="source-a",
                route="direct_fit",
            )

        self.assertEqual(controller.get_total_entries(), 0)

    def test_embedding_uses_left_padded_last_token_pooling(self):
        import torch

        class FakeInputs(dict):
            def to(self, device):
                return self

        class FakeTokenizer:
            def __call__(self, texts, **kwargs):
                return FakeInputs(
                    input_ids=torch.tensor([[0, 1], [2, 3]]),
                    attention_mask=torch.tensor([[0, 1], [1, 1]]),
                )

        class FakeModel:
            def __call__(self, **kwargs):
                return types.SimpleNamespace(
                    last_hidden_state=torch.tensor(
                        [
                            [[9.0, 9.0], [3.0, 4.0]],
                            [[8.0, 8.0], [0.0, 5.0]],
                        ]
                    )
                )

        engine = scs.EmbeddingEngine.__new__(scs.EmbeddingEngine)
        engine.device = "cpu"
        engine.torch = torch
        engine.tokenizer = FakeTokenizer()
        engine.model = FakeModel()

        with patch.object(scs, "EMBEDDING_BATCH_SIZE", 2), patch.object(
            scs, "EMBEDDING_MAX_LENGTH", 8192
        ):
            embeddings = engine.encode(["first", "second"])

        np.testing.assert_allclose(embeddings, np.array([[0.6, 0.8], [0.0, 1.0]]))
        self.assertEqual(engine._last_encode_info["pooling"], "last_token")
        self.assertEqual(engine._last_encode_info["padding_side"], "left")

    def test_data_scope_hash_includes_token_chunk_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs_dir = Path(tmp)
            doc_path = docs_dir / "doc.txt"
            doc_path.write_text("full document text", encoding="utf-8")
            controller = make_controller()

            char_scope = controller._get_data_scope_hash(
                docs_dir,
                [doc_path],
                chunk_unit="chars",
                chunk_size=10000,
                overlap=1000,
            )
            token_scope = controller._get_data_scope_hash(
                docs_dir,
                [doc_path],
                chunk_unit="tokens",
                chunk_size=10000,
                overlap=1000,
                token_chunk_size=6000,
                token_overlap=600,
                tokenizer_model="model-a",
                embedding_max_length=8192,
            )
            changed_token_scope = controller._get_data_scope_hash(
                docs_dir,
                [doc_path],
                chunk_unit="tokens",
                chunk_size=10000,
                overlap=1000,
                token_chunk_size=8000,
                token_overlap=800,
                tokenizer_model="model-a",
                embedding_max_length=8192,
            )
            changed_embedding_scope = controller._get_data_scope_hash(
                docs_dir,
                [doc_path],
                chunk_unit="tokens",
                chunk_size=10000,
                overlap=1000,
                token_chunk_size=6000,
                token_overlap=600,
                tokenizer_model="model-a",
                embedding_max_length=4096,
            )

        self.assertNotEqual(char_scope, token_scope)
        self.assertNotEqual(token_scope, changed_token_scope)
        self.assertNotEqual(token_scope, changed_embedding_scope)

    def test_retrieval_falls_back_to_faiss_when_reranker_returns_no_results(self):
        controller = make_controller()
        first_meta = {"filename": "contract.txt", "chunk_index": 0}
        second_meta = {"filename": "contract.txt", "chunk_index": 1}
        controller.doc_index = FakeSearchIndex(
            [(0.91, first_meta), (0.86, second_meta)]
        )
        controller._doc_chunks = [
            "The contract is governed by New York law.",
            "The agreement expires on December 31, 2028.",
        ]
        controller.reranker = FakeReranker([])

        results = controller.retrieve("What law governs the contract?", top_k=20, rerank_top=5)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], "The contract is governed by New York law.")
        self.assertTrue(controller._last_retrieval_info["reranker_enabled"])
        self.assertEqual(controller._last_retrieval_info["faiss_candidate_count"], 2)
        self.assertEqual(controller._last_retrieval_info["reranker_returned_count"], 0)
        self.assertTrue(controller._last_retrieval_info["reranker_fallback_used"])

    def test_retrieval_keeps_reranker_results_first_and_backfills_when_needed(self):
        controller = make_controller()
        first_meta = {"filename": "contract.txt", "chunk_index": 0}
        second_meta = {"filename": "contract.txt", "chunk_index": 1}
        controller.doc_index = FakeSearchIndex(
            [(0.91, first_meta), (0.86, second_meta)]
        )
        controller._doc_chunks = [
            "The contract is governed by New York law.",
            "The agreement expires on December 31, 2028.",
        ]
        controller.reranker = FakeReranker(
            [(1, 0.77, "The agreement expires on December 31, 2028.")]
        )

        results = controller.retrieve("When does the agreement expire?", top_k=20, rerank_top=5)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], "The agreement expires on December 31, 2028.")
        self.assertEqual(results[0]["score"], 0.77)
        self.assertEqual(results[1]["text"], "The contract is governed by New York law.")
        self.assertTrue(controller._last_retrieval_info["reranker_enabled"])
        self.assertEqual(controller._last_retrieval_info["faiss_candidate_count"], 2)
        self.assertEqual(controller._last_retrieval_info["reranker_returned_count"], 1)
        self.assertTrue(controller._last_retrieval_info["reranker_fallback_used"])

    def test_iterative_scan_budget_respects_ratio_min_max_and_total(self):
        self.assertEqual(
            scs._compute_iterative_scan_budget(
                total_chunks=20,
                min_chunk_ratio=0.30,
                max_chunk_ratio=0.50,
                min_chunks=3,
                max_chunks=0,
            ),
            (6, 10),
        )
        self.assertEqual(
            scs._compute_iterative_scan_budget(
                total_chunks=100,
                min_chunk_ratio=0.30,
                max_chunk_ratio=0.50,
                min_chunks=3,
                max_chunks=0,
            ),
            (30, 50),
        )
        self.assertEqual(
            scs._compute_iterative_scan_budget(
                total_chunks=100,
                min_chunk_ratio=0.30,
                max_chunk_ratio=0.50,
                min_chunks=3,
                max_chunks=24,
            ),
            (30, 30),
        )
        self.assertEqual(
            scs._compute_iterative_scan_budget(
                total_chunks=100,
                min_chunk_ratio=0.30,
                max_chunk_ratio=0.50,
                min_chunks=3,
                max_chunks=24,
            ),
            (30, 30),
        )
        self.assertEqual(
            scs._compute_iterative_scan_budget(
                total_chunks=4,
                min_chunk_ratio=0.30,
                max_chunk_ratio=0.50,
                min_chunks=3,
                max_chunks=24,
            ),
            (3, 3),
        )
        self.assertEqual(
            scs._compute_empty_ledger_fallback_budget(
                total_chunks=10,
                initial_scan_budget=3,
                fallback_ratio=1.0,
            ),
            10,
        )
        self.assertEqual(
            scs._compute_empty_ledger_fallback_budget(
                total_chunks=10,
                initial_scan_budget=3,
                fallback_ratio=0.0,
            ),
            3,
        )

    def test_iterative_scan_order_is_faiss_ranked_without_duplicates(self):
        controller = make_controller()
        controller._doc_chunks = ["chunk zero", "chunk one", "chunk two", "chunk three"]
        controller._doc_chunk_metadata = [{"chunk_index": idx} for idx in range(4)]
        faiss_results = [
            {"text": "chunk two", "score": 0.91, "metadata": {"chunk_index": 2}},
            {"text": "chunk zero", "score": 0.88, "metadata": {"chunk_index": 0}},
        ]

        ordered, total_chunks, faiss_result_count = controller._build_iterative_scan_results(faiss_results)

        self.assertEqual(total_chunks, 4)
        self.assertEqual(faiss_result_count, 2)
        self.assertEqual([row["metadata"]["chunk_index"] for row in ordered], [2, 0])

    def test_iterative_early_stop_requires_answer_and_context_satisfied(self):
        ledger = scs._new_evidence_ledger()
        decision = {
            "status": "answer_found",
            "memory_update": "Chunk directly supports B.",
            "best_choice": "B",
            "best_choice_rationale": "B is directly supported.",
            "open_questions": [],
            "needs_more_context": False,
        }
        scs._update_evidence_ledger(ledger, decision, chunk_index=2)

        self.assertEqual(
            scs._should_stop_iterative_scan(
                decision=decision,
                ledger=ledger,
                visited_count=1,
                min_chunks=3,
                comparative_required_count=2,
                comparative_query=False,
            ),
            (False, "min_chunks_not_reached"),
        )
        self.assertEqual(
            scs._should_stop_iterative_scan(
                decision=decision,
                ledger=ledger,
                visited_count=3,
                min_chunks=3,
                comparative_required_count=2,
                comparative_query=False,
            ),
            (True, "answer_found"),
        )
        needs_more_context = dict(decision, needs_more_context=True)
        self.assertEqual(
            scs._should_stop_iterative_scan(
                decision=needs_more_context,
                ledger=ledger,
                visited_count=3,
                min_chunks=3,
                comparative_required_count=2,
                comparative_query=False,
            ),
            (False, "needs_more_context"),
        )

        contradiction_ledger = scs._new_evidence_ledger()
        contradiction_decision = dict(decision, open_questions=["Need to verify another chunk does not reject B"])
        scs._update_evidence_ledger(contradiction_ledger, contradiction_decision, chunk_index=4)
        self.assertEqual(
            scs._should_stop_iterative_scan(
                decision=contradiction_decision,
                ledger=contradiction_ledger,
                visited_count=3,
                min_chunks=3,
                comparative_required_count=2,
                comparative_query=False,
            ),
            (False, "open_questions_remain"),
        )

    def test_iterative_evidence_ledger_is_bounded_and_preserves_chunk_references(self):
        ledger = scs._new_evidence_ledger()
        decision = {
            "status": "answer_found",
            "memory_update": "A is supported by chunk 7.",
            "target_facts": ["Target event appears in the current chunk."],
            "code_mappings": [{"code": "abb", "relation": "located in", "example": "X -> abb"}],
            "best_choice": "A",
            "best_choice_rationale": "A matches the cumulative memory.",
            "open_questions": [f"open question {idx}" for idx in range(12)],
        }

        scs._update_evidence_ledger(ledger, decision, chunk_index=7)

        self.assertNotIn("A", ledger)
        self.assertEqual(len(ledger["memory_updates"]), 1)
        self.assertIn("[chunk 7]", ledger["memory"])
        self.assertEqual(len(ledger["target_facts"]), 1)
        self.assertEqual(len(ledger["code_mappings"]), 1)
        self.assertEqual(len(ledger["open_questions"]), 5)
        self.assertEqual(ledger["memory_updates"][0]["chunk_index"], 7)
        self.assertEqual(ledger["target_facts"][0]["chunk_index"], 7)
        self.assertEqual(ledger["code_mappings"][0]["chunk_index"], 7)
        self.assertEqual(ledger["visited_chunks"], [7])
        self.assertEqual(ledger["best_choice"], "A")
        self.assertNotIn("confidence", ledger)
        self.assertTrue(scs._ledger_has_useful_memory(ledger))

    def test_iterative_memory_cap_trims_updates_but_preserves_structured_state(self):
        ledger = scs._new_evidence_ledger()
        ledger["parse_failures"].append({"chunk_index": 1, "raw_response": "bad json"})

        with patch.object(scs, "ITERATIVE_MEMORY_MAX_CHARS", 60):
            scs._update_evidence_ledger(
                ledger,
                {
                    "memory_update": "old note " * 20,
                    "best_choice": "B",
                    "best_choice_rationale": "first rationale",
                },
                chunk_index=1,
            )
            scs._update_evidence_ledger(
                ledger,
                {
                    "memory_update": "new note " * 20,
                    "target_facts": ["preserved fact"],
                    "code_mappings": [{"code": "abb", "relation": "located in", "example": "example"}],
                    "best_choice": "C",
                    "best_choice_rationale": "updated rationale",
                },
                chunk_index=2,
            )

        self.assertLessEqual(len(ledger["memory"]), 60)
        self.assertEqual(ledger["memory_updates"][-1]["chunk_index"], 2)
        self.assertEqual(ledger["target_facts"][0]["note"], "preserved fact")
        self.assertEqual(ledger["code_mappings"][0]["code"], "abb")
        self.assertEqual(ledger["best_choice"], "C")
        self.assertEqual(ledger["parse_failures"][0]["chunk_index"], 1)

    def test_iterative_evidence_ledger_records_parse_failures(self):
        ledger = scs._new_evidence_ledger()

        scs._update_evidence_ledger(
            ledger,
            {
                "status": "no_update",
                "parse_failed": True,
                "raw_response": "not-json " * 200,
            },
            chunk_index=3,
        )

        self.assertFalse(scs._ledger_has_useful_memory(ledger))
        self.assertEqual(len(ledger["parse_failures"]), 1)
        self.assertEqual(ledger["parse_failures"][0]["chunk_index"], 3)
        self.assertLessEqual(len(ledger["parse_failures"][0]["raw_response"]), 800)

    def test_relation_ledger_filters_examples_to_current_option_codes(self):
        query = (
            "Document: entity0 A was born in entity3 B.\n\n"
            "Question: Only considering the given document, what is the relation type "
            "between entity0 and entity3?\n\n"
            "Choices:\n"
            "A. abf\n"
            "B. adn\n"
            "C. abb\n"
            "D. aae\n\n"
            "Return only the single best answer choice letter: A, B, C, or D."
        )
        ledger = scs._new_evidence_ledger(query)
        decision = {
            "status": "partial",
            "code_mappings": [
                "Entity0 Rapid Penang, Entity1 State of Penang -> abi (C)",
                "Entity0 Second River, Entity2 New Jersey -> abb (C)",
                "Entity0 Lucius Caesar, Entity5 Agrippa -> aae (B)",
            ],
            "mappings": ["Entity0 developed by Entity1 -> abr"],
        }

        scs._update_evidence_ledger(ledger, decision, chunk_index=6, query=query)

        target_fact_notes = [item["note"] for item in ledger["target_facts"]]
        self.assertTrue(any("entity0, entity3" in note for note in target_fact_notes))
        self.assertTrue(
            any(
                "Current answer option relation codes" in note
                and all(code in note for code in ("abf", "adn", "abb", "aae"))
                for note in target_fact_notes
            )
        )
        mappings = ledger["code_mappings"]
        self.assertEqual([item["code"] for item in mappings], ["abb", "aae"])
        self.assertFalse(any(item["code"] == "abi" for item in mappings))
        self.assertFalse(any(item["code"] == "abr" for item in mappings))

    def test_relation_ledger_requires_mapping_before_accepting_best_choice(self):
        query = (
            "Document: entity0 Astar is a entity1 New Zealand personality.\n\n"
            "Question: Only considering the given document, what is the relation type "
            "between entity0 and entity1?\n\n"
            "Choices:\n"
            "A. aaa\n"
            "B. acy\n"
            "C. abt\n"
            "D. aah\n\n"
            "Return only the single best answer choice letter: A, B, C, or D."
        )
        ledger = scs._new_evidence_ledger(query)
        unsupported_decision = {
            "status": "answer_found",
            "target_facts": ["Code acy corresponds to nationality."],
            "best_choice": "B",
            "best_choice_rationale": "acy is the standard nationality code",
            "open_questions": [],
            "needs_more_context": False,
        }

        scs._update_evidence_ledger(ledger, unsupported_decision, chunk_index=3, query=query)

        self.assertIsNone(ledger["best_choice"])
        self.assertEqual(ledger["best_choice_rationale"], "")
        self.assertFalse(
            any("acy corresponds" in item["note"].lower() for item in ledger["target_facts"])
        )
        self.assertTrue(any("relation code acy" in note for note in ledger["open_questions"]))

        supported_decision = {
            "status": "answer_found",
            "code_mappings": [
                {
                    "code": "acy",
                    "relation": "nationality",
                    "example": "entity0 Example Person -> entity1 Example Country",
                }
            ],
            "best_choice": "B",
            "best_choice_rationale": "acy is demonstrated by a person-country example",
            "open_questions": [],
            "needs_more_context": False,
        }

        scs._update_evidence_ledger(ledger, supported_decision, chunk_index=4, query=query)

        self.assertEqual(ledger["best_choice"], "B")
        self.assertEqual(ledger["best_choice_rationale"], "acy is demonstrated by a person-country example")
        self.assertEqual(ledger["code_mappings"][0]["code"], "acy")

    def test_iterative_inspector_uses_compact_json_contract(self):
        controller = make_controller()
        captured = {}

        class FakeUsage:
            input_tokens = 10
            output_tokens = 5

        class FakeResponse:
            usage = FakeUsage()
            content = [
                types.SimpleNamespace(
                    text='{"status":"no_update","memory_update":"","target_facts":[],"code_mappings":[],"best_choice":null,"best_choice_rationale":"","open_questions":[],"needs_more_context":true}'
                )
            ]

        def fake_create_llm_message(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        with patch.object(scs, "SCAN_MAX_TOKENS", 768), patch.object(
            scs, "create_llm_message", side_effect=fake_create_llm_message
        ):
            parsed = controller._inspect_iterative_chunk(
                (
                    "Document: entity0 A is entity1 B.\n"
                    "Question: what is the relation type between entity0 and entity1?\n\n"
                    "Choices:\nA. aaa\nB. abb\nC. acc\nD. add"
                ),
                scs._new_evidence_ledger(),
                {"text": "chunk text", "metadata": {"chunk_index": 4}},
            )

        self.assertEqual(parsed["chunk_index"], 4)
        self.assertEqual(captured["max_tokens"], 768)
        self.assertIn("Return ONLY one valid compact JSON object", captured["system"])
        self.assertIn("memory_update", captured["system"])
        self.assertIn("code_mappings", captured["system"])
        self.assertIn("best_choice must be A, B, C, D, or null", captured["system"])
        self.assertNotIn("confidence", captured["system"])
        self.assertIn("never replace them with entities from demonstrations", captured["system"])
        self.assertIn("Current symbolic option codes", captured["messages"][0]["content"])

    def test_iterative_batch_inspector_uses_chunk_updates_contract(self):
        controller = make_controller()
        captured = {}

        class FakeUsage:
            input_tokens = 30
            output_tokens = 9

        class FakeResponse:
            usage = FakeUsage()
            content = [
                types.SimpleNamespace(
                    text=json.dumps(
                        {
                            "chunk_updates": [
                                {
                                    "chunk_index": 4,
                                    "status": "partial",
                                    "memory_update": "chunk four evidence",
                                    "target_facts": [],
                                    "code_mappings": [],
                                    "best_choice": None,
                                    "best_choice_rationale": "",
                                    "open_questions": ["need another chunk"],
                                    "needs_more_context": True,
                                },
                                {
                                    "chunk_index": 5,
                                    "status": "answer_found",
                                    "memory_update": "chunk five proves A",
                                    "target_facts": ["A is supported"],
                                    "code_mappings": [],
                                    "best_choice": "A",
                                    "best_choice_rationale": "chunk five proves A",
                                    "open_questions": [],
                                    "needs_more_context": False,
                                },
                            ]
                        }
                    )
                )
            ]

        def fake_create_llm_message(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        results = [
            {"text": "chunk four", "metadata": {"chunk_index": 4}},
            {"text": "chunk five", "metadata": {"chunk_index": 5}},
        ]
        with patch.object(scs, "SCAN_MAX_TOKENS", 768), patch.object(
            scs, "create_llm_message", side_effect=fake_create_llm_message
        ):
            decisions = controller._inspect_iterative_chunk_batch(
                "Question: Which option?\n\nChoices:\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta",
                scs._new_evidence_ledger(),
                results,
            )

        self.assertEqual([decision["chunk_index"] for decision in decisions], [4, 5])
        self.assertEqual(captured["max_tokens"], 768)
        self.assertIn("chunk_updates", captured["system"])
        self.assertIn("Chunks:", captured["messages"][0]["content"])

    def test_iterative_batch_packing_respects_max_chunks_and_keeps_oversized_single_chunk(self):
        controller = make_controller()
        query = "Question: Which option?\n\nChoices:\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta"
        ledger = scs._new_evidence_ledger(query)
        system_prompt = controller._iterative_inspector_system_prompt(query, batched=True)
        results = [
            {"text": "short chunk", "metadata": {"chunk_index": 0}},
            {"text": "second short chunk", "metadata": {"chunk_index": 1}},
            {"text": "third short chunk", "metadata": {"chunk_index": 2}},
        ]

        with patch.object(scs, "ITERATIVE_BATCH_MAX_CHUNKS", 2), patch.object(
            scs, "ITERATIVE_BATCH_INPUT_TOKEN_BUDGET", 100000
        ):
            batch = controller._pack_iterative_inspection_batch(
                query=query,
                ledger=ledger,
                results=results,
                system_prompt=system_prompt,
            )

        self.assertEqual(len(batch), 2)

        with patch.object(scs, "ITERATIVE_BATCH_MAX_CHUNKS", 3), patch.object(
            scs, "ITERATIVE_BATCH_INPUT_TOKEN_BUDGET", 1
        ):
            batch = controller._pack_iterative_inspection_batch(
                query=query,
                ledger=ledger,
                results=results,
                system_prompt=system_prompt,
            )

        self.assertEqual(len(batch), 1)

    def test_iterative_search_batches_chunks_and_stops_before_next_batch(self):
        controller = make_controller()
        query = (
            "Question: What outcome is supported?\n\n"
            "Choices:\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta"
        )
        controller.doc_index = FakeSearchIndex(
            [
                (0.91, {"chunk_index": 0}),
                (0.86, {"chunk_index": 1}),
                (0.82, {"chunk_index": 2}),
                (0.79, {"chunk_index": 3}),
            ]
        )
        controller._doc_chunks = ["alpha", "beta", "gamma", "delta"]
        controller._doc_chunk_metadata = [{"chunk_index": idx} for idx in range(4)]
        batch_decisions = [
            {
                "chunk_index": 0,
                "status": "answer_found",
                "memory_update": "alpha proves A",
                "best_choice": "A",
                "best_choice_rationale": "alpha proves A",
                "open_questions": [],
                "needs_more_context": False,
            },
            {"chunk_index": 1, "status": "partial", "memory_update": "beta extra", "needs_more_context": True},
            {"chunk_index": 2, "status": "partial", "memory_update": "gamma extra", "needs_more_context": True},
        ]

        with patch.object(scs, "ITERATIVE_BATCH_MAX_CHUNKS", 3), patch.object(
            scs, "ITERATIVE_BATCH_INPUT_TOKEN_BUDGET", 100000
        ), patch.object(scs, "SCAN_MIN_CHUNK_RATIO", 0.0), patch.object(
            scs, "SCAN_MAX_CHUNK_RATIO", 1.0
        ), patch.object(scs, "SCAN_MIN_CHUNKS", 1), patch.object(
            controller, "_inspect_iterative_chunk_batch", return_value=batch_decisions
        ), patch.object(
            controller, "_finalize_iterative_answer", side_effect=AssertionError("finalizer should not run")
        ), patch.object(controller, "consensus_verify", return_value={"consensus": "AGREED"}), patch.object(
            controller, "store"
        ):
            result = controller._search_iterative(query, top_k=4, rerank_top=1, synthesize=True)

        self.assertEqual(result["answer"], "A")
        self.assertEqual(result["retrieval"]["iterative_scan_inspector_call_count"], 3)
        self.assertEqual(result["retrieval"]["iterative_scan_inspector_llm_call_count"], 1)
        self.assertEqual(result["retrieval"]["iterative_scan_batch_count"], 1)
        self.assertEqual(result["retrieval"]["iterative_scan_batch_sizes"], [3])
        self.assertEqual(result["retrieval"]["iterative_scan_batch_fallback_count"], 0)

    def test_iterative_search_falls_back_to_single_chunk_when_batch_parse_fails(self):
        controller = make_controller()
        query = (
            "Question: What outcome is supported?\n\n"
            "Choices:\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta"
        )
        controller.doc_index = FakeSearchIndex(
            [(0.91, {"chunk_index": 0}), (0.86, {"chunk_index": 1})]
        )
        controller._doc_chunks = ["alpha", "beta"]
        controller._doc_chunk_metadata = [{"chunk_index": idx} for idx in range(2)]
        inspections = [
            {
                "status": "answer_found",
                "memory_update": "alpha proves A",
                "best_choice": "A",
                "best_choice_rationale": "alpha proves A",
                "open_questions": [],
                "needs_more_context": False,
            },
            {"status": "partial", "memory_update": "beta extra", "needs_more_context": True},
        ]

        with patch.object(scs, "ITERATIVE_BATCH_MAX_CHUNKS", 2), patch.object(
            scs, "ITERATIVE_BATCH_INPUT_TOKEN_BUDGET", 100000
        ), patch.object(scs, "SCAN_MIN_CHUNK_RATIO", 0.0), patch.object(
            scs, "SCAN_MAX_CHUNK_RATIO", 1.0
        ), patch.object(scs, "SCAN_MIN_CHUNKS", 1), patch.object(
            controller, "_inspect_iterative_chunk_batch", return_value=None
        ), patch.object(
            controller, "_inspect_iterative_chunk", side_effect=inspections
        ), patch.object(
            controller, "_finalize_iterative_answer", side_effect=AssertionError("finalizer should not run")
        ), patch.object(controller, "consensus_verify", return_value={"consensus": "AGREED"}), patch.object(
            controller, "store"
        ):
            result = controller._search_iterative(query, top_k=2, rerank_top=1, synthesize=True)

        self.assertEqual(result["answer"], "A")
        self.assertEqual(result["retrieval"]["iterative_scan_inspector_call_count"], 2)
        self.assertEqual(result["retrieval"]["iterative_scan_inspector_llm_call_count"], 3)
        self.assertEqual(result["retrieval"]["iterative_scan_batch_count"], 1)
        self.assertEqual(result["retrieval"]["iterative_scan_batch_fallback_count"], 1)

    def test_event_code_ledger_filters_examples_to_current_option_codes(self):
        query = (
            "Question: Only considering the given document, what is the event type of approach?\n\n"
            "Choices:\n"
            "A. aba\n"
            "B. aai\n"
            "C. acd\n"
            "D. aaz\n\n"
            "Return only the single best answer choice letter: A, B, C, or D."
        )
        ledger = scs._new_evidence_ledger(query)
        decision = {
            "status": "partial",
            "code_mappings": [
                "returning -> aaz",
                "blackouts -> aai",
                "hired -> aew",
            ],
            "target_facts": ["Code aaz corresponds to movement."],
        }

        scs._update_evidence_ledger(ledger, decision, chunk_index=5, query=query)

        self.assertEqual([item["code"] for item in ledger["code_mappings"]], ["aaz", "aai"])
        self.assertFalse(any(item["code"] == "aew" for item in ledger["code_mappings"]))
        self.assertFalse(
            any("corresponds" in item["note"].lower() for item in ledger["target_facts"])
        )
        self.assertTrue(
            any("Current answer option symbolic codes" in item["note"] for item in ledger["target_facts"])
        )

    def test_event_code_ledger_requires_mapping_before_accepting_best_choice(self):
        query = (
            "Question: Only considering the given document, what is the event type of approach?\n\n"
            "Choices:\n"
            "A. aba\n"
            "B. aai\n"
            "C. acd\n"
            "D. aaz\n\n"
            "Return only the single best answer choice letter: A, B, C, or D."
        )
        ledger = scs._new_evidence_ledger(query)
        unsupported_decision = {
            "status": "answer_found",
            "best_choice": "D",
            "best_choice_rationale": "aaz is plausible for approach",
            "open_questions": [],
            "needs_more_context": False,
        }

        scs._update_evidence_ledger(ledger, unsupported_decision, chunk_index=0, query=query)

        self.assertIsNone(ledger["best_choice"])
        self.assertEqual(ledger["best_choice_rationale"], "")
        self.assertTrue(any("symbolic code aaz" in note for note in ledger["open_questions"]))

        supported_decision = {
            "status": "answer_found",
            "code_mappings": [{"code": "aaz", "relation": "movement event", "example": "returning -> aaz"}],
            "best_choice": "D",
            "best_choice_rationale": "aaz has an explicit compatible demonstration",
            "open_questions": [],
            "needs_more_context": False,
        }

        scs._update_evidence_ledger(ledger, supported_decision, chunk_index=3, query=query)

        self.assertEqual(ledger["best_choice"], "D")
        self.assertEqual(ledger["best_choice_rationale"], "aaz has an explicit compatible demonstration")

    def test_relation_early_stop_requires_mapping_for_selected_code(self):
        query = (
            "Document: entity0 Astar is a entity1 New Zealand personality.\n\n"
            "Question: Only considering the given document, what is the relation type "
            "between entity0 and entity1?\n\n"
            "Choices:\n"
            "A. aaa\n"
            "B. acy\n"
            "C. abt\n"
            "D. aah\n\n"
            "Return only the single best answer choice letter: A, B, C, or D."
        )
        ledger = scs._new_evidence_ledger(query)
        ledger["best_choice"] = "B"
        decision = {
            "status": "answer_found",
            "best_choice": "B",
            "open_questions": [],
            "needs_more_context": False,
        }

        should_stop, reason = scs._should_stop_iterative_scan(
            decision=decision,
            ledger=ledger,
            visited_count=4,
            min_chunks=1,
            comparative_required_count=4,
            comparative_query=False,
            query=query,
        )

        self.assertFalse(should_stop)
        self.assertEqual(reason, "relation_code_mapping_missing")

        ledger["code_mappings"] = [
            {"chunk_index": 1, "code": "acy", "relation": "nationality", "example": "person-country example"}
        ]

        should_stop, reason = scs._should_stop_iterative_scan(
            decision=decision,
            ledger=ledger,
            visited_count=4,
            min_chunks=1,
            comparative_required_count=4,
            comparative_query=False,
            query=query,
        )

        self.assertTrue(should_stop)
        self.assertEqual(reason, "answer_found")

    def test_event_code_early_stop_requires_mapping_for_selected_code(self):
        query = (
            "Question: Only considering the given document, what is the event type of approach?\n\n"
            "Choices:\n"
            "A. aba\n"
            "B. aai\n"
            "C. acd\n"
            "D. aaz\n\n"
            "Return only the single best answer choice letter: A, B, C, or D."
        )
        ledger = scs._new_evidence_ledger(query)
        ledger["best_choice"] = "D"
        decision = {
            "status": "answer_found",
            "best_choice": "D",
            "open_questions": [],
            "needs_more_context": False,
        }

        should_stop, reason = scs._should_stop_iterative_scan(
            decision=decision,
            ledger=ledger,
            visited_count=4,
            min_chunks=1,
            comparative_required_count=4,
            comparative_query=False,
            query=query,
        )

        self.assertFalse(should_stop)
        self.assertEqual(reason, "symbolic_code_mapping_missing")

        ledger["code_mappings"] = [
            {"chunk_index": 1, "code": "aaz", "relation": "movement event", "example": "returning -> aaz"}
        ]

        should_stop, reason = scs._should_stop_iterative_scan(
            decision=decision,
            ledger=ledger,
            visited_count=4,
            min_chunks=1,
            comparative_required_count=4,
            comparative_query=False,
            query=query,
        )

        self.assertTrue(should_stop)
        self.assertEqual(reason, "answer_found")

    def test_symbolic_code_early_stop_defers_when_competing_mappings_exist(self):
        query = (
            "Document: entity0 Break the Silence is an album by entity2 van Canto.\n\n"
            "Question: Only considering the given document, what is the relation type "
            "between entity0 and entity2?\n\n"
            "Choices:\n"
            "A. abk\n"
            "B. abp\n"
            "C. aaf\n"
            "D. acd\n\n"
            "Return only the single best answer choice letter: A, B, C, or D."
        )
        ledger = scs._new_evidence_ledger(query)
        ledger["best_choice"] = "B"
        ledger["code_mappings"] = [
            {"chunk_index": 1, "code": "abp", "relation": "album by band", "example": "Album X by Band Y"},
            {"chunk_index": 2, "code": "abk", "relation": "album by artist", "example": "Album Z by Artist W"},
        ]
        decision = {
            "status": "answer_found",
            "best_choice": "B",
            "open_questions": [],
            "needs_more_context": False,
        }

        should_stop, reason = scs._should_stop_iterative_scan(
            decision=decision,
            ledger=ledger,
            visited_count=4,
            min_chunks=1,
            comparative_required_count=4,
            comparative_query=False,
            query=query,
        )

        self.assertFalse(should_stop)
        self.assertEqual(reason, "symbolic_code_final_adjudication_required")

    def test_final_decision_requires_relation_code_contrast_when_competing_mappings_exist(self):
        query = (
            "Document: entity0 Break the Silence is an album by entity2 van Canto.\n\n"
            "Question: Only considering the given document, what is the relation type "
            "between entity0 and entity2?\n\n"
            "Choices:\n"
            "A. abk\n"
            "B. abp\n"
            "C. aaf\n"
            "D. acd\n\n"
            "Return only the single best answer choice letter: A, B, C, or D."
        )
        ledger = scs._new_evidence_ledger(query)
        ledger["code_mappings"] = [
            {"chunk_index": 1, "code": "abp", "relation": "album by band", "example": "Album X by Band Y"},
            {"chunk_index": 2, "code": "abk", "relation": "album by artist", "example": "Album Z by Artist W"},
        ]
        incomplete_decision = {
            "answer": "B",
            "reason": "abp maps to album by band",
            "needs_more_context": False,
            "selected_code": "abp",
            "selected_code_evidence": ["Album X by Band Y -> abp"],
        }

        needs_fallback, reason = scs._final_decision_needs_packed_fallback(
            "B",
            incomplete_decision,
            query=query,
            ledger=ledger,
        )

        self.assertTrue(needs_fallback)
        self.assertEqual(reason, "relation_code_contrast_missing")

        complete_decision = {
            **incomplete_decision,
            "rejected_code_evidence": {
                "abk": "abk examples use individual artists rather than band entities."
            },
        }

        needs_fallback, reason = scs._final_decision_needs_packed_fallback(
            "B",
            complete_decision,
            query=query,
            ledger=ledger,
        )

        self.assertFalse(needs_fallback)
        self.assertEqual(reason, "")

    def test_final_decision_requires_ordering_sequence_evidence(self):
        query = (
            "Question: Put the narratives in chronological order.\n\n"
            "Choices:\n"
            "A. 1234\n"
            "B. 4123\n"
            "C. 4213\n"
            "D. 4132\n\n"
            "Return only the single best answer choice letter: A, B, C, or D."
        )
        incomplete_decision = {
            "answer": "B",
            "reason": "4 starts the current timeline.",
            "needs_more_context": False,
            "chosen_sequence": "4123",
            "ordering_evidence": {
                "4": "Current timeline opens first.",
                "1": "First 1974 event.",
                "2": "Later discovery.",
            },
            "pairwise_order": ["4<1", "1<2"],
        }

        needs_fallback, reason = scs._final_decision_needs_packed_fallback(
            "B",
            incomplete_decision,
            query=query,
            ledger={},
        )

        self.assertTrue(needs_fallback)
        self.assertEqual(reason, "ordering_evidence_incomplete")

        pairwise_without_evidence = {
            **incomplete_decision,
            "ordering_evidence": {
                "4": "Current timeline opens first.",
                "1": "May 1974 event.",
                "2": "After the 1974 events.",
                "3": "Summer 1974 event after narrative 1.",
            },
            "pairwise_order": ["4<1", "1<2", "2<3"],
        }

        needs_fallback, reason = scs._final_decision_needs_packed_fallback(
            "B",
            pairwise_without_evidence,
            query=query,
            ledger={},
        )

        self.assertTrue(needs_fallback)
        self.assertEqual(reason, "ordering_evidence_incomplete")

        complete_decision = {
            **incomplete_decision,
            "ordering_evidence": {
                "4": "Current timeline opens first.",
                "1": "May 1974 event.",
                "2": "After the 1974 events.",
                "3": "Summer 1974 event after narrative 1.",
            },
            "pairwise_order": {
                "4<1": "The current framing event precedes the 1974 memories.",
                "1<2": "Narrative 2 occurs after the 1974 discovery.",
                "2<3": "Narrative 3 is later than narrative 2.",
            },
        }

        needs_fallback, reason = scs._final_decision_needs_packed_fallback(
            "B",
            complete_decision,
            query=query,
            ledger={},
        )

        self.assertFalse(needs_fallback)
        self.assertEqual(reason, "")

    def test_final_adjudication_scores_raw_notes_without_best_choice_fallback(self):
        controller = make_controller()
        ledger = scs._new_evidence_ledger()
        ledger["best_choice"] = "D"
        ledger["memory_updates"].append({"note": "Love symptoms are identical to cholera.", "chunk_index": 0})
        ledger["memory"] = "[chunk 0] Love symptoms are identical to cholera."
        captured = {}

        class FakeUsage:
            input_tokens = 10
            output_tokens = 5

        class FakeResponse:
            usage = FakeUsage()
            content = [
                types.SimpleNamespace(
                    text='{"answer":"C","reason":"raw notes favor love danger"}'
                )
            ]

        def fake_create_llm_message(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        with patch.object(scs, "create_llm_message", side_effect=fake_create_llm_message):
            answer, decision = controller._finalize_iterative_answer(
                "Question: What is symbolized?\n\nChoices:\nA. Confusion\nB. Fate\nC. Love is dangerous\nD. Social indifference",
                ledger,
            )

        self.assertEqual(answer, "C")
        self.assertEqual(decision["reason"], "raw notes favor love danger")
        self.assertNotIn("confidence", captured["system"])
        self.assertIn("Treat ledger.best_choice as a prior", captured["system"])
        self.assertIn("ledger_best_choice_prior", captured["messages"][0]["content"])
        self.assertIn("ledger only", captured["messages"][0]["content"])

    def test_final_adjudication_does_not_fallback_to_ledger_best_choice(self):
        controller = make_controller()
        ledger = scs._new_evidence_ledger()
        ledger["best_choice"] = "D"

        class FakeUsage:
            input_tokens = 10
            output_tokens = 5

        class FakeResponse:
            usage = FakeUsage()
            content = [types.SimpleNamespace(text="unparseable")]

        with patch.object(scs, "create_llm_message", return_value=FakeResponse()):
            answer, decision = controller._finalize_iterative_answer(
                "Question: Which option?\n\nChoices:\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta",
                ledger,
            )

        self.assertEqual(answer, "")
        self.assertEqual(decision["answer"], "")

    def test_iterative_packed_fallback_uses_structured_letter_contract(self):
        controller = make_controller()
        captured = {}

        class FakeUsage:
            input_tokens = 10
            output_tokens = 5

        class FakeResponse:
            usage = FakeUsage()
            content = [types.SimpleNamespace(text='{"answer":"C","reason":"packed chunks support C"}')]

        def fake_create_llm_message(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        with patch.object(scs, "MCQ_SYNTHESIS_MAX_TOKENS", 8), patch.object(
            scs, "create_llm_message", side_effect=fake_create_llm_message
        ):
            answer, decision = controller._fallback_iterative_packed_answer(
                "Question: Which option?\n\nChoices:\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta",
                [{"text": "gamma evidence", "metadata": {"chunk_index": 0}}],
                ledger=scs._new_evidence_ledger(),
                reason="ordering_sequence_mismatch",
            )

        self.assertEqual(answer, "C")
        self.assertEqual(decision["answer"], "C")
        self.assertEqual(decision["reason"], "packed chunks support C")
        self.assertEqual(captured["max_tokens"], 32)
        self.assertIn("compact JSON object", captured["system"])

    def test_iterative_packed_fallback_does_not_return_prose_as_answer(self):
        controller = make_controller()

        class FakeUsage:
            input_tokens = 10
            output_tokens = 5

        class FakeResponse:
            usage = FakeUsage()
            content = [types.SimpleNamespace(text="Based on the provided documents and memory")]

        with patch.object(scs, "create_llm_message", return_value=FakeResponse()):
            answer, decision = controller._fallback_iterative_packed_answer(
                "Question: Which option?\n\nChoices:\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta",
                [{"text": "ambiguous evidence", "metadata": {"chunk_index": 0}}],
                ledger=scs._new_evidence_ledger(),
                reason="ordering_sequence_mismatch",
            )

        self.assertEqual(answer, "")
        self.assertEqual(decision["answer"], "")
        self.assertIn("Based on", decision["raw_response"])

    def test_iterative_packed_fallback_budgets_the_complete_prompt(self):
        controller = make_controller()
        captured = {}

        class FakeUsage:
            input_tokens = 10
            output_tokens = 5

        class FakeResponse:
            usage = FakeUsage()
            content = [types.SimpleNamespace(text='{"answer":"C","reason":"supported"}')]

        def fake_create_llm_message(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        ledger = scs._new_evidence_ledger()
        ledger["memory"] = "remembered evidence " * 100
        budget = 2000
        with patch.object(scs, "ITERATIVE_PACKED_FALLBACK_INPUT_TOKEN_BUDGET", budget), patch.object(
            scs, "create_llm_message", side_effect=fake_create_llm_message
        ):
            controller._fallback_iterative_packed_answer(
                "Question: Which option?\n\nChoices:\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta",
                [{"text": "gamma evidence " * 2000, "metadata": {"chunk_index": 0}}],
                ledger=ledger,
                reason="final_adjudication_needs_more_context",
            )

        complete_prompt = f"{captured['system']}\n\n{captured['messages'][0]['content']}"
        self.assertLessEqual(scs._estimate_llm_input_tokens(complete_prompt), budget)

    def test_iterative_comparative_query_requires_scan_budget_before_stop(self):
        ledger = scs._new_evidence_ledger()
        decision = {
            "status": "answer_found",
            "memory_update": "C has the best trade-off.",
            "best_choice": "C",
            "open_questions": [],
            "needs_more_context": False,
        }
        scs._update_evidence_ledger(ledger, decision, chunk_index=0)

        should_stop, reason = scs._should_stop_iterative_scan(
            decision=decision,
            ledger=ledger,
            visited_count=2,
            min_chunks=1,
            comparative_required_count=3,
            comparative_query=True,
        )

        self.assertFalse(should_stop)
        self.assertEqual(reason, "comparative_scan_budget_not_complete")

    def test_iterative_search_early_stops_and_stores_compact_context(self):
        controller = make_controller()
        controller.doc_index = FakeSearchIndex(
            [(0.91, {"chunk_index": 0}), (0.86, {"chunk_index": 1}), (0.80, {"chunk_index": 2})]
        )
        controller._doc_chunks = ["alpha evidence", "beta evidence", "gamma evidence"]
        controller._doc_chunk_metadata = [{"chunk_index": idx} for idx in range(3)]
        inspections = [
            {
                "status": "partial",
                "memory_update": "alpha is relevant but incomplete",
                "best_choice": None,
                "open_questions": ["need beta"],
                "needs_more_context": True,
            },
            {
                "status": "answer_found",
                "memory_update": "beta evidence supports B",
                "best_choice": "B",
                "best_choice_rationale": "beta evidence supports B",
                "open_questions": [],
                "needs_more_context": False,
            },
        ]
        stored = {}

        def fake_store(query, context, result, model_used="unknown", sources=None, consensus_info=None):
            stored["query"] = query
            stored["context"] = context
            stored["result"] = result
            stored["sources"] = sources
            stored["consensus_info"] = consensus_info

        with patch.object(scs, "SCAN_MIN_CHUNK_RATIO", 0.0), patch.object(
            scs, "SCAN_MAX_CHUNK_RATIO", 0.0
        ), patch.object(scs, "SCAN_MIN_CHUNKS", 2), patch.object(
            scs, "SCAN_MAX_CHUNKS", 0
        ), patch.object(controller, "_inspect_iterative_chunk", side_effect=inspections), patch.object(
            controller, "_finalize_iterative_answer", side_effect=AssertionError("finalizer should not run")
        ), patch.object(controller, "consensus_verify", return_value={"consensus": "AGREED"}), patch.object(
            controller, "store", side_effect=fake_store
        ):
            result = controller._search_iterative("What answer is correct?", top_k=2, rerank_top=1, synthesize=True)

        self.assertEqual(result["answer"], "B")
        self.assertTrue(result["retrieval"]["iterative_scan_early_stop"])
        self.assertEqual(result["retrieval"]["iterative_scan_inspector_call_count"], 2)
        self.assertEqual(result["retrieval"]["iterative_scan_final_adjudication_call_count"], 0)
        self.assertEqual(stored["result"], "B")
        self.assertEqual(stored["consensus_info"], {"consensus": "AGREED"})
        self.assertIn("evidence_ledger", stored["context"])

    def test_iterative_search_runs_final_adjudication_when_no_early_stop(self):
        controller = make_controller()
        controller.doc_index = FakeSearchIndex(
            [(0.91, {"chunk_index": 0}), (0.86, {"chunk_index": 1})]
        )
        controller._doc_chunks = ["alpha evidence", "beta evidence"]
        controller._doc_chunk_metadata = [{"chunk_index": idx} for idx in range(2)]
        inspections = [
            {
                "status": "partial",
                "memory_update": "alpha is relevant but incomplete",
                "best_choice": None,
                "open_questions": ["need beta"],
                "needs_more_context": True,
            },
            {
                "status": "partial",
                "memory_update": "beta is relevant but incomplete",
                "best_choice": None,
                "open_questions": ["need final adjudication"],
                "needs_more_context": True,
            },
        ]

        with patch.object(scs, "SCAN_MIN_CHUNK_RATIO", 0.0), patch.object(
            scs, "SCAN_MAX_CHUNK_RATIO", 0.0
        ), patch.object(scs, "SCAN_MIN_CHUNKS", 1), patch.object(
            scs, "SCAN_MAX_CHUNKS", 0
        ), patch.object(controller, "_inspect_iterative_chunk", side_effect=inspections), patch.object(
            controller, "_finalize_iterative_answer", return_value=("C", {"answer": "C", "reason": "ledger supports C"})
        ), patch.object(
            controller, "_fallback_iterative_packed_answer", side_effect=AssertionError("packed fallback should not run")
        ), patch.object(controller, "consensus_verify", return_value={"consensus": "AGREED"}), patch.object(
            controller, "store"
        ):
            result = controller._search_iterative("Which option is correct?", top_k=2, rerank_top=1, synthesize=True)

        self.assertEqual(result["answer"], "C")
        self.assertFalse(result["retrieval"]["iterative_scan_early_stop"])
        self.assertEqual(result["retrieval"]["iterative_scan_final_adjudication_call_count"], 1)

    def test_iterative_search_scans_more_when_initial_budget_has_empty_ledger(self):
        controller = make_controller()
        controller.doc_index = FakeSearchIndex(
            [
                (0.91, {"chunk_index": 0}),
                (0.86, {"chunk_index": 1}),
                (0.82, {"chunk_index": 2}),
                (0.79, {"chunk_index": 3}),
            ]
        )
        controller._doc_chunks = ["empty one", "empty two", "useful rule", "extra"]
        controller._doc_chunk_metadata = [{"chunk_index": idx} for idx in range(4)]
        inspections = [
            {"status": "no_update", "best_choice": None, "needs_more_context": True},
            {"status": "no_update", "best_choice": None, "needs_more_context": True},
            {
                "status": "partial",
                "memory_update": "The target entity relation appears in this chunk. Relation code abb maps to location containment in examples.",
                "target_facts": ["The target entity relation appears in this chunk."],
                "code_mappings": [{"code": "abb", "relation": "location containment", "example": "Entity0 -> abb"}],
                "best_choice": None,
                "needs_more_context": True,
            },
        ]

        with patch.object(scs, "SCAN_MIN_CHUNK_RATIO", 0.0), patch.object(
            scs, "SCAN_MAX_CHUNK_RATIO", 0.50
        ), patch.object(scs, "SCAN_MIN_CHUNKS", 1), patch.object(
            scs, "SCAN_MAX_CHUNKS", 0
        ), patch.object(scs, "SCAN_EMPTY_LEDGER_FALLBACK_RATIO", 1.0), patch.object(
            controller, "_inspect_iterative_chunk", side_effect=inspections
        ), patch.object(
            controller, "_finalize_iterative_answer", return_value=("C", {"answer": "C", "reason": "ledger supports C"})
        ), patch.object(
            controller, "_fallback_iterative_packed_answer", side_effect=AssertionError("packed fallback should not run")
        ), patch.object(controller, "consensus_verify", return_value={"consensus": "AGREED"}), patch.object(
            controller, "store"
        ):
            result = controller._search_iterative("Which relation type is correct?", top_k=2, rerank_top=1, synthesize=True)

        self.assertEqual(result["answer"], "C")
        self.assertEqual(result["retrieval"]["iterative_scan_budget"], 2)
        self.assertEqual(result["retrieval"]["iterative_scan_empty_ledger_fallback_budget"], 4)
        self.assertTrue(result["retrieval"]["iterative_scan_empty_ledger_fallback_used"])
        self.assertEqual(result["retrieval"]["iterative_scan_inspector_call_count"], 3)
        self.assertEqual(result["retrieval"]["iterative_scan_memory_update_count"], 1)
        self.assertEqual(result["retrieval"]["iterative_scan_target_fact_count"], 2)
        self.assertEqual(result["retrieval"]["iterative_scan_code_mapping_count"], 1)
        self.assertEqual(result["retrieval"]["iterative_scan_packed_fallback_call_count"], 0)

    def test_iterative_search_extends_scan_for_symbolic_code_ambiguity(self):
        controller = make_controller()
        controller.doc_index = FakeSearchIndex(
            [
                (0.91, {"chunk_index": 0}),
                (0.86, {"chunk_index": 1}),
                (0.82, {"chunk_index": 2}),
                (0.79, {"chunk_index": 3}),
            ]
        )
        controller._doc_chunks = ["abk examples", "abp examples", "other examples", "more abp examples"]
        controller._doc_chunk_metadata = [{"chunk_index": idx} for idx in range(4)]
        inspections = [
            {
                "status": "partial",
                "memory_update": "album-by examples for abk",
                "code_mappings": [{"code": "abk", "relation": "album by", "example": "Album A by Artist B"}],
                "best_choice": "A",
                "needs_more_context": True,
            },
            {
                "status": "partial",
                "memory_update": "album-by examples for abp",
                "code_mappings": [{"code": "abp", "relation": "album by", "example": "Album C by Band D"}],
                "best_choice": None,
                "open_questions": ["Need to resolve abk versus abp."],
                "needs_more_context": True,
            },
            {
                "status": "partial",
                "memory_update": "unrelated examples",
                "best_choice": None,
                "needs_more_context": True,
            },
            {
                "status": "partial",
                "memory_update": "more compatible abp examples",
                "code_mappings": [{"code": "abp", "relation": "album by band", "example": "Album E by Band F"}],
                "best_choice": None,
                "needs_more_context": True,
            },
        ]
        query = (
            "Document: entity0 Break the Silence is an album by entity2 van Canto.\n\n"
            "Question: Only considering the given document, what is the relation type "
            "between entity0 and entity2?\n\n"
            "Choices:\n"
            "A. abk\n"
            "B. abp\n"
            "C. aaf\n"
            "D. acd\n\n"
            "Return only the single best answer choice letter: A, B, C, or D."
        )

        final_decision = {
            "answer": "B",
            "reason": "abp has the compatible band examples",
            "needs_more_context": False,
            "selected_code": "abp",
            "selected_code_evidence": "abp maps album-by band examples.",
            "rejected_code_evidence": {"abk": "abk examples are less compatible with the target band relation."},
        }

        with patch.object(scs, "SCAN_MIN_CHUNK_RATIO", 0.0), patch.object(
            scs, "SCAN_MAX_CHUNK_RATIO", 0.50
        ), patch.object(scs, "SCAN_MIN_CHUNKS", 1), patch.object(
            scs, "SCAN_MAX_CHUNKS", 0
        ), patch.object(scs, "SCAN_EMPTY_LEDGER_FALLBACK_RATIO", 1.0), patch.object(
            controller, "_inspect_iterative_chunk", side_effect=inspections
        ), patch.object(
            controller, "_finalize_iterative_answer", return_value=("B", final_decision)
        ), patch.object(
            controller, "_fallback_iterative_packed_answer", side_effect=AssertionError("packed fallback should not run")
        ), patch.object(controller, "consensus_verify", return_value={"consensus": "AGREED"}), patch.object(
            controller, "store"
        ):
            result = controller._search_iterative(query, top_k=4, rerank_top=1, synthesize=True)

        self.assertEqual(result["answer"], "B")
        self.assertEqual(result["retrieval"]["iterative_scan_budget"], 2)
        self.assertEqual(result["retrieval"]["iterative_scan_inspector_call_count"], 4)
        self.assertTrue(result["retrieval"]["iterative_scan_extra_scan_used"])
        self.assertEqual(
            result["retrieval"]["iterative_scan_extra_scan_reason"],
            "symbolic_code_extra_scan_required",
        )
        self.assertEqual(result["retrieval"]["iterative_scan_extra_scan_chunk_count"], 2)

    def test_iterative_search_uses_packed_fallback_when_ledger_stays_empty(self):
        controller = make_controller()
        controller.doc_index = FakeSearchIndex(
            [(0.91, {"chunk_index": 0}), (0.86, {"chunk_index": 1})]
        )
        controller._doc_chunks = ["empty one", "empty two"]
        controller._doc_chunk_metadata = [{"chunk_index": idx} for idx in range(2)]
        inspections = [
            {"status": "no_update", "best_choice": None, "needs_more_context": True},
            {"status": "no_update", "best_choice": None, "needs_more_context": True},
        ]

        with patch.object(scs, "SCAN_MIN_CHUNK_RATIO", 0.0), patch.object(
            scs, "SCAN_MAX_CHUNK_RATIO", 0.50
        ), patch.object(scs, "SCAN_MIN_CHUNKS", 1), patch.object(
            scs, "SCAN_MAX_CHUNKS", 0
        ), patch.object(scs, "SCAN_EMPTY_LEDGER_FALLBACK_RATIO", 1.0), patch.object(
            controller, "_inspect_iterative_chunk", side_effect=inspections
        ), patch.object(
            controller, "_finalize_iterative_answer", side_effect=AssertionError("empty ledger finalizer should not run")
        ), patch.object(
            controller, "_fallback_iterative_packed_answer", return_value=("D", {"answer": "D"})
        ), patch.object(controller, "consensus_verify", return_value={"consensus": "AGREED"}), patch.object(
            controller, "store"
        ):
            result = controller._search_iterative("Which relation type is correct?", top_k=2, rerank_top=1, synthesize=True)

        self.assertEqual(result["answer"], "D")
        self.assertEqual(result["retrieval"]["iterative_scan_stop_reason"], "empty_ledger_packed_fallback")
        self.assertTrue(result["retrieval"]["iterative_scan_empty_ledger_fallback_used"])
        self.assertTrue(result["retrieval"]["iterative_scan_packed_fallback_used"])
        self.assertEqual(result["retrieval"]["iterative_scan_packed_fallback_call_count"], 1)

    def test_iterative_search_uses_final_answer_without_confidence(self):
        controller = make_controller()
        controller.doc_index = FakeSearchIndex(
            [(0.91, {"chunk_index": 0}), (0.86, {"chunk_index": 1})]
        )
        controller._doc_chunks = ["partial one", "partial two"]
        controller._doc_chunk_metadata = [{"chunk_index": idx} for idx in range(2)]
        inspections = [
            {
                "status": "partial",
                "memory_update": "alpha evidence is relevant but incomplete",
                "best_choice": None,
                "needs_more_context": True,
            },
            {
                "status": "partial",
                "memory_update": "beta evidence is relevant but incomplete",
                "best_choice": None,
                "needs_more_context": True,
            },
        ]

        with patch.object(scs, "SCAN_MIN_CHUNK_RATIO", 0.0), patch.object(
            scs, "SCAN_MAX_CHUNK_RATIO", 0.0
        ), patch.object(scs, "SCAN_MIN_CHUNKS", 1), patch.object(
            scs, "SCAN_MAX_CHUNKS", 0
        ), patch.object(controller, "_inspect_iterative_chunk", side_effect=inspections), patch.object(
            controller,
            "_finalize_iterative_answer",
            return_value=("B", {"answer": "B", "reason": "ledger notes support B"}),
        ), patch.object(
            controller, "_fallback_iterative_packed_answer", side_effect=AssertionError("packed fallback should not run")
        ) as packed_fallback, patch.object(
            controller, "consensus_verify", return_value={"consensus": "AGREED"}
        ), patch.object(
            controller, "store"
        ):
            result = controller._search_iterative("Which option is correct?", top_k=2, rerank_top=1, synthesize=True)

        self.assertEqual(result["answer"], "B")
        self.assertEqual(result["retrieval"]["iterative_scan_final_adjudication_call_count"], 1)
        self.assertEqual(result["retrieval"]["iterative_scan_packed_fallback_call_count"], 0)
        self.assertEqual(result["retrieval"]["iterative_scan_final_answer"], "B")
        self.assertEqual(result["retrieval"]["iterative_scan_final_reason"], "ledger notes support B")
        self.assertIsNone(result["retrieval"]["iterative_scan_final_raw_response"])
        packed_fallback.assert_not_called()

    def test_iterative_search_uses_packed_fallback_when_relation_final_lacks_mapping(self):
        controller = make_controller()
        controller.doc_index = FakeSearchIndex(
            [(0.91, {"chunk_index": 0}), (0.86, {"chunk_index": 1})]
        )
        controller._doc_chunks = ["target relation evidence", "more target evidence"]
        controller._doc_chunk_metadata = [{"chunk_index": idx} for idx in range(2)]
        inspections = [
            {
                "status": "partial",
                "memory_update": "The target relation looks like nationality, but no code demonstration is present.",
                "target_facts": ["entity0 Astar is a entity1 New Zealand personality."],
                "best_choice": None,
                "needs_more_context": True,
            },
            {
                "status": "partial",
                "memory_update": "Still no explicit demonstration for the candidate codes.",
                "best_choice": None,
                "needs_more_context": True,
            },
        ]
        query = (
            "Document: entity0 Astar is a entity1 New Zealand personality.\n\n"
            "Question: Only considering the given document, what is the relation type "
            "between entity0 and entity1?\n\n"
            "Choices:\n"
            "A. aaa\n"
            "B. acy\n"
            "C. abt\n"
            "D. aah\n\n"
            "Return only the single best answer choice letter: A, B, C, or D."
        )

        with patch.object(scs, "SCAN_MIN_CHUNK_RATIO", 0.0), patch.object(
            scs, "SCAN_MAX_CHUNK_RATIO", 0.0
        ), patch.object(scs, "SCAN_MIN_CHUNKS", 1), patch.object(
            scs, "SCAN_MAX_CHUNKS", 0
        ), patch.object(controller, "_inspect_iterative_chunk", side_effect=inspections), patch.object(
            controller,
            "_finalize_iterative_answer",
            return_value=("B", {"answer": "B", "reason": "nationality maps to acy"}),
        ), patch.object(
            controller, "_fallback_iterative_packed_answer", return_value=("D", {"answer": "D"})
        ) as packed_fallback, patch.object(
            controller, "consensus_verify", return_value={"consensus": "AGREED"}
        ), patch.object(
            controller, "store"
        ):
            result = controller._search_iterative(query, top_k=2, rerank_top=1, synthesize=True)

        self.assertEqual(result["answer"], "D")
        self.assertEqual(result["retrieval"]["iterative_scan_final_adjudication_call_count"], 1)
        self.assertEqual(result["retrieval"]["iterative_scan_packed_fallback_call_count"], 1)
        self.assertEqual(
            result["retrieval"]["iterative_scan_packed_fallback_reason"],
            "relation_code_mapping_missing",
        )
        self.assertEqual(result["retrieval"]["iterative_scan_stop_reason"], "relation_code_mapping_missing")
        self.assertEqual(result["retrieval"]["iterative_scan_final_answer"], "B")
        packed_fallback.assert_called_once()

    def test_iterative_search_uses_packed_fallback_when_event_code_final_lacks_mapping(self):
        controller = make_controller()
        controller.doc_index = FakeSearchIndex(
            [(0.91, {"chunk_index": 0}), (0.86, {"chunk_index": 1})]
        )
        controller._doc_chunks = ["target event document", "many-shot examples without target code"]
        controller._doc_chunk_metadata = [{"chunk_index": idx} for idx in range(2)]
        inspections = [
            {
                "status": "partial",
                "memory_update": "The target event is approach, but no matching code demo appears.",
                "target_facts": ["approach is the target event."],
                "best_choice": None,
                "needs_more_context": True,
            },
            {
                "status": "partial",
                "memory_update": "Examples contain other option codes only.",
                "code_mappings": [{"code": "aai", "relation": "blackouts", "example": "blackouts -> aai"}],
                "best_choice": None,
                "needs_more_context": True,
            },
        ]
        query = (
            "Question: Only considering the given document, what is the event type of approach?\n\n"
            "Choices:\n"
            "A. aba\n"
            "B. aai\n"
            "C. acd\n"
            "D. aaz\n\n"
            "Return only the single best answer choice letter: A, B, C, or D."
        )

        with patch.object(scs, "SCAN_MIN_CHUNK_RATIO", 0.0), patch.object(
            scs, "SCAN_MAX_CHUNK_RATIO", 0.0
        ), patch.object(scs, "SCAN_MIN_CHUNKS", 1), patch.object(
            scs, "SCAN_MAX_CHUNKS", 0
        ), patch.object(controller, "_inspect_iterative_chunk", side_effect=inspections), patch.object(
            controller,
            "_finalize_iterative_answer",
            return_value=("D", {"answer": "D", "reason": "aaz seems plausible"}),
        ), patch.object(
            controller, "_fallback_iterative_packed_answer", return_value=("B", {"answer": "B"})
        ) as packed_fallback, patch.object(
            controller, "consensus_verify", return_value={"consensus": "AGREED"}
        ), patch.object(
            controller, "store"
        ):
            result = controller._search_iterative(query, top_k=2, rerank_top=1, synthesize=True)

        self.assertEqual(result["answer"], "B")
        self.assertEqual(result["retrieval"]["iterative_scan_final_adjudication_call_count"], 1)
        self.assertEqual(result["retrieval"]["iterative_scan_packed_fallback_call_count"], 1)
        self.assertEqual(
            result["retrieval"]["iterative_scan_packed_fallback_reason"],
            "symbolic_code_mapping_missing",
        )
        self.assertEqual(result["retrieval"]["iterative_scan_stop_reason"], "symbolic_code_mapping_missing")
        self.assertEqual(result["retrieval"]["iterative_scan_final_answer"], "D")
        packed_fallback.assert_called_once()

    def test_iterative_ordering_query_scans_faiss_subset_in_document_order(self):
        controller = make_controller()
        controller.doc_index = FakeSearchIndex(
            [
                (0.91, {"chunk_index": 2}),
                (0.86, {"chunk_index": 0}),
                (0.82, {"chunk_index": 1}),
            ]
        )
        controller._doc_chunks = ["first narrative evidence", "second narrative evidence", "third narrative evidence"]
        controller._doc_chunk_metadata = [{"chunk_index": idx} for idx in range(3)]
        seen_chunk_indices = []

        def fake_inspect(query, ledger, result):
            seen_chunk_indices.append(result["metadata"]["chunk_index"])
            return {
                "status": "partial",
                "memory_update": f"visited chunk {result['metadata']['chunk_index']}",
                "best_choice": None,
                "needs_more_context": True,
            }

        query = (
            "Question: Which order of the narratives is correct?\n\n"
            "Choices:\n"
            "A. 123\n"
            "B. 132\n"
            "C. 213\n"
            "D. 321\n\n"
            "Return only the single best answer choice letter: A, B, C, or D."
        )

        with patch.object(scs, "SCAN_MIN_CHUNK_RATIO", 1.0), patch.object(
            scs, "SCAN_MAX_CHUNK_RATIO", 1.0
        ), patch.object(scs, "SCAN_MIN_CHUNKS", 1), patch.object(
            scs, "SCAN_MAX_CHUNKS", 0
        ), patch.object(controller, "_inspect_iterative_chunk", side_effect=fake_inspect), patch.object(
            controller,
            "_finalize_iterative_answer",
            return_value=(
                "A",
                {
                    "answer": "A",
                    "reason": "ordered",
                    "chosen_sequence": "123",
                    "ordering_evidence": {
                        "1": "first narrative evidence",
                        "2": "second narrative evidence",
                        "3": "third narrative evidence",
                    },
                    "pairwise_order": {
                        "1<2": "first narrative evidence precedes second narrative evidence",
                        "2<3": "second narrative evidence precedes third narrative evidence",
                    },
                },
            ),
        ), patch.object(
            controller, "_fallback_iterative_packed_answer", side_effect=AssertionError("packed fallback should not run")
        ), patch.object(controller, "consensus_verify", return_value={"consensus": "AGREED"}), patch.object(
            controller, "store"
        ):
            result = controller._search_iterative(query, top_k=3, rerank_top=1, synthesize=True)

        self.assertEqual(seen_chunk_indices, [0, 1, 2])
        self.assertEqual(result["answer"], "A")

    def test_iterative_search_uses_packed_fallback_when_final_needs_more_context(self):
        controller = make_controller()
        controller.doc_index = FakeSearchIndex(
            [(0.91, {"chunk_index": 0}), (0.86, {"chunk_index": 1})]
        )
        controller._doc_chunks = ["partial one", "partial two"]
        controller._doc_chunk_metadata = [{"chunk_index": idx} for idx in range(2)]
        inspections = [
            {
                "status": "partial",
                "memory_update": "alpha evidence is relevant but incomplete",
                "best_choice": None,
                "needs_more_context": True,
            },
            {
                "status": "partial",
                "memory_update": "beta evidence is relevant but incomplete",
                "best_choice": None,
                "needs_more_context": True,
            },
        ]

        with patch.object(scs, "SCAN_MIN_CHUNK_RATIO", 0.0), patch.object(
            scs, "SCAN_MAX_CHUNK_RATIO", 0.0
        ), patch.object(scs, "SCAN_MIN_CHUNKS", 1), patch.object(
            scs, "SCAN_MAX_CHUNKS", 0
        ), patch.object(controller, "_inspect_iterative_chunk", side_effect=inspections), patch.object(
            controller,
            "_finalize_iterative_answer",
            return_value=(
                "B",
                {
                    "answer": "B",
                    "reason": "ledger notes are not enough",
                    "needs_more_context": True,
                    "raw_response": '{"answer":"B","reason":"ledger notes are not enough","needs_more_context":true}',
                },
            ),
        ), patch.object(
            controller, "_fallback_iterative_packed_answer", return_value=("C", {"answer": "C"})
        ) as packed_fallback, patch.object(
            controller, "consensus_verify", return_value={"consensus": "AGREED"}
        ), patch.object(
            controller, "store"
        ):
            result = controller._search_iterative("Which option is correct?", top_k=2, rerank_top=1, synthesize=True)

        self.assertEqual(result["answer"], "C")
        self.assertEqual(result["retrieval"]["iterative_scan_final_adjudication_call_count"], 1)
        self.assertEqual(result["retrieval"]["iterative_scan_packed_fallback_call_count"], 1)
        self.assertEqual(
            result["retrieval"]["iterative_scan_packed_fallback_reason"],
            "final_adjudication_needs_more_context",
        )
        self.assertEqual(result["retrieval"]["iterative_scan_stop_reason"], "final_adjudication_needs_more_context")
        self.assertEqual(result["retrieval"]["iterative_scan_final_answer"], "B")
        self.assertEqual(result["retrieval"]["iterative_scan_final_reason"], "ledger notes are not enough")
        self.assertIn("needs_more_context", result["retrieval"]["iterative_scan_final_raw_response"])
        packed_fallback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
