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
                self.assertTrue(legacy.load(legacy_path))
                result = legacy.search(query)
                self.assertFalse(result["from_cache"])
                self.assertEqual(result["answer"], "No relevant documents found.")

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
            )

        self.assertNotEqual(char_scope, token_scope)
        self.assertNotEqual(token_scope, changed_token_scope)

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

    def test_iterative_early_stop_requires_high_confidence_and_context_satisfied(self):
        ledger = scs._new_evidence_ledger()
        decision = {
            "status": "answer_found",
            "supported_choice": "B",
            "confidence": "high",
            "evidence": ["B is directly supported"],
            "contradictions": [],
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
            (True, "high_confidence_answer"),
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
        contradiction_decision = dict(decision, contradictions=["Another chunk rejects B"])
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
            (False, "unresolved_contradictions"),
        )

    def test_iterative_evidence_ledger_is_bounded_and_preserves_chunk_references(self):
        ledger = scs._new_evidence_ledger()
        decision = {
            "status": "answer_found",
            "supported_choice": "A",
            "confidence": "high",
            "evidence": [f"support note {idx}" for idx in range(10)],
            "contradictions": [f"against note {idx}" for idx in range(10)],
            "observations": [f"observation {idx}" for idx in range(20)],
            "rules": [f"rule {idx}" for idx in range(20)],
            "examples": [f"example {idx}" for idx in range(20)],
            "open_questions": [f"open question {idx}" for idx in range(12)],
        }

        scs._update_evidence_ledger(ledger, decision, chunk_index=7)

        self.assertEqual(len(ledger["A"]["support"]), 5)
        self.assertEqual(len(ledger["A"]["against"]), 5)
        self.assertEqual(len(ledger["observations"]), 12)
        self.assertEqual(len(ledger["rules"]), 12)
        self.assertEqual(len(ledger["examples"]), 12)
        self.assertEqual(len(ledger["open_questions"]), 8)
        self.assertEqual({note["chunk_index"] for note in ledger["A"]["support"]}, {7})
        self.assertEqual({note["chunk_index"] for note in ledger["observations"]}, {7})
        self.assertEqual(ledger["visited_chunks"], [7])
        self.assertTrue(scs._ledger_has_useful_memory(ledger))

    def test_iterative_evidence_ledger_records_parse_failures(self):
        ledger = scs._new_evidence_ledger()

        scs._update_evidence_ledger(
            ledger,
            {
                "status": "no_evidence",
                "parse_failed": True,
                "raw_response": "not-json " * 200,
            },
            chunk_index=3,
        )

        self.assertFalse(scs._ledger_has_useful_memory(ledger))
        self.assertEqual(len(ledger["parse_failures"]), 1)
        self.assertEqual(ledger["parse_failures"][0]["chunk_index"], 3)
        self.assertLessEqual(len(ledger["parse_failures"][0]["raw_response"]), 800)

    def test_iterative_inspector_uses_compact_json_contract(self):
        controller = make_controller()
        captured = {}

        class FakeUsage:
            input_tokens = 10
            output_tokens = 5

        class FakeResponse:
            usage = FakeUsage()
            content = [types.SimpleNamespace(text='{"status":"no_evidence","supported_choice":null}')]

        def fake_create_llm_message(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        with patch.object(scs, "SCAN_MAX_TOKENS", 768), patch.object(
            scs, "create_llm_message", side_effect=fake_create_llm_message
        ):
            parsed = controller._inspect_iterative_chunk(
                "Document: entity0 A is entity1 B.\nQuestion: relation type between entity0 and entity1?",
                scs._new_evidence_ledger(),
                {"text": "chunk text", "metadata": {"chunk_index": 4}},
            )

        self.assertEqual(parsed["chunk_index"], 4)
        self.assertEqual(captured["max_tokens"], 768)
        self.assertIn("Return ONLY one valid compact JSON object", captured["system"])
        self.assertIn("at most three items each", captured["system"])
        self.assertIn("twenty words or fewer", captured["system"])
        self.assertIn("never replace them with entities from demonstrations", captured["system"])

    def test_iterative_comparative_query_requires_scan_budget_before_stop(self):
        ledger = scs._new_evidence_ledger()
        decision = {
            "status": "answer_found",
            "supported_choice": "C",
            "confidence": "high",
            "evidence": ["C has the best trade-off"],
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
            {"status": "partial", "supported_choice": None, "confidence": "low", "needs_more_context": True},
            {
                "status": "answer_found",
                "supported_choice": "B",
                "confidence": "high",
                "evidence": ["beta evidence supports B"],
                "contradictions": [],
                "needs_more_context": False,
            },
        ]
        stored = {}

        def fake_store(query, context, result, model_used="unknown", sources=None):
            stored["query"] = query
            stored["context"] = context
            stored["result"] = result
            stored["sources"] = sources

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
                "supported_choice": None,
                "confidence": "low",
                "observations": ["alpha is relevant but incomplete"],
                "needs_more_context": True,
            },
            {
                "status": "partial",
                "supported_choice": None,
                "confidence": "low",
                "observations": ["beta is relevant but incomplete"],
                "needs_more_context": True,
            },
        ]

        with patch.object(scs, "SCAN_MIN_CHUNK_RATIO", 0.0), patch.object(
            scs, "SCAN_MAX_CHUNK_RATIO", 0.0
        ), patch.object(scs, "SCAN_MIN_CHUNKS", 1), patch.object(
            scs, "SCAN_MAX_CHUNKS", 0
        ), patch.object(controller, "_inspect_iterative_chunk", side_effect=inspections), patch.object(
            controller, "_finalize_iterative_answer", return_value=("C", {"answer": "C"})
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
            {"status": "no_evidence", "supported_choice": None, "confidence": "low", "needs_more_context": True},
            {"status": "no_evidence", "supported_choice": None, "confidence": "low", "needs_more_context": True},
            {
                "status": "partial",
                "supported_choice": None,
                "confidence": "medium",
                "observations": ["The target entity relation appears in this chunk."],
                "rules": ["Relation code abb maps to location containment in examples."],
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
            controller, "_finalize_iterative_answer", return_value=("C", {"answer": "C"})
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
        self.assertEqual(result["retrieval"]["iterative_scan_observation_count"], 1)
        self.assertEqual(result["retrieval"]["iterative_scan_rule_count"], 1)
        self.assertEqual(result["retrieval"]["iterative_scan_packed_fallback_call_count"], 0)

    def test_iterative_search_uses_packed_fallback_when_ledger_stays_empty(self):
        controller = make_controller()
        controller.doc_index = FakeSearchIndex(
            [(0.91, {"chunk_index": 0}), (0.86, {"chunk_index": 1})]
        )
        controller._doc_chunks = ["empty one", "empty two"]
        controller._doc_chunk_metadata = [{"chunk_index": idx} for idx in range(2)]
        inspections = [
            {"status": "no_evidence", "supported_choice": None, "confidence": "low", "needs_more_context": True},
            {"status": "no_evidence", "supported_choice": None, "confidence": "low", "needs_more_context": True},
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


if __name__ == "__main__":
    unittest.main()
