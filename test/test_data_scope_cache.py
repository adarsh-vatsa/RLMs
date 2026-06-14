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

    @property
    def metadata(self):
        if self.added:
            return self.added
        return [meta for _, meta in self.results]

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


def make_entry(query, result, scope=None, answer_metadata=None):
    entry = {
        "query": query,
        "result": result,
        "embedding": np.array([1.0, 0.0], dtype="float32"),
        "source_context": result,
        "grounding_info": {},
        "answer_metadata": answer_metadata or {},
    }
    if scope is not None:
        entry["data_scope_hash"] = scope
    return entry


class DataScopedSearchCacheTests(unittest.TestCase):
    def setUp(self):
        self._cache_write_policy = scs.CACHE_WRITE_POLICY
        self._adaptive_reranker = scs.ADAPTIVE_RERANKER

    def tearDown(self):
        scs.CACHE_WRITE_POLICY = self._cache_write_policy
        scs.ADAPTIVE_RERANKER = self._adaptive_reranker

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

    def test_verified_policy_skips_unverified_mcq_cache_hits(self):
        query = "Question: Pick one.\n\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta"
        scs.CACHE_WRITE_POLICY = "verified"
        controller = make_controller()
        controller.cache = {
            "chunk-a": [make_entry(query, "A", "scope-a", {"verification_status": "DIRECT_UNVERIFIED"})],
        }
        controller.data_scope_hash = "scope-a"

        result = controller.search(query)

        self.assertFalse(result["from_cache"])
        self.assertEqual(result["answer"], "No relevant documents found.")

    def test_verified_policy_allows_verified_mcq_cache_hits(self):
        query = "Question: Pick one.\n\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta"
        scs.CACHE_WRITE_POLICY = "verified"
        controller = make_controller()
        controller.cache = {
            "chunk-a": [make_entry(query, "B", "scope-a", {"verification_status": "VERIFIED", "executor_choice": "B"})],
        }
        controller.data_scope_hash = "scope-a"

        result = controller.search(query)

        self.assertTrue(result["from_cache"])
        self.assertEqual(result["answer"], "B")
        self.assertEqual(result["verification_status"], "VERIFIED")

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

    def test_adaptive_reranker_skips_non_competitive_candidate_sets(self):
        scs.ADAPTIVE_RERANKER = True
        controller = make_controller()
        meta = {"filename": "contract.txt", "chunk_index": 0}
        controller.doc_index = FakeSearchIndex([(0.91, meta)])
        controller._doc_chunks = ["The contract is governed by New York law."]
        controller.reranker = FakeReranker([(0, 0.99, "The contract is governed by New York law.")])

        results = controller.retrieve("What law governs the contract?", top_k=5, rerank_top=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(controller.reranker.calls, [])
        self.assertEqual(
            controller._last_retrieval_info["reranker_skipped_reason"],
            "candidate_text_count_lte_rerank_top",
        )

    def test_ingest_adds_hierarchical_chunk_metadata(self):
        controller = make_controller()
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp)
            (docs / "contract.txt").write_text("A" * 120 + "B" * 120 + "C" * 120, encoding="utf-8")
            with patch.object(scs, "FAISSIndex", FakeSearchIndex):
                controller.ingest(docs, chunk_size=120, overlap=0)

        self.assertEqual(len(controller.doc_index.added), 3)
        first = controller.doc_index.added[0]
        second = controller.doc_index.added[1]
        self.assertEqual(first["source_chunk_index"], 0)
        self.assertEqual(first["char_start"], 0)
        self.assertEqual(first["char_end"], 120)
        self.assertEqual(first["chunk_key"], "contract.txt:0")
        self.assertEqual(first["next_chunk_key"], "contract.txt:1")
        self.assertEqual(second["previous_chunk_key"], "contract.txt:0")
        self.assertGreater(first["token_estimate"], 0)

    def test_hierarchical_retrieval_expands_neighbor_windows_and_dedupes(self):
        controller = make_controller()
        metas = [
            {"filename": "contract.txt", "chunk_index": 0, "global_chunk_index": 0, "source_chunk_index": 0, "token_estimate": 3},
            {"filename": "contract.txt", "chunk_index": 1, "global_chunk_index": 1, "source_chunk_index": 1, "token_estimate": 3},
            {"filename": "contract.txt", "chunk_index": 2, "global_chunk_index": 2, "source_chunk_index": 2, "token_estimate": 3},
        ]
        controller.doc_index = FakeSearchIndex([(0.92, metas[1]), (0.91, metas[1])])
        controller.doc_index.added = metas
        controller._doc_chunks = ["before evidence", "main evidence", "after evidence"]

        original_parent = scs.PARENT_WINDOW_CHUNKS
        original_neighbor = scs.NEIGHBOR_WINDOW
        try:
            scs.PARENT_WINDOW_CHUNKS = 3
            scs.NEIGHBOR_WINDOW = 1
            results = controller.retrieve_hierarchical("Where is the evidence?", top_k=2, rerank_top=1)
        finally:
            scs.PARENT_WINDOW_CHUNKS = original_parent
            scs.NEIGHBOR_WINDOW = original_neighbor

        self.assertEqual(len(results), 1)
        self.assertIn("before evidence", results[0]["text"])
        self.assertIn("main evidence", results[0]["text"])
        self.assertIn("after evidence", results[0]["text"])
        self.assertEqual(results[0]["hit_count"], 2)
        self.assertEqual(controller._last_retrieval_info["retrieval_strategy"], "hierarchical")
        self.assertEqual(controller._last_retrieval_info["expanded_window_count"], 1)

    def test_prompt_packer_truncates_oversized_evidence_under_budget(self):
        controller = make_controller()
        original_budget = scs.PROMPT_MAX_INPUT_TOKENS
        original_chunks = scs.SYNTHESIS_MAX_CHUNKS
        try:
            scs.PROMPT_MAX_INPUT_TOKENS = 80
            scs.SYNTHESIS_MAX_CHUNKS = 2
            source_text, packed, info = controller._pack_evidence_for_prompt(
                "Question?",
                [{"text": "x" * 1000, "metadata": {"token_estimate": 250}}],
                system_prompt="System",
                output_reserve_tokens=10,
            )
        finally:
            scs.PROMPT_MAX_INPUT_TOKENS = original_budget
            scs.SYNTHESIS_MAX_CHUNKS = original_chunks

        self.assertTrue(source_text)
        self.assertEqual(len(packed), 1)
        self.assertEqual(info["truncation_reason"], "evidence_window_truncated")
        self.assertLessEqual(
            info["packed_evidence_token_estimate"],
            info["prompt_available_evidence_tokens"],
        )

    def test_option_aware_retrieval_runs_query_for_each_choice(self):
        controller = make_controller()
        meta = {"filename": "contract.txt", "chunk_index": 0, "global_chunk_index": 0, "source_chunk_index": 0}
        controller.doc_index = FakeSearchIndex([(0.91, meta)])
        controller.doc_index.added = [meta]
        controller._doc_chunks = ["Alpha is supported."]
        query = "Which option is correct?\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta"

        original_option_aware = scs.MCQ_OPTION_AWARE_RETRIEVAL
        try:
            scs.MCQ_OPTION_AWARE_RETRIEVAL = True
            results = controller.retrieve_hierarchical(query, top_k=1, rerank_top=1, is_choice_query=True)
        finally:
            scs.MCQ_OPTION_AWARE_RETRIEVAL = original_option_aware

        self.assertEqual(controller._last_retrieval_info["retrieval_query_count"], 5)
        self.assertTrue(controller._last_retrieval_info["mcq_option_aware_retrieval"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["hit_count"], 5)


if __name__ == "__main__":
    unittest.main()
