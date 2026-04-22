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


if __name__ == "__main__":
    unittest.main()
