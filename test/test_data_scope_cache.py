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
from language_memoization import (
    DuckDBMemoStore,
    EvidenceSpan,
    REUSE_AGGREGATION_COMPONENT,
    REUSE_SUPPORTING_FACT,
)


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
        result_a = controller.search(query)
        self.assertEqual(result_a["answer"], "answer from scope a")
        self.assertTrue(result_a["from_memo"])
        self.assertEqual(result_a["cache_type"], "memo_exact")

        controller.data_scope_hash = "scope-b"
        result_b = controller.search(query)
        self.assertEqual(result_b["answer"], "answer from scope b")
        self.assertTrue(result_b["from_memo"])
        self.assertEqual(result_b["cache_type"], "memo_exact")

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

    def test_controller_memoized_subproblem_reuses_partial_coverage(self):
        controller = make_controller()
        task_query = "Find the Q1 ARR"
        scope = controller._memo_scope(
            document_id="finance-model",
            start=0,
            end=6,
            unit="chunk",
            content_hash="scope-a",
        )
        controller.memo_store.add_answer(
            controller._memo_task(task_query),
            controller._memo_scope("finance-model", 0, 2, unit="chunk", content_hash="scope-a"),
            "No ARR in opening notes",
            reusable_as=(REUSE_AGGREGATION_COMPONENT,),
        )
        calls = []

        def solver(task, missing_scope):
            calls.append((missing_scope.start, missing_scope.end))
            if missing_scope.start == 2:
                return {"result": "$12.4M appears in the KPI table"}
            return {"result": "not found", "not_found": True}

        result = controller.memoized_subproblem(task_query, scope, solver)

        self.assertEqual(calls, [(2, 6)])
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["memo_type"], "composed")
        self.assertIn("$12.4M", result["answer"])
        self.assertEqual(result["initial_memo_telemetry"]["coverage_ratio"], 0.333333)
        self.assertEqual(result["memo_telemetry"]["coverage_ratio"], 1.0)

        calls.clear()
        replay = controller.memoized_subproblem(task_query, scope, solver)

        self.assertEqual(calls, [])
        self.assertEqual(replay["model_calls"], 0)
        self.assertEqual(replay["memo_type"], "exact")
        self.assertEqual(replay["answer"], result["answer"])

    def test_search_replays_exact_active_scope_memo_before_retrieval(self):
        controller = make_controller()
        controller.data_scope_hash = "scope-a"
        controller._doc_chunks = ["chunk 0", "chunk 1", "chunk 2"]
        query = "What is the ARR?"
        active_scope = controller._active_search_memo_scope()
        controller.memo_store.add_answer(
            controller._memo_task(
                query,
                task_type="retrieval_search",
                output_contract="synthesized_answer",
            ),
            active_scope,
            "$12.4M",
        )

        result = controller.search(query)

        self.assertTrue(result["from_cache"])
        self.assertTrue(result["from_memo"])
        self.assertEqual(result["cache_type"], "memo_exact")
        self.assertEqual(result["answer"], "$12.4M")

    def test_memo_entries_persist_with_controller_state(self):
        query = "Find ARR"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            controller = make_controller()
            scope = controller._memo_scope("doc", 0, 1, content_hash="scope-a")
            controller.memo_store.add_answer(controller._memo_task(query), scope, "$12.4M")
            controller.save(path)

            with patch.object(scs, "FAISSIndex", FakeSearchIndex):
                loaded = make_controller()
                loaded.load(path)

            plan = loaded.memo_store.plan_reuse(loaded._memo_task(query), scope)
            self.assertTrue(plan.has_exact_replay)
            self.assertEqual(plan.exact_entry.result, "$12.4M")

    def test_solve_with_memo_automatically_decomposes_and_replays(self):
        controller = make_controller()
        chunks = ["a", "b", "c", "d"]
        calls = []
        aggregate_calls = []

        def solver(task, scope):
            calls.append((scope.start, scope.end))
            return {
                "result": f"answer {scope.start}-{scope.end}",
                "evidence": [
                    {
                        "document_id": "doc",
                        "start": scope.start,
                        "end": scope.end,
                        "unit": "chunk",
                        "text": f"evidence {scope.start}-{scope.end}",
                    }
                ],
            }

        def aggregate(query, scope, plan):
            aggregate_calls.append((scope.start, scope.end))
            used = plan.reusable_entries[:1]
            return {
                "result": " | ".join(entry.result for entry in used),
                "used_entry_ids": [entry.entry_id for entry in used],
                "confidence": 0.7,
                "metadata": {"aggregator": "unit-test"},
            }

        result = controller.solve_with_memo(
            "Summarize memo behavior",
            chunks,
            solver,
            document_id="doc",
            chunk_size=2,
            content_hash="scope-a",
            aggregate_fn=aggregate,
        )

        self.assertEqual(calls, [(0, 2), (2, 4)])
        self.assertEqual(aggregate_calls, [(0, 4)])
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(result["aggregate_calls"], 1)
        self.assertEqual(result["solver_scopes"], [(0, 2), (2, 4)])
        self.assertEqual(result["initial_memo_telemetry"]["coverage_ratio"], 0.0)
        self.assertEqual(result["memo_telemetry"]["coverage_ratio"], 1.0)
        self.assertEqual(len(result["window_memo_telemetry"]), 2)
        self.assertEqual(result["window_count"], 2)
        self.assertEqual(result["reused_window_count"], 0)
        self.assertEqual(result["missing_window_count"], 2)
        self.assertIn("answer 0-2", result["answer"])
        exact_plan = controller.memo_store.plan_reuse(
            controller._memo_task(
                "Summarize memo behavior",
                task_type="memoized_document_qa",
                output_contract="supported_answer",
                constraints={"chunk_size": 2},
            ),
            controller._memo_scope("doc", 0, 4, content_hash="scope-a"),
        )
        self.assertEqual(len(exact_plan.exact_entry.dependencies), 1)
        self.assertEqual(exact_plan.exact_entry.metadata["aggregator"], "unit-test")
        self.assertEqual(len(exact_plan.exact_entry.evidence), 1)
        self.assertEqual(exact_plan.exact_entry.evidence[0].text, "evidence 0-2")

        calls.clear()
        replay = controller.solve_with_memo(
            "Summarize memo behavior",
            chunks,
            solver,
            document_id="doc",
            chunk_size=2,
            content_hash="scope-a",
            aggregate_fn=aggregate,
        )

        self.assertEqual(calls, [])
        self.assertEqual(replay["model_calls"], 0)
        self.assertEqual(replay["aggregate_calls"], 0)
        self.assertEqual(replay["memo_type"], "exact")
        self.assertEqual(replay["memo_telemetry"]["coverage_ratio"], 1.0)
        self.assertEqual(replay["window_count"], 0)
        self.assertEqual(replay["answer"], result["answer"])

    def test_solve_with_memo_can_reuse_verified_equivalent_prompt(self):
        controller = make_controller()
        chunks = ["memo stack stores solved subproblems"]

        first = controller.solve_with_memo(
            "What does the memo stack do?",
            chunks,
            lambda task, scope: {"result": "It stores solved subproblems."},
            document_id="doc",
            chunk_size=1,
            content_hash="scope-a",
        )
        self.assertEqual(first["model_calls"], 1)

        calls = []

        def solver(task, scope):
            calls.append((scope.start, scope.end))
            return {"result": "should not be called"}

        def verifier(task, candidate):
            self.assertEqual(task.prompt, "Explain the memo stack purpose.")
            return {"reuse_as": "exact", "confidence": 0.91, "reason": "equivalent enough"}

        reused = controller.solve_with_memo(
            "Explain the memo stack purpose.",
            chunks,
            solver,
            document_id="doc",
            chunk_size=1,
            content_hash="scope-a",
            reuse_verifier=verifier,
        )

        self.assertEqual(calls, [])
        self.assertTrue(reused["semantic_reuse"])
        self.assertEqual(reused["memo_type"], "semantic_exact")
        self.assertEqual(reused["model_calls"], 0)
        self.assertEqual(reused["answer"], first["answer"])

    def test_solve_with_memo_skips_windows_covered_by_larger_fragment(self):
        controller = make_controller()
        chunks = ["a", "b", "c", "d", "e", "f"]
        query = "Find useful facts"
        task = controller._memo_task(
            query,
            task_type="memoized_document_qa",
            output_contract="supported_answer",
            constraints={"chunk_size": 2},
        )
        controller.memo_store.add_answer(
            task,
            controller._memo_scope("doc", 0, 4, unit="chunk", content_hash="scope-a"),
            "chunks 0-4 already solved",
            reusable_as=(REUSE_AGGREGATION_COMPONENT,),
        )
        calls = []

        def solver(task, scope):
            calls.append((scope.start, scope.end))
            return {"result": f"fresh {scope.start}-{scope.end}"}

        result = controller.solve_with_memo(
            query,
            chunks,
            solver,
            document_id="doc",
            chunk_size=2,
            content_hash="scope-a",
        )

        self.assertEqual(calls, [(4, 6)])
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["window_count"], 3)
        self.assertEqual(result["reused_window_count"], 2)
        self.assertEqual(result["missing_window_count"], 1)
        self.assertEqual(
            [item["coverage_ratio"] for item in result["window_memo_telemetry"]],
            [1.0, 1.0, 0.0],
        )
        self.assertIn("chunks 0-4 already solved", result["answer"])
        self.assertIn("fresh 4-6", result["answer"])

    def test_solve_with_memo_reuses_unaffected_windows_after_scope_invalidation(self):
        controller = make_controller()
        query = "Extract launch facts"

        def solver_v1(task, scope):
            text = controller._doc_chunks[scope.start]
            return {"result": f"fact:{text}"}

        first = controller.solve_with_memo(
            query,
            ["alpha", "bravo", "charlie", "delta"],
            solver_v1,
            document_id="doc",
            chunk_size=1,
            content_hash="",
        )

        rejected = controller.memo_store.invalidate_scope(
            controller._memo_scope("doc", 1, 2, unit="chunk", content_hash=""),
            reason="chunk 1 updated",
        )
        calls = []

        def solver_v2(task, scope):
            calls.append((scope.start, scope.end))
            text = controller._doc_chunks[scope.start]
            return {"result": f"fact:{text}"}

        second = controller.solve_with_memo(
            query,
            ["alpha", "bravo-updated", "charlie", "delta"],
            solver_v2,
            document_id="doc",
            chunk_size=1,
            content_hash="",
        )

        self.assertEqual(first["model_calls"], 4)
        self.assertEqual(len(rejected), 2)  # full answer plus the changed chunk.
        self.assertEqual(calls, [(1, 2)])
        self.assertEqual(second["model_calls"], 1)
        self.assertEqual(second["initial_memo_telemetry"]["covered_length"], 3)
        self.assertEqual(second["initial_memo_telemetry"]["missing_length"], 1)
        self.assertEqual(second["reused_window_count"], 3)
        self.assertIn("fact:alpha", second["answer"])
        self.assertIn("fact:bravo-updated", second["answer"])
        self.assertIn("fact:charlie", second["answer"])
        self.assertIn("fact:delta", second["answer"])
        self.assertNotIn("fact:bravo\n", second["answer"])

    def test_solve_with_memo_allows_solver_to_label_fragment_reuse_modes(self):
        controller = make_controller()

        def solver(task, scope):
            return {
                "result": "Katie lives next to the Kiasma museum.",
                "reusable_as": (REUSE_SUPPORTING_FACT, REUSE_AGGREGATION_COMPONENT),
            }

        controller.solve_with_memo(
            "Extract reusable evidence",
            ["Katie lives next to the Kiasma museum."],
            solver,
            document_id="doc",
            chunk_size=1,
            content_hash="scope-a",
        )

        fragment_entries = [
            entry
            for entry in controller.memo_store.entries.values()
            if entry.scope.start == 0 and entry.scope.end == 1 and entry.dependencies == ()
        ]
        self.assertEqual(len(fragment_entries), 1)
        self.assertEqual(
            set(fragment_entries[0].reusable_as),
            {REUSE_SUPPORTING_FACT, REUSE_AGGREGATION_COMPONENT},
        )
        self.assertEqual(fragment_entries[0].fragment_kind, "supporting_fact")

    def test_legacy_cache_and_knowledge_are_adapters_not_return_paths(self):
        controller = make_controller()
        controller.cache = {
            "chunk-a": [make_entry("Where is the launch code?", "ORCHID-57", "scope-a")],
        }
        controller.knowledge = [
            {
                "subject": "launch code",
                "relation": "is",
                "object": "ORCHID-57",
                "source_cache_idx": 0,
                "source_chunk_hash": "chunk-a",
                "data_scope_hash": "scope-a",
            }
        ]
        controller.data_scope_hash = "scope-a"

        packet = controller.reuse_candidate_packets(
            "launch code ORCHID",
            scope=controller._active_search_memo_scope(),
        )

        self.assertEqual(packet["candidate_generators"]["context_chunks"], 0)
        self.assertGreaterEqual(packet["candidate_generators"]["memo_entries"], 1)
        reuse_modes = {
            mode
            for entry in packet["memo_entries"]
            for mode in entry["reusable_as"]
        }
        self.assertIn("exact_answer", reuse_modes)
        self.assertIn("supporting_fact", reuse_modes)
        self.assertTrue(any(entry["fragment_kind"] == "supporting_fact" for entry in packet["memo_entries"]))
        self.assertTrue(all("can_cover_scope" in entry for entry in packet["memo_entries"]))
        self.assertTrue(all("result_truncated" in entry for entry in packet["memo_entries"]))
        self.assertTrue(all("evidence_count" in entry for entry in packet["memo_entries"]))

    def test_candidate_packets_make_truncation_explicit_and_disableable(self):
        controller = make_controller()
        task = controller._memo_task("Extract launch code", task_type="fact", output_contract="text")
        scope = controller._memo_scope("doc", 0, 1, content_hash="scope-a")
        controller.memo_store.add_answer(
            task,
            scope,
            "launch codeword is ORCHID-57 and backup channel is north-bridge-9",
            evidence=[
                EvidenceSpan(
                    document_id="doc",
                    start=0,
                    end=1,
                    unit="chunk",
                    text="launch codeword is ORCHID-57 with a long evidence note",
                )
            ],
        )

        truncated = controller.memo_candidate_packets(
            "launch codeword",
            scope=scope,
            max_result_chars=12,
            max_evidence_chars=10,
        )[0]
        full = controller.memo_candidate_packets(
            "launch codeword",
            scope=scope,
            max_result_chars=None,
            max_evidence_chars=None,
        )[0]

        self.assertTrue(truncated["result_truncated"])
        self.assertTrue(truncated["evidence"][0]["text_truncated"])
        self.assertEqual(truncated["result_chars"], len(full["result"]))
        self.assertFalse(full["result_truncated"])
        self.assertFalse(full["evidence"][0]["text_truncated"])
        self.assertIn("north-bridge-9", full["result"])

    def test_solve_with_memo_persists_context_chunks_when_duckdb_backed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DuckDBMemoStore(Path(tmp) / "memo.duckdb")
            controller = scs.SemanticCacheController(
                metrics=scs.ExecutionMetrics(),
                embedder=FakeEmbedder(),
                reranker=None,
                corpus_id="test",
                memo_store=store,
            )

            controller.solve_with_memo(
                "What is the launch codeword?",
                [
                    "The launch codeword is ORCHID-57.",
                    "The backup contact channel is north-bridge-9.",
                ],
                lambda task, scope: {"result": "ORCHID-57"},
                document_id="doc",
                chunk_size=1,
                content_hash="scope-a",
            )

            self.assertEqual(store.context_chunk_count("test"), 2)
            packets = controller.context_candidate_packets(
                "backup contact channel",
                scope=controller._memo_scope("doc", 0, 2, content_hash="scope-a"),
                max_text_chars=16,
            )
            self.assertEqual(packets[0]["chunk_index"], 1)
            self.assertTrue(packets[0]["text_truncated"])
            self.assertGreater(packets[0]["text_chars"], 16)

            full_packets = controller.context_candidate_packets(
                "backup contact channel",
                scope=controller._memo_scope("doc", 0, 2, content_hash="scope-a"),
                max_text_chars=None,
            )
            self.assertFalse(full_packets[0]["text_truncated"])
            self.assertIn("north-bridge-9", full_packets[0]["text"])
            store.close()

    def test_solve_with_memo_exact_replay_skips_context_chunk_upsert(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DuckDBMemoStore(Path(tmp) / "memo.duckdb")
            controller = scs.SemanticCacheController(
                metrics=scs.ExecutionMetrics(),
                embedder=FakeEmbedder(),
                reranker=None,
                corpus_id="test",
                memo_store=store,
            )
            task = controller._memo_task(
                "What is the launch codeword?",
                task_type="memoized_document_qa",
                output_contract="supported_answer",
                constraints={"chunk_size": 1},
            )
            scope = controller._memo_scope("doc", 0, 1, content_hash="scope-a")
            store.add_answer(task, scope, "ORCHID-57")

            result = controller.solve_with_memo(
                "What is the launch codeword?",
                ["The launch codeword is ORCHID-57."],
                lambda task, scope: {"result": "should not run"},
                document_id="doc",
                chunk_size=1,
                content_hash="scope-a",
            )

            self.assertEqual(result["model_calls"], 0)
            self.assertEqual(result["memo_type"], "exact")
            self.assertEqual(store.context_chunk_count("test"), 0)
            store.close()

    def test_memo_context_planner_answers_from_candidate_packets(self):
        controller = make_controller()
        scope = controller._memo_scope("doc", 0, 4, content_hash="scope-a")
        fact_scope = controller._memo_scope("doc", 1, 2, content_hash="scope-a")
        fact = controller.memo_store.add_answer(
            controller._memo_task(
                "Extract reusable facts",
                task_type="fact_extraction",
                output_contract="fact_bullets",
            ),
            fact_scope,
            "The active launch codeword is ORCHID-57.",
            reusable_as=(REUSE_SUPPORTING_FACT,),
            evidence=[
                EvidenceSpan(
                    document_id="doc",
                    start=1,
                    end=2,
                    unit="chunk",
                    text="The launch codeword is ORCHID-57.",
                )
            ],
        )
        controller.memo_store.add_answer(
            controller._memo_task(
                "Extract reusable facts",
                task_type="fact_extraction",
                output_contract="fact_bullets",
            ),
            controller._memo_scope("doc", 2, 3, content_hash="scope-a"),
            "The old clock was repaired by Marta.",
            reusable_as=(REUSE_SUPPORTING_FACT,),
        )

        def planner(query, candidates):
            self.assertGreaterEqual(len(candidates), 1)
            self.assertTrue(any("ORCHID-57" in candidate["result"] for candidate in candidates))
            return {
                "action": "answer",
                "answer": "ORCHID-57",
                "used_entry_ids": [fact.entry_id],
                "confidence": 0.93,
                "reason": "candidate directly states the active codeword",
            }

        result = controller.answer_from_memo_context(
            "What active launch credential should operations use?",
            planner,
            scope,
        )

        self.assertTrue(result["from_memo"])
        self.assertEqual(result["answer"], "ORCHID-57")
        self.assertEqual(result["dependencies"], [fact.entry_id])
        self.assertEqual(result["planner_calls"], 1)

        replay = controller.answer_from_memo_context(
            "What active launch credential should operations use?",
            planner,
            scope,
        )

        self.assertEqual(replay["planner_calls"], 0)
        self.assertEqual(replay["memo_type"], "exact")
        self.assertEqual(replay["answer"], "ORCHID-57")


if __name__ == "__main__":
    unittest.main()
