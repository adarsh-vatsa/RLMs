import tempfile
import unittest
from pathlib import Path

from language_memoization import (
    DuckDBMemoStore,
    EvidenceSpan,
    REUSE_AGGREGATION_COMPONENT,
    REUSE_SEARCH_HINT,
    VERIFIER_REJECTED,
    TaskSpec,
    ContextScope,
)


def make_task(prompt="Find the Q1 ARR"):
    return TaskSpec(
        prompt=prompt,
        task_type="financial_metric_lookup",
        output_contract="short factual answer",
    )


def make_scope(start, end, *, content_hash="v1"):
    return ContextScope(
        corpus_id="finance",
        document_id="model-a",
        start=start,
        end=end,
        unit="chunk",
        content_hash=content_hash,
    )


class DuckDBMemoStoreTests(unittest.TestCase):
    def test_duckdb_matches_in_memory_planner_semantics(self):
        store = DuckDBMemoStore(":memory:")
        task = make_task()
        store.add_negative(task, make_scope(0, 2), reason="not in opening notes")
        store.add_answer(
            task,
            make_scope(5, 7),
            "$12.4M in KPI table",
            reusable_as=(REUSE_AGGREGATION_COMPONENT,),
        )

        plan = store.plan_reuse(task, make_scope(0, 8))

        self.assertFalse(plan.has_exact_replay)
        self.assertEqual([(s.start, s.end) for s in plan.missing_scopes], [(2, 5), (7, 8)])
        self.assertEqual(len(store.entries), 2)
        store.close()

    def test_duckdb_file_roundtrip_preserves_exact_replay(self):
        task = make_task()
        scope = make_scope(2, 3)
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memo.duckdb"
            store = DuckDBMemoStore(db_path)
            original = store.add_answer(
                task,
                scope,
                "$12.4M",
                evidence=[EvidenceSpan(document_id="model-a", start=10, end=20, text="ARR $12.4M")],
            )
            store.close()

            loaded = DuckDBMemoStore(db_path)
            plan = loaded.plan_reuse(task, scope)

            self.assertTrue(plan.has_exact_replay)
            self.assertEqual(plan.exact_entry.entry_id, original.entry_id)
            self.assertEqual(plan.exact_entry.evidence[0].text, "ARR $12.4M")
            loaded.close()

    def test_duckdb_search_hints_do_not_cover_scope(self):
        store = DuckDBMemoStore(":memory:")
        task = make_task()
        store.add_answer(
            task,
            make_scope(4, 5),
            "Likely around chunk 4",
            reusable_as=(REUSE_SEARCH_HINT,),
        )

        plan = store.plan_reuse(task, make_scope(0, 10))

        self.assertEqual(len(plan.hint_entries), 1)
        self.assertEqual([(s.start, s.end) for s in plan.missing_scopes], [(0, 10)])
        store.close()

    def test_duckdb_rejected_candidates_are_not_planner_hints(self):
        store = DuckDBMemoStore(":memory:")
        task = make_task()
        store.add_answer(
            task,
            make_scope(4, 5),
            "Rejected hint around chunk 4",
            reusable_as=(REUSE_SEARCH_HINT,),
            verifier_status=VERIFIER_REJECTED,
        )

        plan = store.plan_reuse(task, make_scope(0, 10))

        self.assertEqual(plan.hint_entries, ())
        self.assertEqual([(s.start, s.end) for s in plan.missing_scopes], [(0, 10)])
        store.close()

    def test_duckdb_invalidate_scope_persists_rejection(self):
        task = make_task()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memo.duckdb"
            store = DuckDBMemoStore(db_path)
            entry = store.add_answer(task, make_scope(0, 2), "$12.4M")
            rejected = store.invalidate_scope(make_scope(0, 10, content_hash=""), reason="document updated")
            store.close()

            loaded = DuckDBMemoStore(db_path)
            plan = loaded.plan_reuse(task, make_scope(0, 2))
            stats = loaded.stats()

            self.assertEqual([item.entry_id for item in rejected], [entry.entry_id])
            self.assertFalse(plan.has_exact_replay)
            self.assertEqual([(s.start, s.end) for s in plan.missing_scopes], [(0, 2)])
            self.assertEqual(stats["by_verifier_status"]["rejected"], 1)
            loaded.close()

    def test_duckdb_invalidate_scope_rejects_dependency_parents(self):
        task = make_task()
        store = DuckDBMemoStore(":memory:")
        child = store.add_answer(task, make_scope(0, 1), "chunk fact")
        parent = store.add_answer(
            task,
            make_scope(8, 9),
            "derived answer elsewhere",
            dependencies=(child.entry_id,),
        )

        rejected = store.invalidate_scope(make_scope(0, 1), reason="source changed")

        self.assertEqual({entry.entry_id for entry in rejected}, {child.entry_id, parent.entry_id})
        self.assertEqual({entry.entry_id for entry in store.parents(child.entry_id)}, {parent.entry_id})
        self.assertTrue(all(entry.is_rejected for entry in store.parents(child.entry_id)))
        store.close()

    def test_duckdb_scope_candidates_include_different_tasks(self):
        store = DuckDBMemoStore(":memory:")
        old_task = make_task("What does the memo stack do?")
        scope = make_scope(0, 2)
        entry = store.add_answer(old_task, scope, "It stores solved subproblems.")

        candidates = store.scope_candidates(scope)

        self.assertEqual([candidate.entry_id for candidate in candidates], [entry.entry_id])
        store.close()

    def test_duckdb_edges_mirror_dependencies_and_lineage(self):
        store = DuckDBMemoStore(":memory:")
        task = make_task()
        child_a = store.add_answer(task, make_scope(0, 1), "opening fact")
        child_b = store.add_answer(task, make_scope(1, 2), "second fact")
        parent = store.add_answer(
            task,
            make_scope(0, 2),
            "combined answer",
            dependencies=(child_a.entry_id, child_b.entry_id),
        )

        edges = store.graph_edges(parent.entry_id)
        self.assertEqual({edge["child_entry_id"] for edge in edges}, {child_a.entry_id, child_b.entry_id})
        self.assertEqual(
            {entry.entry_id for entry in store.children(parent.entry_id)},
            {child_a.entry_id, child_b.entry_id},
        )
        self.assertEqual([entry.entry_id for entry in store.parents(child_a.entry_id)], [parent.entry_id])
        lineage = store.lineage(parent.entry_id)
        self.assertEqual(lineage["entry_id"], parent.entry_id)
        self.assertEqual({node["entry_id"] for node in lineage["children"]}, {child_a.entry_id, child_b.entry_id})
        store.close()

    def test_duckdb_edges_survive_file_roundtrip(self):
        task = make_task()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memo.duckdb"
            store = DuckDBMemoStore(db_path)
            child = store.add_answer(task, make_scope(0, 1), "fact")
            parent = store.add_answer(
                task,
                make_scope(0, 2),
                "answer",
                dependencies=(child.entry_id,),
            )
            store.close()

            loaded = DuckDBMemoStore(db_path)
            self.assertEqual([entry.entry_id for entry in loaded.children(parent.entry_id)], [child.entry_id])
            loaded.close()

    def test_duckdb_context_chunks_roundtrip_and_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memo.duckdb"
            store = DuckDBMemoStore(db_path)
            stored = store.upsert_context_chunks(
                corpus_id="ops",
                document_id="launch",
                chunks=[
                    "The launch codeword is ORCHID-57.",
                    "The backup contact channel is north-bridge-9.",
                ],
                content_hash="ctx-v1",
                metadata={"fixture": "unit"},
            )
            self.assertEqual(stored, 2)
            self.assertEqual(store.context_chunk_count("ops"), 2)

            fetched = store.fetch_context_range(
                corpus_id="ops",
                document_id="launch",
                start=0,
                end=2,
                content_hash="ctx-v1",
            )
            self.assertEqual([chunk.chunk_index for chunk in fetched], [0, 1])
            self.assertEqual(fetched[0].metadata["fixture"], "unit")

            hits = store.search_context_chunks(
                "Which backup channel should launch use?",
                corpus_id="ops",
                document_id="launch",
                content_hash="ctx-v1",
            )
            self.assertEqual(hits[0][0].chunk_index, 1)
            self.assertIn("backup", hits[0][2])
            store.close()


if __name__ == "__main__":
    unittest.main()
