import tempfile
import unittest
from pathlib import Path

from language_memoization import (
    ContextScope,
    EvidenceSpan,
    MemoEntry,
    MemoStore,
    REUSE_AGGREGATION_COMPONENT,
    REUSE_EXACT_ANSWER,
    REUSE_SEARCH_HINT,
    TaskSpec,
    coerce_confidence,
)


def make_task(prompt="Find the Q1 ARR"):
    return TaskSpec(
        prompt=prompt,
        task_type="financial_metric_lookup",
        output_contract="short factual answer",
    )


def make_scope(start, end, *, doc="doc-a", content_hash="v1"):
    return ContextScope(
        corpus_id="finance",
        document_id=doc,
        start=start,
        end=end,
        unit="chunk",
        content_hash=content_hash,
    )


class LanguageMemoizationTests(unittest.TestCase):
    def test_coerce_confidence_accepts_model_labels(self):
        self.assertEqual(coerce_confidence("high"), 0.9)
        self.assertEqual(coerce_confidence("medium"), 0.6)
        self.assertEqual(coerce_confidence("low"), 0.3)
        self.assertEqual(coerce_confidence("0.42"), 0.42)
        self.assertEqual(coerce_confidence("75"), 0.75)
        self.assertEqual(coerce_confidence("92%"), 0.92)
        self.assertEqual(coerce_confidence("not-a-number", default=0.7), 0.7)

    def test_exact_task_scope_replay(self):
        store = MemoStore()
        task = make_task()
        scope = make_scope(3, 4)
        entry = store.add_answer(task, scope, "$12.4M")

        plan = store.plan_reuse(make_task("  find   the q1 arr "), make_scope(3, 4))

        self.assertTrue(plan.has_exact_replay)
        self.assertTrue(plan.is_complete)
        self.assertEqual(plan.exact_entry.entry_id, entry.entry_id)
        self.assertEqual(plan.missing_scopes, ())
        self.assertEqual(plan.coverage_ratio, 1.0)
        self.assertEqual(plan.to_telemetry()["covered_intervals"], [[3, 4]])

    def test_partial_coverage_returns_missing_subscopes(self):
        store = MemoStore()
        task = make_task()
        store.add_answer(
            task,
            make_scope(0, 2),
            "ARR not mentioned in chunks 0-2",
            reusable_as=(REUSE_AGGREGATION_COMPONENT,),
        )
        store.add_answer(
            task,
            make_scope(5, 7),
            "$12.4M appears in KPI table",
            reusable_as=(REUSE_AGGREGATION_COMPONENT,),
        )

        plan = store.plan_reuse(task, make_scope(0, 8))

        self.assertFalse(plan.is_complete)
        self.assertEqual([(s.start, s.end) for s in plan.missing_scopes], [(2, 5), (7, 8)])
        self.assertEqual(plan.covered_length, 4)
        self.assertEqual(plan.missing_length, 4)
        self.assertEqual(plan.coverage_ratio, 0.5)
        self.assertEqual(plan.to_telemetry()["covered_intervals"], [[0, 2], [5, 7]])
        self.assertEqual(
            plan.to_telemetry()["covering_fragment_kind_counts"],
            {"aggregation_component": 2},
        )

    def test_same_scope_component_and_exact_answer_are_not_double_counted(self):
        store = MemoStore()
        task = make_task()
        scope = make_scope(0, 2)
        store.add_answer(
            task,
            scope,
            "component answer",
            reusable_as=(REUSE_AGGREGATION_COMPONENT,),
        )
        exact = store.add_answer(
            task,
            scope,
            "exact answer",
            dependencies=("component-id",),
            reusable_as=(REUSE_EXACT_ANSWER,),
        )

        plan = store.plan_reuse(task, make_scope(0, 4))

        self.assertEqual([entry.entry_id for entry in plan.reusable_entries], [exact.entry_id])
        self.assertEqual([(s.start, s.end) for s in plan.missing_scopes], [(2, 4)])

    def test_negative_memoization_counts_as_solved_coverage(self):
        store = MemoStore()
        task = make_task()
        store.add_negative(task, make_scope(0, 2), reason="metric not found in opening notes")
        store.add_answer(
            task,
            make_scope(2, 4),
            "$12.4M",
            reusable_as=(REUSE_AGGREGATION_COMPONENT,),
        )

        plan = store.plan_reuse(task, make_scope(0, 4))

        self.assertTrue(plan.is_complete)
        self.assertEqual(len(plan.negative_entries), 1)
        self.assertEqual(plan.missing_scopes, ())
        self.assertEqual(plan.covered_length, 4)
        self.assertEqual(plan.coverage_ratio, 1.0)
        self.assertEqual(plan.negative_entries[0].fragment_kind, "ruled_out_region")
        self.assertEqual(
            plan.to_telemetry()["covering_fragment_kind_counts"],
            {"ruled_out_region": 1, "aggregation_component": 1},
        )

    def test_search_hints_do_not_cover_scope(self):
        store = MemoStore()
        task = make_task()
        hint = MemoEntry(
            task=task,
            scope=make_scope(6, 7),
            result="The KPI table is likely near chunk 6.",
            result_type="hint",
            reusable_as=(REUSE_SEARCH_HINT,),
        )
        store.add(hint)

        plan = store.plan_reuse(task, make_scope(0, 10))

        self.assertEqual([entry.entry_id for entry in plan.hint_entries], [hint.entry_id])
        self.assertEqual([(s.start, s.end) for s in plan.missing_scopes], [(0, 10)])
        self.assertEqual(plan.coverage_ratio, 0.0)
        self.assertEqual(hint.fragment_kind, "search_hint")

    def test_scope_candidates_include_different_tasks(self):
        store = MemoStore()
        old_task = make_task("What does the memo stack do?")
        new_scope = make_scope(0, 2)
        entry = store.add_answer(old_task, new_scope, "It stores solved subproblems.")

        candidates = store.scope_candidates(new_scope)

        self.assertEqual([candidate.entry_id for candidate in candidates], [entry.entry_id])

    def test_content_hash_mismatch_prevents_reuse(self):
        store = MemoStore()
        task = make_task()
        store.add_answer(task, make_scope(0, 2, content_hash="old"), "$12.4M")

        plan = store.plan_reuse(task, make_scope(0, 2, content_hash="new"))

        self.assertFalse(plan.has_exact_replay)
        self.assertEqual([(s.start, s.end) for s in plan.missing_scopes], [(0, 2)])

    def test_invalidate_scope_rejects_overlapping_entries(self):
        store = MemoStore()
        task = make_task()
        entry = store.add_answer(task, make_scope(0, 2), "$12.4M")

        rejected = store.invalidate_scope(make_scope(0, 10, content_hash=""), reason="document updated")
        plan = store.plan_reuse(task, make_scope(0, 2))

        self.assertEqual([item.entry_id for item in rejected], [entry.entry_id])
        self.assertTrue(rejected[0].is_rejected)
        self.assertEqual(rejected[0].metadata["rejection_reason"], "document updated")
        self.assertFalse(plan.has_exact_replay)
        self.assertEqual([(s.start, s.end) for s in plan.missing_scopes], [(0, 2)])

    def test_invalidate_scope_rejects_dependency_parents(self):
        store = MemoStore()
        task = make_task()
        child = store.add_answer(task, make_scope(0, 1), "chunk fact")
        parent = store.add_answer(
            task,
            make_scope(8, 9),
            "derived answer elsewhere",
            dependencies=(child.entry_id,),
        )

        rejected = store.invalidate_scope(make_scope(0, 1), reason="source changed")

        self.assertEqual({entry.entry_id for entry in rejected}, {child.entry_id, parent.entry_id})
        self.assertTrue(store.entries[parent.entry_id].is_rejected)
        self.assertEqual([entry.entry_id for entry in store.parents(child.entry_id)], [parent.entry_id])
        self.assertEqual([entry.entry_id for entry in store.children(parent.entry_id)], [child.entry_id])

    def test_json_roundtrip_preserves_entries_and_evidence(self):
        store = MemoStore()
        task = make_task()
        scope = make_scope(3, 4)
        evidence = [EvidenceSpan(document_id="doc-a", start=120, end=145, text="ARR: $12.4M")]
        original = store.add_answer(
            task,
            scope,
            "$12.4M",
            evidence=evidence,
            reusable_as=(REUSE_EXACT_ANSWER,),
            metadata={"source": "unit-test"},
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = store.save(Path(tmp))
            loaded = MemoStore.load(path)

        plan = loaded.plan_reuse(task, scope)

        self.assertTrue(plan.has_exact_replay)
        self.assertEqual(plan.exact_entry.entry_id, original.entry_id)
        self.assertEqual(plan.exact_entry.evidence[0].text, "ARR: $12.4M")
        self.assertEqual(plan.exact_entry.metadata["source"], "unit-test")

    def test_store_stats_summarize_fragment_mix(self):
        store = MemoStore()
        task = make_task()
        store.add_answer(task, make_scope(0, 1), "$12.4M")
        store.add_negative(task, make_scope(1, 2), reason="not found")
        store.add_answer(
            task,
            make_scope(2, 3),
            "Likely in KPI table",
            reusable_as=(REUSE_SEARCH_HINT,),
        )

        stats = store.stats()

        self.assertEqual(stats["entry_count"], 3)
        self.assertEqual(stats["by_fragment_kind"]["exact_answer"], 1)
        self.assertEqual(stats["by_fragment_kind"]["ruled_out_region"], 1)
        self.assertEqual(stats["by_fragment_kind"]["search_hint"], 1)
        self.assertEqual(stats["by_reuse_mode"]["exact_answer"], 1)
        self.assertEqual(stats["by_reuse_mode"]["ruled_out"], 1)


if __name__ == "__main__":
    unittest.main()
