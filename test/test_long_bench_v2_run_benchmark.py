import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from long_bench_v2.run_benchmark import (
    answer_correct,
    build_cache_reuse_manifest,
    build_query,
    filter_suite_rows,
    load_context_by_source_id,
    load_suite_rows,
    parse_choice,
    resolve_cache_namespace,
)


def _source_row(row_id: str, context: str = "Context text") -> dict:
    return {
        "_id": row_id,
        "context": context,
        "question": "Which option is correct?",
        "choice_A": "Alpha",
        "choice_B": "Beta",
        "choice_C": "Gamma",
        "choice_D": "Delta",
        "answer": "A",
    }


def _suite_row(source_id: str, row_type: str = "original") -> dict:
    return {
        "case_id": f"{source_id}__{row_type}",
        "source_id": source_id,
        "row_type": row_type,
        "is_scored": "true",
        "setup_case_id": "",
        "context_id": "ctx",
        "token_count": "100",
        "expected_cache_type": "miss" if row_type == "original" else row_type,
        "expected_from_cache": "false" if row_type == "original" else "true",
        "depends_on_case_id": "",
        "domain": "Single-Document QA",
        "sub_domain": "Synthetic",
        "difficulty": "easy",
        "length": "short",
        "question": "Which option is correct?",
        "choice_A": "Alpha",
        "choice_B": "Beta",
        "choice_C": "Gamma",
        "choice_D": "Delta",
        "answer": "A",
    }


class LongBenchV2RunBenchmarkTests(unittest.TestCase):
    def test_load_suite_rows_resolves_context_from_source_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_path = tmp / "data.json"
            suite_path = tmp / "suite.csv"
            source_path.write_text(json.dumps([_source_row("row_1", "Full long context")]), encoding="utf-8")
            with suite_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(_suite_row("row_1").keys()))
                writer.writeheader()
                writer.writerow(_suite_row("row_1"))

            contexts = load_context_by_source_id(source_path)
            rows = load_suite_rows(suite_path, contexts)

        self.assertEqual(contexts["row_1"], "Full long context")
        self.assertEqual(rows[0]["context"], "Full long context")

    def test_filter_suite_rows_by_type_and_max_rows(self):
        rows = [
            _suite_row("row_1", "original"),
            _suite_row("row_1", "exact"),
            _suite_row("row_1", "semantic"),
            _suite_row("row_2", "original"),
        ]

        selected = filter_suite_rows(rows, row_types=["original", "semantic"], max_rows=2)

        self.assertEqual([row["case_id"] for row in selected], ["row_1__original", "row_1__semantic"])

    def test_resolve_cache_namespace_is_deterministic(self):
        rows = [_suite_row("row_1", "original"), _suite_row("row_1", "exact")]

        first = resolve_cache_namespace("suite-sha", "source-sha", rows, "model-a", 20, 5, ["original", "exact"])
        second = resolve_cache_namespace("suite-sha", "source-sha", list(reversed(rows)), "model-a", 20, 5, ["exact", "original"])
        changed = resolve_cache_namespace("suite-sha", "source-sha", rows, "model-b", 20, 5, ["original", "exact"])

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_parse_choice_and_answer_correct(self):
        self.assertEqual(parse_choice("A"), "A")
        self.assertEqual(parse_choice("Final answer: C"), "C")
        self.assertEqual(parse_choice("(D) because the passage says so"), "D")
        self.assertTrue(answer_correct("Option B", "B"))
        self.assertFalse(answer_correct("Option B", "C"))

    def test_build_query_preserves_multiple_choice_fields(self):
        query = build_query(_suite_row("row_1"))

        self.assertIn("Question: Which option is correct?", query)
        self.assertIn("A. Alpha", query)
        self.assertIn("Return only the single best answer choice letter", query)

    def test_cache_reuse_manifest_baseline_and_cache_modes(self):
        baseline = build_cache_reuse_manifest(enabled=False)
        cold = build_cache_reuse_manifest(
            enabled=True,
            cache_namespace="ns",
            dataset_signature="sig",
            cache_state_existed_before_run=False,
            cache_hits=2,
            row_count=4,
        )
        warm = build_cache_reuse_manifest(
            enabled=True,
            cache_namespace="ns",
            dataset_signature="sig",
            cache_state_existed_before_run=True,
            cache_hits=3,
            row_count=4,
        )

        self.assertFalse(baseline["enabled"])
        self.assertEqual(cold["run_start_type"], "cold_start")
        self.assertEqual(cold["cache_hit_rate"], 0.5)
        self.assertEqual(warm["run_start_type"], "warm_start")


if __name__ == "__main__":
    unittest.main()
