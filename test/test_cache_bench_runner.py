import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache_bench.run_benchmark import (
    _answer_matches,
    _filter_cases,
    _load_fixture_suite,
    _normalize_cache_type,
    _parse_csv_values,
    _summarize_rows,
)


FIXTURE_PATH = Path("benchmark_fixtures/cache_bench/cache_mode_suite_v1.json")


class CacheBenchRunnerTests(unittest.TestCase):
    def test_fixture_suite_loads_with_expected_labels(self):
        suite = _load_fixture_suite(FIXTURE_PATH)
        self.assertEqual(suite.suite_name, "cache_mode_suite_v1")
        self.assertIn("finance_q1", suite.corpora)
        self.assertGreaterEqual(len(suite.cases), 20)
        self.assertTrue(any(case.expected_cache_type == "knowledge" for case in suite.cases))
        self.assertTrue(any(case.expected_cache_type == "semantic" for case in suite.cases))
        self.assertTrue(any(case.expected_cache_type == "miss" for case in suite.cases))

    def test_case_filter_preserves_requested_order(self):
        suite = _load_fixture_suite(FIXTURE_PATH)
        selected = _filter_cases(
            suite.cases,
            case_ids=["doctor_semantic_director", "finance_exact_arr"],
            max_cases=0,
        )
        self.assertEqual(
            [case.case_id for case in selected],
            ["doctor_semantic_director", "finance_exact_arr"],
        )

    def test_answer_matching_modes(self):
        self.assertTrue(_answer_matches("ARR was $12.4M.", ["$12.4M"], "contains"))
        self.assertTrue(_answer_matches("Scott Derrickson", ["scott derrickson"], "exact"))
        self.assertTrue(
            _answer_matches("Reasoning...\nFinal answer: 2016", ["2016"], "lastline_contains")
        )
        self.assertTrue(
            _answer_matches("Reasoning...\nScott Derrickson", ["Scott Derrickson"], "lastline_exact")
        )
        self.assertFalse(_answer_matches("Benedict Cumberbatch", ["Scott Derrickson"], "contains"))

    def test_cache_type_normalization(self):
        self.assertEqual(_normalize_cache_type({"from_cache": True, "cache_type": "semantic"}), "semantic")
        self.assertEqual(_normalize_cache_type({"from_cache": False, "cache_type": ""}), "miss")

    def test_parse_csv_values(self):
        self.assertEqual(
            _parse_csv_values("finance_exact_arr, doctor_semantic_director"),
            ["finance_exact_arr", "doctor_semantic_director"],
        )

    def test_row_summary_groups_expected_metrics(self):
        rows = [
            {"scenario": "semantic_positive", "answer_correct": True, "cache_type_match": True, "overall_pass": True},
            {"scenario": "semantic_positive", "answer_correct": False, "cache_type_match": True, "overall_pass": False},
            {"scenario": "knowledge_negative", "answer_correct": True, "cache_type_match": False, "overall_pass": False},
        ]
        summary = _summarize_rows(rows, "scenario")
        self.assertEqual(summary["semantic_positive"]["cases"], 2)
        self.assertEqual(summary["semantic_positive"]["route_match_rate"], 1.0)
        self.assertEqual(summary["knowledge_negative"]["overall_pass_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
