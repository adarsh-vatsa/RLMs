import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache_bench.run_benchmark import _load_fixture_suite
from legal_bench.run_benchmark import (
    ARTIFACT_SUBDIR,
    _build_legal_eval_report,
    _resolve_case_cache_namespace,
)
from legal_bench.suite import (
    build_legal_cache_suite,
    build_suite_from_path,
    load_cuad_source_records,
    load_raw_records,
    normalize_cuad_records,
    select_legal_records,
)


FIXTURE_PATH = Path("benchmark_fixtures/legal_bench/cuad_qa_tiny.json")


def _raw_cuad_row(title: str, clause_label: str, row_id: str) -> dict:
    answer = f"{clause_label} answer"
    return {
        "id": row_id,
        "title": title,
        "context": f"{title} contains {answer}.",
        "question": (
            f'Highlight the parts (if any) of this contract related to "{clause_label}" '
            "that should be reviewed by a lawyer."
        ),
        "answers": {"text": [answer], "answer_start": [0]},
    }


class LegalBenchSuiteTests(unittest.TestCase):
    def test_cuad_fixture_normalizes_records(self):
        raw_records = load_raw_records(FIXTURE_PATH)
        records = normalize_cuad_records(raw_records)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].title, "ACME_DISTRIBUTION_AGREEMENT")
        self.assertEqual(records[0].clause_label, "Governing Law")
        self.assertIn("State of New York", records[0].answers[0])

    def test_build_suite_generates_all_route_labels(self):
        records = normalize_cuad_records(load_raw_records(FIXTURE_PATH))
        suite = build_legal_cache_suite(records)
        routes = {case["expected_cache_type"] for case in suite["cases"]}
        self.assertEqual(routes, {"exact", "semantic", "knowledge", "miss"})
        self.assertEqual(len(suite["corpora"]), 1)
        self.assertTrue(all("metadata" in case for case in suite["cases"]))

    def test_normalization_filters_document_name_field(self):
        raw_records = [
            {
                "id": "doc_name",
                "title": "Contract A",
                "context": "This distribution agreement is between ACME and Beta.",
                "question": (
                    'Highlight the parts (if any) of this contract related to "Document Name" '
                    "that should be reviewed by a lawyer. Details: The name of the contract"
                ),
                "answers": {"text": ["distribution agreement"], "answer_start": [5]},
            },
            {
                "id": "governing_law",
                "title": "Contract A",
                "context": "This agreement is governed by New York law.",
                "question": (
                    'Highlight the parts (if any) of this contract related to "Governing Law" '
                    "that should be reviewed by a lawyer."
                ),
                "answers": {"text": ["governed by New York law"], "answer_start": [18]},
            },
        ]

        records = normalize_cuad_records(raw_records)

        self.assertEqual([record.clause_label for record in records], ["Governing Law"])

    def test_normalization_keeps_one_record_per_contract_clause_label(self):
        raw_records = [
            {
                "id": "parties_1",
                "title": "Contract A",
                "context": "The parties are ACME and Beta.",
                "question": (
                    'Highlight the parts (if any) of this contract related to "Parties" '
                    "that should be reviewed by a lawyer."
                ),
                "answers": {"text": ["ACME and Beta"], "answer_start": [16]},
            },
            {
                "id": "parties_2",
                "title": "Contract A",
                "context": "The parties are ACME and Beta.",
                "question": (
                    'Highlight the parts (if any) of this contract related to "Parties" '
                    "that should be reviewed by a lawyer."
                ),
                "answers": {"text": ["Beta"], "answer_start": [25]},
            },
            {
                "id": "agreement_date",
                "title": "Contract A",
                "context": "The agreement is dated January 1, 2020.",
                "question": (
                    'Highlight the parts (if any) of this contract related to "Agreement Date" '
                    "that should be reviewed by a lawyer."
                ),
                "answers": {"text": ["January 1, 2020"], "answer_start": [23]},
            },
        ]

        records = normalize_cuad_records(raw_records)

        self.assertEqual(
            [record.clause_label for record in records],
            ["Parties", "Agreement Date"],
        )

    def test_generated_suite_loads_with_cache_bench_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "legal_suite.json"
            suite = build_suite_from_path(FIXTURE_PATH, out_path)
            loaded = _load_fixture_suite(out_path)

        self.assertEqual(loaded.suite_name, suite["suite_name"])
        self.assertGreaterEqual(len(loaded.cases), 6)
        self.assertTrue(any(case.expected_cache_type == "miss" for case in loaded.cases))

    def test_balanced_selection_skips_contracts_with_too_few_labels(self):
        raw_records = [
            _raw_cuad_row("Contract A", "Parties", "a_parties"),
            _raw_cuad_row("Contract A", "Agreement Date", "a_date"),
            _raw_cuad_row("Contract A", "Effective Date", "a_effective"),
            _raw_cuad_row("Contract B", "Parties", "b_parties"),
            _raw_cuad_row("Contract C", "Parties", "c_parties"),
            _raw_cuad_row("Contract C", "Agreement Date", "c_date"),
            _raw_cuad_row("Contract C", "Effective Date", "c_effective"),
        ]
        records = normalize_cuad_records(raw_records)

        selected = select_legal_records(
            records,
            selection_strategy="balanced-corpora",
            max_corpora=2,
            records_per_corpus=3,
        )

        self.assertEqual([record.title for record in selected], ["Contract A"] * 3 + ["Contract C"] * 3)
        self.assertNotIn("Contract B", {record.title for record in selected})

    def test_balanced_suite_generates_expected_shape(self):
        raw_records = []
        for corpus_idx in range(1, 6):
            for clause_label in ("Parties", "Agreement Date", "Effective Date"):
                raw_records.append(
                    _raw_cuad_row(
                        f"Contract {corpus_idx}",
                        clause_label,
                        f"contract_{corpus_idx}_{clause_label}",
                    )
                )
        records = normalize_cuad_records(raw_records)

        suite = build_legal_cache_suite(
            records,
            selection_strategy="balanced-corpora",
            max_corpora=5,
            records_per_corpus=3,
        )

        self.assertEqual(suite["source"]["records_selected"], 15)
        self.assertEqual(suite["source"]["corpora_selected"], 5)
        self.assertEqual(suite["source"]["route_counts"], {
            "exact": 15,
            "semantic": 15,
            "knowledge": 15,
            "miss": 15,
        })
        self.assertEqual(len(suite["corpora"]), 5)
        self.assertEqual(len(suite["cases"]), 60)

    def test_cuad_source_loader_flattens_official_shape(self):
        source_payload = {
            "data": [
                {
                    "title": "Contract A",
                    "paragraphs": [
                        {
                            "context": "The agreement is governed by New York law.",
                            "qas": [
                                {
                                    "id": "row_1",
                                    "question": 'Highlight the parts related to "Governing Law" that should be reviewed.',
                                    "answers": [
                                        {
                                            "text": "governed by New York law",
                                            "answer_start": 17,
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train_separate_questions.json").write_text(
                json.dumps(source_payload), encoding="utf-8"
            )
            (root / "test.json").write_text(json.dumps(source_payload), encoding="utf-8")
            rows = load_cuad_source_records(source_path=root)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["split"], "train")
        self.assertEqual(rows[1]["split"], "test")
        self.assertEqual(rows[0]["answers"]["text"], ["governed by New York law"])

    def test_legal_artifact_namespace_is_separate(self):
        self.assertEqual(ARTIFACT_SUBDIR, "legal_cache_suite")

    def test_legal_runner_enables_reranker_by_default(self):
        from legal_bench.run_benchmark import build_arg_parser

        args = build_arg_parser().parse_args([])
        self.assertFalse(args.disable_reranker)
        args = build_arg_parser().parse_args(["--disable-reranker"])
        self.assertTrue(args.disable_reranker)

    def test_legal_eval_report_includes_miss_rates(self):
        manifest = {
            "answer_accuracy": 1.0,
            "cache_type_match_rate": 0.5,
            "overall_pass_rate": 0.5,
            "expected_route_counts": {"exact": 1, "semantic": 0, "knowledge": 0, "miss": 1},
            "actual_route_counts": {"exact": 0, "semantic": 0, "knowledge": 0, "miss": 2},
            "by_scenario": {},
            "by_expected_cache_type": {},
            "by_actual_cache_type": {},
            "case_cache_isolation": {"enabled": True},
            "expected_miss_rate": 0.5,
            "actual_miss_rate": 1.0,
            "unexpected_miss_rate": 0.5,
            "positive_case_miss_rate": 1.0,
        }
        rows = [
            {
                "case_id": "exact_case",
                "title": "Exact",
                "scenario": "legal_exact_positive",
                "expected_cache_type": "exact",
                "actual_cache_type": "miss",
                "expected_from_cache": True,
                "actual_from_cache": False,
                "answer_correct": True,
                "cache_type_match": False,
                "from_cache_match": False,
                "overall_pass": False,
                "prediction": "answer",
            },
            {
                "case_id": "miss_case",
                "title": "Miss",
                "scenario": "legal_semantic_negative",
                "expected_cache_type": "miss",
                "actual_cache_type": "miss",
                "expected_from_cache": False,
                "actual_from_cache": False,
                "answer_correct": True,
                "cache_type_match": True,
                "from_cache_match": True,
                "overall_pass": True,
                "prediction": "answer",
            },
        ]

        report = _build_legal_eval_report(
            Path("run"),
            "suite",
            Path("suite.json"),
            rows,
            manifest,
        )

        self.assertEqual(report["actual_miss_rate"], 1.0)
        self.assertEqual(report["unexpected_miss_rate"], 0.5)
        self.assertEqual(report["positive_case_miss_rate"], 1.0)

    def test_case_cache_namespace_isolates_cases(self):
        records = normalize_cuad_records(load_raw_records(FIXTURE_PATH))
        suite_dict = build_legal_cache_suite(records)
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = Path(tmpdir) / "suite.json"
            suite_path.write_text(json.dumps(suite_dict), encoding="utf-8")
            loaded = _load_fixture_suite(suite_path)

        first = loaded.cases[0]
        second = loaded.cases[1]
        self.assertNotEqual(
            _resolve_case_cache_namespace("base", first, True),
            _resolve_case_cache_namespace("base", second, True),
        )
        self.assertEqual(_resolve_case_cache_namespace("base", first, False), "base")


if __name__ == "__main__":
    unittest.main()
