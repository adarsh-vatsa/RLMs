import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from aa_lcr.intervals import main as aa_lcr_main
from execution.intervals import compare, resample_weights
from long_bench_v2.intervals import main as long_bench_main
from mrcr_v2.intervals import main as mrcr_main


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def condition(window, outcomes, middle=False):
    return {"window": window, "middle": middle, "outcomes": dict(zip("abcd", outcomes))}


def read_csv(path):
    with path.open() as handle:
        return {(row["condition"], row["baseline"]): row for row in csv.DictReader(handle)}


class ResampleTests(unittest.TestCase):
    def test_clusters_are_drawn_whole(self):
        questions, weights = resample_weights({"a": "x", "b": "x", "c": "y", "d": "z"}, dict.fromkeys("abcd", ""),
                                              replicates=500)
        self.assertEqual(questions, ["a", "b", "c", "d"])
        np.testing.assert_array_equal(weights[:, 0], weights[:, 1])
        np.testing.assert_array_equal(weights[:, 1] + weights[:, 2] + weights[:, 3], 3)
        np.testing.assert_array_equal(weights, resample_weights({"a": "x", "b": "x", "c": "y", "d": "z"},
                                                                dict.fromkeys("abcd", ""), replicates=500)[1])

    def test_strata_keep_their_size(self):
        strata = {"a": "short", "b": "short", "c": "short", "d": "long"}
        _, weights = resample_weights({question: question for question in strata}, strata, replicates=500)
        np.testing.assert_array_equal(weights[:, :3].sum(axis=1), 3)
        np.testing.assert_array_equal(weights[:, 3], 1)


class CompareTests(unittest.TestCase):
    def test_each_condition_against_middle_at_its_window_then_listed_pairs(self):
        conditions = {
            "middle": condition(65536, [0, 0, 1, 0], middle=True), "dense": condition(65536, [1, 1, 1, 0]),
            "position": condition(65536, [0, 0, 1, 0]), "middle_32k": condition(32768, [0, 0, 0, 0], middle=True),
            "bm25_32k": condition(32768, [1, 0, 0, 0]), "full": condition(262144, [1, 1, 1, 1]),
        }
        columns, rows = compare(conditions, {"a": "x", "b": "x", "c": "y", "d": "z"}, metric="accuracy",
                                replicates=2000)
        self.assertIn("baseline accuracy", columns)
        self.assertEqual([(row["condition"], row["baseline"]) for row in rows],
                         [("dense", "middle"), ("position", "middle"), ("bm25_32k", "middle_32k"),
                          ("full", "middle"), ("dense", "position")])
        dense = rows[0]
        self.assertEqual((dense["accuracy"], dense["baseline accuracy"], dense["difference"]), (0.75, 0.25, 0.5))
        self.assertEqual((dense["gained"], dense["lost"]), (2, 0))
        self.assertLessEqual(dense["95% low"], 0.5)
        self.assertGreaterEqual(dense["95% high"], 0.5)
        self.assertEqual((rows[1]["95% low"], rows[1]["95% high"]), (0, 0))
        self.assertEqual((rows[3]["window"], rows[3]["baseline window"]), (262144, 65536))
        self.assertNotIn("reference", columns)

    def test_gap_recovered_has_an_interval_only_when_the_gap_stays_positive(self):
        conditions = {
            "middle": condition(65536, [0, 0, 1, 0], middle=True), "dense": condition(65536, [1, 1, 1, 0]),
            "full": condition(262144, [1, 1, 1, 1]), "middle_32k": condition(32768, [0, 0, 1, 0], middle=True),
            "bm25_32k": condition(32768, [1, 0, 1, 0]),
        }
        references = {65536: "full", 32768: "full"}
        columns, rows = compare(conditions, {"a": "x", "b": "x", "c": "y", "d": "y"}, metric="accuracy",
                                references=references, replicates=2000)
        rows = {(row["condition"], row["baseline"]): row for row in rows}
        self.assertIn("gap recovered 95% low", columns)
        dense = rows[("dense", "middle")]
        self.assertEqual((dense["reference"], dense["gap recovered"]), ("full", 2 / 3))
        self.assertLessEqual(dense["gap recovered 95% low"], 2 / 3)
        self.assertNotIn("gap recovered", rows[("full", "middle")])
        _, rows = compare(conditions, {"a": "a", "b": "b", "c": "c", "d": "d"}, metric="accuracy",
                          references=references, replicates=2000)
        dense = next(row for row in rows if row["condition"] == "dense")
        self.assertEqual(dense["gap recovered"], 2 / 3)
        self.assertNotIn("gap recovered 95% low", dense)

    def test_rejects_a_condition_missing_a_question(self):
        conditions = {"middle": condition(65536, [0, 1, 1, 0], middle=True), "dense": condition(65536, [1, 1, 1])}
        with self.assertRaisesRegex(ValueError, "dense does not have a valid outcome"):
            compare(conditions, dict.fromkeys("abcd", "x"), metric="accuracy")


class SuiteTests(unittest.TestCase):
    def test_aa_lcr_resamples_document_sets_with_the_chosen_grader(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory) / "S"
            for mode, name, window, overflow, correct in (
                    ("direct", "full", 262144, "unsupported", [True, True, True, True]),
                    ("direct", "middle", 65536, "middle", [False, False, True, False]),
                    ("hybrid", "dense", 65536, "unsupported", [True, True, True, None])):
                path = suite / mode / f"S_{name}"
                write_rows(path / "bridge_rows.jsonl", [{"question_id": str(index), "document_set_id": f"set{index // 2}"}
                                                        for index in range(4)])
                (path / "manifest.json").write_text(json.dumps({
                    "run_id": f"S_{name}", "started_at": name, "context_window_tokens": window,
                    "execution": {"mode": mode, "direct_overflow": overflow}}))
                grades = suite / "regrades" / "luna" / mode / f"S_{name}"
                write_rows(grades / "bridge_rows.jsonl", [{"question_id": str(index), "grade_valid": True,
                                                           "answer_correct": bool(value)}
                                                          for index, value in enumerate(correct)])
                (grades / "manifest.json").write_text("{}")
            aa_lcr_main([str(suite), "--grader", "luna"])
            row = read_csv(suite / "intervals-luna.csv")[("dense", "middle")]
            self.assertEqual((row["difference"], row["gained"], row["lost"]), ("0.5", "2", "0"))
            self.assertEqual((row["reference"], row["gap recovered"]), ("full", "0.6667"))
            self.assertNotEqual(row["gap recovered 95% low"], "")
            self.assertIn("resamples of the 2 document sets", (suite / "intervals-luna.md").read_text())

    def test_mrcr_resamples_within_conversations_against_the_oracle_and_full_context(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory) / "S"
            for mode, name, window, overflow, oracle, matches in (
                    ("direct", "middle", 262144, "middle", False, [1, 0, 1, 0]),
                    ("direct", "middle_32k", 32768, "middle", False, [0, 0, 0, 0]),
                    ("hybrid", "upgraded_32k", 32768, "unsupported", False, [1, 1, 0, 0]),
                    ("hybrid", "oracle_32k", 32768, "unsupported", True, [1, 1, 1, 1])):
                path = suite / mode / f"S_{name}"
                # Conversation c0 (questions 0 and 2) fits the 262K window whole.
                write_rows(path / "predictions.jsonl", [{
                    "case_id": str(index), "source_id": f"c{index % 2}", "exact_match": bool(value),
                    "full_rendered_input_tokens": 1064000 if index % 2 else 133000,
                    "route": "direct_fit" if window == 262144 and index % 2 == 0 else "truncated"}
                    for index, value in enumerate(matches)])
                (path / "manifest.json").write_text(json.dumps({
                    "run_id": f"S_{name}", "status": "complete", "created_at": name, "context_window_tokens": window,
                    "oracle": oracle, "execution": {"mode": mode, "direct_overflow": overflow}}))
            mrcr_main([str(suite)])
            rows = read_csv(suite / "intervals.csv")
            row = rows[("upgraded_32k", "middle_32k")]
            self.assertEqual((row["exact match"], row["baseline exact match"], row["difference"]), ("0.5", "0.0", "0.5"))
            self.assertEqual((row["reference"], row["gap recovered"]), ("oracle_32k", "0.5"))
            self.assertEqual(rows[("oracle_32k", "middle_32k")]["gap recovered"], "")
            self.assertIn("2 conversations", (suite / "intervals.md").read_text())
            rows = read_csv(suite / "intervals-full-context.csv")
            self.assertEqual((rows[("upgraded_32k", "middle_32k")]["reference"],
                              rows[("upgraded_32k", "middle_32k")]["gap recovered"]), ("middle", "0.5"))
            self.assertIn("the 2 questions on the 133K conversation that `middle` sends whole",
                          (suite / "intervals-full-context.md").read_text())

    def test_long_bench_resamples_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory) / "S"
            for folder, name, mode, overflow, correct in (
                    ("longbench_v2_api", "middle", "direct", "middle", [0, 1, 0]),
                    ("longbench_v2", "bm25", "hybrid", "unsupported", [1, 1, 0])):
                path = suite / folder / f"S_{name}"
                write_rows(path / "execution.jsonl", [{"case_id": str(index), "source_content_sha256": f"doc{index // 2}"}
                                                      for index in range(3)])
                write_rows(path / "evaluation.jsonl", [{"case_id": str(index), "scores": {"accuracy": value}}
                                                       for index, value in enumerate(correct)])
                (path / "manifest.json").write_text(json.dumps({"run_id": f"S_{name}", "started_at": name}))
                (path / "execution_manifest.json").write_text(json.dumps({
                    "mode": mode, "direct_overflow": overflow, "context_window_tokens": 65536}))
            long_bench_main([str(suite)])
            row = read_csv(suite / "intervals.csv")[("bm25", "middle")]
            self.assertEqual((row["gained"], row["lost"]), ("1", "0"))
            self.assertIn("resamples of the 2 documents", (suite / "intervals.md").read_text())


if __name__ == "__main__":
    unittest.main()
