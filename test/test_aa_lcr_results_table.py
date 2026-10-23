import csv
import json
from pathlib import Path
import tempfile
import unittest

from aa_lcr.results_table import main


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def run(suite, mode, name, started, window, execution, answers):
    path = suite / mode / f"S_{name}"
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({
        "run_id": f"S_{name}", "started_at": started, "context_window_tokens": window, "elapsed_seconds": 780,
        "execution": {"mode": mode, "direct_overflow": "unsupported", "child_tokens": 7500, **execution}}))
    write_rows(path / "bridge_rows.jsonl", [{
        "question_id": str(index), "final_rendered_input_tokens": 50, "full_rendered_input_tokens": 100,
        "executor_output_tokens": 10, "executor_finish_reason": "stop", "executor_attempts": 1,
        "document_embedding_ms": 0, "data_source_filenames": ["a", "b"], "selected_evidence_ranges": [], **answer}
        for index, answer in enumerate(answers)])
    write_rows(path / "execution.jsonl", [{"generation_ms": 2000, "retrieval_ms": 100, "packing_ms": 400}] * len(answers))


def grade(suite, grader, mode, name, correct):
    path = suite / "regrades" / grader / mode / f"S_{name}"
    path.mkdir(parents=True)
    (path / "manifest.json").write_text("{}")
    write_rows(path / "bridge_rows.jsonl", [{"question_id": str(index), "grade_valid": value is not None,
                                             "answer_correct": value} for index, value in enumerate(correct)])


class ResultsTableTests(unittest.TestCase):
    def test_one_row_per_run_with_paired_and_grader_comparisons(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary) / "S"
            run(suite, "direct", "full", "2026-10-23T00", 262144, {}, [{}] * 3)
            run(suite, "direct", "middle", "2026-10-23T01", 65536, {"direct_overflow": "middle"},
                [{"executor_finish_reason": "length"}, {}, {"executor_attempts": 2}])
            covered = [{"document_id": "a"}, {"document_id": "b"}]
            run(suite, "hybrid", "upgraded", "2026-10-23T02", 65536, {
                "evidence_selection": "random", "selection_seed": 1, "child_tokens": 1024, "neighbor_children": 1,
                "packing": "fill", "document_floor": True}, [
                {"selected_evidence_ranges": covered, "document_embedding_ms": 120000},
                {"selected_evidence_ranges": covered[:1]}, {"selected_evidence_ranges": covered}])
            grade(suite, "G", "direct", "middle", [True, False, None])
            grade(suite, "G", "hybrid", "upgraded", [True, True, False])
            grade(suite, "Other", "hybrid", "upgraded", [True, False, False])
            main([str(suite), "--grader", "G"])

            rows = {row["condition"]: row for row in csv.DictReader((suite / "results-G.csv").open())}
            self.assertEqual(list(rows), ["full", "middle", "upgraded"])
            self.assertEqual((rows["full"]["policy"], rows["full"]["window"], rows["full"]["accuracy"]),
                             ("full prompt", "262144", "not graded"))
            middle, upgraded = rows["middle"], rows["upgraded"]
            self.assertEqual((middle["policy"], middle["correct"], middle["accuracy"]),
                             ("truncation: middle", "1/2 (1 invalid)", "0.5"))
            self.assertEqual((middle["output limit hit"], middle["retried requests"], middle["gained vs middle"]),
                             ("1", "1", ""))
            self.assertEqual(upgraded["policy"], "random seed 1, 1024-token chunks, 1 neighbour each side, "
                                                 "fill packing, document floor")
            self.assertEqual((upgraded["documents covered"], upgraded["questions missing a document"]), ("0.8333", "1"))
            self.assertEqual((upgraded["gained vs middle"], upgraded["lost vs middle"]), ("1", "0"))
            self.assertEqual((upgraded["agreement with Other"], upgraded["embedding minutes"]), ("0.6667", "2"))
            self.assertEqual(upgraded["retrieval + packing s/question"], "0.5")
            table = (suite / "results-G.md").read_text()
            self.assertIn("| middle | truncation: middle | 65,536 | 50% | 50 |  |  | 50.0% | 1/2 (1 invalid) |", table)
            self.assertIn("| 83% | 1 | 66.7% | 2/3 | 1 | 0 | 67% |", table)


if __name__ == "__main__":
    unittest.main()
