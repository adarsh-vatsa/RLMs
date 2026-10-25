import csv
import json
from pathlib import Path
import tempfile
import unittest

from mrcr_v2.results_table import main


def run(suite, mode, name, created, status, execution, answers, retrieval_query="request"):
    path = suite / mode / f"S_{name}"
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({
        "run_id": f"S_{name}", "status": status, "created_at": created, "completed_at": "2026-10-23T02:00:00",
        "context_window_tokens": 262144, "selected_example_ids": [str(i) for i in range(4)],
        "retrieval_query": retrieval_query,
        "execution": {"mode": mode, "direct_overflow": "unsupported", "child_tokens": 3500, **execution}}))
    (path / "predictions.jsonl").write_text("".join(json.dumps({
        "case_id": str(index), "source_id": "small" if index < 2 else "large",
        "full_rendered_input_tokens": 133000 if index < 2 else 1064000, "final_rendered_input_tokens": 133000,
        "mrcr_score": 0.5, "prefix_compliant": True, "output_tokens": 10, "finish_reason": "stop", "attempts": 1,
        "generation_ms": 3000, "retrieval_ms": 500, "packing_ms": 500, "document_embedding_ms": 60000,
        **answer}) + "\n" for index, answer in enumerate(answers)))


class ResultsTableTests(unittest.TestCase):
    def test_one_row_per_run_with_conversation_bands_and_paired_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary) / "S"
            run(suite, "direct", "middle", "2026-10-23T00:00:00", "complete", {"direct_overflow": "middle"},
                [{"exact_match": True}, {"exact_match": False}, {"exact_match": False}, {"exact_match": False}])
            run(suite, "hybrid", "dense_question", "2026-10-23T01:00:00", "complete", {"evidence_selection": "dense"},
                [{"exact_match": True}, {"exact_match": True}, {"exact_match": True, "finish_reason": "length"},
                 {"exact_match": False, "attempts": 2}], retrieval_query="question")
            run(suite, "hybrid", "ladder1", "2026-10-23T03:00:00", "running",
                {"evidence_selection": "fusion", "child_tokens": 1024}, [{"exact_match": True}])
            main([str(suite)])

            rows = {row["condition"]: row for row in csv.DictReader((suite / "results.csv").open())}
            self.assertEqual(list(rows), ["middle", "dense_question", "ladder1"])
            dense = rows["dense_question"]
            self.assertEqual((dense["policy"], dense["exact match"], dense["correct"]),
                             ("dense, full-question query", "0.75", "3/4"))
            self.assertEqual((dense["gained vs middle"], dense["lost vs middle"], dense["p vs middle"]), ("2", "0", "0.5"))
            self.assertEqual((dense["exact match at 133K"], dense["exact match at 1.06M"]), ("1.0", "0.5"))
            self.assertEqual((dense["output limit hit"], dense["retried requests"], dense["embedding minutes"],
                              dense["run minutes"]), ("1", "1", "4", "60"))
            self.assertEqual((rows["middle"]["gained vs middle"], rows["middle"]["embedding minutes"]), ("", ""))
            self.assertEqual((rows["ladder1"]["policy"], rows["ladder1"]["questions"], rows["ladder1"]["exact match"]),
                             ("fusion, 1024-token chunks", "1/4", "running"))
            self.assertIn("| dense_question | dense, full-question query | 262,144 | 4/4 | 56% |",
                          (suite / "results.md").read_text())


if __name__ == "__main__":
    unittest.main()
