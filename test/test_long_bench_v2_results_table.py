import csv
import json
from pathlib import Path
import tempfile
import unittest

from long_bench_v2.results_table import main


def run(suite, folder, name, started, config, records, correct):
    path = suite / folder / f"S_{name}"
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"run_id": f"S_{name}", "started_at": started,
                                                     "elapsed_seconds": 1200}))
    (path / "execution_manifest.json").write_text(json.dumps({
        "mode": "direct", "direct_overflow": "unsupported", "child_tokens": 7500, "context_window_tokens": 65536,
        **config}))
    (path / "execution.jsonl").write_text("".join(json.dumps({
        "case_id": str(index), "prediction": "A", "full_rendered_input_tokens": 200000,
        "final_rendered_input_tokens": 50000, "generation_ms": 4000, "retrieval_ms": 200, "packing_ms": 300,
        "document_embedding_ms": 0, "attempts": 1, **record}) + "\n" for index, record in enumerate(records)))
    (path / "evaluation.jsonl").write_text("".join(json.dumps({
        "case_id": str(index), "status": "ok", "scores": {"accuracy": value}}) + "\n"
        for index, value in enumerate(correct)))


class ResultsTableTests(unittest.TestCase):
    def test_direct_and_hybrid_runs_share_one_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary) / "S"
            run(suite, "longbench_v2_api", "middle", "2026-10-26T00:00:00", {"direct_overflow": "middle"},
                [{}, {"prediction": "X"}, {"attempts": 2}, {}], [True, False, False, False])
            run(suite, "longbench_v2", "ladder3", "2026-10-26T01:00:00", {
                "mode": "hybrid", "evidence_selection": "fusion", "child_tokens": 1024, "neighbor_children": 1,
                "packing": "fill"}, [{"document_embedding_ms": 90000}, {}, {}, {}], [True, True, True, False])
            run(suite, "longbench_v2_api", "full", "2026-10-25T23:00:00", {"context_window_tokens": 262144},
                [{"final_rendered_input_tokens": 200000}] * 4, [True, True, False, True])
            main([str(suite)])

            rows = {row["condition"]: row for row in csv.DictReader((suite / "results.csv").open())}
            self.assertEqual(list(rows), ["full", "middle", "ladder3"])
            self.assertEqual((rows["full"]["policy"], rows["full"]["prompt kept"], rows["full"]["gained vs middle"]),
                             ("full prompt", "1.0", ""))
            middle = rows["middle"]
            self.assertEqual((middle["accuracy"], middle["invalid answers"], middle["retried requests"],
                              middle["retrieval + packing s/question"]), ("0.25", "1", "1", ""))
            ladder = rows["ladder3"]
            self.assertEqual(ladder["policy"], "fusion, 1024-token chunks, 1 neighbour each side, fill packing")
            self.assertEqual((ladder["correct"], ladder["gained vs middle"], ladder["lost vs middle"],
                              ladder["p vs middle"]), ("3/4", "2", "0", "0.5"))
            self.assertEqual((ladder["embedding minutes"], ladder["retrieval + packing s/question"],
                              ladder["run minutes"]), ("2", "0.5", "20"))
            self.assertIn("| ladder3 |", (suite / "results.md").read_text())


if __name__ == "__main__":
    unittest.main()
