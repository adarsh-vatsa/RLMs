import tempfile
import unittest
from pathlib import Path

from scripts.run_dp_memo_mutable_workload import run_workload


class MutableWorkloadBenchmarkTests(unittest.TestCase):
    def test_mutable_workload_reuses_unaffected_chunks_and_replays_warm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = run_workload(
                duckdb_path=root / "memo.duckdb",
                output_dir=root / "out",
                reset_db=True,
            )

        totals = manifest["totals"]

        self.assertEqual(totals["v1_model_calls"], 4)
        self.assertEqual(totals["invalidated_entries"], 2)
        self.assertEqual(totals["v2_model_calls"], 1)
        self.assertEqual(totals["v2_reused_windows"], 3)
        self.assertEqual(totals["v2_missing_windows"], 1)
        self.assertEqual(totals["warm_model_calls"], 0)
        self.assertIn("fact:Backup channel is east-relay-3.", manifest["answers"]["v2"])
        self.assertNotIn("fact:Backup channel is north-bridge-9.", manifest["answers"]["v2"])


if __name__ == "__main__":
    unittest.main()
