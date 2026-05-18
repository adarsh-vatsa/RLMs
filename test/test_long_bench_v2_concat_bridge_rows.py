import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from long_bench_v2.concat_bridge_rows import (
    _iter_bridge_csv_paths,
    concatenate_bridge_rows,
    main,
)


def _write_run(root: Path, namespace: str, run_id: str, manifest: dict, rows: list[dict]) -> Path:
    run_dir = root / namespace / run_id
    run_dir.mkdir(parents=True)
    manifest = {"run_id": run_id, **manifest}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    csv_path = run_dir / "bridge_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "row_type", "answer_correct"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


class LongBenchV2ConcatBridgeRowsTests(unittest.TestCase):
    def test_iter_bridge_csv_paths_only_uses_requested_longbench_namespaces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            longbench_path = _write_run(
                root,
                "longbench_v2",
                "run_a",
                {"benchmark_target": "longbench_v2", "mode": "cache"},
                [{"case_id": "a", "row_type": "original", "answer_correct": "true"}],
            )
            _write_run(
                root,
                "official_ruler_v2",
                "run_b",
                {"benchmark_target": "ruler_v2", "mode": "cache"},
                [{"case_id": "b", "row_type": "ruler", "answer_correct": "true"}],
            )

            paths = _iter_bridge_csv_paths(root, ("longbench_v2", "longbench_v2_rlm", "longbench_v2_api"))

        self.assertEqual(paths, [longbench_path])

    def test_concatenate_bridge_rows_adds_run_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path = _write_run(
                root,
                "longbench_v2",
                "run_cache",
                {"benchmark_target": "longbench_v2", "mode": "cache", "executor_model": "model-a"},
                [{"case_id": "a", "row_type": "original", "answer_correct": "true"}],
            )
            api_path = _write_run(
                root,
                "longbench_v2_api",
                "run_api",
                {"benchmark_target": "longbench_v2_api", "baseline_type": "plain_api"},
                [{"case_id": "b", "row_type": "semantic", "answer_correct": "false"}],
            )

            rows, columns = concatenate_bridge_rows([cache_path, api_path])

        self.assertEqual(len(rows), 2)
        self.assertIn("experiment_namespace", columns)
        self.assertIn("case_id", columns)
        self.assertEqual(rows[0]["experiment_namespace"], "longbench_v2")
        self.assertEqual(rows[0]["mode"], "cache")
        self.assertEqual(rows[1]["experiment_namespace"], "longbench_v2_api")
        self.assertEqual(rows[1]["baseline_type"], "plain_api")

    def test_main_rejects_non_longbench_namespace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                main([
                    "--artifact-root",
                    tmpdir,
                    "--namespaces",
                    "official_ruler_v2",
                ])


if __name__ == "__main__":
    unittest.main()
