import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_dp_memo_runs import compact_nulls, resolve_manifest, summarize_manifest


class SummarizeDPMemoRunsTests(unittest.TestCase):
    def test_resolve_manifest_finds_latest_nested_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "20260101T000000Z"
            new = root / "20260102T000000Z"
            old.mkdir()
            new.mkdir()
            (old / "manifest.json").write_text('{"run": "old"}', encoding="utf-8")
            (new / "manifest.json").write_text('{"run": "new"}', encoding="utf-8")

            self.assertEqual(resolve_manifest(root), new / "manifest.json")

    def test_summarize_manifest_extracts_core_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "benchmark": "dp_memo_nolima",
                        "data_kind": "official_or_downloaded",
                        "corpus_id": "dp_memo_nolima_stable_test",
                        "model": "q3",
                        "settings": {"lengths": [8000], "solver_mode": "evidence"},
                        "totals": {
                            "samples": 1,
                            "accuracy_contains": 1.0,
                            "model_calls": 21,
                            "aggregate_calls": 1,
                            "avg_latency_ms": 123.4,
                            "initial_coverage_ratio": 0.0,
                            "final_coverage_ratio": 1.0,
                            "exact_replay_checks": 1,
                        },
                        "memo_entries": 22,
                        "memo_stats": {
                            "entry_count": 22,
                            "by_fragment_kind": {"supporting_fact": 1},
                            "dependency_edge_count": 1,
                            "evidence_span_count": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_manifest(path)

        self.assertEqual(summary["benchmark"], "dp_memo_nolima")
        self.assertEqual(summary["data_kind"], "official_or_downloaded")
        self.assertEqual(summary["corpus_id"], "dp_memo_nolima_stable_test")
        self.assertEqual(summary["lengths"], [8000])
        self.assertEqual(summary["solver_mode"], "evidence")
        self.assertEqual(summary["avg_latency_ms"], 123.4)
        self.assertEqual(summary["initial_coverage_ratio"], 0.0)
        self.assertEqual(summary["fragment_mix"], {"supporting_fact": 1})
        self.assertEqual(summary["dependency_edge_count"], 1)
        self.assertEqual(summary["evidence_span_count"], 2)

    def test_summarize_manifest_extracts_mutable_workload_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "benchmark": "dp_memo_mutable_workload",
                        "corpus_id": "dp_memo_mutable_workload",
                        "totals": {
                            "v1_model_calls": 4,
                            "invalidated_entries": 2,
                            "v2_model_calls": 1,
                            "v2_initial_coverage_ratio": 0.75,
                            "v2_reused_windows": 3,
                            "v2_missing_windows": 1,
                            "warm_model_calls": 0,
                        },
                        "memo_stats": {"entry_count": 7},
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_manifest(path)

        self.assertEqual(summary["benchmark"], "dp_memo_mutable_workload")
        self.assertEqual(summary["corpus_id"], "dp_memo_mutable_workload")
        self.assertEqual(summary["memo_entries"], 7)
        self.assertEqual(summary["v1_model_calls"], 4)
        self.assertEqual(summary["invalidated_entries"], 2)
        self.assertEqual(summary["v2_model_calls"], 1)
        self.assertEqual(summary["v2_initial_coverage_ratio"], 0.75)
        self.assertEqual(summary["v2_reused_windows"], 3)
        self.assertEqual(summary["v2_missing_windows"], 1)
        self.assertEqual(summary["warm_model_calls"], 0)

    def test_compact_nulls_omits_null_fields(self):
        self.assertEqual(compact_nulls({"a": 1, "b": None, "c": 0}), {"a": 1, "c": 0})


if __name__ == "__main__":
    unittest.main()
