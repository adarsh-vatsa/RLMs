import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from long_bench_v2.run_rlm_benchmark import (
    ARTIFACT_SUBDIR,
    build_arg_parser,
    run_longbench_rlm_benchmark,
)


def _source_row(row_id: str, context: str = "The correct answer is Gamma.") -> dict:
    return {
        "_id": row_id,
        "context": context,
        "question": "Which option is correct?",
        "choice_A": "Alpha",
        "choice_B": "Beta",
        "choice_C": "Gamma",
        "choice_D": "Delta",
        "answer": "C",
    }


def _suite_row(source_id: str, row_type: str = "original") -> dict:
    return {
        "case_id": f"{source_id}__{row_type}",
        "source_id": source_id,
        "row_type": row_type,
        "is_scored": "true",
        "setup_case_id": "",
        "context_id": "ctx",
        "token_count": "123",
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
        "answer": "C",
    }


class FakeUsageObject:
    def __init__(self):
        self.total_calls = 2
        self.input_tokens = 500
        self.output_tokens = 25


class FakeRLMResult:
    response = "Final answer: C"
    execution_time = 0.75
    root_model = "claude-sonnet-4-5"
    usage_summary = FakeUsageObject()


class FakeRLM:
    def __init__(self, calls):
        self.calls = calls

    def completion(self, prompt, root_prompt=None):
        self.calls.append({"prompt": prompt, "root_prompt": root_prompt})
        return FakeRLMResult()


class LongBenchV2RlmBenchmarkTests(unittest.TestCase):
    def test_cli_defaults_to_anthropic_sonnet(self):
        parser = build_arg_parser()
        args = parser.parse_args([])

        self.assertEqual(args.rlm_backend, "anthropic")
        self.assertEqual(args.rlm_model, "claude-sonnet-4-5")

    def test_fake_rlm_run_writes_uncached_artifacts(self):
        calls = []

        def fake_factory():
            return FakeRLM(calls)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "data.json"
            suite_path = root / "suite.csv"
            source_path.write_text(json.dumps([_source_row("row_1")]), encoding="utf-8")
            with suite_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(_suite_row("row_1").keys()))
                writer.writeheader()
                writer.writerow(_suite_row("row_1", "original"))

            args = types.SimpleNamespace(
                suite_csv=suite_path,
                source_json_path=source_path,
                row_types="original,exact,semantic",
                max_rows=0,
                rlm_backend="anthropic",
                rlm_model="claude-sonnet-4-5",
                rlm_environment="local",
                rlm_max_iterations=30,
                rlm_max_depth=1,
                rlm_api_key_env="ANTHROPIC_API_KEY",
                rlm_verbose=False,
                rlm_log_trajectories=False,
                output_dir=root / "artifacts",
                manifest_note="test run",
            )

            run_longbench_rlm_benchmark(args, rlm_factory=fake_factory)

            run_root = root / "artifacts" / ARTIFACT_SUBDIR
            run_dirs = list(run_root.iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            manifest = json.loads((run_dir / "manifest.json").read_text())
            report = json.loads((run_dir / "official_longbench_v2_rlm_eval_report.json").read_text())
            bridge_row = json.loads((run_dir / "bridge_rows.jsonl").read_text().splitlines()[0])
            prediction_row = json.loads((run_dir / "predictions.jsonl").read_text().splitlines()[0])

        self.assertEqual(manifest["benchmark_target"], "longbench_v2_rlm")
        self.assertEqual(manifest["baseline_type"], "rlm_uncached")
        self.assertFalse(manifest["cache_reuse"]["enabled"])
        self.assertEqual(manifest["row_type_counts"], {"original": 1})
        self.assertEqual(manifest["answer_accuracy"], 1.0)
        self.assertEqual(manifest["total_api_calls"], 2)
        self.assertEqual(manifest["total_input_tokens"], 500)
        self.assertEqual(manifest["total_output_tokens"], 25)
        self.assertEqual(report["benchmark_target"], "longbench_v2_rlm")
        self.assertTrue(bridge_row["answer_correct"])
        self.assertEqual(bridge_row["token_count"], "123")
        self.assertEqual(bridge_row["delta_input_tokens"], 500)
        self.assertEqual(bridge_row["delta_output_tokens"], 25)
        self.assertEqual(bridge_row["rlm_root_model"], "claude-sonnet-4-5")
        self.assertEqual(prediction_row["prediction"], "C")
        self.assertEqual(prediction_row["answer"], "C")
        self.assertIn("Choices:", calls[0]["prompt"]["question"])
        self.assertIn("A, B, C, or D", calls[0]["root_prompt"])


if __name__ == "__main__":
    unittest.main()
