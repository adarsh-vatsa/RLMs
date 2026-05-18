import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from long_bench_v2.run_api_benchmark import (
    ARTIFACT_SUBDIR,
    build_arg_parser,
    parse_api_usage,
    run_longbench_api_benchmark,
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


class FakeUsage:
    def __init__(self):
        self.input_tokens = 500
        self.output_tokens = 25


class FakeTextBlock:
    text = "Final answer: C"


class FakeResponse:
    def __init__(self):
        self.usage = FakeUsage()
        self.content = [FakeTextBlock()]


class FakeMessages:
    def __init__(self, calls, fail=False):
        self.calls = calls
        self.fail = fail

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("context length exceeded")
        return FakeResponse()


class FakeClient:
    def __init__(self, calls, fail=False):
        self.messages = FakeMessages(calls, fail=fail)


class LongBenchV2ApiBenchmarkTests(unittest.TestCase):
    def test_cli_defaults_to_anthropic_sonnet(self):
        parser = build_arg_parser()
        args = parser.parse_args([])

        self.assertEqual(args.api_provider, "anthropic")
        self.assertEqual(args.api_model, "claude-sonnet-4-5")
        self.assertEqual(args.max_output_tokens, 256)

    def test_parse_api_usage_from_anthropic_response(self):
        usage = parse_api_usage(FakeResponse(), model="claude-sonnet-4-5", success=True)

        self.assertEqual(usage["calls"], 1)
        self.assertEqual(usage["input_tokens"], 500)
        self.assertEqual(usage["output_tokens"], 25)
        self.assertEqual(usage["total_tokens"], 525)
        self.assertEqual(usage["cost_usd"], 0.001875)
        self.assertEqual(usage["usage_parse_status"], "parsed")

    def test_fake_api_run_writes_uncached_artifacts(self):
        calls = []

        def fake_factory():
            return FakeClient(calls)

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
                api_provider="anthropic",
                api_model="claude-sonnet-4-5",
                api_key_env="ANTHROPIC_API_KEY",
                max_output_tokens=256,
                fail_fast=False,
                output_dir=root / "artifacts",
                manifest_note="test run",
            )

            run_longbench_api_benchmark(args, client_factory=fake_factory)

            run_root = root / "artifacts" / ARTIFACT_SUBDIR
            run_dirs = list(run_root.iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            manifest = json.loads((run_dir / "manifest.json").read_text())
            report = json.loads((run_dir / "official_longbench_v2_api_eval_report.json").read_text())
            bridge_row = json.loads((run_dir / "bridge_rows.jsonl").read_text().splitlines()[0])
            prediction_row = json.loads((run_dir / "predictions.jsonl").read_text().splitlines()[0])

        self.assertEqual(manifest["benchmark_target"], "longbench_v2_api")
        self.assertEqual(manifest["baseline_type"], "plain_api_uncached_full_context")
        self.assertFalse(manifest["cache_reuse"]["enabled"])
        self.assertEqual(manifest["row_type_counts"], {"original": 1})
        self.assertEqual(manifest["answer_accuracy"], 1.0)
        self.assertEqual(manifest["api_error_count"], 0)
        self.assertEqual(manifest["total_api_calls"], 1)
        self.assertEqual(manifest["total_input_tokens"], 500)
        self.assertEqual(manifest["total_output_tokens"], 25)
        self.assertEqual(report["benchmark_target"], "longbench_v2_api")
        self.assertTrue(bridge_row["answer_correct"])
        self.assertEqual(bridge_row["token_count"], "123")
        self.assertEqual(bridge_row["api_provider"], "anthropic")
        self.assertEqual(bridge_row["api_model"], "claude-sonnet-4-5")
        self.assertEqual(bridge_row["api_status"], "ok")
        self.assertEqual(prediction_row["prediction"], "C")
        self.assertEqual(prediction_row["answer"], "C")
        self.assertEqual(calls[0]["model"], "claude-sonnet-4-5")
        self.assertEqual(calls[0]["max_tokens"], 256)
        self.assertIn("Context:", calls[0]["messages"][0]["content"])
        self.assertIn("Choices:", calls[0]["messages"][0]["content"])

    def test_api_error_records_failed_row_and_continues(self):
        calls = []

        def fake_factory():
            return FakeClient(calls, fail=True)

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
                row_types="original",
                max_rows=0,
                api_provider="anthropic",
                api_model="claude-sonnet-4-5",
                api_key_env="ANTHROPIC_API_KEY",
                max_output_tokens=256,
                fail_fast=False,
                output_dir=root / "artifacts",
                manifest_note="",
            )

            run_longbench_api_benchmark(args, client_factory=fake_factory)

            run_dir = next((root / "artifacts" / ARTIFACT_SUBDIR).iterdir())
            manifest = json.loads((run_dir / "manifest.json").read_text())
            bridge_row = json.loads((run_dir / "bridge_rows.jsonl").read_text().splitlines()[0])

        self.assertEqual(manifest["api_error_count"], 1)
        self.assertEqual(manifest["total_api_calls"], 0)
        self.assertEqual(bridge_row["api_status"], "error")
        self.assertIn("context length exceeded", bridge_row["api_error"])
        self.assertFalse(bridge_row["answer_correct"])


if __name__ == "__main__":
    unittest.main()
