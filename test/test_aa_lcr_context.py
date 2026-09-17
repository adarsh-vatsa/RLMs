import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aa_lcr.run_benchmark import build_arg_parser, resolve_experiment
from execution.client import served_context_window


class ContextBudgetTests(unittest.TestCase):
    def test_default_uses_served_context_and_reserves_output(self):
        for mode in ("direct", "hybrid"):
            with self.subTest(mode=mode):
                args = build_arg_parser().parse_args(["--mode", mode, "--api-key-env", "TEST_EXECUTOR_KEY"])
                with patch("aa_lcr.run_benchmark.served_context_window", return_value=131072) as discover, patch.dict(os.environ, TEST_EXECUTOR_KEY="test-key"):
                    experiment = resolve_experiment(args)
                self.assertEqual(experiment.mode, mode)
                self.assertEqual(experiment.context_window_tokens, 131072)
                self.assertEqual(experiment.max_input_tokens, 131072 - 16384)
                self.assertFalse(experiment.allow_direct_truncation)
                self.assertEqual(discover.call_args.kwargs["api_key"], "test-key")
                self.assertEqual(discover.call_args.kwargs["model"], args.executor_model)

    def test_explicit_limit_is_offline_and_input_can_be_narrowed(self):
        for extra, expected in (([], 61440), (["--max-input-tokens", "60000"], 60000)):
            args = build_arg_parser().parse_args([
                "--mode", "direct", "--context-window-tokens", "65536",
                "--max-output-tokens", "4096", *extra,
            ])
            with patch("aa_lcr.run_benchmark.served_context_window", side_effect=AssertionError("offline")):
                experiment = resolve_experiment(args)
            self.assertEqual(experiment.context_window_tokens, 65536)
            self.assertEqual(experiment.max_input_tokens, expected)

    def test_invalid_budgets(self):
        for extra in (
            ["--context-window-tokens", "0"],
            ["--max-input-tokens", "-1"],
            ["--max-output-tokens", "0"],
            ["--context-window-tokens", "16384"],
            ["--context-window-tokens", "65536", "--max-input-tokens", "60000"],
        ):
            with self.subTest(extra=extra), patch("aa_lcr.run_benchmark.served_context_window", side_effect=AssertionError("unexpected discovery")):
                args = build_arg_parser().parse_args(["--mode", "direct", *extra])
                with self.assertRaises(ValueError):
                    resolve_experiment(args)

    def test_legacy_budgets_remain_offline(self):
        for name, context, input_limit, output in (
            ("direct_64k", 65536, 60000, 512),
            ("hybrid_64k", 65536, 60000, 512),
            ("direct_262k", 262144, 240000, 16384),
            ("hybrid_262k", 262144, 240000, 16384),
        ):
            args = build_arg_parser().parse_args(["--experiment", name])
            with patch("aa_lcr.run_benchmark.served_context_window", side_effect=AssertionError("offline")):
                experiment = resolve_experiment(args)
            self.assertEqual((experiment.context_window_tokens, experiment.max_input_tokens, args.max_output_tokens), (context, input_limit, output))

    def test_discovery_selects_exact_model_and_passes_credentials(self):
        def opener(request, timeout):
            self.assertEqual(request.full_url, "http://executor/v1/models")
            self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
            self.assertEqual(timeout, 12)
            return io.BytesIO(json.dumps({"data": [
                {"id": "other", "max_model_len": 999999},
                {"id": "selected", "max_model_len": 65536},
            ]}).encode())
        self.assertEqual(served_context_window(
            base_url="http://executor/v1/", model="selected", api_key="test-key",
            timeout_seconds=12, opener=opener,
        ), 65536)

    def test_discovery_failure_requires_explicit_limit(self):
        for payload in (
            {"data": []}, {"data": [{"id": "other", "max_model_len": 65536}]},
            *({"data": [{"id": "selected", "max_model_len": value}]} for value in (None, 0, -1, True, "65536")),
        ):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "--context-window-tokens"):
                served_context_window(base_url="http://executor/v1", model="selected",
                    opener=lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))
        def unavailable(*args, **kwargs):
            raise OSError("offline")
        with self.assertRaisesRegex(ValueError, "--context-window-tokens"):
            served_context_window(base_url="http://executor/v1", model="selected", opener=unavailable)

    def test_launcher_forwards_separate_mode_and_optional_budgets(self):
        env = {**os.environ, "AA_LCR_LAUNCH_DRY_RUN": "1", "OPENAI_COMPAT_EXECUTOR_BASE_URL": "http://executor/v1"}
        for mode in ("direct", "hybrid"):
            for extra in ([], ["--context-window-tokens", "65536", "--max-input-tokens", "60000", "--max-output-tokens", "4096"]):
                result = subprocess.run(["bash", "jarvis/run_aa_lcr.sh", mode,
                    "--execution-profile", "common", "--execution-only", *extra],
                    cwd=Path(__file__).resolve().parents[1], env=env, check=True, capture_output=True, text=True)
                command = shlex.split(next(line.split("command=", 1)[1] for line in result.stdout.splitlines() if "command=" in line))
                parsed = build_arg_parser().parse_args(command[command.index("aa_lcr.run_benchmark") + 1:])
                self.assertEqual(parsed.mode, mode)
                self.assertIsNone(parsed.experiment)
                self.assertEqual(parsed.context_window_tokens, 65536 if extra else None)
                self.assertEqual(parsed.max_input_tokens, 60000 if extra else None)
                self.assertIn("allocation=client-gpu" if mode == "hybrid" else "allocation=client mem=", result.stdout)


if __name__ == "__main__":
    unittest.main()
