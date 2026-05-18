import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ruler_v2.run_rlm_benchmark import (
    ARTIFACT_SUBDIR,
    _build_default_rlm_factory,
    _filter_official_samples,
    _load_official_samples,
    _normalize_official_sample,
    build_arg_parser,
    normalize_rlm_args,
    parse_rlm_usage_summary,
    run_rlm_benchmark,
)


class FakeUsageObject:
    def __init__(self):
        self.total_calls = 3
        self.input_tokens = 1200
        self.output_tokens = 80


class FakeRLMResult:
    response = "ORCHID-57"
    execution_time = 1.25
    root_model = "claude-haiku-4-5-20251001"
    usage_summary = FakeUsageObject()


class FakeRLM:
    def __init__(self, calls):
        self.calls = calls

    def completion(self, prompt, root_prompt=None):
        self.calls.append({"prompt": prompt, "root_prompt": root_prompt})
        return FakeRLMResult()


class CapturingRLM:
    last_kwargs = None

    def __init__(self, **kwargs):
        CapturingRLM.last_kwargs = kwargs


class RulerRlmBenchmarkTests(unittest.TestCase):
    def test_load_normalize_and_filter_ruler_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = (
                Path(tmpdir)
                / "data_8192"
                / "mk_niah_basic"
                / "test.jsonl"
            )
            test_path.parent.mkdir(parents=True)
            test_path.write_text(
                json.dumps(
                    {
                        "id": "sample_a",
                        "context": "The launch codeword is ORCHID-57.",
                        "question": "What is the launch codeword?",
                        "answer": ["ORCHID-57"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            raw = _load_official_samples(Path(tmpdir))
            normalized = [_normalize_official_sample(raw[0], 1, "")]
            selected = _filter_official_samples(
                normalized,
                selected_tasks=["mk_niah_basic"],
                selected_lengths=["8192"],
                max_samples_per_task=1,
            )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["task"], "mk_niah_basic")
        self.assertEqual(selected[0]["length"], "8192")
        self.assertEqual(selected[0]["expected_answer"], ["ORCHID-57"])

    def test_parse_usage_summary_from_object(self):
        parsed = parse_rlm_usage_summary(
            FakeUsageObject(),
            model="claude-haiku-4-5-20251001",
        )

        self.assertEqual(parsed["calls"], 3)
        self.assertEqual(parsed["input_tokens"], 1200)
        self.assertEqual(parsed["output_tokens"], 80)
        self.assertEqual(parsed["total_tokens"], 1280)
        self.assertEqual(parsed["cost_usd"], 0.0016)
        self.assertEqual(parsed["usage_parse_status"], "parsed")

    def test_cli_defaults_to_anthropic_haiku(self):
        parser = build_arg_parser()
        args = normalize_rlm_args(parser.parse_args(["--official-prepared-data", "benchmark_data/ruler2"]))

        self.assertEqual(args.rlm_backend, "anthropic")
        self.assertEqual(args.rlm_model, "claude-haiku-4-5-20251001")
        self.assertEqual(args.rlm_base_url, "")
        self.assertEqual(args.rlm_api_key_env, "ANTHROPIC_API_KEY")

    def test_cli_openrouter_base_url_defaults_to_openrouter_key_env(self):
        parser = build_arg_parser()
        args = normalize_rlm_args(
            parser.parse_args(
                [
                    "--official-prepared-data",
                    "benchmark_data/ruler2",
                    "--rlm-base-url",
                    "https://openrouter.ai/api/v1",
                ]
            )
        )

        self.assertEqual(args.rlm_api_key_env, "OPENROUTER_API_KEY")

    def test_rlm_factory_passes_base_url_to_backend_kwargs(self):
        fake_rlm_module = types.ModuleType("rlm")
        fake_rlm_module.RLM = CapturingRLM
        fake_logger_module = types.ModuleType("rlm.logger")

        class FakeLogger:
            def __init__(self, log_dir):
                self.log_dir = log_dir

        fake_logger_module.RLMLogger = FakeLogger
        args = types.SimpleNamespace(
            rlm_backend="openai",
            rlm_model="anthropic/claude-sonnet-4.5",
            rlm_environment="local",
            rlm_max_depth=1,
            rlm_max_iterations=30,
            rlm_api_key_env=None,
            rlm_base_url="https://openrouter.ai/api/v1",
            rlm_log_trajectories=False,
            rlm_verbose=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            previous_rlm = sys.modules.get("rlm")
            previous_logger = sys.modules.get("rlm.logger")
            sys.modules["rlm"] = fake_rlm_module
            sys.modules["rlm.logger"] = fake_logger_module
            try:
                factory = _build_default_rlm_factory(args, Path(tmpdir))
                factory()
            finally:
                if previous_rlm is None:
                    sys.modules.pop("rlm", None)
                else:
                    sys.modules["rlm"] = previous_rlm
                if previous_logger is None:
                    sys.modules.pop("rlm.logger", None)
                else:
                    sys.modules["rlm.logger"] = previous_logger

        backend_kwargs = CapturingRLM.last_kwargs["backend_kwargs"]
        self.assertEqual(CapturingRLM.last_kwargs["backend"], "openai")
        self.assertEqual(backend_kwargs["model_name"], "anthropic/claude-sonnet-4.5")
        self.assertEqual(backend_kwargs["base_url"], "https://openrouter.ai/api/v1")

    def test_fake_rlm_run_writes_uncached_artifacts(self):
        calls = []

        def fake_factory():
            return FakeRLM(calls)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared = root / "prepared" / "data_8192" / "mk_niah_basic"
            prepared.mkdir(parents=True)
            (prepared / "test.jsonl").write_text(
                json.dumps(
                    {
                        "id": "sample_a",
                        "context": "The launch codeword is ORCHID-57.",
                        "question": "What is the launch codeword?",
                        "answer": ["ORCHID-57"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = types.SimpleNamespace(
                official_prepared_data=str(root / "prepared"),
                official_tasks="mk_niah_basic",
                official_lengths="8192",
                official_max_samples_per_task=1,
                official_eval_command="",
                rlm_backend="anthropic",
                rlm_model="claude-haiku-4-5-20251001",
                rlm_environment="local",
                rlm_max_iterations=30,
                rlm_max_depth=1,
                rlm_api_key_env="ANTHROPIC_API_KEY",
                rlm_verbose=False,
                rlm_log_trajectories=False,
                output_dir=str(root / "artifacts"),
                manifest_note="",
            )

            run_rlm_benchmark(args, rlm_factory=fake_factory)

            run_root = root / "artifacts" / ARTIFACT_SUBDIR
            run_dirs = list(run_root.iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            manifest = json.loads((run_dir / "manifest.json").read_text())
            bridge_row = json.loads((run_dir / "bridge_rows.jsonl").read_text().splitlines()[0])
            prediction_row = json.loads((run_dir / "predictions.jsonl").read_text().splitlines()[0])

        self.assertEqual(manifest["official_target"], "ruler_v2_rlm")
        self.assertEqual(manifest["baseline_type"], "rlm_uncached")
        self.assertFalse(manifest["cache_reuse"]["enabled"])
        self.assertEqual(manifest["total_api_calls"], 3)
        self.assertEqual(bridge_row["delta_input_tokens"], 1200)
        self.assertEqual(bridge_row["delta_output_tokens"], 80)
        self.assertEqual(bridge_row["rlm_root_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(prediction_row["generation"], "ORCHID-57")
        self.assertEqual(calls[0]["prompt"]["question"], "What is the launch codeword?")
        self.assertIn("provided context", calls[0]["root_prompt"])


if __name__ == "__main__":
    unittest.main()
