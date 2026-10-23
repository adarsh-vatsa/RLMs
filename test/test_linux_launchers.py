import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


class LinuxLauncherTests(unittest.TestCase):
    def launch(self, script, *args, env=None, cwd=None, linux=REPO / "linux"):
        environment = {key: value for key, value in os.environ.items() if not key.startswith((
            "OPENAI_COMPAT_", "SEMANTIC_CACHE_", "AA_LCR_", "MRCR_", "EXECUTOR_", "EVALUATOR_", "ONLY",
        ))}
        environment.update(DRY_RUN="1", BENCHMARK_VENV="/tmp/client env", VLLM_VENV="/tmp/server env")
        environment.update(env or {})
        return subprocess.run(["bash", str(linux / script), *args],
            cwd=cwd or REPO, env=environment, capture_output=True, text=True)

    def command(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return shlex.split(next(line.split("command:", 1)[1] for line in result.stdout.splitlines() if "command:" in line))

    def test_all_modes_forward_budgets_and_parse_with_existing_runners(self):
        for benchmark in ("aa_lcr", "mrcr_v2", "longbench_v2"):
            for mode in ("direct", "hybrid"):
                with self.subTest(benchmark=benchmark, mode=mode):
                    command = self.command(self.launch("run_benchmark.sh", benchmark, mode,
                        "--context-window-tokens", "131072", "--max-input-tokens", "120000",
                        "--max-output-tokens", "4096", "--min-source-tokens", "100000",
                        "--max-source-tokens", "200000", "--max-rows", "2"))
                    self.assertEqual(command[0], "/tmp/client env/bin/python")
                    parser = importlib.import_module(command[2]).build_arg_parser()
                    args = parser.parse_args(command[3:])
                    self.assertEqual(args.execution_profile, "common")
                    self.assertEqual((args.context_window_tokens, args.max_input_tokens, args.max_output_tokens), (131072, 120000, 4096))
                    self.assertEqual((args.min_source_tokens, args.max_source_tokens, args.max_rows), (100000, 200000, 2))
                    if benchmark == "longbench_v2":
                        self.assertEqual(args.suite_csv, Path("benchmark_data/long_bench_v2/data.csv"))
                    self.assertEqual(args.mode if benchmark != "longbench_v2" else command[2], mode if benchmark != "longbench_v2" else ("long_bench_v2.run_benchmark" if mode == "hybrid" else "long_bench_v2.run_api_benchmark"))

    def test_remote_endpoints_and_literal_arguments(self):
        command = self.command(self.launch("run_benchmark.sh", "aa_lcr", "direct",
            "--execution-only", "--run-id", "literal $(do-not-run); with spaces",
            env={"OPENAI_COMPAT_EXECUTOR_BASE_URL": "http://remote:9000/v1", "AA_LCR_DATA_DIR": "/data/path with spaces"}))
        from aa_lcr.run_benchmark import build_arg_parser
        args = build_arg_parser().parse_args(command[3:])
        self.assertEqual(args.executor_base_url, "http://remote:9000/v1")
        self.assertEqual(args.questions_csv, Path("/data/path with spaces/AA-LCR_Dataset.csv"))
        self.assertEqual(args.run_id, "literal $(do-not-run); with spaces")
        self.assertTrue(args.execution_only)
        self.assertIsNone(args.context_window_tokens)
        self.assertIsNone(args.max_input_tokens)

    def test_services_have_separate_environments_and_context_options(self):
        command = self.command(self.launch("serve_vllm.sh", "executor", "--limit-mm-per-prompt", '{"image": 0}',
            env={"EXECUTOR_TP_SIZE": "4", "EXECUTOR_MAX_MODEL_LEN": "65536", "EXECUTOR_PORT": "9000"}))
        self.assertEqual(command[0], "/tmp/server env/bin/python")
        self.assertEqual(command[command.index("--max-model-len") + 1], "65536")
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "4")
        self.assertEqual(command[command.index("--port") + 1], "9000")
        self.assertEqual(command[-1], '{"image": 0}')
        default = self.command(self.launch("serve_vllm.sh", "executor"))
        self.assertNotIn("--max-model-len", default)
        evaluator = self.command(self.launch("serve_vllm.sh", "evaluator"))
        self.assertEqual(evaluator[evaluator.index("--max-model-len") + 1], "32768")

    def test_real_launch_executes_only_python_and_propagates_exit_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            python = Path(temporary) / "bin/python"
            python.parent.mkdir()
            python.write_text(f"#!{sys.executable}\nimport json, os, sys\nprint(json.dumps({{'argv': sys.argv[1:], 'cwd': os.getcwd(), 'search': os.getenv('SEMANTIC_CACHE_SEARCH_MODE'), 'device': os.getenv('SEMANTIC_CACHE_EMBEDDING_DEVICE')}}))\nsys.exit(17)\n")
            python.chmod(0o755)
            result = self.launch("run_benchmark.sh", "longbench_v2", "hybrid",
                env={"DRY_RUN": "0", "BENCHMARK_VENV": temporary, "SEMANTIC_CACHE_EMBEDDING_DEVICE": "cpu"}, cwd=temporary)
            self.assertEqual(result.returncode, 17, result.stderr)
            captured = json.loads(result.stdout)
            self.assertEqual(captured["cwd"], str(REPO))
            self.assertEqual(captured["search"], "hybrid")
            self.assertEqual(captured["device"], "cpu")
            self.assertEqual(captured["argv"][:2], ["-m", "long_bench_v2.run_benchmark"])

    def test_setup_dry_run_does_not_create_environments(self):
        with tempfile.TemporaryDirectory() as temporary:
            for target, variable in (("client", "BENCHMARK_VENV"), ("server", "VLLM_VENV")):
                path = Path(temporary) / target
                result = self.launch("setup.sh", target, env={variable: str(path)})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("uv venv", result.stdout)
                self.assertIn("uv pip install", result.stdout)
                self.assertFalse(path.exists())

    def test_server_setup_matches_cuda_wheel_and_checks_extension(self):
        for backend, expected in (("", "cu129"), ("cu128", "cu128")):
            result = self.launch("setup.sh", "server", env={"TORCH_BACKEND": backend})
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [shlex.split(line.split("command:", 1)[1])
                        for line in result.stdout.splitlines() if "command:" in line]
            install = next(command for command in commands if command[:3] == ["uv", "pip", "install"])
            self.assertEqual(install[install.index("--torch-backend") + 1], expected)
            self.assertIn("vllm==0.19.1", install)
            self.assertEqual(commands[-1][0], "/tmp/server env/bin/python")
            self.assertIn("import vllm._C", commands[-1][-1])

    def test_invalid_modes_and_help(self):
        for script in ("run_benchmark.sh", "serve_vllm.sh", "setup.sh"):
            self.assertEqual(self.launch(script, "--help").returncode, 0)
            self.assertEqual(self.launch(script, "invalid").returncode, 2)
        self.assertEqual(self.launch("run_benchmark.sh", "invalid", "direct").returncode, 2)
        self.assertEqual(self.launch("run_benchmark.sh", "aa_lcr", "invalid").returncode, 2)

    def suite_commands(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return [shlex.split(line.split("command:", 1)[1]) for line in result.stdout.splitlines() if "command:" in line]

    def test_aa_lcr_suite_runs_and_grades_every_condition(self):
        commands = self.suite_commands(self.launch("run_aa_lcr_suite.sh", "S", env={
            "AA_LCR_RECIPE": "--evidence-selection fusion --packing fill"}))
        runner = importlib.import_module("aa_lcr.run_benchmark").build_arg_parser()
        regrader = importlib.import_module("aa_lcr.regrade").build_arg_parser()
        runs = {args.run_id: args for args in (runner.parse_args(command[3:])
                                                for command in commands if command[2] == "aa_lcr.run_benchmark")}
        grades = [regrader.parse_args(command[3:]) for command in commands if command[2] == "aa_lcr.regrade"]
        self.assertEqual((len(runs), len(grades)), (18, 18))
        self.assertTrue(all(args.execution_only and args.fail_fast for args in runs.values()))
        self.assertEqual({args.output_root for args in runs.values()}, {Path("benchmark_artifacts/aa_lcr/suites/S")})
        self.assertEqual((runs["S_full"].max_input_tokens, runs["S_head"].direct_overflow), (258048, "head"))
        self.assertEqual((runs["S_random2"].evidence_selection, runs["S_random2"].selection_seed), ("random", 2))
        upgraded = runs["S_upgraded_32k"]
        self.assertEqual((upgraded.evidence_selection, upgraded.packing, upgraded.max_input_tokens), ("fusion", "fill", 28672))
        self.assertEqual((grades[0].source_run, grades[0].output_dir, grades[0].grader_api_style), (
            Path("benchmark_artifacts/aa_lcr/suites/S/direct/S_full"),
            Path("benchmark_artifacts/aa_lcr/suites/S/regrades/Qwen3.6-35B-A3B/direct/S_full"), "vllm"))
        self.assertEqual(commands[-1][2:], ["aa_lcr.results_table", "benchmark_artifacts/aa_lcr/suites/S",
                                            "--grader", "Qwen3.6-35B-A3B"])
        self.assertEqual(self.launch("run_aa_lcr_suite.sh", "--help").returncode, 0)

    def test_aa_lcr_suite_resumes_and_filters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(REPO / "linux", root / "linux")
            artifacts = root / "benchmark_artifacts/aa_lcr/suites/S"
            for finished in ("direct/S_full", "regrades/Qwen3.6-35B-A3B/direct/S_full"):
                (artifacts / finished).mkdir(parents=True)
                (artifacts / finished / "manifest.json").write_text("{}")
            (artifacts / "direct/S_middle").mkdir()
            result = self.launch("run_aa_lcr_suite.sh", "S", env={"ONLY": "full,middle"}, linux=root / "linux")
            commands = self.suite_commands(result)
            self.assertIn("run: already finished", result.stdout)
            self.assertIn("grade: already done", result.stdout)
            self.assertIn("moving unfinished benchmark_artifacts/aa_lcr/suites/S/direct/S_middle", result.stdout)
            self.assertTrue((artifacts / "direct/S_middle").is_dir())
            self.assertEqual([command[-1] for command in commands if command[2] == "aa_lcr.run_benchmark"], ["S_middle"])
            self.assertEqual(self.launch("run_aa_lcr_suite.sh", "S", env={"ONLY": "fulll"}).returncode, 1)

    def test_aa_lcr_suite_luna_checks_one_call_then_regrades(self):
        commands = self.suite_commands(self.launch("run_aa_lcr_suite.sh", "S", env={
            "AA_LCR_GRADER": "luna", "ONLY": "full,position"}))
        regrader = importlib.import_module("aa_lcr.regrade").build_arg_parser()
        *regrades, table = commands
        self.assertTrue(all(command[2] == "aa_lcr.regrade" for command in regrades))
        self.assertEqual(table[2:], ["aa_lcr.results_table", "benchmark_artifacts/aa_lcr/suites/S", "--grader", "gpt-5.6-luna"])
        check, *grades = [regrader.parse_args(command[3:]) for command in regrades]
        self.assertEqual((check.question_ids, len(grades)), ("1", 2))
        for args in (check, *grades):
            self.assertEqual((args.evaluator_model, args.evaluator_base_url, args.grader_api_style,
                              args.grader_reasoning_effort, args.evaluator_api_key_env),
                             ("openai/gpt-5.6-luna", "https://openrouter.ai/api/v1", "openai", "medium",
                              "OPENROUTER_API_KEY"))
        self.assertEqual(grades[1].output_dir, Path("benchmark_artifacts/aa_lcr/suites/S/regrades/gpt-5.6-luna/hybrid/S_position"))
        everything = self.launch("run_aa_lcr_suite.sh", "S", env={"AA_LCR_GRADER": "luna"})
        self.assertIn("Suite S: 18 conditions", everything.stdout)
        self.assertIn("upgraded_80k", everything.stdout)


if __name__ == "__main__":
    unittest.main()
