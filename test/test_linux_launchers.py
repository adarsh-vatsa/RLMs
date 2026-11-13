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
        self.assertEqual((len(runs), len(grades)), (31, 31))
        self.assertTrue(all(args.execution_only and args.fail_fast for args in runs.values()))
        self.assertEqual({args.output_root for args in runs.values()}, {Path("benchmark_artifacts/aa_lcr/suites/S")})
        self.assertEqual((runs["S_full"].max_input_tokens, runs["S_head"].direct_overflow), (258048, "head"))
        self.assertEqual((runs["S_random2"].evidence_selection, runs["S_random2"].selection_seed), ("random", 2))
        upgraded = runs["S_upgraded_32k"]
        self.assertEqual((upgraded.evidence_selection, upgraded.packing, upgraded.max_input_tokens), ("fusion", "fill", 28672))
        self.assertEqual((runs["S_bm25_80k"].evidence_selection, runs["S_bm25_80k"].max_input_tokens), ("bm25", 77824))
        ladder3 = runs["S_ladder3_32k"]
        self.assertEqual((ladder3.evidence_selection, ladder3.child_tokens, ladder3.neighbor_children, ladder3.packing,
                          ladder3.document_floor, ladder3.max_input_tokens), ("fusion", 1024, 1, "fill", False, 28672))
        for name, neighbours in (("S_ladder1", 0), ("S_ladder2", 1)):
            rung = runs[name]
            self.assertEqual((rung.evidence_selection, rung.child_tokens, rung.child_overlap_tokens, rung.neighbor_children,
                              rung.packing, rung.document_floor, rung.max_input_tokens),
                             ("fusion", 1024, 128, neighbours, "stop", False, 61440))
        for name, selection, budget in (("S_position_recipe_32k", "position", 28672), ("S_bm25_recipe", "bm25", 61440),
                                         ("S_bm25_recipe_80k", "bm25", 77824)):
            control = runs[name]
            self.assertEqual((control.evidence_selection, control.child_tokens, control.child_overlap_tokens,
                              control.neighbor_children, control.packing, control.document_floor,
                              control.max_input_tokens), (selection, 1024, 128, 1, "fill", True, budget))
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
        self.assertIn("Suite S: 31 conditions", everything.stdout)
        self.assertIn("upgraded_80k", everything.stdout)


    def test_mrcr_suite_runs_every_condition_with_the_reference_settings(self):
        commands = self.suite_commands(self.launch("run_mrcr_suite.sh", "S", env={
            "MRCR_RECIPE": "--evidence-selection fusion --child-tokens 1024 --child-overlap-tokens 128"}))
        runner = importlib.import_module("mrcr_v2.run_benchmark").build_arg_parser()
        runs = {args.run_id: args for args in (runner.parse_args(command[3:])
                                                for command in commands if command[2] == "mrcr_v2.run_benchmark")}
        self.assertEqual(len(runs), 37)
        self.assertEqual([runs[f"S_random{seed}"].selection_seed for seed in range(3)], [0, 1, 2])
        self.assertEqual({(args.output_root, args.data_dir, args.max_rows_per_source, args.fail_fast)
                          for args in runs.values()},
                         {(Path("benchmark_artifacts/mrcr_v2/suites/S"),
                           Path("benchmark_data/mrcr_v2_4needle_100k_1200k"), 30, True)})
        self.assertEqual((runs["S_dense"].child_tokens, runs["S_dense"].max_input_tokens), (3500, 258048))
        self.assertEqual((runs["S_dense_7500"].child_tokens, runs["S_dense_question"].retrieval_query), (7500, "question"))
        ladder = runs["S_ladder4"]
        self.assertEqual((ladder.evidence_selection, ladder.child_tokens, ladder.neighbor_children, ladder.packing,
                          ladder.document_floor), ("fusion", 1024, 1, "fill", True))
        self.assertEqual((runs["S_position_128k"].max_input_tokens, runs["S_upgraded_64k"].child_tokens), (126976, 1024))
        bm25 = runs["S_bm25_64k"]
        self.assertEqual((bm25.evidence_selection, bm25.child_tokens, bm25.max_input_tokens), ("bm25", 3500, 61440))
        oracle = runs["S_oracle_32k"]
        self.assertEqual((oracle.mode, oracle.oracle, oracle.evidence_selection, oracle.child_tokens, oracle.packing,
                          oracle.document_floor, oracle.max_input_tokens), ("hybrid", True, "fusion", 1024, "fill", True, 28672))
        self.assertEqual(sum(args.oracle for args in runs.values()), 4)
        for name, budget in (("S_bm25_recipe", 258048), ("S_bm25_recipe_64k", 61440)):
            control = runs[name]
            self.assertEqual((control.evidence_selection, control.child_tokens, control.neighbor_children, control.packing,
                              control.document_floor, control.oracle, control.max_input_tokens),
                             ("bm25", 1024, 1, "fill", True, False, budget))
        self.assertFalse(any(name.startswith("S_position_recipe") for name in runs))
        self.assertEqual(commands[-2][2:], ["mrcr_v2.results_table", "benchmark_artifacts/mrcr_v2/suites/S"])
        self.assertEqual(commands[-1][2:], ["mrcr_v2.visibility", "benchmark_artifacts/mrcr_v2/suites/S",
                                            "--data-dir", "benchmark_data/mrcr_v2_4needle_100k_1200k"])
        for rows in ("100", "0"):
            sized = self.suite_commands(self.launch("run_mrcr_suite.sh", "S", env={"MRCR_ROWS_PER_SOURCE": rows}))
            self.assertEqual({runner.parse_args(command[3:]).max_rows_per_source for command in sized
                              if command[2] == "mrcr_v2.run_benchmark"}, {int(rows)})
        self.assertEqual(self.launch("run_mrcr_suite.sh", "--help").returncode, 0)

    def test_mrcr_suite_retries_runs_that_did_not_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(REPO / "linux", root / "linux")
            suite = root / "benchmark_artifacts/mrcr_v2/suites/S"
            for run, status in (("direct/S_middle", "complete"), ("hybrid/S_dense", "running")):
                (suite / run).mkdir(parents=True)
                (suite / run / "manifest.json").write_text(json.dumps({"status": status}, indent=2))
            result = self.launch("run_mrcr_suite.sh", "S", env={"ONLY": "middle,dense"}, linux=root / "linux")
            commands = self.suite_commands(result)
            self.assertIn("run: already finished", result.stdout)
            self.assertIn("moving unfinished benchmark_artifacts/mrcr_v2/suites/S/hybrid/S_dense", result.stdout)
            self.assertEqual([command[-1] for command in commands if command[2] == "mrcr_v2.run_benchmark"], ["S_dense"])

    def test_longbench_suite_runs_every_condition_on_the_shared_question_set(self):
        commands = self.suite_commands(self.launch("run_longbench_suite.sh", "S", env={
            "LONGBENCH_RECIPE": "--evidence-selection fusion --packing fill"}))
        parsers = {module: importlib.import_module(module).build_arg_parser()
                   for module in ("long_bench_v2.run_api_benchmark", "long_bench_v2.run_benchmark")}
        runs = {args.run_id: (command[2], args) for command in commands if command[2] in parsers
                for args in [parsers[command[2]].parse_args(command[3:])]}
        self.assertEqual(len(runs), 27)
        self.assertEqual([runs[f"S_random{seed}"][1].selection_seed for seed in range(3)], [0, 1, 2])
        self.assertEqual({(args.output_dir, args.min_source_tokens, args.max_source_tokens, args.fail_fast,
                           args.execution_profile, args.row_types) for _, args in runs.values()},
                         {(Path("benchmark_artifacts/longbench_v2/suites/S"), 65529, 262136, True, "common", "original")})
        module, full = runs["S_full"]
        self.assertEqual((module, full.max_input_tokens, full.max_output_tokens), ("long_bench_v2.run_api_benchmark", 262136, 8))
        module, middle = runs["S_middle"]
        self.assertEqual((module, middle.max_input_tokens, middle.direct_overflow), ("long_bench_v2.run_api_benchmark", 65528, "middle"))
        module, ladder = runs["S_ladder4"]
        self.assertEqual((module, ladder.mode, ladder.evidence_selection, ladder.child_tokens, ladder.packing,
                          ladder.document_floor), ("long_bench_v2.run_benchmark", "baseline", "fusion", 1024, "fill", True))
        self.assertEqual((runs["S_position_128k"][1].max_input_tokens, runs["S_upgraded_32k"][1].packing), (131064, "fill"))
        module, bm25 = runs["S_bm25_128k"]
        self.assertEqual((module, bm25.evidence_selection, bm25.max_input_tokens), ("long_bench_v2.run_benchmark", "bm25", 131064))
        for name, budget in (("S_bm25_recipe", 65528), ("S_bm25_recipe_32k", 32760), ("S_bm25_recipe_128k", 131064)):
            module, control = runs[name]
            self.assertEqual((module, control.evidence_selection, control.child_tokens, control.child_overlap_tokens,
                              control.neighbor_children, control.packing, control.document_floor, control.max_input_tokens),
                             ("long_bench_v2.run_benchmark", "bm25", 1024, 128, 1, "fill", True, budget))
        self.assertEqual(commands[-1][2:], ["long_bench_v2.results_table", "benchmark_artifacts/longbench_v2/suites/S"])
        self.assertEqual(self.launch("run_longbench_suite.sh", "--help").returncode, 0)

    def test_jarvis_suite_submits_one_job_that_waits_for_the_executor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bin").mkdir()
            (root / "bin/sbatch").write_text("#!/usr/bin/env bash\n"
                f"printf '%s\\n' \"$*\" \"$CLIENT_CMD\" \"${{JARVIS_REQUIRED_SERVICES_SET:-}}\" "
                f"\"${{WAIT_FOR_ENDPOINTS:-}}\" \"$SUITE_RESUME\" > {root}/submitted\n")
            (root / "bin/sbatch").chmod(0o755)
            (root / "serving.json").write_text("{}")
            env = {"DRY_RUN": "0", "PATH": f"{root / 'bin'}:{os.environ['PATH']}", "PROJECT_LOG_DIR": str(root / "logs"),
                   "JARVIS_LOAD_DOTENV": "0", "SLURM_JOB_ID": "", "AA_LCR_SERVING_METADATA": str(root / "serving.json"),
                   "MRCR_SERVING_METADATA": str(root / "serving.json"), "LONGBENCH_SERVING_METADATA": str(root / "serving.json")}
            jarvis = REPO / "jarvis"
            self.assertEqual(self.launch("run_suite.sh", "unknown", linux=jarvis, env=env).returncode, 2)
            missing = self.launch("run_suite.sh", "longbench", "S", linux=jarvis, env=env)
            self.assertEqual(missing.returncode, 1)
            self.assertIn("OPENAI_COMPAT_EXECUTOR_BASE_URL", missing.stderr)
            env["OPENAI_COMPAT_EXECUTOR_BASE_URL"] = "http://g101:8000/v1"
            no_metadata = self.launch("run_suite.sh", "mrcr", "S", linux=jarvis,
                                      env={**env, "MRCR_SERVING_METADATA": str(root / "absent.json")})
            self.assertEqual(no_metadata.returncode, 1)
            self.assertIn("Serving metadata not found", no_metadata.stderr)
            for benchmark in ("aa_lcr", "mrcr", "longbench"):
                with self.subTest(benchmark=benchmark):
                    result = self.launch("run_suite.sh", benchmark, "S", linux=jarvis, env=env)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    arguments, command, required, wait, resume = (root / "submitted").read_text().splitlines()
                    self.assertIn("--gres=gpu:l40s:1", arguments)
                    self.assertTrue(arguments.endswith("jarvis/run_client.sh"))
                    self.assertEqual((command, required, wait), (f"bash linux/run_{benchmark}_suite.sh S", "1", "1"))
                    self.assertTrue(resume.endswith(f"jarvis/run_suite.sh {benchmark} S"))
            luna = self.launch("run_suite.sh", "aa_lcr", "S", linux=jarvis,
                               env={**env, "AA_LCR_GRADER": "luna", "OPENAI_COMPAT_EXECUTOR_BASE_URL": ""})
            self.assertEqual(luna.returncode, 0, luna.stderr)
            arguments, command, required, wait, resume = (root / "submitted").read_text().splitlines()
            self.assertIn("--partition=compute-short", arguments)
            self.assertNotIn("--gres", arguments)
            self.assertEqual((command, required, wait), ("bash linux/run_aa_lcr_suite.sh S", "", ""))
            self.assertTrue(resume.startswith("AA_LCR_GRADER=luna bash "))
            full = self.launch("run_suite.sh", "mrcr", "S", linux=jarvis, env={**env, "MRCR_ROWS_PER_SOURCE": "0"})
            self.assertEqual(full.returncode, 0, full.stderr)
            self.assertTrue((root / "submitted").read_text().splitlines()[-1].startswith("MRCR_ROWS_PER_SOURCE=0 bash "))
            preview = self.launch("run_suite.sh", "mrcr", "S", linux=jarvis)
            self.assertEqual(preview.returncode, 0, preview.stderr)
            self.assertIn("[MRCR SUITE] Suite S: 33 conditions", preview.stdout)

    def test_jarvis_executor_can_be_pinned_to_a_node(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bin").mkdir()
            (root / "bin/sbatch").write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > {root}/submitted\n")
            (root / "bin/sbatch").chmod(0o755)
            env = {"DRY_RUN": "0", "PATH": f"{root / 'bin'}:{os.environ['PATH']}", "PROJECT_LOG_DIR": str(root / "logs"),
                   "JARVIS_LOAD_DOTENV": "0", "SLURM_JOB_ID": ""}
            for node, expected in (("", None), ("g202", "--nodelist=g202")):
                with self.subTest(node=node):
                    result = self.launch("run.sh", "submit", "executor", linux=REPO / "jarvis",
                                         env={**env, "EXECUTOR_NODE": node})
                    self.assertEqual(result.returncode, 0, result.stderr)
                    arguments = (root / "submitted").read_text().splitlines()
                    self.assertIn("--gres=gpu:l40s:4", arguments)
                    self.assertEqual([argument for argument in arguments if argument.startswith("--nodelist")],
                                     [expected] if expected else [])

    def test_g202_rerun_moves_only_g202_runs_and_reruns_them_with_the_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(REPO / "linux", root / "linux")
            shutil.copytree(REPO / "jarvis", root / "jarvis")
            suite = root / "benchmark_artifacts/aa_lcr/suites/20261029"
            for run, node in (("hybrid/20261029_upgraded", "g202"), ("regrades/gpt-5.6-luna/hybrid/20261029_upgraded", "g202"),
                              ("hybrid/20261029_upgraded_32k", "g101")):
                (suite / run).mkdir(parents=True)
                (suite / run / "manifest.json").write_text(json.dumps({"executor_base_url": f"http://{node}.x:8000/v1"}, indent=2))
            (suite / "logs").mkdir()
            for log in ("upgraded.log", "upgraded.grade-gpt-5.6-luna.log", "upgraded_32k.log"):
                (suite / "logs" / log).write_text("")

            preview = self.launch("rerun_g202_runs.sh", linux=root / "jarvis", cwd=root)
            commands = self.suite_commands(preview)
            moved = {line.split()[2] for line in preview.stdout.splitlines() if line.startswith("[RERUN] moving")}
            self.assertEqual(moved, {f"benchmark_artifacts/aa_lcr/suites/20261029/{path}" for path in (
                "hybrid/20261029_upgraded", "regrades/gpt-5.6-luna/hybrid/20261029_upgraded",
                "logs/upgraded.log", "logs/upgraded.grade-gpt-5.6-luna.log")})
            self.assertTrue((suite / "hybrid/20261029_upgraded").exists())
            runs = [command[-1] for command in commands if command[2].endswith(".run_benchmark")]
            self.assertEqual(len(runs), 26 - 2)
            self.assertIn("20261029_bm25_recipe_128k", runs)
            self.assertIn("20261029_position_recipe_80k", runs)
            self.assertEqual([command[2] for command in commands if command[2].endswith(".intervals")],
                             ["aa_lcr.intervals", "long_bench_v2.intervals", "mrcr_v2.intervals"])

            moving = subprocess.run(["bash", "-c", "source jarvis/rerun_g202_runs.sh help > /dev/null; "
                                     "move_old_runs benchmark_artifacts/aa_lcr/suites/20261029 hybrid upgraded,upgraded_32k"],
                                    cwd=root, env={**os.environ, "DRY_RUN": "0"}, capture_output=True, text=True)
            self.assertEqual(moving.returncode, 0, moving.stderr)
            old = root / "benchmark_artifacts/aa_lcr/suites/20261029_g202"
            self.assertEqual(sorted(str(path.relative_to(old)) for path in old.rglob("*") if path.is_file()), [
                "hybrid/20261029_upgraded/manifest.json", "logs/upgraded.grade-gpt-5.6-luna.log", "logs/upgraded.log",
                "regrades/gpt-5.6-luna/hybrid/20261029_upgraded/manifest.json"])
            self.assertTrue((suite / "hybrid/20261029_upgraded_32k/manifest.json").exists())
            self.assertTrue((suite / "logs/upgraded_32k.log").exists())

    def test_g202_rerun_submits_a_pinned_executor_and_a_job_that_starts_with_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bin").mkdir()
            (root / "bin/sbatch").write_text("#!/usr/bin/env bash\n"
                f"printf '%s|%s|%s|%s\\n' \"$*\" \"${{CLIENT_CMD:-}}\" "
                f"\"${{RERUN_EXECUTOR_JOB:-}}\" \"${{RERUN_EXECUTOR_URL_FILE:-}}\" >> {root}/submitted\n"
                "echo 'Submitted batch job 42'\n")
            (root / "bin/sbatch").chmod(0o755)
            env = {"DRY_RUN": "0", "PATH": f"{root / 'bin'}:{os.environ['PATH']}", "PROJECT_LOG_DIR": str(root / "logs"),
                   "JARVIS_LOAD_DOTENV": "0", "SLURM_JOB_ID": ""}
            result = self.launch("rerun_g202_runs.sh", linux=REPO / "jarvis", env={**env, "RERUN_TIME": "10:00:00"})
            self.assertEqual(result.returncode, 0, result.stderr)
            executor, client = [line.split("|") for line in (root / "submitted").read_text().splitlines()]
            self.assertIn("--time=10:00:00", executor[0])
            self.assertIn("--time=10:00:00", client[0])
            self.assertIn("--nodelist=g101", executor[0])
            self.assertTrue(executor[0].endswith("jarvis/serve_vllm.sh"))
            self.assertIn("--gres=gpu:l40s:1", client[0])
            self.assertTrue(client[0].endswith("jarvis/run_client.sh"))
            self.assertIn("--dependency=after:42", client[0])
            self.assertNotIn("--dependency", executor[0])
            self.assertEqual(client[1:], ["bash jarvis/rerun_g202_runs.sh job", "42", f"{root}/logs/executor-42.url"])

    def test_g202_rerun_stops_waiting_when_the_executor_job_has_ended(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bin").mkdir()
            (root / "bin/squeue").write_text("#!/usr/bin/env bash\n")
            (root / "bin/scancel").write_text(f"#!/usr/bin/env bash\necho \"$@\" > {root}/cancelled\n")
            for tool in ("squeue", "scancel"):
                (root / "bin" / tool).chmod(0o755)
            result = self.launch("rerun_g202_runs.sh", "job", linux=REPO / "jarvis", env={
                "DRY_RUN": "0", "PATH": f"{root / 'bin'}:{os.environ['PATH']}", "RERUN_EXECUTOR_JOB": "42",
                "RERUN_EXECUTOR_URL_FILE": str(root / "executor-42.url")})
            self.assertEqual(result.returncode, 1)
            self.assertIn("Executor job 42 ended (no longer queued) before it was ready", result.stderr)
            self.assertEqual((root / "cancelled").read_text().strip(), "42")

    def test_parity_runs_preview_lists_only_the_new_runs_and_skips_finished_ones(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(REPO / "linux", root / "linux")
            shutil.copytree(REPO / "jarvis", root / "jarvis")
            done = root / "benchmark_artifacts/mrcr_v2/suites/20261029/hybrid/20261029_random1"
            done.mkdir(parents=True)
            (done / "manifest.json").write_text(json.dumps({"status": "complete"}, indent=2))

            preview = self.launch("parity_runs.sh", linux=root / "jarvis", cwd=root)
            commands = self.suite_commands(preview)
            runs = sorted((command[2].split(".")[0], command[-1]) for command in commands
                          if command[2].endswith(".run_benchmark"))
            self.assertEqual(runs, sorted(
                [("aa_lcr", "20261029_ladder1"), ("aa_lcr", "20261029_ladder2"),
                 ("mrcr_v2", "20261029_random2"), ("mrcr_v2", "20261029_bm25_recipe")]
                + [("long_bench_v2", f"20261029_{name}")
                   for name in ("random1", "random2", "bm25_recipe", "bm25_recipe_32k", "bm25_recipe_128k")]))
            self.assertIn("run: already finished", preview.stdout)
            regrader = importlib.import_module("aa_lcr.regrade").build_arg_parser()
            grades = [regrader.parse_args(command[3:]) for command in commands if command[2] == "aa_lcr.regrade"]
            self.assertEqual(sorted(args.grader_api_style for args in grades), ["openai"] * 3 + ["vllm"] * 2)
            self.assertEqual([command[2] for command in commands if command[2].endswith(".intervals")],
                             ["aa_lcr.intervals", "long_bench_v2.intervals", "mrcr_v2.intervals"])

    def test_parity_runs_submit_a_pinned_executor_and_a_job_that_waits_for_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bin").mkdir()
            (root / "bin/sbatch").write_text("#!/usr/bin/env bash\n"
                f"printf '%s|%s|%s|%s\\n' \"$*\" \"${{CLIENT_CMD:-}}\" "
                f"\"${{PARITY_EXECUTOR_JOB:-}}\" \"${{PARITY_EXECUTOR_URL_FILE:-}}\" >> {root}/submitted\n"
                "echo 'Submitted batch job 42'\n")
            (root / "bin/squeue").write_text("#!/usr/bin/env bash\n")
            (root / "bin/scancel").write_text(f"#!/usr/bin/env bash\necho \"$@\" > {root}/cancelled\n")
            for tool in ("sbatch", "squeue", "scancel"):
                (root / "bin" / tool).chmod(0o755)
            env = {"DRY_RUN": "0", "PATH": f"{root / 'bin'}:{os.environ['PATH']}", "PROJECT_LOG_DIR": str(root / "logs"),
                   "JARVIS_LOAD_DOTENV": "0", "SLURM_JOB_ID": ""}
            jarvis = REPO / "jarvis"
            refused = self.launch("parity_runs.sh", linux=jarvis, env={**env, "EXECUTOR_NODE": "g202"})
            self.assertEqual(refused.returncode, 1)
            self.assertIn("EXECUTOR_NODE is g202", refused.stderr)
            self.assertFalse((root / "submitted").exists())

            result = self.launch("parity_runs.sh", linux=jarvis, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            executor, client = [line.split("|") for line in (root / "submitted").read_text().splitlines()]
            self.assertIn("--time=08:00:00", executor[0])
            self.assertIn("--time=08:00:00", client[0])
            self.assertIn("--nodelist=g101", executor[0])
            self.assertTrue(executor[0].endswith("jarvis/serve_vllm.sh"))
            self.assertIn("--dependency=after:42", client[0])
            self.assertTrue(client[0].endswith("jarvis/run_client.sh"))
            self.assertEqual(client[1:], ["bash jarvis/parity_runs.sh job", "42", f"{root}/logs/executor-42.url"])

            waiting = self.launch("parity_runs.sh", "job", linux=jarvis, env={
                **env, "PARITY_EXECUTOR_JOB": "42", "PARITY_EXECUTOR_URL_FILE": str(root / "executor-42.url")})
            self.assertEqual(waiting.returncode, 1)
            self.assertIn("Executor job 42 ended (no longer queued) before it was ready", waiting.stderr)
            self.assertEqual((root / "cancelled").read_text().strip(), "42")

    def test_longbench_suite_finds_runs_in_each_runners_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(REPO / "linux", root / "linux")
            suite = root / "benchmark_artifacts/longbench_v2/suites/S"
            (suite / "longbench_v2_api/S_middle").mkdir(parents=True)
            (suite / "longbench_v2_api/S_middle/manifest.json").write_text("{}")
            (suite / "longbench_v2/S_dense").mkdir(parents=True)
            result = self.launch("run_longbench_suite.sh", "S", env={"ONLY": "middle,dense"}, linux=root / "linux")
            commands = self.suite_commands(result)
            self.assertIn("run: already finished", result.stdout)
            self.assertIn("moving unfinished benchmark_artifacts/longbench_v2/suites/S/longbench_v2/S_dense", result.stdout)
            self.assertEqual([command[-1] for command in commands if command[2].startswith("long_bench_v2.run")], ["S_dense"])


if __name__ == "__main__":
    unittest.main()
