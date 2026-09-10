import csv
import io
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

import test_aa_lcr
from test_aa_lcr import (
    FakeCompletionCaller,
    FakeTokenizer,
    write_fixture,
)
from aa_lcr.api import Completion, call_chat_completion
from aa_lcr.compare_conditions import compare_conditions
from aa_lcr.grading import Grader
from aa_lcr.prompting import (
    GRADER_SYSTEM_V1_1,
    GRADER_USER_V1_1,
    build_grader_messages,
    parse_grade,
    prompt_contract_metadata,
)
from aa_lcr.regrade import build_arg_parser as regrade_parser, run_regrade
from aa_lcr.run_benchmark import build_arg_parser, run_benchmark
from aa_lcr.dataset import sha256_file


class JsonCaller(FakeCompletionCaller):
    def __init__(self, finish_reason="stop", verdict="CORRECT"):
        super().__init__()
        self.finish_reason = finish_reason
        self.verdict = verdict

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        is_grader = kwargs["model"] != "Qwen/Qwen3.6-35B-A3B"
        answer = json.dumps({"verdict": self.verdict}) if is_grader else "An answer"
        return Completion(
            answer,
            10,
            8 if is_grader else kwargs["max_tokens"],
            {
                "prompt_tokens": 10,
                "completion_tokens": 8 if is_grader else kwargs["max_tokens"],
            },
            "stop" if is_grader else self.finish_reason,
        )


class RerunTests(unittest.TestCase):
    def make_run(self, root, *, output_tokens=None, caller=None, run_id="original"):
        questions, documents, manifest = write_fixture(root)
        args = test_aa_lcr.AALCRRunnerTests()._args(
            root, "direct_262k", questions, documents, manifest, run_id
        )
        args.max_output_tokens = output_tokens
        result = run_benchmark(
            args,
            tokenizer_factory=lambda _: FakeTokenizer(),
            completion_caller=caller or FakeCompletionCaller(),
        )
        return result, args

    def test_http_request_styles_and_finish_reason(self):
        requests = []

        def opener(request, **kwargs):
            requests.append(json.loads(request.data))
            return io.BytesIO(
                json.dumps(
                    {
                        "choices": [
                            {"message": {"content": "done"}, "finish_reason": "length"}
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 16384},
                    }
                ).encode()
            )

        for style in ("vllm", "openai"):
            result = call_chat_completion(
                base_url="http://test/v1",
                model="model",
                messages=[{"role": "user", "content": "question"}],
                max_tokens=16384,
                api_style=style,
                opener=opener,
            )
            self.assertEqual(result.finish_reason, "length")
            self.assertEqual(result.output_tokens, 16384)
        self.assertEqual(requests[0]["max_tokens"], 16384)
        self.assertEqual(requests[0]["temperature"], 0)
        self.assertEqual(requests[1]["max_completion_tokens"], 16384)
        self.assertNotIn("max_tokens", requests[1])
        self.assertNotIn("temperature", requests[1])

    def test_generation_budget_and_length_termination_are_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            caller = JsonCaller("length")
            questions, documents, manifest = write_fixture(root)
            args = test_aa_lcr.AALCRRunnerTests()._args(
                root, "direct_262k", questions, documents, manifest, ""
            )
            args.grader_prompt_version = "aa_lcr_equality_v1.1"
            result = run_benchmark(
                args,
                tokenizer_factory=lambda _: FakeTokenizer(),
                completion_caller=caller,
            )
            self.assertIn("out16384_r1_", result.name)
            metadata = json.loads((result / "manifest.json").read_text())
            self.assertEqual(metadata["max_output_tokens"], 16384)
            self.assertEqual(metadata["output_length_terminated_count"], 2)
            self.assertEqual(metadata["api_error_count"], 0)
            self.assertEqual(metadata["answer_accuracy"], 1.0)
            self.assertEqual(caller.calls[0]["max_tokens"], 16384)
            self.assertFalse(
                caller.calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"]
            )
            for filename in ("predictions.jsonl", "bridge_rows.jsonl"):
                row = json.loads((result / filename).read_text().splitlines()[0])
                self.assertEqual(row["executor_finish_reason"], "length")
                self.assertEqual(row["executor_output_tokens"], 16384)

    def test_explicit_legacy_budget_and_invalid_budgets(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, args = self.make_run(Path(temporary), output_tokens=512)
            self.assertEqual(
                json.loads((result / "manifest.json").read_text())["max_output_tokens"],
                512,
            )
            for value in (0, -1, 30000):
                args.max_output_tokens = value
                with self.subTest(value=value), self.assertRaises(ValueError):
                    run_benchmark(args)
            args.experiment, args.max_output_tokens = "direct_64k", 16384
            with self.assertRaisesRegex(ValueError, "context window"):
                run_benchmark(args)
            args.max_output_tokens, args.run_id = None, "64k"
            result = run_benchmark(
                args,
                tokenizer_factory=lambda _: FakeTokenizer(),
                completion_caller=FakeCompletionCaller(),
            )
            self.assertEqual(
                json.loads((result / "manifest.json").read_text())["max_output_tokens"],
                512,
            )

    def test_prompt_versions_and_json_verdict_parsing(self):
        legacy = prompt_contract_metadata()
        original = json.loads(
            Path(
                "benchmark_artifacts/aa_lcr/direct_262k/20260830T235804Z/manifest.json"
            ).read_text()
        )
        self.assertEqual(
            legacy["grader_prompt_template_sha256"],
            original["grader_prompt_template_sha256"],
        )
        self.assertEqual(
            legacy["prompt_template_sha256"], original["prompt_template_sha256"]
        )
        messages = build_grader_messages(
            "question", "0.14", "14%", "aa_lcr_equality_v1.1"
        )
        self.assertEqual(messages[0]["content"], GRADER_SYSTEM_V1_1)
        self.assertEqual(
            messages[1]["content"],
            GRADER_USER_V1_1.format(
                question="question", official_answer="0.14", candidate_answer="14%"
            ),
        )
        self.assertNotEqual(
            legacy["grader_prompt_template_sha256"],
            prompt_contract_metadata("aa_lcr_equality_v1.1")[
                "grader_prompt_template_sha256"
            ],
        )
        self.assertEqual(
            parse_grade('{"verdict":"CORRECT"}', "aa_lcr_equality_v1.1"), "CORRECT"
        )
        for text in (
            "CORRECT",
            "[]",
            "null",
            '{"verdict":true}',
            '{"verdict":"probably"}',
            '```json\n{"verdict":"CORRECT"}\n```',
        ):
            self.assertEqual(parse_grade(text, "aa_lcr_equality_v1.1"), "")

    def test_grader_context_and_hosted_parameters(self):
        args = build_arg_parser().parse_args(["--experiment", "direct_262k"])
        args.grader_context_window = 1024
        caller = JsonCaller()
        grader = Grader(
            args, tokenizer_factory=lambda _: FakeTokenizer(), completion_caller=caller
        )
        with self.assertRaisesRegex(ValueError, "answer not truncated"):
            grader.grade("question", "key", "x" * 1024)
        self.assertEqual(caller.calls, [])
        args.grader_api_style = "openai"
        args.grader_prompt_version = "aa_lcr_equality_v1.1"
        args.evaluator_model = "gpt-5.6-luna"
        args.evaluator_base_url = "https://api.openai.com/v1"
        grader = Grader(
            args,
            tokenizer_factory=lambda _: self.fail(
                "Hosted grader loaded a local tokenizer"
            ),
            completion_caller=caller,
        )
        _, _, grade = grader.grade("question", "key", "x" * 20000)
        self.assertEqual(grade, "CORRECT")
        call = caller.calls[0]
        self.assertEqual(call["api_style"], "openai")
        self.assertEqual(call["max_tokens"], 16384)
        self.assertEqual(
            call["extra_body"],
            {"reasoning_effort": "medium", "response_format": {"type": "json_object"}},
        )

    def test_incomplete_grader_verdict_is_invalid(self):
        args = build_arg_parser().parse_args(["--experiment", "direct_262k"])

        def caller(**kwargs):
            return Completion("CORRECT", 1, 8, {}, "length")

        grader = Grader(
            args, tokenizer_factory=lambda _: FakeTokenizer(), completion_caller=caller
        )
        _, _, grade = grader.grade("q", "a", "a")
        self.assertEqual(grade, "")

    def regrade_args(self, source, args, output):
        return regrade_parser().parse_args(
            [
                "--source-run",
                str(source),
                "--output-dir",
                str(output),
                "--questions-csv",
                str(args.questions_csv),
                "--documents-root",
                str(args.documents_root),
                "--dataset-manifest",
                str(args.dataset_manifest),
                "--grader-prompt-version",
                "aa_lcr_equality_v1.1",
                "--max-retries",
                "1",
            ]
        )

    def test_regrade_preserves_generation_and_separates_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, args = self.make_run(root)
            before = {p: p.read_bytes() for p in source.iterdir()}
            with args.questions_csv.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fields, rows = reader.fieldnames, list(reader)
            rows[0]["answer"] = "Revised key"
            with args.questions_csv.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            dataset = json.loads(args.dataset_manifest.read_text())
            dataset.update(
                questions_sha256=sha256_file(args.questions_csv),
                dataset_revision="new-fixture-revision",
            )
            args.dataset_manifest.write_text(json.dumps(dataset))
            options = self.regrade_args(source, args, root / "regraded")
            caller = JsonCaller(verdict="INCORRECT")
            output = run_regrade(
                options,
                tokenizer_factory=lambda _: FakeTokenizer(),
                completion_caller=caller,
            )
            self.assertTrue(all(p.read_bytes() == data for p, data in before.items()))
            self.assertEqual(len(caller.calls), 2)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                manifest["source_predictions_sha256"],
                sha256_file(source / "predictions.jsonl"),
            )
            self.assertEqual(manifest["source_dataset_revision"], "fixture-revision")
            self.assertEqual(manifest["dataset_revision"], "new-fixture-revision")
            self.assertEqual(manifest["executor_calls"], 0)
            self.assertEqual(manifest["answer_accuracy"], 0)
            report = compare_conditions([source], [output], replicates=10)
            self.assertEqual(report["delta"], -1)
            self.assertEqual(len(report["regressed_question_ids"]), 2)
            self.assertEqual(
                report["baseline"]["executor_finish_reasons"], {"unknown": 2}
            )
            self.assertIn("questions_sha256", report["settings_differences"])
            with self.assertRaises(FileExistsError):
                run_regrade(options)
            options.output_dir = source / "child"
            with self.assertRaisesRegex(ValueError, "separate"):
                run_regrade(options)

    def test_regrade_rejects_bad_source_before_calls(self):
        for corruption in ("missing", "duplicate", "question", "answer", "scope"):
            with (
                self.subTest(corruption=corruption),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                source, args = self.make_run(root)
                file = source / (
                    "bridge_rows.jsonl"
                    if corruption == "scope"
                    else "predictions.jsonl"
                )
                rows = [json.loads(line) for line in file.read_text().splitlines()]
                if corruption == "missing":
                    rows.pop()
                elif corruption == "duplicate":
                    rows.append(rows[0])
                else:
                    rows[0][
                        {
                            "question": "question",
                            "answer": "candidate_answer",
                            "scope": "source_scope_hash",
                        }[corruption]
                    ] = "wrong"
                file.write_text("".join(json.dumps(row) + "\n" for row in rows))
                options = self.regrade_args(source, args, root / "regraded")
                caller = JsonCaller()
                with self.assertRaises(ValueError):
                    run_regrade(
                        options,
                        tokenizer_factory=lambda _: FakeTokenizer(),
                        completion_caller=caller,
                    )
                self.assertFalse(options.output_dir.exists())
                self.assertEqual(caller.calls, [])

    def test_regrade_subset_metadata_and_unknown_terminations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, args = self.make_run(root)
            options = self.regrade_args(source, args, root / "subset")
            options.question_ids = "2"
            options.validate_only = True
            run_regrade(options)
            self.assertFalse(options.output_dir.exists())
            options.validate_only = False
            output = run_regrade(
                options,
                tokenizer_factory=lambda _: FakeTokenizer(),
                completion_caller=JsonCaller(),
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["rows_selected"], 1)
            self.assertEqual(manifest["question_ids"], ["2"])
            self.assertEqual(manifest["source_question_ids"], ["1", "2"])
            self.assertEqual(manifest["executor_finish_reason_counts"], {"unknown": 1})
            self.assertEqual(manifest["route_counts"], {"direct_fit": 1})
            self.assertFalse(manifest["full_run"])

    def test_regrade_failures_remain_visible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, args = self.make_run(root)
            options = self.regrade_args(source, args, root / "failed")

            def fail(**kwargs):
                raise RuntimeError("grader unavailable")

            output = run_regrade(
                options,
                tokenizer_factory=lambda _: FakeTokenizer(),
                completion_caller=fail,
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertIsNone(manifest["answer_accuracy"])
            self.assertEqual(manifest["api_error_count"], 2)
            self.assertEqual(manifest["total_request_attempts"], 2)
            with self.assertRaisesRegex(ValueError, "valid grades"):
                compare_conditions([source], [output], replicates=10)

    def test_repeats_use_mean_pass_rate_and_reject_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, args = self.make_run(root, run_id="first")
            second, _ = self.make_run(root / "second-fixture", run_id="second")
            file = second / "bridge_rows.jsonl"
            rows = [json.loads(line) for line in file.read_text().splitlines()]
            for row in rows:
                row.update(answer_correct=False, grade="INCORRECT")
            file.write_text("".join(json.dumps(row) + "\n" for row in rows))
            report = compare_conditions([first], [first, second], replicates=10)
            self.assertEqual(report["candidate"]["accuracy"], 0.5)
            self.assertEqual(report["delta"], -0.5)
            with self.assertRaisesRegex(ValueError, "counted twice"):
                compare_conditions([first], [first, first], replicates=10)
            rows[0]["question"] = "different"
            file.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                compare_conditions([first], [second], replicates=10)

    def test_launcher_quotes_paths_and_forwards_options(self):
        env = {
            **os.environ,
            "OPENAI_COMPAT_EXECUTOR_BASE_URL": "http://executor/v1",
            "OPENAI_COMPAT_EVALUATOR_BASE_URL": "http://evaluator/v1",
            "AA_LCR_LAUNCH_DRY_RUN": "1",
            "AA_LCR_DATA_DIR": "data with spaces;literal",
            "AA_LCR_RUN_ID": "run $(literal)",
            "AA_LCR_MAX_OUTPUT_TOKENS": "16384",
            "AA_LCR_REPEAT_ID": "3",
        }
        result = subprocess.run(
            [
                "bash",
                "jarvis/run_aa_lcr.sh",
                "direct_262k",
                "--grader-prompt-version",
                "aa_lcr_equality_v1.1",
            ],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        command = next(
            line.split("command=", 1)[1]
            for line in result.stdout.splitlines()
            if "command=" in line
        )
        args = shlex.split(command)
        self.assertEqual(
            args[args.index("--questions-csv") + 1],
            "data with spaces;literal/AA-LCR_Dataset.csv",
        )
        self.assertEqual(args[args.index("--run-id") + 1], "run $(literal)")
        self.assertEqual(args[args.index("--repeat-id") + 1], "3")
        self.assertIn("aa_lcr_equality_v1.1", args)


if __name__ == "__main__":
    unittest.main()
