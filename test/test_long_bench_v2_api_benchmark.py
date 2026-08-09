import csv
import io
import json
import os
import sys
import tempfile
import types
import urllib.error
import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from long_bench_v2.run_api_benchmark import (
    ARTIFACT_SUBDIR,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_OPENAI_COMPAT_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    OPENAI_COMPAT_SYSTEM_PROMPT,
    _build_default_api_client_factory,
    _chat_token_count,
    _token_ids,
    build_arg_parser,
    normalize_api_args,
    parse_api_usage,
    prepare_openai_compatible_messages,
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
            raise RuntimeError("input too long")
        return FakeResponse()


class FakeClient:
    def __init__(self, calls, fail=False):
        self.messages = FakeMessages(calls, fail=fail)


class FakeOpenRouterHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class FakeOpenRouterOpener:
    def __init__(self, calls):
        self.calls = calls

    def __call__(self, request, timeout=120):
        self.calls.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "headers": dict(request.header_items()),
                "body": json.loads(request.data.decode("utf-8")),
            }
        )
        return FakeOpenRouterHTTPResponse(
            {
                "choices": [{"message": {"content": "Final answer: C"}}],
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 25,
                    "total_tokens": 525,
                },
            }
        )


class FakeOpenRouterErrorOpener:
    def __init__(self, calls):
        self.calls = calls

    def __call__(self, request, timeout=120):
        self.calls.append(
            {
                "url": request.full_url,
                "body": json.loads(request.data.decode("utf-8")),
            }
        )
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            hdrs={},
            fp=io.BytesIO(b'{"error":{"message":"input too long"}}'),
        )


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=True,
    ):
        self.enable_thinking = enable_thinking
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        input_ids = self.encode(rendered)
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class FakeBatchEncoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class FakeOpenAICompatibleOpener(FakeOpenRouterOpener):
    pass


class FakeOpenAICompatibleErrorOpener:
    def __init__(self, calls):
        self.calls = calls

    def __call__(self, request, timeout=120):
        self.calls.append(json.loads(request.data.decode("utf-8")))
        raise urllib.error.HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            hdrs={},
            fp=io.BytesIO(b'{"error":{"message":"temporary failure"}}'),
        )


class LongBenchV2ApiBenchmarkTests(unittest.TestCase):
    def test_cli_defaults_to_anthropic_sonnet(self):
        parser = build_arg_parser()
        args = normalize_api_args(parser.parse_args([]))

        self.assertEqual(args.api_provider, "anthropic")
        self.assertEqual(args.api_model, "claude-sonnet-4-5")
        self.assertEqual(args.api_key_env, "ANTHROPIC_API_KEY")
        self.assertEqual(args.max_output_tokens, 256)

    def test_cli_openrouter_defaults(self):
        parser = build_arg_parser()
        args = normalize_api_args(parser.parse_args(["--api-provider", "openrouter"]))

        self.assertEqual(args.api_provider, "openrouter")
        self.assertEqual(args.api_model, DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(args.api_key_env, "OPENROUTER_API_KEY")

    def test_cli_openai_compatible_defaults(self):
        parser = build_arg_parser()
        args = normalize_api_args(parser.parse_args(["--api-provider", "openai_compatible"]))

        self.assertEqual(args.api_model, DEFAULT_OPENAI_COMPAT_MODEL)
        self.assertEqual(args.api_key_env, "")
        self.assertEqual(args.api_base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(args.context_window_tokens, DEFAULT_CONTEXT_WINDOW_TOKENS)
        self.assertEqual(args.max_input_tokens, DEFAULT_MAX_INPUT_TOKENS)
        self.assertEqual(args.max_retries, 5)

    def test_token_ids_normalizes_supported_tokenizer_shapes(self):
        self.assertEqual(_token_ids([1, 2, 3]), [1, 2, 3])
        self.assertEqual(_token_ids([[1, 2, 3]]), [1, 2, 3])
        self.assertEqual(_token_ids(FakeTensor([1, 2, 3])), [1, 2, 3])
        self.assertEqual(
            _token_ids({"input_ids": FakeTensor([[1, 2, 3]])}),
            [1, 2, 3],
        )
        self.assertEqual(
            _token_ids(FakeBatchEncoding(FakeTensor([[1, 2, 3]]))),
            [1, 2, 3],
        )

    def test_token_ids_rejects_mapping_without_input_ids(self):
        with self.assertRaisesRegex(ValueError, "input_ids"):
            _token_ids({"attention_mask": [1, 1]})

    def test_middle_truncation_keeps_prompt_beginning_and_end(self):
        row = _suite_row("row_1", "original")
        row["context"] = "BEGIN-" + ("middle " * 400) + "-END"
        tokenizer = FakeTokenizer()
        empty_messages = [
            {"role": "system", "content": OPENAI_COMPAT_SYSTEM_PROMPT},
            {"role": "user", "content": ""},
        ]
        chat_overhead = _chat_token_count(tokenizer, empty_messages)
        max_input_tokens = chat_overhead + 700
        context_window = max_input_tokens + 108

        messages, metadata = prepare_openai_compatible_messages(
            row,
            tokenizer,
            context_window_tokens=context_window,
            max_input_tokens=max_input_tokens,
            max_output_tokens=8,
        )

        self.assertTrue(metadata["prompt_truncated"])
        self.assertGreater(metadata["prompt_tokens_before_truncation"], max_input_tokens)
        self.assertEqual(metadata["prompt_tokens_after_truncation"], max_input_tokens)
        self.assertGreater(metadata["prompt_tokens_removed"], 0)
        self.assertEqual(metadata["input_token_budget"], max_input_tokens)
        self.assertEqual(metadata["context_window_safety_margin_tokens"], 100)
        self.assertFalse(tokenizer.enable_thinking)
        self.assertTrue(messages[1]["content"].startswith("Context:\nBEGIN-"))
        self.assertIn("-END", messages[1]["content"])
        self.assertTrue(messages[1]["content"].endswith("A, B, C, or D."))

    def test_middle_truncation_rejects_input_cap_outside_context_window(self):
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            prepare_openai_compatible_messages(
                _suite_row("row_1", "original"),
                FakeTokenizer(),
                context_window_tokens=1000,
                max_input_tokens=995,
                max_output_tokens=8,
            )

    def test_openrouter_client_factory_reads_environment_key(self):
        parser = build_arg_parser()
        args = normalize_api_args(parser.parse_args(["--api-provider", "openrouter"]))

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-openrouter-key"}):
            factory = _build_default_api_client_factory(args)
            client = factory()

        self.assertEqual(client["api_key"], "test-openrouter-key")

    def test_parse_api_usage_from_anthropic_response(self):
        usage = parse_api_usage(FakeResponse(), model="claude-sonnet-4-5", success=True)

        self.assertEqual(usage["calls"], 1)
        self.assertEqual(usage["input_tokens"], 500)
        self.assertEqual(usage["output_tokens"], 25)
        self.assertEqual(usage["total_tokens"], 525)
        self.assertEqual(usage["cost_usd"], 0.001875)
        self.assertEqual(usage["usage_parse_status"], "parsed")

    def test_parse_api_usage_from_openrouter_response(self):
        response = {
            "choices": [{"message": {"content": "Final answer: C"}}],
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 25,
                "total_tokens": 525,
            },
        }

        usage = parse_api_usage(response, model="anthropic/claude-sonnet-4.5", success=True)

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

    def test_fake_openrouter_run_uses_chat_completions_shape(self):
        calls = []

        def fake_factory():
            return {
                "api_key": "test-openrouter-key",
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "opener": FakeOpenRouterOpener(calls),
            }

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
                api_provider="openrouter",
                api_model="anthropic/claude-sonnet-4.5",
                api_key_env="OPENROUTER_API_KEY",
                max_output_tokens=256,
                fail_fast=False,
                output_dir=root / "artifacts",
                manifest_note="openrouter test",
            )

            run_longbench_api_benchmark(args, client_factory=fake_factory)

            run_dir = next((root / "artifacts" / ARTIFACT_SUBDIR).iterdir())
            manifest = json.loads((run_dir / "manifest.json").read_text())
            bridge_row = json.loads((run_dir / "bridge_rows.jsonl").read_text().splitlines()[0])
            prediction_row = json.loads((run_dir / "predictions.jsonl").read_text().splitlines()[0])

        self.assertEqual(calls[0]["url"], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(calls[0]["body"]["model"], "anthropic/claude-sonnet-4.5")
        self.assertEqual(calls[0]["body"]["max_tokens"], 256)
        self.assertEqual(calls[0]["body"]["messages"][0]["role"], "user")
        self.assertIn("Choices:", calls[0]["body"]["messages"][0]["content"])
        self.assertEqual(manifest["api_provider"], "openrouter")
        self.assertEqual(manifest["api_model"], "anthropic/claude-sonnet-4.5")
        self.assertEqual(manifest["answer_accuracy"], 1.0)
        self.assertEqual(bridge_row["api_provider"], "openrouter")
        self.assertEqual(bridge_row["delta_input_tokens"], 500)
        self.assertEqual(bridge_row["delta_output_tokens"], 25)
        self.assertEqual(prediction_row["prediction"], "C")

    def test_fake_openai_compatible_run_uses_qwen_direct_payload(self):
        calls = []

        def fake_factory():
            return {
                "api_key": "",
                "url": "http://executor.example:8000/v1/chat/completions",
                "opener": FakeOpenAICompatibleOpener(calls),
            }

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
                api_provider="openai_compatible",
                api_model="Qwen/Qwen3.6-35B-A3B",
                api_base_url="http://executor.example:8000/v1",
                api_key_env="",
                max_output_tokens=8,
                context_window_tokens=4096,
                max_input_tokens=4000,
                max_retries=2,
                fail_fast=False,
                output_dir=root / "artifacts",
                manifest_note="local direct test",
            )

            run_longbench_api_benchmark(
                args,
                client_factory=fake_factory,
                tokenizer_factory=lambda _: FakeTokenizer(),
            )

            run_dir = next((root / "artifacts" / ARTIFACT_SUBDIR).iterdir())
            manifest = json.loads((run_dir / "manifest.json").read_text())
            bridge_row = json.loads((run_dir / "bridge_rows.jsonl").read_text().splitlines()[0])

        payload = calls[0]["body"]
        self.assertEqual(payload["model"], "Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(payload["max_tokens"], 8)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["messages"][0], {"role": "system", "content": OPENAI_COMPAT_SYSTEM_PROMPT})
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertIn("Context:", payload["messages"][1]["content"])
        self.assertIn("Choices:", payload["messages"][1]["content"])
        self.assertEqual(manifest["api_provider"], "openai_compatible")
        self.assertEqual(manifest["context_window_tokens"], 4096)
        self.assertEqual(manifest["max_input_tokens"], 4000)
        self.assertEqual(manifest["input_token_budget"], 4000)
        self.assertEqual(manifest["context_window_safety_margin_tokens"], 88)
        self.assertEqual(
            manifest["truncation_policy"],
            "longbench_v2_middle_keep_first_last",
        )
        self.assertEqual(manifest["system_prompt_style"], "strict")
        self.assertFalse(manifest["thinking_enabled"])
        self.assertEqual(manifest["total_request_attempts"], 1)
        self.assertFalse(bridge_row["prompt_truncated"])
        self.assertGreater(bridge_row["prompt_tokens_before_truncation"], 2)
        self.assertEqual(bridge_row["input_token_budget"], 4000)
        self.assertEqual(bridge_row["context_window_safety_margin_tokens"], 88)
        self.assertEqual(bridge_row["api_attempt_count"], 1)

    def test_openai_compatible_retry_exhaustion_is_counted_as_incorrect(self):
        calls = []

        def fake_factory():
            return {
                "api_key": "",
                "url": "http://executor.example:8000/v1/chat/completions",
                "opener": FakeOpenAICompatibleErrorOpener(calls),
            }

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
                api_provider="openai_compatible",
                api_model="Qwen/Qwen3.6-35B-A3B",
                api_base_url="http://executor.example:8000/v1",
                api_key_env="",
                max_output_tokens=8,
                context_window_tokens=4096,
                max_input_tokens=4000,
                max_retries=3,
                fail_fast=False,
                output_dir=root / "artifacts",
                manifest_note="",
            )

            with patch("long_bench_v2.run_api_benchmark.time.sleep"):
                run_longbench_api_benchmark(
                    args,
                    client_factory=fake_factory,
                    tokenizer_factory=lambda _: FakeTokenizer(),
                )

            run_dir = next((root / "artifacts" / ARTIFACT_SUBDIR).iterdir())
            manifest = json.loads((run_dir / "manifest.json").read_text())
            bridge_row = json.loads((run_dir / "bridge_rows.jsonl").read_text().splitlines()[0])

        self.assertEqual(len(calls), 3)
        self.assertEqual(manifest["api_error_count"], 1)
        self.assertEqual(manifest["total_request_attempts"], 3)
        self.assertEqual(bridge_row["api_attempt_count"], 3)
        self.assertEqual(bridge_row["api_status"], "error")
        self.assertIn("temporary failure", bridge_row["api_error"])
        self.assertFalse(bridge_row["answer_correct"])

    def test_openrouter_error_records_provider_response_text(self):
        calls = []

        def fake_factory():
            return {
                "api_key": "test-openrouter-key",
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "opener": FakeOpenRouterErrorOpener(calls),
            }

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
                api_provider="openrouter",
                api_model="anthropic/claude-sonnet-4.5",
                api_key_env="OPENROUTER_API_KEY",
                max_output_tokens=256,
                fail_fast=False,
                output_dir=root / "artifacts",
                manifest_note="",
            )

            run_longbench_api_benchmark(args, client_factory=fake_factory)

            run_dir = next((root / "artifacts" / ARTIFACT_SUBDIR).iterdir())
            manifest = json.loads((run_dir / "manifest.json").read_text())
            bridge_row = json.loads((run_dir / "bridge_rows.jsonl").read_text().splitlines()[0])

        self.assertEqual(calls[0]["url"], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(manifest["api_error_count"], 1)
        self.assertEqual(bridge_row["api_status"], "error")
        self.assertIn("input too long", bridge_row["api_error"])

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
        self.assertIn("input too long", bridge_row["api_error"])
        self.assertFalse(bridge_row["answer_correct"])

    def test_existing_run_directory_is_not_reused(self):
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
            fixed_time = datetime(2026, 8, 2, 20, 0, tzinfo=timezone.utc)

            with patch("long_bench_v2.run_api_benchmark.datetime") as mock_datetime:
                mock_datetime.now.return_value = fixed_time
                run_longbench_api_benchmark(args, client_factory=fake_factory)
                with self.assertRaises(FileExistsError):
                    run_longbench_api_benchmark(args, client_factory=fake_factory)


if __name__ == "__main__":
    unittest.main()
