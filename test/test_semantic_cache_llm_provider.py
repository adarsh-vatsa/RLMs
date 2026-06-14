import io
import json
import os
import sys
import urllib.error
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import semantic_cache_system as scs


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class SemanticCacheLLMProviderTests(unittest.TestCase):
    def tearDown(self):
        for name in scs.OPENAI_COMPAT_EXTRA_BODY_ENVS:
            os.environ.pop(name, None)
        scs.MCQ_PROMPT_STYLE = "default"
        scs.configure_llm_provider(
            provider="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            executor_model="claude-sonnet-4-20250514",
            evaluator_model="claude-haiku-4-5-20251001",
            openai_compat_base_url="http://127.0.0.1:8000/v1",
            openai_compat_executor_base_url="",
            openai_compat_evaluator_base_url="",
            openai_compat_api_key_env="",
            openai_compat_structured_outputs=True,
        )

    def test_openrouter_message_wrapper_parses_response_and_usage(self):
        calls = []

        def fake_urlopen(request, timeout=120):
            calls.append(
                {
                    "url": request.full_url,
                    "timeout": timeout,
                    "body": json.loads(request.data.decode("utf-8")),
                }
            )
            return FakeHTTPResponse(
                {
                    "choices": [{"message": {"content": "Final answer: C"}}],
                    "usage": {
                        "prompt_tokens": 500,
                        "completion_tokens": 25,
                    },
                }
            )

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), patch(
            "semantic_cache_system.urllib.request.urlopen",
            fake_urlopen,
        ):
            scs.configure_llm_provider(
                provider="openrouter",
                api_key_env="OPENROUTER_API_KEY",
                executor_model="anthropic/claude-sonnet-4.5",
                evaluator_model="anthropic/claude-haiku-4.5",
                openrouter_base_url="https://openrouter.ai/api/v1",
            )
            response = scs.create_llm_message(
                model="anthropic/claude-sonnet-4.5",
                max_tokens=10,
                temperature=0,
                system="Answer briefly.",
                messages=[{"role": "user", "content": "Question"}],
            )

        self.assertEqual(calls[0]["url"], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(calls[0]["body"]["model"], "anthropic/claude-sonnet-4.5")
        self.assertEqual(calls[0]["body"]["messages"][0]["role"], "system")
        self.assertEqual(response.content[0].text, "Final answer: C")
        self.assertEqual(response.usage.input_tokens, 500)
        self.assertEqual(response.usage.output_tokens, 25)

    def test_openrouter_message_wrapper_preserves_error_body(self):
        def fake_urlopen(request, timeout=120):
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "Bad Request",
                hdrs={},
                fp=io.BytesIO(b'{"error":{"message":"input too long"}}'),
            )

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), patch(
            "semantic_cache_system.urllib.request.urlopen",
            fake_urlopen,
        ):
            scs.configure_llm_provider(provider="openrouter", api_key_env="OPENROUTER_API_KEY")
            with self.assertRaisesRegex(RuntimeError, "input too long"):
                scs.create_llm_message(
                    model="anthropic/claude-sonnet-4.5",
                    max_tokens=10,
                    temperature=0,
                    messages=[{"role": "user", "content": "Question"}],
                )

    def test_openai_compatible_shared_endpoint_has_optional_auth_and_missing_usage_defaults(self):
        calls = []

        def fake_urlopen(request, timeout=120):
            calls.append(
                {
                    "url": request.full_url,
                    "headers": dict(request.header_items()),
                    "body": json.loads(request.data.decode("utf-8")),
                }
            )
            return FakeHTTPResponse({"choices": [{"message": {"content": "local answer"}}]})

        with patch("semantic_cache_system.urllib.request.urlopen", fake_urlopen):
            scs.configure_llm_provider(
                provider="openai_compatible",
                openai_compat_base_url="http://llm-node:8000/v1",
                openai_compat_executor_base_url="",
                openai_compat_evaluator_base_url="",
                openai_compat_api_key_env="",
            )
            response = scs.create_llm_message(
                model=scs.EXECUTOR_MODEL,
                max_tokens=10,
                temperature=0,
                system="Answer briefly.",
                messages=[{"role": "user", "content": "Question"}],
            )

        self.assertEqual(calls[0]["url"], "http://llm-node:8000/v1/chat/completions")
        self.assertNotIn("Authorization", calls[0]["headers"])
        self.assertEqual(calls[0]["body"]["messages"][0]["role"], "system")
        self.assertEqual(response.content[0].text, "local answer")
        self.assertEqual(response.usage.input_tokens, 0)
        self.assertEqual(response.usage.output_tokens, 0)

    def test_openai_compatible_routes_executor_and_evaluator_to_role_endpoints(self):
        calls = []

        def fake_urlopen(request, timeout=120):
            calls.append({"url": request.full_url, "body": json.loads(request.data.decode("utf-8"))})
            return FakeHTTPResponse(
                {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                }
            )

        with patch("semantic_cache_system.urllib.request.urlopen", fake_urlopen):
            scs.configure_llm_provider(
                provider="openai_compatible",
                executor_model="meta-llama/Llama-3.3-70B-Instruct",
                evaluator_model="mistralai/Mistral-Small-24B-Instruct-2501",
                openai_compat_base_url="http://shared:8000/v1",
                openai_compat_executor_base_url="http://executor:8000/v1",
                openai_compat_evaluator_base_url="http://evaluator:8001/v1",
            )
            scs.create_llm_message(
                model=scs.EXECUTOR_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": "Synthesize"}],
            )
            scs.create_llm_message(
                model=scs.EVALUATOR_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": "Verify"}],
            )
            scs.create_llm_message(
                model="other/model",
                max_tokens=10,
                messages=[{"role": "user", "content": "Fallback"}],
            )

        self.assertEqual(calls[0]["url"], "http://executor:8000/v1/chat/completions")
        self.assertEqual(calls[1]["url"], "http://evaluator:8001/v1/chat/completions")
        self.assertEqual(calls[2]["url"], "http://shared:8000/v1/chat/completions")
        self.assertEqual(calls[0]["body"]["model"], "meta-llama/Llama-3.3-70B-Instruct")
        self.assertEqual(calls[1]["body"]["model"], "mistralai/Mistral-Small-24B-Instruct-2501")

    def test_openai_compatible_auth_header_is_added_when_key_env_is_configured(self):
        calls = []

        def fake_urlopen(request, timeout=120):
            calls.append({"headers": dict(request.header_items())})
            return FakeHTTPResponse({"choices": [{"message": {"content": "ok"}}]})

        with patch.dict(os.environ, {"LOCAL_VLLM_API_KEY": "local-secret"}), patch(
            "semantic_cache_system.urllib.request.urlopen",
            fake_urlopen,
        ):
            scs.configure_llm_provider(
                provider="openai_compatible",
                openai_compat_base_url="http://llm-node:8000/v1",
                openai_compat_api_key_env="LOCAL_VLLM_API_KEY",
            )
            scs.create_llm_message(
                model=scs.EXECUTOR_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": "Question"}],
            )

        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer local-secret")

    def test_openai_compatible_structured_output_payload_is_passed_through(self):
        calls = []

        def fake_urlopen(request, timeout=120):
            calls.append(json.loads(request.data.decode("utf-8")))
            return FakeHTTPResponse({"choices": [{"message": {"content": "{\"hit\": false, \"id\": null}"}}]})

        with patch("semantic_cache_system.urllib.request.urlopen", fake_urlopen):
            scs.configure_llm_provider(
                provider="openai_compatible",
                openai_compat_base_url="http://llm-node:8000/v1",
            )
            scs.create_llm_message(
                model=scs.EVALUATOR_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": "JSON"}],
                response_format={"type": "json_object"},
                extra_body={"seed": 123},
            )

        self.assertEqual(calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(calls[0]["seed"], 123)

    def test_openai_compatible_extra_body_envs_merge_by_role(self):
        calls = []

        def fake_urlopen(request, timeout=120):
            calls.append(json.loads(request.data.decode("utf-8")))
            return FakeHTTPResponse({"choices": [{"message": {"content": "ok"}}]})

        env = {
            "OPENAI_COMPAT_EXTRA_BODY_JSON": json.dumps(
                {
                    "seed": 11,
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                        "keep": "common",
                    },
                }
            ),
            "OPENAI_COMPAT_EXECUTOR_EXTRA_BODY_JSON": json.dumps(
                {
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            ),
            "OPENAI_COMPAT_EVALUATOR_EXTRA_BODY_JSON": json.dumps(
                {
                    "seed": 22,
                    "chat_template_kwargs": {"evaluator_only": True},
                }
            ),
        }

        with patch.dict(os.environ, env), patch(
            "semantic_cache_system.urllib.request.urlopen",
            fake_urlopen,
        ):
            scs.configure_llm_provider(
                provider="openai_compatible",
                executor_model="executor/model",
                evaluator_model="evaluator/model",
                openai_compat_base_url="http://llm-node:8000/v1",
            )
            scs.create_llm_message(
                model=scs.EXECUTOR_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": "Synthesize"}],
            )
            scs.create_llm_message(
                model=scs.EVALUATOR_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": "Judge"}],
            )
            scs.create_llm_message(
                model="other/model",
                max_tokens=10,
                messages=[{"role": "user", "content": "Fallback"}],
            )

        self.assertEqual(calls[0]["seed"], 11)
        self.assertEqual(calls[0]["top_k"], 20)
        self.assertEqual(
            calls[0]["chat_template_kwargs"],
            {"enable_thinking": False, "keep": "common"},
        )
        self.assertEqual(calls[1]["seed"], 22)
        self.assertEqual(
            calls[1]["chat_template_kwargs"],
            {"enable_thinking": True, "keep": "common", "evaluator_only": True},
        )
        self.assertEqual(calls[2]["seed"], 11)
        self.assertNotIn("top_k", calls[2])

    def test_openai_compatible_extra_body_env_invalid_json_fails_closed(self):
        def fake_urlopen(request, timeout=120):
            self.fail("request should not be sent when extra-body JSON is invalid")

        with patch.dict(os.environ, {"OPENAI_COMPAT_EXTRA_BODY_JSON": "not-json"}), patch(
            "semantic_cache_system.urllib.request.urlopen",
            fake_urlopen,
        ):
            scs.configure_llm_provider(provider="openai_compatible")
            with self.assertRaisesRegex(RuntimeError, "OPENAI_COMPAT_EXTRA_BODY_JSON"):
                scs.create_llm_message(
                    model=scs.EXECUTOR_MODEL,
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Question"}],
                )

    def test_openai_compatible_extra_body_config_redacts_manifest_secrets(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_COMPAT_EXTRA_BODY_JSON": json.dumps(
                    {
                        "api_key": "secret",
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                ),
                "OPENAI_COMPAT_EXECUTOR_EXTRA_BODY_JSON": "",
                "OPENAI_COMPAT_EVALUATOR_EXTRA_BODY_JSON": "",
            },
        ):
            config = scs.get_openai_compatible_extra_body_config(redact=True)

        self.assertEqual(config["common"]["api_key"], "[REDACTED]")
        self.assertEqual(config["common"]["chat_template_kwargs"]["enable_thinking"], False)

    def test_llm_json_parser_strips_thinking_blocks_and_extracts_first_object(self):
        parsed = scs._extract_llm_json_object(
            '<think>{"hit": false, "id": 99}</think>\nText before {"hit": true, "id": "3"} trailing'
        )

        self.assertEqual(parsed, {"hit": True, "id": "3"})
        self.assertIsNone(scs._extract_llm_json_object("<think>unfinished"))
        self.assertIsNone(scs._extract_llm_json_object("not json"))

    def test_mcq_prompt_style_can_use_strict_elimination_prompt(self):
        original_style = scs.MCQ_PROMPT_STYLE
        try:
            scs.MCQ_PROMPT_STYLE = "strict"
            strict_prompt = scs._mcq_system_prompt()
            scs.MCQ_PROMPT_STYLE = "default"
            default_prompt = scs._mcq_system_prompt()
        finally:
            scs.MCQ_PROMPT_STYLE = original_style

        self.assertIn("Reject choices", strict_prompt)
        self.assertIn("satisfy every constraint", strict_prompt)
        self.assertIn("overstate the evidence", strict_prompt)
        self.assertNotEqual(strict_prompt, default_prompt)

    def test_sniper_fails_closed_on_malformed_or_out_of_range_json(self):
        metrics = scs.ExecutionMetrics()
        controller = scs.SemanticCacheController(metrics=metrics)
        candidates = [({"query": "What is ARR?"}, 0, 0.95)]

        def fake_message(**kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(text='<think>checking</think>{"hit": true, "id": 99}')],
                usage=SimpleNamespace(input_tokens=5, output_tokens=2),
            )

        with patch("semantic_cache_system.create_llm_message", fake_message):
            decision = controller._llm_sniper_evaluate("Report ARR", candidates)

        self.assertEqual(decision, {"hit": False, "id": None})

    def test_autonomous_agent_mocked_miss_store_then_hit_flow(self):
        class FakeRouter:
            def select_model(self, query):
                return scs.EXECUTOR_MODEL

        class FakeCache:
            def __init__(self):
                self.stored = None

            def check(self, query, context):
                if self.stored:
                    return {"result": self.stored, "ephemeral": False, "was_summarized": False}
                return None

            def consensus_verify(self, query, context, result, model):
                return {"consensus": "AGREED", "divergent_facts": []}

            def store(self, query, context, result, model_used="unknown"):
                self.stored = result

            def _apply_context_collapse_guard(self, result, query, source_context=""):
                return {"result": result, "ephemeral": False, "was_summarized": False}

        def fake_message(**kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(text="fresh local answer")],
                usage=SimpleNamespace(input_tokens=11, output_tokens=4),
            )

        agent = scs.AutonomousAgent.__new__(scs.AutonomousAgent)
        agent.metrics = scs.ExecutionMetrics()
        agent.cache = FakeCache()
        agent.router = FakeRouter()

        with patch("semantic_cache_system.create_llm_message", fake_message):
            first = agent.cached_query("Summarize", "Document")
            second = agent.cached_query("Summarize", "Document")

        self.assertFalse(first["from_cache"])
        self.assertEqual(first["result"], "fresh local answer")
        self.assertTrue(second["from_cache"])
        self.assertEqual(second["result"], "fresh local answer")


if __name__ == "__main__":
    unittest.main()
