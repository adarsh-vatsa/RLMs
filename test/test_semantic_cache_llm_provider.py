import io
import json
import os
import sys
import urllib.error
import unittest
from pathlib import Path
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
        scs.configure_llm_provider(provider="anthropic", api_key_env="ANTHROPIC_API_KEY")

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


if __name__ == "__main__":
    unittest.main()
