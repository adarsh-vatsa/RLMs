import sys
import types
import unittest
from unittest.mock import patch

from local_llm import LocalLLMConfig, MLXLocalLLM, clean_generation_text


class LocalLLMTests(unittest.TestCase):
    def test_mlx_wrapper_loads_once_and_generates(self):
        fake_module = types.ModuleType("mlx_lm")
        calls = {"load": 0, "generate": 0}

        def fake_load(model_name):
            calls["load"] += 1
            return "model", "tokenizer"

        def fake_generate(**kwargs):
            calls["generate"] += 1
            self.assertEqual(kwargs["model"], "model")
            self.assertEqual(kwargs["tokenizer"], "tokenizer")
            self.assertEqual(kwargs["prompt"], "hello")
            self.assertEqual(kwargs["max_tokens"], 5)
            return " world "

        fake_module.load = fake_load
        fake_module.generate = fake_generate

        with patch.dict(sys.modules, {"mlx_lm": fake_module}):
            llm = MLXLocalLLM(LocalLLMConfig(model="local-test", max_tokens=5))
            self.assertEqual(llm.generate("hello"), "world")
            self.assertEqual(llm.generate("hello"), "world")

        self.assertEqual(calls["load"], 1)
        self.assertEqual(calls["generate"], 2)

    def test_default_generation_does_not_forward_repo_token_cap(self):
        fake_module = types.ModuleType("mlx_lm")
        seen_kwargs = {}

        def fake_load(model_name):
            return "model", "tokenizer"

        def fake_generate(**kwargs):
            seen_kwargs.update(kwargs)
            return " done "

        fake_module.load = fake_load
        fake_module.generate = fake_generate

        with patch.dict(sys.modules, {"mlx_lm": fake_module}):
            llm = MLXLocalLLM(LocalLLMConfig(model="local-test", max_tokens=None))
            self.assertEqual(llm.generate("hello"), "done")

        self.assertNotIn("max_tokens", seen_kwargs)

    def test_clean_generation_text_removes_thinking_blocks(self):
        self.assertEqual(
            clean_generation_text("<think>internal notes</think>\nFinal answer."),
            "Final answer.",
        )

    def test_chat_generation_uses_tokenizer_template_when_available(self):
        fake_module = types.ModuleType("mlx_lm")

        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize=True, **kwargs):
                self.kwargs = kwargs
                return "CHAT:" + messages[-1]["content"]

        fake_tokenizer = FakeTokenizer()

        def fake_load(model_name):
            return "model", fake_tokenizer

        def fake_generate(**kwargs):
            self.assertEqual(kwargs["prompt"], "CHAT:hello")
            return "world"

        fake_module.load = fake_load
        fake_module.generate = fake_generate

        with patch.dict(sys.modules, {"mlx_lm": fake_module}):
            llm = MLXLocalLLM(LocalLLMConfig(model="local-test", max_tokens=5))
            self.assertEqual(llm.generate_chat([{"role": "user", "content": "hello"}]), "world")
            self.assertFalse(fake_tokenizer.kwargs["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
