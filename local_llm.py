"""Local LLM adapter used by repo experiments.

The default backend is MLX via `mlx_lm`. It is intentionally small and simple:
callers pass a prompt and receive text. Higher-level code decides how to use it
as a solver, verifier, or synthesizer.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

RECOMMENDED_MLX_MODEL = "Brooooooklyn/Qwen3.6-27B-UD-Q3_K_XL-mlx"
SMOKE_MLX_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def _optional_int_from_env(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    if raw.strip().lower() in {"none", "null", "off", "false"}:
        return None
    return int(raw)


DEFAULT_MLX_MODEL = os.getenv("RLMS_MLX_MODEL", RECOMMENDED_MLX_MODEL)
DEFAULT_MAX_TOKENS = _optional_int_from_env("RLMS_MLX_MAX_TOKENS", None)
DEFAULT_MAX_KV_SIZE = _optional_int_from_env("RLMS_MLX_MAX_KV_SIZE", 2048)


def clean_generation_text(text: str) -> str:
    """Remove common reasoning wrappers from local model output."""
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0].strip()
    return cleaned.strip()


@dataclass
class LocalLLMConfig:
    model: str = DEFAULT_MLX_MODEL
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    temperature: float = 0.0
    max_kv_size: Optional[int] = DEFAULT_MAX_KV_SIZE


@dataclass
class GenerationResult:
    text: str
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    generation_tokens: Optional[int] = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class MLXLocalLLM:
    """Thin wrapper around mlx_lm.load/generate."""

    def __init__(self, config: LocalLLMConfig | None = None):
        self.config = config or LocalLLMConfig()
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            from mlx_lm import load
        except ImportError as exc:
            raise RuntimeError(
                "MLXLocalLLM requires mlx-lm. Install with `python -m pip install mlx-lm`."
            ) from exc
        self._model, self._tokenizer = load(self.config.model)

    def _generation_kwargs(
        self,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict:
        effective_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        effective_temperature = temperature if temperature is not None else self.config.temperature

        kwargs = {}
        if effective_max_tokens is not None:
            kwargs["max_tokens"] = effective_max_tokens

        # mlx_lm's Python API accepts a sampler object rather than the CLI's
        # `--temp` argument. For deterministic local tests, omit the sampler and
        # use greedy decoding. Non-zero temperature can be added later with
        # mlx_lm.sample_utils.make_sampler.
        if effective_temperature not in (0, 0.0):
            try:
                from mlx_lm.sample_utils import make_sampler
            except ImportError as exc:
                raise RuntimeError("This mlx-lm version does not expose make_sampler") from exc
            kwargs["sampler"] = make_sampler(temp=effective_temperature)
        if self.config.max_kv_size is not None:
            kwargs["max_kv_size"] = self.config.max_kv_size
        return kwargs

    def generate_result(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        self.load()
        kwargs = self._generation_kwargs(max_tokens=max_tokens, temperature=temperature)

        try:
            from mlx_lm import stream_generate
        except ImportError:
            stream_generate = None

        if stream_generate is not None:
            text = ""
            last_response = None
            for response in stream_generate(
                self._model,
                self._tokenizer,
                prompt,
                **kwargs,
            ):
                text += response.text
                last_response = response
            return GenerationResult(
                text=clean_generation_text(text),
                finish_reason=getattr(last_response, "finish_reason", None),
                prompt_tokens=getattr(last_response, "prompt_tokens", None),
                generation_tokens=getattr(last_response, "generation_tokens", None),
            )

        try:
            from mlx_lm import generate
        except ImportError as exc:
            raise RuntimeError(
                "MLXLocalLLM requires mlx-lm. Install with `python -m pip install mlx-lm`."
            ) from exc

        kwargs = {
            "model": self._model,
            "tokenizer": self._tokenizer,
            "prompt": prompt,
            "verbose": False,
            **kwargs,
        }
        return GenerationResult(text=clean_generation_text(generate(**kwargs).strip()))

    def generate(self, prompt: str, *, max_tokens: int | None = None, temperature: float | None = None) -> str:
        return self.generate_result(prompt, max_tokens=max_tokens, temperature=temperature).text

    def render_chat_prompt(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Render chat messages with the model tokenizer when possible."""
        self.load()
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self._tokenizer.apply_chat_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=True,
                )

        rendered = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            rendered.append(f"{role.upper()}: {content}")
        rendered.append("ASSISTANT:")
        return "\n\n".join(rendered)

    def generate_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate from chat messages using the tokenizer chat template."""
        return self.generate(
            self.render_chat_prompt(messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate_chat_result(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        """Generate from chat messages and return backend finish metadata."""
        return self.generate_result(
            self.render_chat_prompt(messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )


def quick_generate(
    prompt: str,
    model: str = DEFAULT_MLX_MODEL,
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
    max_kv_size: Optional[int] = DEFAULT_MAX_KV_SIZE,
) -> str:
    """One-shot helper for scripts and smoke tests."""
    llm = MLXLocalLLM(LocalLLMConfig(model=model, max_tokens=max_tokens, max_kv_size=max_kv_size))
    return llm.generate(prompt)
