"""Shared OpenAI-compatible HTTP client and retry accounting."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


# Qwen3.6's recommended thinking settings for precise tasks. Its general ones add a presence penalty
# of 1.5, which would penalize MRCR's verbatim answers.
THINKING_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    raw_usage: dict
    finish_reason: str | None = None
    reasoning: str = ""


def with_thinking(body: dict, config, case_id: str) -> dict:
    """`body` unchanged without thinking. With thinking, sampling is seeded by the question and
    `config.sampling_seed`, so every policy draws the same seed for a question and a rerun repeats it."""
    if not config.thinking:
        return body
    seed = int(hashlib.sha256(f"{case_id}:{config.sampling_seed}".encode()).hexdigest()[:8], 16)
    template = {**body.get("chat_template_kwargs", {}), "enable_thinking": True}
    return {**body, **THINKING_SAMPLING, "seed": seed, "chat_template_kwargs": template}


def response_reasoning(payload: dict) -> str:
    """The reasoning returned beside the answer; vLLM versions name the field either way."""
    message = (payload.get("choices") or [{}])[0].get("message") or {}
    return str(message.get("reasoning") or message.get("reasoning_content") or "").strip()


class CompletionRetryError(RuntimeError):
    def __init__(self, attempts: int, cause: Exception):
        super().__init__(str(cause))
        self.attempts = attempts
        self.__cause__ = cause


def served_context_window(
    *, base_url: str, model: str, api_key: str = "", timeout_seconds: int = 30,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> int:
    """Read the selected vLLM model's served limit, not its training limit."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(base_url.rstrip("/") + "/models", headers=headers)
    try:
        with opener(request, timeout=timeout_seconds) as response:
            models = json.loads(response.read().decode("utf-8"))["data"]
        selected = next(item for item in models if item.get("id") == model)
        limit = selected.get("max_model_len")
        if type(limit) is not int or limit <= 0:
            raise ValueError("selected model has no positive max_model_len")
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        raise ValueError(
            f"Cannot determine served context for {model!r}; "
            "set --context-window-tokens to the executor's served limit"
        ) from exc
    return limit


def _response_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    return str(choices[0].get("text") or "").strip()


def call_chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    extra_body: dict | None = None,
    api_key: str = "",
    timeout_seconds: int = 1800,
    api_style: str = "vllm",
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> Completion:
    payload = {
        "model": model,
        "messages": messages,
    }
    if api_style == "openai":
        payload["max_completion_tokens"] = max_tokens
    elif api_style == "vllm":
        payload.update(max_tokens=max_tokens, temperature=0)
    else:
        raise ValueError(f"Unsupported API style: {api_style}")
    if extra_body:
        payload.update(extra_body)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc
    usage = result.get("usage") or {}
    return Completion(
        text=_response_text(result),
        input_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        output_tokens=int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        ),
        raw_usage=usage,
        finish_reason=(result.get("choices") or [{}])[0].get("finish_reason"),
        reasoning=response_reasoning(result),
    )


def call_with_retries(
    call: Callable[[], Completion],
    *,
    max_retries: int,
) -> tuple[Completion, int]:
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    for attempt in range(1, max_retries + 1):
        try:
            return call(), attempt
        except Exception as exc:
            if attempt == max_retries:
                raise CompletionRetryError(attempt, exc) from exc
            time.sleep(1)
    raise AssertionError("unreachable")
