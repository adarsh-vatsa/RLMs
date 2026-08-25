"""Minimal OpenAI-compatible HTTP client used by AA-LCR."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    raw_usage: dict


class CompletionRetryError(RuntimeError):
    def __init__(self, attempts: int, cause: Exception):
        super().__init__(str(cause))
        self.attempts = attempts
        self.__cause__ = cause


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
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> Completion:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
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
