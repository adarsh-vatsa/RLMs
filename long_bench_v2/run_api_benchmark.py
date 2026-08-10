"""Plain full-context API baseline runner for prepared LongBench-v2 CSV suites."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")

from long_bench_v2.run_benchmark import (  # noqa: E402
    DEFAULT_ROW_TYPES,
    DEFAULT_SOURCE_JSON,
    DEFAULT_SUITE_CSV,
    _coerce_text,
    _parse_csv_values,
    _sha256_file,
    answer_correct,
    build_query,
    filter_suite_rows,
    load_context_by_source_id,
    load_suite_rows,
    parse_choice,
    summarize_rows,
)
from long_bench_v2.rlm_helpers import (  # noqa: E402
    _aggregate_bridge_row_totals,
    _estimate_cost_usd,
    _get_nested_number,
    _json_safe,
    _write_csv_rows,
)
from long_bench_v2.qwen_prompt import (  # noqa: E402
    MCQ_ALLOWED_CHOICES,
    STRICT_MCQ_SYSTEM_PROMPT,
    build_openai_compatible_mcq_extra_body,
    build_strict_mcq_messages,
    chat_token_count as _chat_token_count,
    mcq_decoder_constraint_metadata,
    token_ids as _token_ids,
)


ARTIFACT_SUBDIR = "longbench_v2_api"
REPORT_FILENAME = "official_longbench_v2_api_eval_report.json"
DEFAULT_API_PROVIDER = "anthropic"
DEFAULT_API_MODEL = "claude-sonnet-4-5"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_OPENAI_COMPAT_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_OPENAI_COMPAT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_CONTEXT_WINDOW_TOKENS = 65536
DEFAULT_MAX_INPUT_TOKENS = 60000
OPENAI_COMPAT_SYSTEM_PROMPT = STRICT_MCQ_SYSTEM_PROMPT
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
API_SYSTEM_PROMPT = (
    "Answer the multiple-choice question using only the provided context. "
    "Return only the final answer choice letter: A, B, C, or D."
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_api_prompt(row: dict) -> str:
    return f"{API_SYSTEM_PROMPT}\n\nContext:\n{row['context']}\n\n{build_query(row)}"


def build_openai_compatible_messages(row: dict) -> list[dict]:
    return build_strict_mcq_messages(row["context"], build_query(row))


def _middle_tokens(token_ids: list[int], limit: int) -> list[int]:
    if len(token_ids) <= limit:
        return list(token_ids)
    if limit <= 0:
        return []
    head = limit // 2
    tail = limit - head
    return token_ids[:head] + token_ids[-tail:]


def prepare_openai_compatible_messages(
    row: dict,
    tokenizer: Any,
    context_window_tokens: int,
    max_input_tokens: int,
    max_output_tokens: int,
) -> tuple[list[dict], dict]:
    if context_window_tokens <= max_output_tokens:
        raise ValueError("--context-window-tokens must exceed --max-output-tokens")
    if max_input_tokens <= 0:
        raise ValueError("--max-input-tokens must be positive")
    if max_input_tokens + max_output_tokens > context_window_tokens:
        raise ValueError(
            "--max-input-tokens plus --max-output-tokens must not exceed "
            "--context-window-tokens"
        )

    messages = build_openai_compatible_messages(row)
    input_budget = max_input_tokens
    safety_margin_tokens = context_window_tokens - max_input_tokens - max_output_tokens
    original_prompt_tokens = _chat_token_count(tokenizer, messages)
    if original_prompt_tokens <= input_budget:
        return messages, {
            "prompt_truncated": False,
            "prompt_tokens_before_truncation": original_prompt_tokens,
            "prompt_tokens_after_truncation": original_prompt_tokens,
            "prompt_tokens_removed": 0,
            "context_window_tokens": context_window_tokens,
            "input_token_budget": input_budget,
            "context_window_safety_margin_tokens": safety_margin_tokens,
        }

    user_content = messages[1]["content"]
    user_token_ids = _token_ids(tokenizer.encode(user_content, add_special_tokens=False))
    empty_messages = [messages[0], {"role": "user", "content": ""}]
    chat_overhead = _chat_token_count(tokenizer, empty_messages)
    keep_tokens = min(len(user_token_ids), max(0, input_budget - chat_overhead))

    while True:
        truncated_user_ids = _middle_tokens(user_token_ids, keep_tokens)
        truncated_user = tokenizer.decode(truncated_user_ids, skip_special_tokens=True)
        truncated_messages = [messages[0], {"role": "user", "content": truncated_user}]
        final_prompt_tokens = _chat_token_count(tokenizer, truncated_messages)
        if final_prompt_tokens <= input_budget:
            break
        overflow = final_prompt_tokens - input_budget
        next_keep_tokens = max(0, keep_tokens - overflow)
        if next_keep_tokens == keep_tokens:
            next_keep_tokens = max(0, keep_tokens - 1)
        if keep_tokens == 0:
            raise ValueError("System prompt and chat template exceed the configured input budget")
        keep_tokens = next_keep_tokens

    return truncated_messages, {
        "prompt_truncated": True,
        "prompt_tokens_before_truncation": original_prompt_tokens,
        "prompt_tokens_after_truncation": final_prompt_tokens,
        "prompt_tokens_removed": original_prompt_tokens - final_prompt_tokens,
        "context_window_tokens": context_window_tokens,
        "input_token_budget": input_budget,
        "context_window_safety_margin_tokens": safety_margin_tokens,
    }


def _extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message")
                if isinstance(message, dict):
                    content_value = message.get("content")
                    if isinstance(content_value, str) and content_value.strip():
                        return content_value.strip()
                text_value = first_choice.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    return text_value.strip()
        for key in ("text", "content", "response", "answer", "output"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        content = response.get("content")
    else:
        content = getattr(response, "content", None)

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    return _coerce_text(response)


def parse_api_usage(response: Any, model: str, success: bool) -> dict:
    usage = getattr(response, "usage", None)
    if isinstance(response, dict):
        usage = response.get("usage", usage)
    safe_usage = _json_safe(usage)
    input_tokens = int(
        _get_nested_number(safe_usage, {"input_tokens", "prompt_tokens", "total_input_tokens"})
    )
    output_tokens = int(
        _get_nested_number(safe_usage, {"output_tokens", "completion_tokens", "total_output_tokens"})
    )
    total_tokens = int(_get_nested_number(safe_usage, {"total_tokens", "tokens_total"}))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    status = "parsed" if usage is not None and (input_tokens or output_tokens or total_tokens) else "missing_usage_summary"
    return {
        "raw": safe_usage,
        "calls": 1 if success else 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(_estimate_cost_usd(model, input_tokens, output_tokens), 8),
        "usage_parse_status": status,
    }


def build_eval_report(run_dir: Path, bridge_rows: list[dict], manifest: dict) -> dict:
    failing_rows = [
        {
            "case_id": row["case_id"],
            "source_id": row["source_id"],
            "row_type": row["row_type"],
            "expected_answer": row["expected_answer"],
            "prediction": row["prediction"],
            "generation": row["generation"],
            "api_status": row["api_status"],
            "api_error": row.get("api_error", ""),
        }
        for row in bridge_rows
        if not row.get("answer_correct")
    ]
    return {
        "run_dir": str(run_dir),
        "benchmark_target": "longbench_v2_api",
        "baseline_type": "plain_api_uncached_full_context",
        "scored_rows": len(bridge_rows),
        "answer_accuracy": manifest["answer_accuracy"],
        "row_type_counts": manifest["row_type_counts"],
        "by_row_type": manifest["by_row_type"],
        "api_error_count": manifest["api_error_count"],
        "truncated_row_count": manifest["truncated_row_count"],
        "valid_choice_count": manifest["valid_choice_count"],
        "invalid_choice_count": manifest["invalid_choice_count"],
        "valid_choice_accuracy": manifest["valid_choice_accuracy"],
        "failing_rows": failing_rows,
    }


def _build_default_api_client_factory(args: argparse.Namespace) -> Callable[[], Any]:
    if args.api_provider == "openrouter":
        api_key = os.getenv(args.api_key_env) if args.api_key_env else ""
        if not api_key:
            raise RuntimeError(
                f"OpenRouter API key not found. Set {args.api_key_env} or pass --api-key-env."
            )

        def factory():
            return {"api_key": api_key, "url": OPENROUTER_CHAT_COMPLETIONS_URL}

        return factory

    if args.api_provider == "openai_compatible":
        api_key = os.getenv(args.api_key_env) if args.api_key_env else ""

        def factory():
            return {
                "api_key": api_key,
                "url": args.api_base_url.rstrip("/") + "/chat/completions",
            }

        return factory

    if args.api_provider != "anthropic":
        raise ValueError(f"Unsupported --api-provider: {args.api_provider}")

    try:
        from anthropic import Anthropic
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Anthropic package is not installed. Install it first, for example: "
            "`pip install anthropic`."
        ) from exc

    api_key = os.getenv(args.api_key_env) if args.api_key_env else ""

    def factory():
        if api_key:
            return Anthropic(api_key=api_key)
        return Anthropic()

    return factory


def _call_anthropic(client: Any, args: argparse.Namespace, prompt: str) -> Any:
    return client.messages.create(
        model=args.api_model,
        max_tokens=args.max_output_tokens,
        messages=[{"role": "user", "content": prompt}],
    )


def normalize_api_args(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "api_model", None) is None:
        if args.api_provider == "openrouter":
            args.api_model = DEFAULT_OPENROUTER_MODEL
        elif args.api_provider == "openai_compatible":
            args.api_model = DEFAULT_OPENAI_COMPAT_MODEL
        else:
            args.api_model = DEFAULT_API_MODEL
    if getattr(args, "api_key_env", None) is None:
        if args.api_provider == "openrouter":
            args.api_key_env = "OPENROUTER_API_KEY"
        elif args.api_provider == "openai_compatible":
            args.api_key_env = os.getenv("OPENAI_COMPAT_API_KEY_ENV", "")
        else:
            args.api_key_env = "ANTHROPIC_API_KEY"
    if getattr(args, "api_base_url", None) is None:
        args.api_base_url = (
            os.getenv("OPENAI_COMPAT_EXECUTOR_BASE_URL")
            or os.getenv("OPENAI_COMPAT_BASE_URL")
            or DEFAULT_OPENAI_COMPAT_BASE_URL
        )
    if not hasattr(args, "context_window_tokens"):
        args.context_window_tokens = DEFAULT_CONTEXT_WINDOW_TOKENS
    if not hasattr(args, "max_input_tokens"):
        args.max_input_tokens = DEFAULT_MAX_INPUT_TOKENS
    if getattr(args, "max_retries", None) is None:
        args.max_retries = 5 if args.api_provider == "openai_compatible" else 1
    return args


def _call_openrouter(client: Any, args: argparse.Namespace, prompt: str) -> dict:
    if isinstance(client, dict):
        api_key = _coerce_text(client.get("api_key"))
        url = _coerce_text(client.get("url")) or OPENROUTER_CHAT_COMPLETIONS_URL
        opener = client.get("opener") or urllib.request.urlopen
    else:
        api_key = _coerce_text(getattr(client, "api_key", ""))
        url = _coerce_text(getattr(client, "url", "")) or OPENROUTER_CHAT_COMPLETIONS_URL
        opener = getattr(client, "opener", urllib.request.urlopen)
    if not api_key:
        raise RuntimeError("OpenRouter API key is missing")

    payload = {
        "model": args.api_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_output_tokens,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with opener(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
    return json.loads(raw)


def _call_openai_compatible(client: Any, args: argparse.Namespace, messages: list[dict]) -> dict:
    if isinstance(client, dict):
        api_key = _coerce_text(client.get("api_key"))
        url = _coerce_text(client.get("url"))
        opener = client.get("opener") or urllib.request.urlopen
    else:
        api_key = _coerce_text(getattr(client, "api_key", ""))
        url = _coerce_text(getattr(client, "url", ""))
        opener = getattr(client, "opener", urllib.request.urlopen)
    if not url:
        url = args.api_base_url.rstrip("/") + "/chat/completions"

    payload = {
        "model": args.api_model,
        "messages": messages,
        "max_tokens": args.max_output_tokens,
        "temperature": 0,
    }
    payload.update(build_openai_compatible_mcq_extra_body())
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with opener(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc
    return json.loads(raw)


def _call_api(
    client: Any,
    args: argparse.Namespace,
    prompt: str,
    messages: Optional[list[dict]] = None,
) -> Any:
    if args.api_provider == "anthropic":
        return _call_anthropic(client, args, prompt)
    if args.api_provider == "openrouter":
        return _call_openrouter(client, args, prompt)
    if args.api_provider == "openai_compatible":
        if messages is None:
            raise ValueError("OpenAI-compatible requests require chat messages")
        return _call_openai_compatible(client, args, messages)
    raise ValueError(f"Unsupported --api-provider: {args.api_provider}")


def _load_tokenizer(model: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Transformers is required for OpenAI-compatible prompt truncation. "
            "Install the Jarvis client dependencies first."
        ) from exc
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def run_longbench_api_benchmark(
    args: argparse.Namespace,
    client_factory: Optional[Callable[[], Any]] = None,
    tokenizer_factory: Optional[Callable[[str], Any]] = None,
) -> None:
    args = normalize_api_args(args)
    if args.max_retries < 1:
        raise ValueError("--max-retries must be at least 1")
    suite_csv = Path(args.suite_csv)
    source_json_path = Path(args.source_json_path)
    row_types = _parse_csv_values(args.row_types)
    if not row_types:
        raise ValueError("--row-types must include at least one row type")

    contexts = load_context_by_source_id(source_json_path)
    all_rows = load_suite_rows(suite_csv, contexts)
    source_ids = _parse_csv_values(getattr(args, "source_ids", ""))
    selected_rows = filter_suite_rows(
        all_rows,
        row_types=row_types,
        max_rows=args.max_rows,
        source_ids=source_ids,
    )
    if not selected_rows:
        raise ValueError("No LongBench-v2 rows matched the requested filters")

    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / ARTIFACT_SUBDIR / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    predictions_path = out_dir / "predictions.jsonl"
    bridge_rows_path = out_dir / "bridge_rows.jsonl"
    bridge_rows_csv_path = out_dir / "bridge_rows.csv"
    manifest_path = out_dir / "manifest.json"
    report_path = out_dir / REPORT_FILENAME

    print(f"\n[LONGBENCH-V2-API] Run id: {run_id}")
    print(f"[LONGBENCH-V2-API] Rows: {len(selected_rows)}")
    print(f"[LONGBENCH-V2-API] Row types: {row_types}")
    print(f"[LONGBENCH-V2-API] Provider/model: {args.api_provider}/{args.api_model}")
    print(f"[LONGBENCH-V2-API] Output dir: {out_dir}")

    if client_factory is None:
        client_factory = _build_default_api_client_factory(args)
    client = client_factory()
    tokenizer = None
    if args.api_provider == "openai_compatible":
        tokenizer_factory = tokenizer_factory or _load_tokenizer
        tokenizer = tokenizer_factory(args.api_model)

    prediction_rows: list[dict] = []
    bridge_rows: list[dict] = []

    for idx, row in enumerate(selected_rows, start=1):
        print(f"[LONGBENCH-V2-API] Row {idx}/{len(selected_rows)}: {row['case_id']}")
        prompt = build_api_prompt(row)
        messages = None
        truncation = {
            "prompt_truncated": False,
            "prompt_tokens_before_truncation": 0,
            "prompt_tokens_after_truncation": 0,
            "prompt_tokens_removed": 0,
            "context_window_tokens": 0,
            "input_token_budget": 0,
            "context_window_safety_margin_tokens": 0,
        }
        if tokenizer is not None:
            messages, truncation = prepare_openai_compatible_messages(
                row,
                tokenizer,
                args.context_window_tokens,
                args.max_input_tokens,
                args.max_output_tokens,
            )
        generation = ""
        api_status = "ok"
        api_error = ""
        response = None
        api_attempt_count = 0
        t0 = time.time()
        try:
            while api_attempt_count < args.max_retries:
                api_attempt_count += 1
                try:
                    response = _call_api(client, args, prompt, messages=messages)
                    break
                except Exception:
                    if api_attempt_count >= args.max_retries:
                        raise
                    time.sleep(1)
            generation = _extract_response_text(response)
            usage = parse_api_usage(response, args.api_model, success=True)
        except Exception as exc:
            latency_ms = (time.time() - t0) * 1000.0
            if args.fail_fast:
                raise
            api_status = "error"
            api_error = f"{type(exc).__name__}: {exc}"
            usage = {
                "raw": {},
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "usage_parse_status": "api_error",
            }
        else:
            latency_ms = (time.time() - t0) * 1000.0

        prediction = parse_choice(generation)
        correct = answer_correct(generation, row.get("answer", ""))
        valid_choice = generation.strip() in MCQ_ALLOWED_CHOICES
        prediction_row = {
            "id": row["case_id"],
            "sample_id": row["case_id"],
            "source_id": row["source_id"],
            "case_id": row["case_id"],
            "row_type": row["row_type"],
            "question": row["question"],
            "generation": generation,
            "prediction": prediction,
            "answer": row.get("answer", ""),
            "expected_answer": row.get("answer", ""),
            "choice_A": row.get("choice_A", ""),
            "choice_B": row.get("choice_B", ""),
            "choice_C": row.get("choice_C", ""),
            "choice_D": row.get("choice_D", ""),
        }
        prediction_rows.append(prediction_row)

        bridge_rows.append(
            {
                **prediction_row,
                "token_count": row.get("token_count", ""),
                "expected_cache_type": row.get("expected_cache_type", ""),
                "expected_from_cache": row.get("expected_from_cache", ""),
                "answer_correct": correct,
                "valid_choice": valid_choice,
                "latency_ms": round(latency_ms, 3),
                "delta_calls": usage["calls"],
                "delta_input_tokens": usage["input_tokens"],
                "delta_output_tokens": usage["output_tokens"],
                "delta_cost_usd": usage["cost_usd"],
                "api_provider": args.api_provider,
                "api_model": args.api_model,
                "api_status": api_status,
                "api_error": api_error,
                "api_attempt_count": api_attempt_count,
                "api_usage_summary": usage["raw"],
                "usage_parse_status": usage["usage_parse_status"],
                **truncation,
                **(
                    mcq_decoder_constraint_metadata()
                    if args.api_provider == "openai_compatible"
                    else {}
                ),
            }
        )

    _write_jsonl(predictions_path, prediction_rows)
    _write_jsonl(bridge_rows_path, bridge_rows)
    _write_csv_rows(bridge_rows_csv_path, bridge_rows)

    totals = _aggregate_bridge_row_totals(bridge_rows)
    finished_at = datetime.now(timezone.utc)
    elapsed_seconds = round((finished_at - started_at).total_seconds(), 3)
    correct_count = sum(1 for row in bridge_rows if row["answer_correct"])
    row_type_counts = dict(sorted(Counter(row["row_type"] for row in bridge_rows).items()))
    error_count = sum(1 for row in bridge_rows if row["api_status"] == "error")
    truncated_count = sum(1 for row in bridge_rows if row["prompt_truncated"])
    total_request_attempts = sum(row["api_attempt_count"] for row in bridge_rows)
    valid_choice_count = sum(bool(row["valid_choice"]) for row in bridge_rows)
    invalid_choice_count = len(bridge_rows) - valid_choice_count
    valid_choice_correct_count = sum(
        bool(row["valid_choice"] and row["answer_correct"]) for row in bridge_rows
    )

    manifest = {
        "run_id": run_id,
        "created_at": finished_at.isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "benchmark_target": "longbench_v2_api",
        "baseline_type": "plain_api_uncached_full_context",
        "suite_csv": str(suite_csv),
        "suite_csv_sha256": _sha256_file(suite_csv),
        "source_json_path": str(source_json_path),
        "source_json_sha256": _sha256_file(source_json_path),
        "row_types_requested": row_types,
        "source_ids_requested": source_ids,
        "max_rows": args.max_rows,
        "rows_selected": len(bridge_rows),
        "row_type_counts": row_type_counts,
        "answer_correct_count": correct_count,
        "answer_accuracy": round(correct_count / len(bridge_rows), 6) if bridge_rows else 0.0,
        "by_row_type": summarize_rows(bridge_rows, "row_type"),
        "api_provider": args.api_provider,
        "api_model": args.api_model,
        "max_output_tokens": args.max_output_tokens,
        "max_retries": args.max_retries,
        "api_error_count": error_count,
        "total_request_attempts": total_request_attempts,
        "truncated_row_count": truncated_count,
        "valid_choice_count": valid_choice_count,
        "invalid_choice_count": invalid_choice_count,
        "valid_choice_accuracy": (
            round(valid_choice_correct_count / valid_choice_count, 6)
            if valid_choice_count
            else 0.0
        ),
        "artifacts": {
            "predictions": str(predictions_path),
            "bridge_rows": str(bridge_rows_path),
            "bridge_rows_csv": str(bridge_rows_csv_path),
            "eval_report": str(report_path),
        },
        "total_api_calls": totals["calls"],
        "total_input_tokens": totals["input_tokens"],
        "total_output_tokens": totals["output_tokens"],
        "total_tokens": totals["total_tokens"],
        "total_estimated_cost_usd": round(totals["cost"], 8),
        "cache_reuse": {
            "enabled": False,
            "reason": "Plain API baseline is intentionally uncached",
        },
    }
    if args.api_provider == "openai_compatible":
        manifest.update(
            {
                "api_base_url": args.api_base_url,
                "context_window_tokens": args.context_window_tokens,
                "max_input_tokens": args.max_input_tokens,
                "input_token_budget": args.max_input_tokens,
                "context_window_safety_margin_tokens": (
                    args.context_window_tokens
                    - args.max_input_tokens
                    - args.max_output_tokens
                ),
                "truncation_policy": "longbench_v2_middle_keep_first_last",
                "system_prompt_style": "strict",
                "temperature": 0,
                "thinking_enabled": False,
                **mcq_decoder_constraint_metadata(),
            }
        )
    if args.manifest_note:
        manifest["note"] = args.manifest_note

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = build_eval_report(out_dir, bridge_rows, manifest)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[LONGBENCH-V2-API] Completed.")
    print(f"[LONGBENCH-V2-API] Predictions : {predictions_path}")
    print(f"[LONGBENCH-V2-API] Bridge rows : {bridge_rows_path}")
    print(f"[LONGBENCH-V2-API] Bridge CSV  : {bridge_rows_csv_path}")
    print(f"[LONGBENCH-V2-API] Manifest    : {manifest_path}")
    print(f"[LONGBENCH-V2-API] Eval report : {report_path}")
    print(f"[LONGBENCH-V2-API] Accuracy    : {manifest['answer_accuracy']:.3f}")
    if error_count:
        print(f"[LONGBENCH-V2-API] API errors  : {error_count}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plain full-context API baseline runner for prepared LongBench-v2 CSV suites"
    )
    parser.add_argument("--suite-csv", type=Path, default=DEFAULT_SUITE_CSV)
    parser.add_argument("--source-json-path", type=Path, default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--row-types", type=str, default=DEFAULT_ROW_TYPES)
    parser.add_argument(
        "--source-ids",
        type=str,
        default="",
        help="Comma-separated source IDs to select before applying --max-rows.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Cap selected rows after filtering (0 means all)")
    parser.add_argument(
        "--api-provider",
        choices=["anthropic", "openrouter", "openai_compatible"],
        default=DEFAULT_API_PROVIDER,
    )
    parser.add_argument("--api-model", type=str, default=None)
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=None,
        help="OpenAI-compatible /v1 base URL. Defaults to the executor endpoint environment variable.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=None,
        help="Environment variable used for the API key. Local OpenAI-compatible endpoints may leave this unset.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument(
        "--context-window-tokens",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW_TOKENS,
        help="Served model context limit used for OpenAI-compatible middle truncation.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help="Maximum rendered input tokens before LongBench-v2 middle truncation.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Request attempts per row. Defaults to 5 locally and 1 for hosted providers.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_artifacts"))
    parser.add_argument("--manifest-note", type=str, default="")
    return parser


def main() -> None:
    start = time.time()
    parser = build_arg_parser()
    args = parser.parse_args()
    args = normalize_api_args(args)
    run_longbench_api_benchmark(args)
    print(f"\nTotal elapsed time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
