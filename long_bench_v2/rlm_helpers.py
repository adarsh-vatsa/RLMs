"""Shared helpers for LongBench-v2 uncached RLM and API baselines."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Callable


# Mirrors semantic_cache_system.MODEL_FAMILY_PRICING_USD_PER_1K without importing
# the cache stack. The LongBench baseline runners are intentionally cache-free.
MODEL_FAMILY_PRICING_USD_PER_1K = {
    "sonnet": {"input": 0.003, "output": 0.015},
    "haiku": {"input": 0.001, "output": 0.005},
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return repr(value)


def _get_nested_number(value: Any, key_options: set[str]) -> float:
    safe = _json_safe(value)
    if isinstance(safe, dict):
        for key, item in safe.items():
            if key in key_options and isinstance(item, (int, float)):
                return float(item)
        total = 0.0
        for item in safe.values():
            total += _get_nested_number(item, key_options)
        return total
    if isinstance(safe, list):
        return sum(_get_nested_number(item, key_options) for item in safe)
    return 0.0


def _model_family(model: str) -> str:
    model_lower = model.lower()
    if "haiku" in model_lower:
        return "haiku"
    if "sonnet" in model_lower:
        return "sonnet"
    return ""


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    family = _model_family(model)
    if not family:
        return 0.0
    rates = MODEL_FAMILY_PRICING_USD_PER_1K[family]
    return (input_tokens / 1000.0 * rates["input"]) + (
        output_tokens / 1000.0 * rates["output"]
    )


def parse_rlm_usage_summary(usage_summary: Any, model: str) -> dict:
    safe_usage = _json_safe(usage_summary)
    input_tokens = int(
        _get_nested_number(
            safe_usage,
            {"input_tokens", "prompt_tokens", "total_input_tokens"},
        )
    )
    output_tokens = int(
        _get_nested_number(
            safe_usage,
            {"output_tokens", "completion_tokens", "total_output_tokens"},
        )
    )
    total_tokens = int(_get_nested_number(safe_usage, {"total_tokens", "tokens_total"}))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    calls = int(
        _get_nested_number(
            safe_usage,
            {"calls", "api_calls", "request_count", "requests", "num_calls", "total_calls"},
        )
    )

    if usage_summary is None:
        status = "missing_usage_summary"
    elif input_tokens or output_tokens or total_tokens or calls:
        status = "parsed"
    else:
        status = "no_supported_usage_fields"

    return {
        "raw": safe_usage,
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(_estimate_cost_usd(model, input_tokens, output_tokens), 8),
        "usage_parse_status": status,
    }


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def _write_csv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})


def _aggregate_bridge_row_totals(rows: list[dict]) -> dict[str, float]:
    total_calls = sum(int(row.get("delta_calls", 0) or 0) for row in rows)
    total_input_tokens = sum(int(row.get("delta_input_tokens", 0) or 0) for row in rows)
    total_output_tokens = sum(int(row.get("delta_output_tokens", 0) or 0) for row in rows)
    total_cost = sum(float(row.get("delta_cost_usd", 0.0) or 0.0) for row in rows)
    return {
        "calls": total_calls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "cost": total_cost,
    }


def _build_default_rlm_factory(args: argparse.Namespace, out_dir: Path) -> Callable[[], Any]:
    try:
        from rlm import RLM
        from rlm.logger import RLMLogger
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RLM package is not installed. Install the upstream package first, "
            "for example: `pip install rlms` or install the local checkout of "
            "https://github.com/alexzhang13/rlm."
        ) from exc

    args = normalize_rlm_args(args)
    backend_kwargs = {"model_name": args.rlm_model}
    api_key = os.getenv(args.rlm_api_key_env) if args.rlm_api_key_env else ""
    if api_key:
        backend_kwargs["api_key"] = api_key
    base_url = getattr(args, "rlm_base_url", "")
    if base_url:
        backend_kwargs["base_url"] = base_url

    logger = None
    if args.rlm_log_trajectories:
        log_dir = out_dir / "rlm_trajectories"
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = RLMLogger(log_dir=str(log_dir))

    def factory():
        return RLM(
            backend=args.rlm_backend,
            backend_kwargs=dict(backend_kwargs),
            environment=args.rlm_environment,
            max_depth=args.rlm_max_depth,
            max_iterations=args.rlm_max_iterations,
            logger=logger,
            verbose=bool(args.rlm_verbose),
        )

    return factory


def normalize_rlm_args(args: argparse.Namespace) -> argparse.Namespace:
    api_key_env = getattr(args, "rlm_api_key_env", None)
    base_url = (getattr(args, "rlm_base_url", "") or "").lower()
    if api_key_env is None:
        args.rlm_api_key_env = (
            "OPENROUTER_API_KEY" if "openrouter.ai" in base_url else "ANTHROPIC_API_KEY"
        )
    return args
