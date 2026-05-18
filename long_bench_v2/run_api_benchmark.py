"""Plain full-context API baseline runner for prepared LongBench-v2 CSV suites."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
from ruler_v2.run_rlm_benchmark import (  # noqa: E402
    _aggregate_bridge_row_totals,
    _estimate_cost_usd,
    _get_nested_number,
    _json_safe,
    _write_csv_rows,
)


ARTIFACT_SUBDIR = "longbench_v2_api"
REPORT_FILENAME = "official_longbench_v2_api_eval_report.json"
DEFAULT_API_PROVIDER = "anthropic"
DEFAULT_API_MODEL = "claude-sonnet-4-5"
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


def _extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, dict):
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
        "failing_rows": failing_rows,
    }


def _build_default_api_client_factory(args: argparse.Namespace) -> Callable[[], Any]:
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


def run_longbench_api_benchmark(
    args: argparse.Namespace,
    client_factory: Optional[Callable[[], Any]] = None,
) -> None:
    suite_csv = Path(args.suite_csv)
    source_json_path = Path(args.source_json_path)
    row_types = _parse_csv_values(args.row_types)
    if not row_types:
        raise ValueError("--row-types must include at least one row type")

    contexts = load_context_by_source_id(source_json_path)
    all_rows = load_suite_rows(suite_csv, contexts)
    selected_rows = filter_suite_rows(all_rows, row_types=row_types, max_rows=args.max_rows)
    if not selected_rows:
        raise ValueError("No LongBench-v2 rows matched the requested filters")

    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / ARTIFACT_SUBDIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

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

    prediction_rows: list[dict] = []
    bridge_rows: list[dict] = []

    for idx, row in enumerate(selected_rows, start=1):
        print(f"[LONGBENCH-V2-API] Row {idx}/{len(selected_rows)}: {row['case_id']}")
        prompt = build_api_prompt(row)
        generation = ""
        api_status = "ok"
        api_error = ""
        response = None
        t0 = time.time()
        try:
            response = _call_anthropic(client, args, prompt)
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
                "latency_ms": round(latency_ms, 3),
                "delta_calls": usage["calls"],
                "delta_input_tokens": usage["input_tokens"],
                "delta_output_tokens": usage["output_tokens"],
                "delta_cost_usd": usage["cost_usd"],
                "api_provider": args.api_provider,
                "api_model": args.api_model,
                "api_status": api_status,
                "api_error": api_error,
                "api_usage_summary": usage["raw"],
                "usage_parse_status": usage["usage_parse_status"],
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
        "max_rows": args.max_rows,
        "rows_selected": len(bridge_rows),
        "row_type_counts": row_type_counts,
        "answer_correct_count": correct_count,
        "answer_accuracy": round(correct_count / len(bridge_rows), 6) if bridge_rows else 0.0,
        "by_row_type": summarize_rows(bridge_rows, "row_type"),
        "api_provider": args.api_provider,
        "api_model": args.api_model,
        "max_output_tokens": args.max_output_tokens,
        "api_error_count": error_count,
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
    parser.add_argument("--max-rows", type=int, default=0, help="Cap selected rows after filtering (0 means all)")
    parser.add_argument("--api-provider", type=str, default=DEFAULT_API_PROVIDER)
    parser.add_argument("--api-model", type=str, default=DEFAULT_API_MODEL)
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="ANTHROPIC_API_KEY",
        help="Environment variable used for the Anthropic API key when present",
    )
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_artifacts"))
    parser.add_argument("--manifest-note", type=str, default="")
    return parser


def main() -> None:
    start = time.time()
    parser = build_arg_parser()
    args = parser.parse_args()
    run_longbench_api_benchmark(args)
    print(f"\nTotal elapsed time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
