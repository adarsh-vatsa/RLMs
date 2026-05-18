"""Uncached RLM baseline runner for prepared LongBench-v2 CSV suites."""

from __future__ import annotations

import argparse
import json
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
    _build_default_rlm_factory,
    _write_csv_rows,
    normalize_rlm_args,
    parse_rlm_usage_summary,
)


ARTIFACT_SUBDIR = "longbench_v2_rlm"
REPORT_FILENAME = "official_longbench_v2_rlm_eval_report.json"
DEFAULT_RLM_BACKEND = "anthropic"
DEFAULT_RLM_MODEL = "claude-sonnet-4-5"
RLM_ROOT_PROMPT = (
    "Answer the multiple-choice question using the provided context. "
    "Return only the final answer choice letter: A, B, C, or D."
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_eval_report(run_dir: Path, bridge_rows: list[dict], manifest: dict) -> dict:
    failing_rows = [
        {
            "case_id": row["case_id"],
            "source_id": row["source_id"],
            "row_type": row["row_type"],
            "expected_answer": row["expected_answer"],
            "prediction": row["prediction"],
            "generation": row["generation"],
        }
        for row in bridge_rows
        if not row.get("answer_correct")
    ]
    return {
        "run_dir": str(run_dir),
        "benchmark_target": "longbench_v2_rlm",
        "baseline_type": "rlm_uncached",
        "scored_rows": len(bridge_rows),
        "answer_accuracy": manifest["answer_accuracy"],
        "row_type_counts": manifest["row_type_counts"],
        "by_row_type": manifest["by_row_type"],
        "failing_rows": failing_rows,
    }


def _coerce_generation(result: Any) -> str:
    response = getattr(result, "response", result)
    if isinstance(response, dict):
        for key in ("response", "output", "answer", "text", "content"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(response, str):
        return response.strip()
    return _coerce_text(response)


def run_longbench_rlm_benchmark(
    args: argparse.Namespace,
    rlm_factory: Optional[Callable[[], Any]] = None,
) -> None:
    args = normalize_rlm_args(args)
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

    print(f"\n[LONGBENCH-V2-RLM] Run id: {run_id}")
    print(f"[LONGBENCH-V2-RLM] Rows: {len(selected_rows)}")
    print(f"[LONGBENCH-V2-RLM] Row types: {row_types}")
    print(f"[LONGBENCH-V2-RLM] Backend/model: {args.rlm_backend}/{args.rlm_model}")
    print(f"[LONGBENCH-V2-RLM] Output dir: {out_dir}")

    if rlm_factory is None:
        rlm_factory = _build_default_rlm_factory(args, out_dir)

    prediction_rows: list[dict] = []
    bridge_rows: list[dict] = []

    for idx, row in enumerate(selected_rows, start=1):
        print(f"[LONGBENCH-V2-RLM] Row {idx}/{len(selected_rows)}: {row['case_id']}")
        query = build_query(row)
        prompt = {"context": row["context"], "question": query}
        rlm_client = rlm_factory()

        t0 = time.time()
        result = rlm_client.completion(prompt=prompt, root_prompt=RLM_ROOT_PROMPT)
        latency_ms = (time.time() - t0) * 1000.0

        generation = _coerce_generation(result)
        prediction = parse_choice(generation)
        correct = answer_correct(generation, row.get("answer", ""))
        root_model = _coerce_text(getattr(result, "root_model", "")) or args.rlm_model
        execution_time = getattr(result, "execution_time", None)
        usage = parse_rlm_usage_summary(
            getattr(result, "usage_summary", None),
            model=root_model or args.rlm_model,
        )

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
                "rlm_backend": args.rlm_backend,
                "rlm_model": args.rlm_model,
                "rlm_root_model": root_model,
                "rlm_execution_time": execution_time,
                "rlm_usage_summary": usage["raw"],
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

    manifest = {
        "run_id": run_id,
        "created_at": finished_at.isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "benchmark_target": "longbench_v2_rlm",
        "baseline_type": "rlm_uncached",
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
        "rlm_backend": args.rlm_backend,
        "rlm_model": args.rlm_model,
        "rlm_environment": args.rlm_environment,
        "rlm_max_iterations": args.rlm_max_iterations,
        "rlm_max_depth": args.rlm_max_depth,
        "rlm_base_url": getattr(args, "rlm_base_url", ""),
        "rlm_log_trajectories": bool(args.rlm_log_trajectories),
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
            "reason": "RLM baseline is intentionally uncached",
        },
    }
    if args.rlm_log_trajectories:
        manifest["artifacts"]["rlm_trajectories"] = str(out_dir / "rlm_trajectories")
    if args.manifest_note:
        manifest["note"] = args.manifest_note

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = build_eval_report(out_dir, bridge_rows, manifest)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[LONGBENCH-V2-RLM] Completed.")
    print(f"[LONGBENCH-V2-RLM] Predictions : {predictions_path}")
    print(f"[LONGBENCH-V2-RLM] Bridge rows : {bridge_rows_path}")
    print(f"[LONGBENCH-V2-RLM] Bridge CSV  : {bridge_rows_csv_path}")
    print(f"[LONGBENCH-V2-RLM] Manifest    : {manifest_path}")
    print(f"[LONGBENCH-V2-RLM] Eval report : {report_path}")
    print(f"[LONGBENCH-V2-RLM] Accuracy    : {manifest['answer_accuracy']:.3f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Uncached RLM baseline runner for prepared LongBench-v2 CSV suites"
    )
    parser.add_argument("--suite-csv", type=Path, default=DEFAULT_SUITE_CSV)
    parser.add_argument("--source-json-path", type=Path, default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--row-types", type=str, default=DEFAULT_ROW_TYPES)
    parser.add_argument("--max-rows", type=int, default=0, help="Cap selected rows after filtering (0 means all)")
    parser.add_argument("--rlm-backend", type=str, default=DEFAULT_RLM_BACKEND)
    parser.add_argument("--rlm-model", type=str, default=DEFAULT_RLM_MODEL)
    parser.add_argument("--rlm-environment", type=str, default="local")
    parser.add_argument("--rlm-max-iterations", type=int, default=30)
    parser.add_argument("--rlm-max-depth", type=int, default=1)
    parser.add_argument(
        "--rlm-base-url",
        type=str,
        default="",
        help="Optional OpenAI-compatible backend base URL, e.g. https://openrouter.ai/api/v1",
    )
    parser.add_argument(
        "--rlm-api-key-env",
        type=str,
        default=None,
        help=(
            "Environment variable used as backend_kwargs['api_key'] when present. "
            "Defaults to ANTHROPIC_API_KEY, or OPENROUTER_API_KEY when --rlm-base-url uses openrouter.ai."
        ),
    )
    parser.add_argument("--rlm-verbose", action="store_true")
    parser.add_argument("--rlm-log-trajectories", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_artifacts"))
    parser.add_argument("--manifest-note", type=str, default="")
    return parser


def main() -> None:
    start = time.time()
    parser = build_arg_parser()
    args = normalize_rlm_args(parser.parse_args())
    run_longbench_rlm_benchmark(args)
    print(f"\nTotal elapsed time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
