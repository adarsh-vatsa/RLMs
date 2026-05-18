"""LongBench-v2 benchmark runner for retrieval baseline and cache reuse runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


ARTIFACT_SUBDIR = "longbench_v2"
REPORT_FILENAME = "official_longbench_v2_eval_report.json"
DEFAULT_SUITE_CSV = Path("benchmark_data/long_bench_v2/data_cache_suite.csv")
DEFAULT_SOURCE_JSON = Path("benchmark_data/long_bench_v2/data.json")
DEFAULT_ROW_TYPES = "original,exact,semantic"
VALID_CACHE_TYPES = {"exact", "semantic", "knowledge", "miss", "unknown"}
CHOICE_LETTERS = {"A", "B", "C", "D"}


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    return ""


def _parse_csv_values(raw: str) -> list[str]:
    values = [part.strip() for part in (raw or "").split(",")]
    return [part for part in values if part]


def _csv_cell(value: object) -> object:
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


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sanitize_path_segment(raw: str, max_len: int = 180) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", _coerce_text(raw)).strip("._-")
    return (value or "default")[:max_len]


def load_context_by_source_id(source_json_path: Path) -> dict[str, str]:
    raw = json.loads(source_json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected {source_json_path} to contain a JSON list")

    contexts: dict[str, str] = {}
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Row {idx} in {source_json_path} is not a JSON object")
        source_id = _coerce_text(item.get("_id"))
        context = _coerce_text(item.get("context"))
        if not source_id:
            raise ValueError(f"Row {idx} in {source_json_path} is missing _id")
        if not context:
            raise ValueError(f"Row {idx} in {source_json_path} is missing context")
        if source_id in contexts:
            raise ValueError(f"Duplicate _id in {source_json_path}: {source_id}")
        contexts[source_id] = context
    return contexts


def load_suite_rows(suite_csv: Path, context_by_source_id: dict[str, str]) -> list[dict]:
    with suite_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "case_id",
            "source_id",
            "row_type",
            "question",
            "choice_A",
            "choice_B",
            "choice_C",
            "choice_D",
            "answer",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{suite_csv} is missing required columns: {', '.join(missing)}")

        rows: list[dict] = []
        seen_case_ids: set[str] = set()
        for idx, row in enumerate(reader, start=2):
            normalized = {key: _coerce_text(value) for key, value in row.items()}
            case_id = normalized.get("case_id", "")
            source_id = normalized.get("source_id", "")
            if not case_id:
                raise ValueError(f"{suite_csv}:{idx} has empty case_id")
            if case_id in seen_case_ids:
                raise ValueError(f"Duplicate case_id in {suite_csv}: {case_id}")
            if source_id not in context_by_source_id:
                raise ValueError(f"{suite_csv}:{idx} source_id not found in source JSON: {source_id}")
            seen_case_ids.add(case_id)
            normalized["context"] = context_by_source_id[source_id]
            rows.append(normalized)
    return rows


def filter_suite_rows(rows: list[dict], row_types: Sequence[str], max_rows: int = 0) -> list[dict]:
    row_type_filter = {row_type.strip() for row_type in row_types if row_type.strip()}
    selected = [row for row in rows if not row_type_filter or row.get("row_type") in row_type_filter]
    if max_rows > 0:
        selected = selected[:max_rows]
    return selected


def build_query(row: dict) -> str:
    return (
        f"Question: {row['question']}\n\n"
        "Choices:\n"
        f"A. {row.get('choice_A', '')}\n"
        f"B. {row.get('choice_B', '')}\n"
        f"C. {row.get('choice_C', '')}\n"
        f"D. {row.get('choice_D', '')}\n\n"
        "Return only the single best answer choice letter: A, B, C, or D."
    )


def parse_choice(text: str) -> str:
    raw = _coerce_text(text).upper()
    if not raw:
        return ""
    stripped = raw.strip()
    if stripped in CHOICE_LETTERS:
        return stripped

    patterns = [
        r"(?:FINAL\s+ANSWER|ANSWER|CHOICE|OPTION)\s*(?:IS|:)?\s*[\(\[]?\s*([A-D])\s*[\)\]]?",
        r"^\s*[\(\[]?\s*([A-D])\s*[\)\].:-]",
        r"\b([A-D])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, stripped)
        if match:
            return match.group(1)
    return ""


def answer_correct(prediction: str, expected_answer: str) -> bool:
    return parse_choice(prediction) == parse_choice(expected_answer)


def _build_dataset_signature(rows: list[dict]) -> str:
    hasher = hashlib.sha256()
    ordered = sorted(rows, key=lambda row: (_coerce_text(row.get("row_type")), _coerce_text(row.get("case_id"))))
    for row in ordered:
        payload = {
            "case_id": _coerce_text(row.get("case_id")),
            "source_id": _coerce_text(row.get("source_id")),
            "row_type": _coerce_text(row.get("row_type")),
            "question": _coerce_text(row.get("question")),
            "answer": _coerce_text(row.get("answer")),
            "choice_A": _coerce_text(row.get("choice_A")),
            "choice_B": _coerce_text(row.get("choice_B")),
            "choice_C": _coerce_text(row.get("choice_C")),
            "choice_D": _coerce_text(row.get("choice_D")),
        }
        hasher.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()[:16]


def resolve_cache_namespace(
    suite_csv_sha256: str,
    source_json_sha256: str,
    selected_rows: list[dict],
    executor_model: str,
    top_k: int,
    rerank_top: int,
    row_types: Sequence[str],
) -> tuple[str, str]:
    dataset_signature = _build_dataset_signature(selected_rows)
    row_type_sig = "-".join(sorted({row_type.lower() for row_type in row_types if row_type}))
    digest = hashlib.sha256(
        (
            f"{suite_csv_sha256}\n{source_json_sha256}\n{dataset_signature}\n"
            f"{executor_model}\n{top_k}\n{rerank_top}\n{row_type_sig}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    return _sanitize_path_segment(f"longbench_v2__{row_type_sig}__{digest}"), dataset_signature


def summarize_rows(rows: list[dict], key: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        label = _coerce_text(row.get(key)) or "unlabeled"
        grouped.setdefault(label, []).append(row)

    summary: dict[str, dict] = {}
    for label, bucket in sorted(grouped.items()):
        total = len(bucket)
        correct = sum(1 for row in bucket if row.get("answer_correct"))
        cached = sum(1 for row in bucket if row.get("from_cache"))
        summary[label] = {
            "rows": total,
            "answer_accuracy": round(correct / total, 6) if total else 0.0,
            "cache_hit_rate": round(cached / total, 6) if total else 0.0,
        }
    return summary


def aggregate_bridge_row_totals(rows: list[dict]) -> dict[str, float]:
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


def build_eval_report(run_dir: Path, bridge_rows: list[dict], manifest: dict) -> dict:
    failing_rows = [
        {
            "case_id": row["case_id"],
            "source_id": row["source_id"],
            "row_type": row["row_type"],
            "expected_answer": row["expected_answer"],
            "prediction": row["prediction"],
            "generation": row["generation"],
            "actual_cache_type": row["cache_type"],
        }
        for row in bridge_rows
        if not row.get("answer_correct")
    ]
    return {
        "run_dir": str(run_dir),
        "benchmark_target": "longbench_v2",
        "scored_rows": len(bridge_rows),
        "answer_accuracy": manifest["answer_accuracy"],
        "row_type_counts": manifest["row_type_counts"],
        "actual_route_counts": manifest["actual_route_counts"],
        "by_row_type": manifest["by_row_type"],
        "failing_rows": failing_rows,
    }


def build_cache_reuse_manifest(
    *,
    enabled: bool,
    cache_namespace: str = "",
    dataset_signature: str = "",
    cache_state_root: Path | None = None,
    cache_state_path: Path | None = None,
    cache_state_existed_before_reset: bool = False,
    cache_state_existed_before_run: bool = False,
    cache_reset_requested: bool = False,
    cache_load_attempts: int = 0,
    cache_load_successes: int = 0,
    cache_entries_before_run: int = 0,
    cache_entries_after_run: int = 0,
    cache_hits: int = 0,
    row_count: int = 0,
) -> dict:
    if not enabled:
        return {
            "enabled": False,
            "reason": "mode is baseline",
        }
    return {
        "enabled": True,
        "cache_namespace": cache_namespace,
        "dataset_signature": dataset_signature,
        "cache_state_root": str(cache_state_root) if cache_state_root else "",
        "cache_state_path": str(cache_state_path) if cache_state_path else "",
        "cache_state_existed_before_reset": cache_state_existed_before_reset,
        "cache_state_existed_before_run": cache_state_existed_before_run,
        "cache_reset_requested": bool(cache_reset_requested),
        "run_start_type": "warm_start" if cache_state_existed_before_run else "cold_start",
        "cache_load_attempts": cache_load_attempts,
        "cache_load_successes": cache_load_successes,
        "cache_entries_before_run": cache_entries_before_run,
        "cache_entries_after_run": cache_entries_after_run,
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / row_count, 6) if row_count else 0.0,
    }


def _snapshot_metrics(metrics) -> dict[str, float]:
    return metrics.get_totals()


def _normalize_cache_type(output: dict) -> str:
    if output.get("from_cache"):
        cache_type = _coerce_text(output.get("cache_type"))
        return cache_type if cache_type in VALID_CACHE_TYPES else "unknown"
    return "miss"


def _write_context_doc(out_dir: Path, row: dict) -> Path:
    safe_case_id = _sanitize_path_segment(row["case_id"])
    docs_dir = out_dir / "sample_docs" / safe_case_id
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "context.txt").write_text(row["context"].rstrip() + "\n", encoding="utf-8")
    return docs_dir


def _import_semantic_cache_system():
    import semantic_cache_system as scs  # Local import keeps unit tests lightweight.

    return scs


def run_longbench_benchmark(args: argparse.Namespace) -> None:
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

    cache_state_enabled = args.mode == "cache"
    cache_namespace = ""
    dataset_signature = ""
    cache_state_root: Path | None = None
    cache_state_path: Path | None = None
    cache_state_existed_before_reset = False
    cache_state_existed_before_run = False
    cache_entries_before_run = 0
    cache_entries_after_run = 0
    cache_load_attempts = 0
    cache_load_successes = 0
    cache_hits = 0

    suite_csv_sha256 = _sha256_file(suite_csv)
    source_json_sha256 = _sha256_file(source_json_path)

    if cache_state_enabled:
        cache_namespace, dataset_signature = resolve_cache_namespace(
            suite_csv_sha256=suite_csv_sha256,
            source_json_sha256=source_json_sha256,
            selected_rows=selected_rows,
            executor_model=args.executor_model,
            top_k=args.top_k,
            rerank_top=args.rerank_top,
            row_types=row_types,
        )
        cache_state_root = Path(args.cache_state_root) if args.cache_state_root else Path(args.output_dir) / ARTIFACT_SUBDIR / "cache_state"
        cache_state_path = cache_state_root / cache_namespace
        cache_state_existed_before_reset = cache_state_path.exists()
        if args.cache_reset and cache_state_existed_before_reset:
            shutil.rmtree(cache_state_path)
            print(f"[LONGBENCH-V2] Cache reset requested. Removed state at: {cache_state_path}")
        cache_state_root.mkdir(parents=True, exist_ok=True)
        cache_state_existed_before_run = cache_state_path.exists()

    print(f"\n[LONGBENCH-V2] Run id: {run_id}")
    print(f"[LONGBENCH-V2] Rows: {len(selected_rows)}")
    print(f"[LONGBENCH-V2] Row types: {row_types}")
    print(f"[LONGBENCH-V2] Mode: {args.mode}")
    print(f"[LONGBENCH-V2] Output dir: {out_dir}")
    if cache_state_enabled:
        print(f"[LONGBENCH-V2] Cache state root: {cache_state_root}")
        print(f"[LONGBENCH-V2] Cache namespace : {cache_namespace}")
        print(f"[LONGBENCH-V2] Cache state: {'warm start' if cache_state_existed_before_run else 'cold start'}")

    scs = _import_semantic_cache_system()
    scs.EXECUTOR_MODEL = args.executor_model
    shared_embedder = scs.EmbeddingEngine()
    shared_reranker = None if args.disable_reranker else scs.Reranker()

    prediction_rows: list[dict] = []
    bridge_rows: list[dict] = []

    for idx, row in enumerate(selected_rows, start=1):
        print(f"[LONGBENCH-V2] Row {idx}/{len(selected_rows)}: {row['case_id']}")
        docs_dir = _write_context_doc(out_dir, row)
        controller_corpus_id = cache_namespace if cache_state_enabled else f"longbench_v2_{idx}"
        controller = scs.SemanticCacheController(
            metrics=scs.ExecutionMetrics(),
            embedder=shared_embedder,
            reranker=shared_reranker,
            corpus_id=controller_corpus_id,
            corpus_domain="longbench_v2",
        )

        if cache_state_enabled and cache_state_path is not None:
            cache_load_attempts += 1
            if cache_state_path.exists() and controller.load(cache_state_path):
                cache_load_successes += 1
            loaded_entries = controller.get_total_entries()
            if idx == 1:
                cache_entries_before_run = loaded_entries

        controller.ingest(docs_dir)
        query = build_query(row)
        before = _snapshot_metrics(controller.metrics)
        t0 = time.time()
        output = controller.search(
            query,
            top_k=args.top_k,
            rerank_top=args.rerank_top,
            synthesize=True,
            cache_read=cache_state_enabled,
        )
        latency_ms = (time.time() - t0) * 1000.0
        after = _snapshot_metrics(controller.metrics)

        generation = _coerce_text(output.get("answer"))
        prediction = parse_choice(generation)
        correct = answer_correct(generation, row.get("answer", ""))
        actual_cache_type = _normalize_cache_type(output)
        actual_from_cache = bool(output.get("from_cache"))
        if actual_from_cache:
            cache_hits += 1

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

        retrieval = output.get("retrieval") or {}
        bridge_row = {
            **prediction_row,
            "mode": args.mode,
            "token_count": row.get("token_count", ""),
            "expected_cache_type": row.get("expected_cache_type", ""),
            "expected_from_cache": row.get("expected_from_cache", ""),
            "from_cache": actual_from_cache,
            "cache_type": actual_cache_type,
            "answer_correct": correct,
            "latency_ms": round(latency_ms, 3),
            "delta_calls": after["calls"] - before["calls"],
            "delta_input_tokens": after["input_tokens"] - before["input_tokens"],
            "delta_output_tokens": after["output_tokens"] - before["output_tokens"],
            "delta_cost_usd": round(after["cost"] - before["cost"], 8),
            "faiss_candidate_count": retrieval.get("faiss_candidate_count"),
            "candidate_text_count": retrieval.get("candidate_text_count"),
            "reranker_enabled": retrieval.get("reranker_enabled"),
            "reranker_returned_count": retrieval.get("reranker_returned_count"),
            "reranker_fallback_used": retrieval.get("reranker_fallback_used"),
        }
        bridge_rows.append(bridge_row)

        if cache_state_enabled and cache_state_path is not None:
            controller.save(cache_state_path)
            cache_entries_after_run = controller.get_total_entries()

    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with bridge_rows_path.open("w", encoding="utf-8") as handle:
        for row in bridge_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_csv_rows(bridge_rows_csv_path, bridge_rows)

    totals = aggregate_bridge_row_totals(bridge_rows)
    finished_at = datetime.now(timezone.utc)
    elapsed_seconds = round((finished_at - started_at).total_seconds(), 3)
    correct_count = sum(1 for row in bridge_rows if row["answer_correct"])
    row_type_counts = dict(sorted(Counter(row["row_type"] for row in bridge_rows).items()))
    actual_route_counts = {
        route: sum(1 for row in bridge_rows if row["cache_type"] == route)
        for route in sorted(VALID_CACHE_TYPES)
    }

    manifest = {
        "run_id": run_id,
        "created_at": finished_at.isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "benchmark_target": "longbench_v2",
        "suite_csv": str(suite_csv),
        "suite_csv_sha256": suite_csv_sha256,
        "source_json_path": str(source_json_path),
        "source_json_sha256": source_json_sha256,
        "mode": args.mode,
        "executor_model": args.executor_model,
        "top_k": args.top_k,
        "rerank_top": args.rerank_top,
        "reranker_disabled": bool(args.disable_reranker),
        "row_types_requested": row_types,
        "max_rows": args.max_rows,
        "rows_selected": len(bridge_rows),
        "row_type_counts": row_type_counts,
        "answer_correct_count": correct_count,
        "answer_accuracy": round(correct_count / len(bridge_rows), 6) if bridge_rows else 0.0,
        "actual_route_counts": actual_route_counts,
        "by_row_type": summarize_rows(bridge_rows, "row_type"),
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
    }
    manifest["cache_reuse"] = build_cache_reuse_manifest(
        enabled=cache_state_enabled,
        cache_namespace=cache_namespace,
        dataset_signature=dataset_signature,
        cache_state_root=cache_state_root,
        cache_state_path=cache_state_path,
        cache_state_existed_before_reset=cache_state_existed_before_reset,
        cache_state_existed_before_run=cache_state_existed_before_run,
        cache_reset_requested=bool(args.cache_reset),
        cache_load_attempts=cache_load_attempts,
        cache_load_successes=cache_load_successes,
        cache_entries_before_run=cache_entries_before_run,
        cache_entries_after_run=cache_entries_after_run,
        cache_hits=cache_hits,
        row_count=len(bridge_rows),
    )
    if args.manifest_note:
        manifest["note"] = args.manifest_note

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = build_eval_report(out_dir, bridge_rows, manifest)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[LONGBENCH-V2] Completed.")
    print(f"[LONGBENCH-V2] Predictions : {predictions_path}")
    print(f"[LONGBENCH-V2] Bridge rows : {bridge_rows_path}")
    print(f"[LONGBENCH-V2] Bridge CSV  : {bridge_rows_csv_path}")
    print(f"[LONGBENCH-V2] Manifest    : {manifest_path}")
    print(f"[LONGBENCH-V2] Eval report : {report_path}")
    print(f"[LONGBENCH-V2] Accuracy    : {manifest['answer_accuracy']:.3f}")
    if cache_state_enabled:
        print(
            "[LONGBENCH-V2] Cache reuse: "
            f"loads={cache_load_successes}/{cache_load_attempts}, "
            f"hits={cache_hits}/{len(bridge_rows)}, "
            f"entries_before={cache_entries_before_run}, "
            f"entries_after={cache_entries_after_run}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LongBench-v2 retrieval/cache benchmark runner")
    parser.add_argument("--suite-csv", type=Path, default=DEFAULT_SUITE_CSV)
    parser.add_argument("--source-json-path", type=Path, default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--mode", choices=["baseline", "cache"], default="cache")
    parser.add_argument("--cache-reset", action="store_true")
    parser.add_argument("--cache-state-root", type=Path, default=None)
    parser.add_argument("--row-types", type=str, default=DEFAULT_ROW_TYPES)
    parser.add_argument("--max-rows", type=int, default=0, help="Cap selected rows after filtering (0 means all)")
    parser.add_argument("--executor-model", type=str, default="claude-sonnet-4-5")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--rerank-top", type=int, default=5)
    parser.add_argument("--disable-reranker", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_artifacts"))
    parser.add_argument("--manifest-note", type=str, default="")
    return parser


def main() -> None:
    start = time.time()
    parser = build_arg_parser()
    args = parser.parse_args()
    run_longbench_benchmark(args)
    print(f"\nTotal elapsed time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
