"""Legal real-dataset cache-route benchmark runner.

This runner keeps LegalBench/CUAD artifacts separate from the small synthetic
cache_bench suite while preserving the same prediction/bridge/manifest shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


from cache_bench.run_benchmark import (  # noqa: E402
    VALID_CACHE_TYPES,
    FixtureCase,
    FixtureCorpus,
    _aggregate_eval_totals,
    _answer_matches,
    _build_eval_report,
    _coerce_text,
    _filter_cases,
    _import_semantic_cache_system,
    _load_fixture_suite,
    _normalize_cache_type,
    _parse_csv_values,
    _resolve_cache_namespace,
    _sanitize_path_segment,
    _sha256_file,
    _snapshot_metrics,
    _summarize_rows,
    _write_corpus_docs,
    _write_csv_rows,
)


ARTIFACT_SUBDIR = "legal_cache_suite"
REPORT_FILENAME = "official_legal_cache_eval_report.json"


def _load_case_metadata(suite_path: Path) -> dict[str, dict]:
    raw = json.loads(Path(suite_path).read_text(encoding="utf-8"))
    metadata: dict[str, dict] = {}
    for case in raw.get("cases", []):
        if not isinstance(case, dict):
            continue
        case_id = _coerce_text(case.get("case_id"))
        case_metadata = case.get("metadata")
        if case_id and isinstance(case_metadata, dict):
            metadata[case_id] = case_metadata
    return metadata


def _resolve_case_cache_namespace(base_namespace: str, case: FixtureCase, isolate_cases: bool) -> str:
    if not isolate_cases:
        return base_namespace
    digest = hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()[:12]
    return _sanitize_path_segment(f"{base_namespace}__case_{digest}")


def _build_legal_eval_report(
    run_dir: Path,
    suite_name: str,
    suite_path: Path,
    bridge_rows: list[dict],
    manifest: dict,
) -> dict:
    report = _build_eval_report(run_dir, suite_name, suite_path, bridge_rows, manifest)
    report["benchmark_target"] = "legal_cache_suite"
    report["case_cache_isolation"] = manifest.get("case_cache_isolation", {})
    report["expected_miss_rate"] = manifest.get("expected_miss_rate", 0.0)
    report["actual_miss_rate"] = manifest.get("actual_miss_rate", 0.0)
    report["unexpected_miss_rate"] = manifest.get("unexpected_miss_rate", 0.0)
    report["positive_case_miss_rate"] = manifest.get("positive_case_miss_rate", 0.0)
    return report


def _manifest_route_counts(rows: Sequence[dict], key: str) -> dict[str, int]:
    return {route: sum(1 for row in rows if row[key] == route) for route in VALID_CACHE_TYPES}


def run_legal_benchmark(args: argparse.Namespace) -> None:
    suite_path = Path(args.suite_path)
    suite = _load_fixture_suite(suite_path)
    case_metadata = _load_case_metadata(suite_path)
    suite_sha256 = _sha256_file(suite_path)
    selected_cases = _filter_cases(
        suite.cases,
        case_ids=_parse_csv_values(args.case_ids),
        max_cases=args.max_cases,
    )

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
    base_cache_namespace = ""
    cache_state_root: Path | None = None
    cache_state_existed_before_reset = False
    cache_load_attempts = 0
    cache_load_successes = 0
    cache_hits = 0
    cache_entries_before_run = 0
    cache_entries_after_run = 0
    cache_namespaces_used: list[str] = []

    if cache_state_enabled:
        base_cache_namespace = _resolve_cache_namespace(
            suite_name=suite.suite_name,
            suite_sha256=suite_sha256,
            selected_cases=selected_cases,
            executor_model=args.executor_model,
            top_k=args.top_k,
            rerank_top=args.rerank_top,
        )
        cache_state_root = (
            Path(args.cache_state_root)
            if args.cache_state_root
            else Path(args.output_dir) / ARTIFACT_SUBDIR / "cache_state"
        )
        cache_state_root.mkdir(parents=True, exist_ok=True)

        cache_state_existed_before_reset = any(
            (cache_state_root / _resolve_case_cache_namespace(
                base_cache_namespace, case, args.case_cache_isolation
            )).exists()
            for case in selected_cases
        )
        if args.cache_reset:
            for case in selected_cases:
                namespace = _resolve_case_cache_namespace(
                    base_cache_namespace, case, args.case_cache_isolation
                )
                case_state_path = cache_state_root / namespace
                if case_state_path.exists():
                    shutil.rmtree(case_state_path)

    print(f"\n[LEGAL-BENCH] Run id: {run_id}")
    print(f"[LEGAL-BENCH] Suite: {suite.suite_name}")
    print(f"[LEGAL-BENCH] Cases: {len(selected_cases)}")
    print(f"[LEGAL-BENCH] Mode: {args.mode}")
    print(f"[LEGAL-BENCH] Case cache isolation: {args.case_cache_isolation}")
    print(f"[LEGAL-BENCH] Output dir: {out_dir}")
    if cache_state_enabled and cache_state_root is not None:
        print(f"[LEGAL-BENCH] Cache state root: {cache_state_root}")
        print(f"[LEGAL-BENCH] Base namespace  : {base_cache_namespace}")

    scs = _import_semantic_cache_system()
    scs.EXECUTOR_MODEL = args.executor_model
    shared_embedder = scs.EmbeddingEngine()
    shared_reranker = None if args.disable_reranker else scs.Reranker()

    corpus_dirs: Dict[str, Path] = {}
    for case in selected_cases:
        if case.corpus_id not in corpus_dirs:
            corpus_dirs[case.corpus_id] = _write_corpus_docs(out_dir, suite.corpora[case.corpus_id])

    predictions_rows: list[dict] = []
    bridge_rows: list[dict] = []
    total_setup_calls = 0
    total_setup_input_tokens = 0
    total_setup_output_tokens = 0
    total_setup_cost = 0.0

    for idx, case in enumerate(selected_cases, start=1):
        case_namespace = (
            _resolve_case_cache_namespace(base_cache_namespace, case, args.case_cache_isolation)
            if cache_state_enabled
            else f"{suite.suite_name}_{case.case_id}"
        )
        if cache_state_enabled and case_namespace not in cache_namespaces_used:
            cache_namespaces_used.append(case_namespace)
        cache_state_path = cache_state_root / case_namespace if cache_state_enabled and cache_state_root else None

        controller = scs.SemanticCacheController(
            metrics=scs.ExecutionMetrics(),
            embedder=shared_embedder,
            reranker=shared_reranker,
            corpus_id=case_namespace,
            corpus_domain="legal_bench",
        )

        cache_state_existed_before_run = bool(cache_state_path and cache_state_path.exists())
        if cache_state_enabled and cache_state_path is not None:
            cache_load_attempts += 1
            if cache_state_path.exists() and controller.load(cache_state_path):
                cache_load_successes += 1
            cache_entries_before_run += controller.get_total_entries()

        controller.ingest(corpus_dirs[case.corpus_id])

        for seed_query in case.seed_queries:
            controller.search(
                seed_query,
                top_k=args.top_k,
                rerank_top=args.rerank_top,
                synthesize=True,
                cache_read=cache_state_enabled,
            )

        setup_totals = _snapshot_metrics(controller.metrics)
        total_setup_calls += int(setup_totals["calls"])
        total_setup_input_tokens += int(setup_totals["input_tokens"])
        total_setup_output_tokens += int(setup_totals["output_tokens"])
        total_setup_cost += float(setup_totals["cost"])

        before_eval = dict(setup_totals)
        t0 = time.time()
        output = controller.search(
            case.eval_query,
            top_k=args.top_k,
            rerank_top=args.rerank_top,
            synthesize=True,
            cache_read=cache_state_enabled,
        )
        latency_ms = (time.time() - t0) * 1000.0
        after_eval = _snapshot_metrics(controller.metrics)
        retrieval = output.get("retrieval") or {}

        prediction = _coerce_text(output.get("answer"))
        actual_cache_type = _normalize_cache_type(output)
        actual_from_cache = bool(output.get("from_cache"))
        if actual_from_cache:
            cache_hits += 1

        answer_correct = _answer_matches(
            prediction,
            case.expected_answers,
            case.expected_answer_mode,
        )
        cache_type_match = actual_cache_type == case.expected_cache_type
        from_cache_match = actual_from_cache == case.expected_from_cache
        overall_pass = answer_correct and cache_type_match and from_cache_match

        metadata = case_metadata.get(case.case_id, {})
        prediction_row = {
            "id": case.case_id,
            "case_id": case.case_id,
            "title": case.title,
            "scenario": case.scenario,
            "corpus_id": case.corpus_id,
            "mode": args.mode,
            "query": case.eval_query,
            "generation": prediction,
            "prediction": prediction,
            "answer": prediction,
            "expected_answers": case.expected_answers,
            "expected_answer_mode": case.expected_answer_mode,
            "expected_cache_type": case.expected_cache_type,
            "actual_cache_type": actual_cache_type,
            "expected_from_cache": case.expected_from_cache,
            "actual_from_cache": actual_from_cache,
            "answer_correct": answer_correct,
            "cache_type_match": cache_type_match,
            "from_cache_match": from_cache_match,
            "overall_pass": overall_pass,
            "metadata": metadata,
            "retrieval": retrieval,
        }
        predictions_rows.append(prediction_row)

        delta_calls = int(after_eval["calls"] - before_eval["calls"])
        delta_input_tokens = int(after_eval["input_tokens"] - before_eval["input_tokens"])
        delta_output_tokens = int(after_eval["output_tokens"] - before_eval["output_tokens"])
        delta_cost = float(after_eval["cost"] - before_eval["cost"])

        bridge_row = {
            "case_id": case.case_id,
            "title": case.title,
            "scenario": case.scenario,
            "corpus_id": case.corpus_id,
            "mode": args.mode,
            "case_index": idx,
            "case_cache_namespace": case_namespace,
            "case_cache_state_existed_before_run": cache_state_existed_before_run,
            "seed_query_count": len(case.seed_queries),
            "query": case.eval_query,
            "expected_answers": case.expected_answers,
            "expected_answer_mode": case.expected_answer_mode,
            "expected_cache_type": case.expected_cache_type,
            "actual_cache_type": actual_cache_type,
            "expected_from_cache": case.expected_from_cache,
            "actual_from_cache": actual_from_cache,
            "answer_correct": answer_correct,
            "cache_type_match": cache_type_match,
            "from_cache_match": from_cache_match,
            "overall_pass": overall_pass,
            "prediction": prediction,
            "latency_ms": round(latency_ms, 3),
            "setup_calls": int(setup_totals["calls"]),
            "setup_input_tokens": int(setup_totals["input_tokens"]),
            "setup_output_tokens": int(setup_totals["output_tokens"]),
            "setup_cost_usd": round(float(setup_totals["cost"]), 8),
            "delta_calls": delta_calls,
            "delta_input_tokens": delta_input_tokens,
            "delta_output_tokens": delta_output_tokens,
            "delta_cost_usd": round(delta_cost, 8),
            "knowledge_top_score": output.get("knowledge_top_score"),
            "knowledge_margin": output.get("knowledge_margin"),
            "knowledge_verifier_called": output.get("knowledge_verifier_called"),
            "knowledge_verifier_allow": output.get("knowledge_verifier_allow"),
            "knowledge_verifier_reason": output.get("knowledge_verifier_reason"),
            "knowledge_verifier_confidence": output.get("knowledge_verifier_confidence"),
            "knowledge_verifier_trigger_reasons": output.get("knowledge_verifier_trigger_reasons"),
            "knowledge_lexical_support": output.get("knowledge_lexical_support"),
            "faiss_candidate_count": retrieval.get("faiss_candidate_count"),
            "candidate_text_count": retrieval.get("candidate_text_count"),
            "reranker_enabled": retrieval.get("reranker_enabled"),
            "reranker_returned_count": retrieval.get("reranker_returned_count"),
            "reranker_fallback_used": retrieval.get("reranker_fallback_used"),
        }
        for key, value in metadata.items():
            bridge_row[f"legal_{key}"] = value
        bridge_rows.append(bridge_row)

        if cache_state_enabled and cache_state_path is not None:
            controller.save(cache_state_path)
            cache_entries_after_run += controller.get_total_entries()

    with predictions_path.open("w", encoding="utf-8") as predictions_file:
        for row in predictions_rows:
            predictions_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with bridge_rows_path.open("w", encoding="utf-8") as bridge_file:
        for row in bridge_rows:
            bridge_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_csv_rows(bridge_rows_csv_path, bridge_rows)

    eval_totals = _aggregate_eval_totals(bridge_rows)
    total_calls = total_setup_calls + int(eval_totals["calls"])
    total_input_tokens = total_setup_input_tokens + int(eval_totals["input_tokens"])
    total_output_tokens = total_setup_output_tokens + int(eval_totals["output_tokens"])
    total_cost = total_setup_cost + float(eval_totals["cost"])
    answer_correct_count = sum(1 for row in bridge_rows if row["answer_correct"])
    route_match_count = sum(1 for row in bridge_rows if row["cache_type_match"])
    overall_pass_count = sum(1 for row in bridge_rows if row["overall_pass"])
    expected_miss_count = sum(1 for row in bridge_rows if row["expected_cache_type"] == "miss")
    actual_miss_count = sum(1 for row in bridge_rows if row["actual_cache_type"] == "miss")
    unexpected_miss_count = sum(
        1
        for row in bridge_rows
        if row["expected_cache_type"] != "miss" and row["actual_cache_type"] == "miss"
    )
    expected_positive_count = len(bridge_rows) - expected_miss_count
    reranker_enabled_count = sum(1 for row in bridge_rows if row.get("reranker_enabled"))
    reranker_fallback_count = sum(1 for row in bridge_rows if row.get("reranker_fallback_used"))

    finished_at = datetime.now(timezone.utc)
    elapsed_seconds = round((finished_at - started_at).total_seconds(), 3)
    manifest = {
        "run_id": run_id,
        "created_at": finished_at.isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "benchmark_target": "legal_cache_suite",
        "suite_name": suite.suite_name,
        "suite_path": str(suite_path),
        "suite_sha256": suite_sha256,
        "cases_selected": len(selected_cases),
        "mode": args.mode,
        "executor_model": args.executor_model,
        "top_k": args.top_k,
        "rerank_top": args.rerank_top,
        "disable_reranker": bool(args.disable_reranker),
        "retrieval": {
            "reranker_enabled": not bool(args.disable_reranker),
            "reranker_enabled_rows": reranker_enabled_count,
            "reranker_fallback_count": reranker_fallback_count,
            "reranker_fallback_rate": (
                round(reranker_fallback_count / len(bridge_rows), 6)
                if bridge_rows
                else 0.0
            ),
        },
        "artifacts": {
            "predictions": str(predictions_path),
            "bridge_rows": str(bridge_rows_path),
            "bridge_rows_csv": str(bridge_rows_csv_path),
            "eval_report": str(report_path),
        },
        "case_cache_isolation": {
            "enabled": bool(args.case_cache_isolation),
            "reason": "prevents unrelated legal cases from contaminating route labels",
            "namespaces_used": len(cache_namespaces_used),
        },
        "total_api_calls": total_calls,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_estimated_cost_usd": round(total_cost, 8),
        "setup_api_calls": total_setup_calls,
        "setup_input_tokens": total_setup_input_tokens,
        "setup_output_tokens": total_setup_output_tokens,
        "setup_total_tokens": total_setup_input_tokens + total_setup_output_tokens,
        "setup_estimated_cost_usd": round(total_setup_cost, 8),
        "evaluation_api_calls": int(eval_totals["calls"]),
        "evaluation_input_tokens": int(eval_totals["input_tokens"]),
        "evaluation_output_tokens": int(eval_totals["output_tokens"]),
        "evaluation_total_tokens": int(eval_totals["total_tokens"]),
        "evaluation_estimated_cost_usd": round(float(eval_totals["cost"]), 8),
        "answer_accuracy": round(answer_correct_count / len(bridge_rows), 6) if bridge_rows else 0.0,
        "cache_type_match_rate": round(route_match_count / len(bridge_rows), 6) if bridge_rows else 0.0,
        "overall_pass_rate": round(overall_pass_count / len(bridge_rows), 6) if bridge_rows else 0.0,
        "expected_miss_rate": round(expected_miss_count / len(bridge_rows), 6) if bridge_rows else 0.0,
        "actual_miss_rate": round(actual_miss_count / len(bridge_rows), 6) if bridge_rows else 0.0,
        "unexpected_miss_rate": round(unexpected_miss_count / len(bridge_rows), 6) if bridge_rows else 0.0,
        "positive_case_miss_rate": (
            round(unexpected_miss_count / expected_positive_count, 6)
            if expected_positive_count
            else 0.0
        ),
        "expected_route_counts": _manifest_route_counts(bridge_rows, "expected_cache_type"),
        "actual_route_counts": _manifest_route_counts(bridge_rows, "actual_cache_type"),
        "by_scenario": _summarize_rows(bridge_rows, "scenario"),
        "by_expected_cache_type": _summarize_rows(bridge_rows, "expected_cache_type"),
        "by_actual_cache_type": _summarize_rows(bridge_rows, "actual_cache_type"),
    }
    if cache_state_enabled:
        manifest["cache_reuse"] = {
            "enabled": True,
            "base_cache_namespace": base_cache_namespace,
            "cache_state_root": str(cache_state_root) if cache_state_root else "",
            "cache_state_existed_before_reset": cache_state_existed_before_reset,
            "cache_reset_requested": bool(args.cache_reset),
            "cache_load_attempts": cache_load_attempts,
            "cache_load_successes": cache_load_successes,
            "cache_entries_before_run": cache_entries_before_run,
            "cache_entries_after_run": cache_entries_after_run,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / len(bridge_rows), 6) if bridge_rows else 0.0,
        }
    else:
        manifest["cache_reuse"] = {
            "enabled": False,
            "reason": "mode is baseline",
        }
    if args.manifest_note:
        manifest["note"] = args.manifest_note

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = _build_legal_eval_report(out_dir, suite.suite_name, suite_path, bridge_rows, manifest)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[LEGAL-BENCH] Completed.")
    print(f"[LEGAL-BENCH] Predictions : {predictions_path}")
    print(f"[LEGAL-BENCH] Bridge rows : {bridge_rows_path}")
    print(f"[LEGAL-BENCH] Bridge CSV  : {bridge_rows_csv_path}")
    print(f"[LEGAL-BENCH] Manifest    : {manifest_path}")
    print(f"[LEGAL-BENCH] Eval report : {report_path}")
    print(
        "[LEGAL-BENCH] Summary: "
        f"answer_accuracy={manifest['answer_accuracy']:.3f}, "
        f"cache_type_match_rate={manifest['cache_type_match_rate']:.3f}, "
        f"overall_pass_rate={manifest['overall_pass_rate']:.3f}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LegalBench/CUAD cache-route benchmark runner"
    )
    parser.add_argument(
        "--suite-path",
        type=str,
        default="benchmark_data/legal_bench/legal_cache_suite_cuad_v1.json",
        help="Path to a generated legal cache-route suite JSON file",
    )
    parser.add_argument("--case-ids", type=str, default="")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--executor-model", type=str, default="claude-sonnet-4-5")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "cache"],
        default="cache",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--rerank-top", type=int, default=5)
    parser.add_argument(
        "--disable-reranker",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Disable Qwen reranking for legal CUAD runs. By default, reranking is "
            "enabled and retrieval falls back to FAISS candidates when the reranker "
            "returns zero chunks."
        ),
    )
    parser.add_argument(
        "--cache-state-root",
        type=str,
        default="",
        help=(
            "Root directory for persistent legal cache state. "
            "If omitted, defaults to <output-dir>/legal_cache_suite/cache_state"
        ),
    )
    parser.add_argument("--cache-reset", action="store_true")
    parser.add_argument(
        "--no-case-cache-isolation",
        dest="case_cache_isolation",
        action="store_false",
        help="Use one shared namespace for all legal cases instead of per-case namespaces",
    )
    parser.set_defaults(case_cache_isolation=True)
    parser.add_argument("--output-dir", type=str, default="benchmark_artifacts")
    parser.add_argument("--manifest-note", type=str, default="")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_legal_benchmark(args)


if __name__ == "__main__":
    main()
