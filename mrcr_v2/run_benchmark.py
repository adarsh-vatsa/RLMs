"""Run MRCR v2 against an OpenAI-compatible executor, without answer caching."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from execution.client import call_chat_completion
from mrcr_v2.dataset import (
    DEFAULT_DATA_DIR, DEFAULT_MODEL, file_hash, load_prepared, load_tokenizer,
    read_source, text_hash, tokenizer_metadata, validate_bounds,
)
from mrcr_v2.prompting import PROMPT_VERSION
from mrcr_v2.scoring import UPSTREAM_REVISION, score_prediction
from mrcr_v2.adapter import solver_task
from execution.contracts import add_arguments, resolve_config
from execution.pipeline import Pipeline
from execution.artifacts import record_evaluation, finalize
from execution.tokens import route as execution_route


def _import_semantic_cache_system():
    import semantic_cache_system

    return semantic_cache_system


def _validate_args(args, manifest: dict) -> tuple[int, int]:
    minimum = args.min_source_tokens if args.min_source_tokens is not None else manifest["min_source_tokens"]
    maximum = args.max_source_tokens if args.max_source_tokens is not None else manifest["max_source_tokens"]
    validate_bounds(minimum, maximum)
    if minimum < manifest["min_source_tokens"] or maximum > manifest["max_source_tokens"]:
        raise ValueError("Cannot widen prepared source bounds; prepare a new dataset")
    if args.max_rows < 0:
        raise ValueError("--max-rows must be non-negative")
    if min(args.context_window_tokens, args.max_input_tokens, args.max_output_tokens) <= 0:
        raise ValueError("Executor token budgets must be positive")
    if args.max_input_tokens + args.max_output_tokens > args.context_window_tokens:
        raise ValueError("Input plus output budget must not exceed the context window")
    if args.child_tokens <= 0 or not 0 <= args.child_overlap_tokens < args.child_tokens:
        raise ValueError("Child tokens must be positive with 0 <= overlap < child tokens")
    if args.max_retries < 1 or args.request_timeout_seconds <= 0:
        raise ValueError("Retry count and request timeout must be positive")
    return minimum, maximum


def _route(mode: str, full_tokens: int, budget: int) -> str:
    return execution_route(mode, full_tokens, budget)


def _summary(rows: list[dict]) -> dict:
    supported = [row for row in rows if row["status"] != "unsupported_context"]
    count = len(supported)
    return {
        "selected_count": len(rows), "supported_count": count,
        "supported_fraction": count / len(rows) if rows else None,
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "mean_mrcr_score": sum(row["mrcr_score"] for row in supported) / count if count else None,
        "exact_match_accuracy": sum(row["exact_match"] for row in supported) / count if count else None,
        "prefix_compliance_rate": sum(row["prefix_compliant"] for row in supported) / count if count else None,
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "ingest_ms": sum(row["ingest_ms"] for row in rows),
        "retrieval_ms": sum(row["retrieval_ms"] for row in rows),
        "generation_ms": sum(row["generation_ms"] for row in rows),
    }


def _source_band(tokens: int) -> str:
    lower = 1 << (tokens.bit_length() - 1)
    return f"[{lower},{lower * 2})"


def build_report(rows: list[dict]) -> dict:
    return {
        **_summary(rows),
        "metric_scale": "0_to_1",
        "quality_denominator": "supported_examples_including_execution_failures_as_zero",
        "source_band_tokenizer": "executor_full_rendered_prompt",
        "by_route": {key: _summary([row for row in rows if row["route"] == key])
                     for key in sorted({row["route"] for row in rows})},
        "by_source_length_band": {
            key: _summary([row for row in rows if _source_band(row["full_rendered_input_tokens"]) == key])
            for key in sorted({_source_band(row["full_rendered_input_tokens"]) for row in rows})
        },
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_benchmark(args, *, tokenizer_factory=load_tokenizer,
                  completion_caller=call_chat_completion, scs_module=None) -> Path | None:
    data_dir = Path(args.data_dir)
    dataset_manifest, all_rows = load_prepared(data_dir)
    minimum, maximum = _validate_args(args, dataset_manifest)
    revision = args.tokenizer_revision or dataset_manifest["tokenizer"]["revision"]
    tokenizer = tokenizer_factory(args.executor_model, revision)
    config = resolve_config(args, mode=args.mode, legacy_source_order=True,
                            executor_model=args.executor_model, max_input_tokens=args.max_input_tokens,
                            max_output_tokens=args.max_output_tokens, context_window_tokens=args.context_window_tokens)
    preflight_pipeline = Pipeline(config, tokenizer, None)
    token_metadata = tokenizer_metadata(tokenizer, args.executor_model)
    if token_metadata != dataset_manifest["tokenizer"]:
        raise ValueError("Executor tokenizer or chat template changed; prepare a new dataset")
    selected = [row for row in all_rows if minimum <= row["full_rendered_input_tokens"] <= maximum]
    if args.max_rows:
        selected = selected[:args.max_rows]
    if not selected:
        raise ValueError("No MRCR examples matched the requested filters")
    # Retain selection order within each source while processing only one source at a time.
    source_order = {source_id: index for index, source_id in enumerate(dict.fromkeys(row["source_id"] for row in selected))}
    selected.sort(key=lambda row: source_order[row["source_id"]])
    audited = []
    previous_source = None
    prefix = body = ""
    for row in selected:
        if row["source_id"] != previous_source:
            prefix, body = read_source(data_dir, row["source_id"])
            previous_source = row["source_id"]
        prompt = prefix + body + row["question"]
        if text_hash(prompt) != row["case_id"]:
            raise ValueError("Prepared prompt does not match its example ID")
        audit_row = preflight_pipeline.preflight(solver_task(row["case_id"], row["source_id"], prefix, body, row["question"]))
        full_tokens = audit_row["full_rendered_input_tokens"]
        if full_tokens != row["full_rendered_input_tokens"]:
            raise ValueError("Prepared token count changed; prepare a new dataset")
        audited.append({
            "case_id": row["case_id"], "full_rendered_input_tokens": full_tokens,
            "route": audit_row["route"],
        })
    audit = {
        "execution": config.metadata(), "cache_assumption": "miss",
        "mode": args.mode, "min_source_tokens": minimum, "max_source_tokens": maximum,
        "selected_count": len(selected), "rows": audited,
        "route_counts": dict(Counter(row["route"] for row in audited)),
        "tokenizer": token_metadata, "max_input_tokens": args.max_input_tokens,
        "max_output_tokens": args.max_output_tokens, "context_window_tokens": args.context_window_tokens,
    }
    if args.preflight_only:
        if args.preflight_output:
            _write_json(Path(args.preflight_output), audit)
        print(json.dumps(audit, indent=2))
        return None

    serving_metadata = json.loads(Path(args.serving_metadata).read_text()) if args.serving_metadata else {}
    if not isinstance(serving_metadata, dict):
        raise ValueError("--serving-metadata must contain a JSON object")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("--run-id must be a single directory name")
    run_dir = Path(args.output_root) / args.mode / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id, "repeat_id": args.repeat_id, "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode, "prompt_version": PROMPT_VERSION, "scorer_revision": UPSTREAM_REVISION,
        "dataset_manifest_sha256": file_hash(data_dir / "dataset_manifest.json"),
        "dataset": dataset_manifest, "tokenizer": token_metadata,
        "min_source_tokens": minimum, "max_source_tokens": maximum,
        "selected_example_ids": [row["case_id"] for row in selected],
        "max_input_tokens": args.max_input_tokens, "max_output_tokens": args.max_output_tokens,
        "context_window_tokens": args.context_window_tokens,
        "child_tokens": args.child_tokens, "child_overlap_tokens": args.child_overlap_tokens,
        "executor_model": args.executor_model, "executor_base_url": args.executor_base_url,
        "temperature": 0, "enable_thinking": False, "answer_cache_enabled": config.cache_read or config.cache_write,
        "execution": config.metadata(),
        "reranker_enabled": config.rerank_top > 0, "knowledge_lookup_enabled": False,
        "retrieval_assisted": args.mode == "hybrid", "serving_metadata": serving_metadata,
        "max_retries": args.max_retries, "request_timeout_seconds": args.request_timeout_seconds,
        "api_usage_note": "Usage is for returned completions; failed HTTP attempts may have unreported usage.",
    }
    _write_json(run_dir / "manifest.json", manifest)
    predictions = []
    previous_source = None
    api_key = os.getenv(args.api_key_env, "") if args.api_key_env else ""

    def backend():
        scs = scs_module or _import_semantic_cache_system()
        controller = scs.SemanticCacheController(
            metrics=scs.ExecutionMetrics(), embedder=scs.EmbeddingEngine(),
            reranker=scs.Reranker() if config.rerank_top else None,
            corpus_id=f"mrcr_v2_{run_id}", corpus_domain="mrcr_v2")
        if config.cache_read and config.cache_matching == "semantic":
            from execution.cache import configure_verifier
            configure_verifier(controller, args)
        manifest["embedding"] = {
            "model": scs.EMBEDDING_MODEL, "max_length": scs.EMBEDDING_MAX_LENGTH,
            "tokenizer": tokenizer_metadata(controller.embedder.tokenizer, scs.EMBEDDING_MODEL),
            "query_instruction": scs.EMBEDDING_QUERY_INSTRUCTION, "contract_version": scs.EMBEDDING_CONTRACT_VERSION,
            "batch_size": scs.EMBEDDING_BATCH_SIZE, "device": str(controller.embedder.device),
            "dtype": controller.embedder.torch_dtype_name,
        }
        return scs, controller

    def complete(request):
        return completion_caller(base_url=args.executor_base_url, model=args.executor_model,
            messages=request, max_tokens=args.max_output_tokens,
            extra_body={"temperature": 0, "chat_template_kwargs": {"enable_thinking": False}},
            api_key=api_key, timeout_seconds=args.request_timeout_seconds)

    pipeline = Pipeline(config, tokenizer, complete, backend_factory=backend, output_dir=run_dir)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as output, \
            (run_dir / "bridge_rows.csv").open("w", encoding="utf-8", newline="") as bridge:
        writer = None
        for row in selected:
            started = time.perf_counter()
            route = _route(args.mode, row["full_rendered_input_tokens"], args.max_input_tokens)
            record = {
                "case_id": row["case_id"], "source_id": row["source_id"],
                "mode": args.mode, "route": route, "status": "ok", "error": "",
                "prediction": "", "answer": row["answer"],
                "mrcr_score": None, "exact_match": None, "prefix_compliant": None,
                "full_rendered_input_tokens": row["full_rendered_input_tokens"],
                "published_context_len": row["published_context_len"],
                "final_rendered_input_tokens": 0, "selected_evidence_ranges": [],
                "selected_child_indices": [], "candidate_count": 0,
                "ingested_chunks": 0, "child_encoded_length_max": 0,
                "ingest_ms": 0.0, "document_embedding_ms": 0.0,
                "retrieval_ms": 0.0, "packing_ms": 0.0, "generation_ms": 0.0,
                "input_tokens": 0, "output_tokens": 0, "raw_usage": {},
                "attempts": 0, "finish_reason": None, "total_ms": 0.0,
                "upstream_metadata": row["upstream_metadata"],
            }
            if row["source_id"] != previous_source:
                prefix, body = read_source(data_dir, row["source_id"])
                previous_source = row["source_id"]
            result = pipeline.execute(solver_task(row["case_id"], row["source_id"], prefix, body, row["question"]))
            record.update(result)
            route = result["route"]
            if result["status"] == "ok":
                record.update(score_prediction(result["prediction"], row["answer"]))
            elif result["status"] == "error":
                record.update(mrcr_score=0.0, exact_match=False, prefix_compliant=False)
            record_evaluation(run_dir, row["case_id"], record["status"],
                {key: record[key] for key in ("mrcr_score", "exact_match", "prefix_compliant") if record[key] is not None})
            record["total_ms"] = (time.perf_counter() - started) * 1000
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            if writer is None:
                writer = csv.DictWriter(bridge, fieldnames=list(record))
                writer.writeheader()
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in record.items()})
            bridge.flush()
            predictions.append(record)
            print(f"[MRCR] {len(predictions)}/{len(selected)} {route} {record['status']} score={record['mrcr_score']}")
            if args.fail_fast and record["status"] == "error":
                break
    finalize(run_dir)
    report = build_report(predictions)
    report["planned_count"] = len(selected)
    manifest.update(status="complete" if len(predictions) == len(selected) else "stopped_early",
                    completed_at=datetime.now(timezone.utc).isoformat(), processed_count=len(predictions))
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "eval_report.json", report)
    if args.fail_fast and any(row["status"] == "error" for row in predictions):
        raise RuntimeError(f"MRCR stopped on error; results saved in {run_dir}")
    print(f"[MRCR] Results: {run_dir}")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["direct", "hybrid"], required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--min-source-tokens", type=int)
    parser.add_argument("--max-source-tokens", type=int)
    parser.add_argument("--executor-model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--executor-base-url", default=os.getenv("OPENAI_COMPAT_EXECUTOR_BASE_URL") or os.getenv("OPENAI_COMPAT_BASE_URL") or "http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default=os.getenv("OPENAI_COMPAT_API_KEY_ENV", ""))
    parser.add_argument("--context-window-tokens", type=int, default=65536)
    parser.add_argument("--max-input-tokens", type=int, default=60000)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--child-tokens", type=int, default=7500)
    parser.add_argument("--child-overlap-tokens", type=int, default=750)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--request-timeout-seconds", type=int, default=1800)
    parser.add_argument("--output-root", type=Path, default=Path("benchmark_artifacts/mrcr_v2"))
    parser.add_argument("--run-id")
    parser.add_argument("--repeat-id", type=int, default=1)
    parser.add_argument("--serving-metadata", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    add_arguments(parser)
    return parser


if __name__ == "__main__":
    run_benchmark(build_arg_parser().parse_args())
