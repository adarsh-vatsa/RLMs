"""Run MRCR v2 against an OpenAI-compatible executor, without answer caching."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from aa_lcr.api import call_chat_completion, call_with_retries
from aa_lcr.prompting import chat_token_count
from mrcr_v2.dataset import (
    DEFAULT_DATA_DIR, DEFAULT_MODEL, file_hash, load_prepared, load_tokenizer,
    read_source, text_hash, tokenizer_metadata, validate_bounds,
)
from mrcr_v2.prompting import PROMPT_VERSION, messages, pack_evidence
from mrcr_v2.scoring import UPSTREAM_REVISION, score_prediction


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
    if full_tokens <= budget:
        return "direct_fit"
    return "unsupported_context" if mode == "direct" else "dense_child_packed"


def _index_source(scs, controller, body: str, source_id: str, args) -> dict:
    started = time.perf_counter()
    if args.child_tokens >= scs.EMBEDDING_MAX_LENGTH:
        raise ValueError("--child-tokens must be below the embedding input limit")
    # Call the exact chunker directly: ingest() otherwise permits estimated-token fallback.
    chunks = scs._chunk_text_with_tokenizer(
        body, tokenizer=controller.embedder.tokenizer,
        chunk_tokens=args.child_tokens, overlap_tokens=args.child_overlap_tokens,
    )
    if not chunks:
        raise ValueError("Exact embedding-tokenizer offsets are required for MRCR")
    texts, metadata = [], []
    for index, (text, meta) in enumerate(chunks):
        start, end = meta["char_start"], meta["char_end"]
        if not 0 <= start < end <= len(body) or text != body[start:end]:
            raise ValueError("Embedding tokenizer returned invalid source offsets")
        texts.append(text)
        metadata.append({
            **meta, "chunk_index": index, "child_index": index,
            "source_id": source_id, "tokenizer_model": scs.EMBEDDING_MODEL,
        })
    embedding_started = time.perf_counter()
    embeddings = controller.embedder.encode_documents(texts)
    embedding_ms = (time.perf_counter() - embedding_started) * 1000
    encoded_max = controller.embedder._last_encode_info["max_sequence_tokens"]
    if encoded_max >= scs.EMBEDDING_MAX_LENGTH:
        raise ValueError("A child reached the embedding input limit; reduce --child-tokens")
    controller.doc_index = scs.FAISSIndex()
    controller.doc_index.add(embeddings, metadata)
    controller._doc_chunks = texts
    controller._doc_chunk_metadata = metadata
    return {
        "ingested_chunks": len(texts), "document_embedding_ms": embedding_ms,
        "ingest_ms": (time.perf_counter() - started) * 1000,
        "child_encoded_length_max": encoded_max,
    }


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
        full_tokens = chat_token_count(tokenizer, messages(prompt))
        if full_tokens != row["full_rendered_input_tokens"]:
            raise ValueError("Prepared token count changed; prepare a new dataset")
        audited.append({
            "case_id": row["case_id"], "full_rendered_input_tokens": full_tokens,
            "route": _route(args.mode, full_tokens, args.max_input_tokens),
        })
    audit = {
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
        "temperature": 0, "enable_thinking": False, "answer_cache_enabled": False,
        "reranker_enabled": False, "knowledge_lookup_enabled": False,
        "retrieval_assisted": args.mode == "hybrid", "serving_metadata": serving_metadata,
        "max_retries": args.max_retries, "request_timeout_seconds": args.request_timeout_seconds,
        "api_usage_note": "Usage is for returned completions; failed HTTP attempts may have unreported usage.",
    }
    _write_json(run_dir / "manifest.json", manifest)
    predictions = []
    controller = scs = None
    active_source = previous_source = None
    api_key = os.getenv(args.api_key_env, "") if args.api_key_env else ""
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
            try:
                if route == "unsupported_context":
                    record["status"] = "unsupported_context"
                else:
                    if row["source_id"] != previous_source:
                        prefix, body = read_source(data_dir, row["source_id"])
                        previous_source = row["source_id"]
                    request = messages(prefix + body + row["question"])
                    if route == "dense_child_packed":
                        if controller is None:
                            scs = scs_module or _import_semantic_cache_system()
                            controller = scs.SemanticCacheController(
                                metrics=scs.ExecutionMetrics(), embedder=scs.EmbeddingEngine(),
                                reranker=None, corpus_id=f"mrcr_v2_{run_id}", corpus_domain="mrcr_v2",
                            )
                            manifest["embedding"] = {
                                "model": scs.EMBEDDING_MODEL, "max_length": scs.EMBEDDING_MAX_LENGTH,
                                "tokenizer": tokenizer_metadata(controller.embedder.tokenizer, scs.EMBEDDING_MODEL),
                                "query_instruction": scs.EMBEDDING_QUERY_INSTRUCTION,
                                "contract_version": scs.EMBEDDING_CONTRACT_VERSION,
                                "batch_size": scs.EMBEDDING_BATCH_SIZE,
                                "device": str(controller.embedder.device),
                                "dtype": controller.embedder.torch_dtype_name,
                            }
                        if active_source != row["source_id"]:
                            record.update(_index_source(scs, controller, body, row["source_id"], args))
                            active_source = row["source_id"]
                        retrieved_at = time.perf_counter()
                        results = controller.retrieve(row["question"], top_k=controller.doc_index.total,
                                                      rerank_top=0, use_reranker=False)
                        record["retrieval_ms"] = (time.perf_counter() - retrieved_at) * 1000
                        packed_at = time.perf_counter()
                        request, packing = pack_evidence(tokenizer, prefix, body, row["question"],
                                                        results, args.max_input_tokens)
                        record.update(packing)
                        record["packing_ms"] = (time.perf_counter() - packed_at) * 1000
                    record["final_rendered_input_tokens"] = chat_token_count(tokenizer, request)
                    generation_started = time.perf_counter()
                    try:
                        result, attempts = call_with_retries(
                            lambda: completion_caller(
                                base_url=args.executor_base_url, model=args.executor_model,
                                messages=request, max_tokens=args.max_output_tokens,
                                extra_body={"temperature": 0, "chat_template_kwargs": {"enable_thinking": False}},
                                api_key=api_key, timeout_seconds=args.request_timeout_seconds,
                            ), max_retries=args.max_retries,
                        )
                    finally:
                        record["generation_ms"] = (time.perf_counter() - generation_started) * 1000
                    record.update(
                        prediction=result.text, input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens, raw_usage=result.raw_usage,
                        finish_reason=result.finish_reason, attempts=attempts,
                    )
                    record.update(score_prediction(result.text, row["answer"]))
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}",
                              mrcr_score=0.0, exact_match=False, prefix_compliant=False,
                              attempts=int(getattr(exc, "attempts", 0)))
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
    return parser


if __name__ == "__main__":
    run_benchmark(build_arg_parser().parse_args())
