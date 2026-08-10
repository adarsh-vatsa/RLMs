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
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from long_bench_v2.qwen_prompt import (  # noqa: E402
    build_openai_compatible_mcq_extra_body,
    build_strict_mcq_messages,
    chat_token_count,
    mcq_decoder_constraint_metadata,
)


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


ARTIFACT_SUBDIR = "longbench_v2"
REPORT_FILENAME = "official_longbench_v2_eval_report.json"
DEFAULT_SUITE_CSV = Path("benchmark_data/long_bench_v2/data_cache_suite.csv")
DEFAULT_SOURCE_JSON = Path("benchmark_data/long_bench_v2/data.json")
DEFAULT_ROW_TYPES = "original,exact,semantic"
DEFAULT_OPENAI_COMPAT_EXECUTOR_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_OPENAI_COMPAT_EVALUATOR_MODEL = "mistralai/Mistral-Small-24B-Instruct-2501"
DEFAULT_TOP_K = 10
DEFAULT_RERANK_TOP = 3
DEFAULT_SYNTHESIS_MAX_CHUNKS = 3
DEFAULT_ROW_ORDER = "source_grouped"
DEFAULT_CACHE_SAVE_INTERVAL = 10
DEFAULT_CONTEXT_WINDOW_TOKENS = 262144
DEFAULT_MAX_INPUT_TOKENS = 240000
DEFAULT_MAX_OUTPUT_TOKENS = 8
DEFAULT_CHILD_TOKENS = 7500
DEFAULT_CHILD_OVERLAP_TOKENS = 750
HYBRID_ROUTE_VERSION = "longbench_v2_fast_hybrid_v1"
HYBRID_PROMPT_VERSION = "strict_qwen_mcq_v1"
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


def _coerce_int(value: object, default: int = 0) -> int:
    text = _coerce_text(value)
    if not text:
        return default
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


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


def filter_suite_rows(
    rows: list[dict],
    row_types: Sequence[str],
    max_rows: int = 0,
    source_ids: Sequence[str] = (),
) -> list[dict]:
    row_type_filter = {row_type.strip() for row_type in row_types if row_type.strip()}
    source_id_filter = {source_id.strip() for source_id in source_ids if source_id.strip()}
    selected = [
        row
        for row in rows
        if (not row_type_filter or row.get("row_type") in row_type_filter)
        and (not source_id_filter or row.get("source_id") in source_id_filter)
    ]
    if max_rows > 0:
        selected = selected[:max_rows]
    return selected


def order_suite_rows(rows: list[dict], row_order: str) -> list[dict]:
    if row_order == "input":
        return list(rows)
    if row_order != "source_grouped":
        raise ValueError(f"Unsupported --row-order: {row_order}")

    source_order: list[str] = []
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        source_id = _coerce_text(row.get("source_id")) or _coerce_text(row.get("case_id"))
        if source_id not in grouped:
            source_order.append(source_id)
            grouped[source_id] = []
        grouped[source_id].append(row)
    return [row for source_id in source_order for row in grouped[source_id]]


def estimate_context_tokens(row: dict) -> int:
    token_count = _coerce_int(row.get("token_count"), 0)
    if token_count > 0:
        return token_count
    return int(round(len(_coerce_text(row.get("context")).split()) * 1.33))


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

    explicit_pattern = (
        r"(?:FINAL\s+ANSWER|CORRECT\s+(?:ANSWER|CHOICE|OPTION)|ANSWER|CHOICE|OPTION)"
        r"\s*(?:IS\s*:|IS|:)?\s*[\(\[]?\s*([A-D])\b\s*[\)\]]?"
    )
    explicit_matches = list(re.finditer(explicit_pattern, stripped))
    if explicit_matches:
        return explicit_matches[-1].group(1)

    boundary_patterns = [
        r"^\s*[\(\[]?\s*([A-D])\s*[\)\].:-]",
        r"(?:^|\n)\s*[\(\[]?\s*([A-D])\s*[\)\].:-]?\s*$",
    ]
    for pattern in boundary_patterns:
        match = re.search(pattern, stripped)
        if match:
            return match.group(1)
    return ""


def answer_correct(prediction: str, expected_answer: str) -> bool:
    return parse_choice(prediction) == parse_choice(expected_answer)


def build_source_scope_hash(source_id: str, context: str) -> str:
    normalized_context = str(context or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    payload = f"longbench_v2_source_scope_v1\0{source_id.strip()}\0{normalized_context}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strict_choice(text: str) -> str:
    normalized = str(text or "").strip()
    return normalized if normalized in CHOICE_LETTERS else ""


def _response_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            value = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        return "\n".join(parts).strip()
    return ""


def _load_executor_tokenizer(model: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Transformers is required for LongBench-v2 hybrid token accounting."
        ) from exc
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def executor_tokenizer_metadata(tokenizer: Any, requested_model: str) -> dict:
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    if not isinstance(chat_template, str):
        chat_template = json.dumps(chat_template, sort_keys=True, default=str)
    return {
        "executor_tokenizer_model": requested_model,
        "executor_tokenizer_name_or_path": _coerce_text(
            getattr(tokenizer, "name_or_path", requested_model)
        ) or requested_model,
        "executor_tokenizer_class": type(tokenizer).__name__,
        "executor_chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
    }


def pack_hybrid_children(
    *,
    tokenizer: Any,
    query: str,
    results: list[dict],
    max_input_tokens: int,
) -> tuple[list[dict], dict]:
    """Pack FAISS-ranked children until the next exact chat request would overflow."""
    separator = "\n\n---\n\n"
    selected: list[dict] = []
    selected_texts: list[str] = []
    rendered_tokens = chat_token_count(tokenizer, build_strict_mcq_messages("", query))

    for result in results:
        candidate_texts = [*selected_texts, result["text"]]
        candidate_context = separator.join(candidate_texts)
        candidate_messages = build_strict_mcq_messages(candidate_context, query)
        candidate_tokens = chat_token_count(tokenizer, candidate_messages)
        if candidate_tokens > max_input_tokens:
            break
        selected.append(result)
        selected_texts = candidate_texts
        rendered_tokens = candidate_tokens

    if not selected:
        raise RuntimeError("No retrieved child fits within the configured hybrid input budget")

    selected_ranges = []
    for result in selected:
        metadata = result.get("metadata") or {}
        selected_ranges.append(
            {
                "child_index": metadata.get("child_index", metadata.get("chunk_index")),
                "token_start": metadata.get("token_start"),
                "token_end": metadata.get("token_end"),
                "char_start": metadata.get("char_start"),
                "char_end": metadata.get("char_end"),
                "score": round(float(result.get("score") or 0.0), 8),
            }
        )

    packed_context = separator.join(selected_texts)
    return build_strict_mcq_messages(packed_context, query), {
        "rendered_input_tokens": rendered_tokens,
        "faiss_candidate_count": len(results),
        "selected_child_count": len(selected),
        "dropped_child_count": max(0, len(results) - len(selected)),
        "selected_child_indices": [item["child_index"] for item in selected_ranges],
        "selected_evidence_ranges": selected_ranges,
    }


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
    llm_provider: str = "anthropic",
    evaluator_model: str = "",
    synthesis_max_chunks: int = 0,
    openai_compatible_extra_body: dict | None = None,
    mcq_prompt_style: str = "default",
    mcq_synthesis_max_tokens: int = 0,
    reranker_disabled: bool = False,
    reranker_relevance_threshold: float = 0.0,
    reranker_max_length: int = 0,
    min_reranked_results: int = 0,
    doc_chunk_size: int = 0,
    doc_chunk_overlap: int = 0,
    doc_chunk_tokens: int = 0,
    doc_chunk_overlap_tokens: int = 0,
    doc_chunk_tokenizer_model: str = "",
    embedding_query_instruction: str = "",
    embedding_max_length: int = 0,
    embedding_contract_version: str = "",
    synthesis_input_token_budget: int = 0,
    search_mode: str = "packed",
    iterative_reader_version: int = 0,
    scan_min_chunk_ratio: float = 0.0,
    scan_max_chunk_ratio: float = 0.0,
    scan_min_chunks: int = 0,
    scan_max_chunks: int = 0,
    scan_max_tokens: int = 0,
    scan_empty_ledger_fallback_ratio: float = 0.0,
    iterative_packed_fallback_input_token_budget: int = 0,
    iterative_memory_max_chars: int = 0,
    iterative_batch_max_chunks: int = 1,
    iterative_batch_input_token_budget: int = 0,
    scan_order: str = "",
    hybrid_policy: dict | None = None,
) -> tuple[str, str]:
    dataset_signature = _build_dataset_signature(selected_rows)
    row_type_sig = "-".join(sorted({row_type.lower() for row_type in row_types if row_type}))
    extra_body_sig = json.dumps(openai_compatible_extra_body or {}, sort_keys=True)
    hybrid_policy_sig = json.dumps(hybrid_policy or {}, sort_keys=True)
    normalized_search_mode = (_coerce_text(search_mode) or "packed").lower()
    fixed_retrieval_knobs_ignored = normalized_search_mode in {"iterative", "hybrid"}
    namespace_reranker_disabled = bool(reranker_disabled) if normalized_search_mode != "iterative" else False
    namespace_top_k = 0 if fixed_retrieval_knobs_ignored else top_k
    namespace_rerank_top = (
        0 if normalized_search_mode == "iterative" or namespace_reranker_disabled else rerank_top
    )
    namespace_synthesis_max_chunks = 0 if fixed_retrieval_knobs_ignored else synthesis_max_chunks
    namespace_synthesis_input_budget = (
        0 if fixed_retrieval_knobs_ignored else synthesis_input_token_budget
    )
    namespace_reranker_relevance_threshold = (
        reranker_relevance_threshold
        if normalized_search_mode != "iterative" and not namespace_reranker_disabled
        else 0.0
    )
    namespace_reranker_max_length = (
        reranker_max_length
        if normalized_search_mode != "iterative" and not namespace_reranker_disabled
        else 0
    )
    namespace_min_reranked_results = (
        min_reranked_results
        if normalized_search_mode != "iterative" and not namespace_reranker_disabled
        else 0
    )
    namespace_iterative_reader_version = (
        iterative_reader_version if normalized_search_mode == "iterative" else 0
    )
    namespace_scan_empty_ledger_fallback_ratio = (
        scan_empty_ledger_fallback_ratio if normalized_search_mode == "iterative" else 0.0
    )
    namespace_iterative_packed_fallback_input_token_budget = (
        iterative_packed_fallback_input_token_budget if normalized_search_mode == "iterative" else 0
    )
    namespace_iterative_memory_max_chars = (
        iterative_memory_max_chars if normalized_search_mode == "iterative" else 0
    )
    namespace_iterative_batch_max_chunks = (
        iterative_batch_max_chunks if normalized_search_mode == "iterative" else 1
    )
    namespace_iterative_batch_input_token_budget = (
        iterative_batch_input_token_budget if normalized_search_mode == "iterative" else 0
    )
    digest = hashlib.sha256(
        (
            f"{suite_csv_sha256}\n{source_json_sha256}\n{dataset_signature}\n"
            f"{llm_provider}\n{executor_model}\n{evaluator_model}\n"
            f"{namespace_top_k}\n{namespace_rerank_top}\n{namespace_synthesis_max_chunks}\n{row_type_sig}\n"
            f"{extra_body_sig}\n{mcq_prompt_style}\n"
            f"{mcq_synthesis_max_tokens}\n"
            f"{namespace_reranker_disabled}\n{namespace_reranker_relevance_threshold}\n"
            f"{namespace_reranker_max_length}\n{namespace_min_reranked_results}\n"
            f"{doc_chunk_size}\n{doc_chunk_overlap}\n{doc_chunk_tokens}\n"
            f"{doc_chunk_overlap_tokens}\n{doc_chunk_tokenizer_model}\n"
            f"{embedding_query_instruction}\n{embedding_max_length}\n"
            f"{embedding_contract_version}\n"
            f"{namespace_synthesis_input_budget}\n"
            f"{normalized_search_mode}\n{scan_min_chunk_ratio}\n{scan_max_chunk_ratio}\n"
            f"{scan_min_chunks}\n{scan_max_chunks}\n"
            f"{scan_max_tokens}\n{namespace_scan_empty_ledger_fallback_ratio}\n"
            f"{namespace_iterative_packed_fallback_input_token_budget}\n"
            f"{namespace_iterative_memory_max_chars}\n"
            f"{namespace_iterative_batch_max_chunks}\n"
            f"{namespace_iterative_batch_input_token_budget}\n"
            f"{namespace_iterative_reader_version}\n{scan_order}\n{hybrid_policy_sig}"
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
    total_context_token_estimate = sum(int(row.get("context_token_estimate", 0) or 0) for row in rows)
    unique_source_context: dict[str, int] = {}
    for row in rows:
        source_key = _coerce_text(row.get("source_id")) or _coerce_text(row.get("case_id")) or str(len(unique_source_context))
        if source_key not in unique_source_context:
            unique_source_context[source_key] = int(row.get("context_token_estimate", 0) or 0)
    total_unique_source_context_estimate = sum(unique_source_context.values())
    total_cost = sum(float(row.get("delta_cost_usd", 0.0) or 0.0) for row in rows)
    return {
        "calls": total_calls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "context_token_estimate": total_context_token_estimate,
        "unique_source_context_token_estimate": total_unique_source_context_estimate,
        "cost": total_cost,
    }


def pct_savings(baseline_value: int | float, used_value: int | float) -> float | None:
    baseline = float(baseline_value or 0)
    if baseline == 0:
        return None
    return round(((baseline - float(used_value)) / baseline) * 100.0, 6)


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


def build_effective_config(scs, args: argparse.Namespace) -> dict:
    mcq_style_normalizer = getattr(scs, "_normalize_mcq_prompt_style", None)
    if callable(mcq_style_normalizer):
        mcq_prompt_style = mcq_style_normalizer(
            getattr(scs, "MCQ_PROMPT_STYLE", "default"),
            warn=False,
        )
    else:
        mcq_prompt_style = _coerce_text(getattr(scs, "MCQ_PROMPT_STYLE", "default")) or "default"

    scs.SYNTHESIS_MAX_CHUNKS = args.synthesis_max_chunks
    search_mode = _coerce_text(getattr(scs, "SEARCH_MODE", "packed")) or "packed"
    hybrid_enabled = search_mode == "hybrid"
    if hybrid_enabled:
        scs.DOCUMENT_CHUNK_TOKENS = args.child_tokens
        scs.DOCUMENT_CHUNK_OVERLAP_TOKENS = args.child_overlap_tokens
        scs.DOCUMENT_CHUNK_TOKENIZER_MODEL = _coerce_text(getattr(scs, "EMBEDDING_MODEL", ""))
    doc_chunk_tokens = int(getattr(scs, "DOCUMENT_CHUNK_TOKENS", 0))
    doc_chunk_tokenizer_model = _coerce_text(getattr(scs, "DOCUMENT_CHUNK_TOKENIZER_MODEL", ""))
    if doc_chunk_tokens > 0 and not doc_chunk_tokenizer_model:
        doc_chunk_tokenizer_model = args.executor_model

    reranker_disabled = bool(args.disable_reranker or search_mode in {"iterative", "hybrid"})
    rerank_top_effective = None if reranker_disabled else args.rerank_top
    retrieval_knobs_ignored = search_mode in {"iterative", "hybrid"}

    artifact_fields = {
        "top_k": args.top_k,
        "top_k_effective": None if retrieval_knobs_ignored else args.top_k,
        "iterative_top_k_ignored": search_mode == "iterative",
        "hybrid_top_k_ignored": hybrid_enabled,
        "rerank_top": args.rerank_top,
        "rerank_top_effective": rerank_top_effective,
        "iterative_rerank_top_ignored": search_mode == "iterative",
        "hybrid_rerank_top_ignored": hybrid_enabled,
        "synthesis_max_chunks": None if hybrid_enabled else int(scs.SYNTHESIS_MAX_CHUNKS),
        "synthesis_max_chunks_configured": int(scs.SYNTHESIS_MAX_CHUNKS),
        "mcq_prompt_style": mcq_prompt_style,
        "mcq_synthesis_max_tokens": int(getattr(scs, "MCQ_SYNTHESIS_MAX_TOKENS", 0)),
        "search_mode": search_mode,
        "iterative_reader_version": int(getattr(scs, "ITERATIVE_READER_VERSION", 0)),
        "scan_min_chunk_ratio": float(getattr(scs, "SCAN_MIN_CHUNK_RATIO", 0.0)),
        "scan_max_chunk_ratio": float(getattr(scs, "SCAN_MAX_CHUNK_RATIO", 0.0)),
        "scan_min_chunks": int(getattr(scs, "SCAN_MIN_CHUNKS", 0)),
        "scan_max_chunks": int(getattr(scs, "SCAN_MAX_CHUNKS", 0)),
        "scan_max_tokens": int(getattr(scs, "SCAN_MAX_TOKENS", 0)),
        "scan_empty_ledger_fallback_ratio": float(getattr(scs, "SCAN_EMPTY_LEDGER_FALLBACK_RATIO", 0.0)),
        "iterative_packed_fallback_input_token_budget": int(
            getattr(scs, "ITERATIVE_PACKED_FALLBACK_INPUT_TOKEN_BUDGET", 0)
        ),
        "iterative_memory_max_chars": int(getattr(scs, "ITERATIVE_MEMORY_MAX_CHARS", 0)),
        "iterative_batch_max_chunks": int(getattr(scs, "ITERATIVE_BATCH_MAX_CHUNKS", 1)),
        "iterative_batch_input_token_budget": int(
            getattr(scs, "ITERATIVE_BATCH_INPUT_TOKEN_BUDGET", 0)
        ),
        "scan_order": _coerce_text(getattr(scs, "SCAN_ORDER", "")),
        "doc_chunk_size": int(getattr(scs, "DOCUMENT_CHUNK_SIZE", 0)),
        "doc_chunk_overlap": int(getattr(scs, "DOCUMENT_CHUNK_OVERLAP", 0)),
        "doc_chunk_tokens": doc_chunk_tokens,
        "doc_chunk_overlap_tokens": int(getattr(scs, "DOCUMENT_CHUNK_OVERLAP_TOKENS", 0)),
        "doc_chunk_tokenizer_model": doc_chunk_tokenizer_model,
        "embedding_query_instruction": _coerce_text(getattr(scs, "EMBEDDING_QUERY_INSTRUCTION", "")),
        "embedding_batch_size": int(getattr(scs, "EMBEDDING_BATCH_SIZE", 0)),
        "embedding_max_length": int(getattr(scs, "EMBEDDING_MAX_LENGTH", 0)),
        "embedding_contract_version": _coerce_text(
            getattr(scs, "EMBEDDING_CONTRACT_VERSION", "")
        ),
        "embedding_device": _coerce_text(getattr(scs, "EMBEDDING_DEVICE", "")),
        "embedding_dtype": _coerce_text(getattr(scs, "EMBEDDING_DTYPE", "")),
        "reranker_relevance_threshold": float(getattr(scs, "RERANKER_RELEVANCE_THRESHOLD", 0.0)),
        "reranker_batch_size": int(getattr(scs, "RERANKER_BATCH_SIZE", 0)),
        "reranker_max_length": int(getattr(scs, "RERANKER_MAX_LENGTH", 0)),
        "min_reranked_results": int(getattr(scs, "MIN_RERANKED_RESULTS", 0)),
        "synthesis_input_token_budget": int(getattr(scs, "SYNTHESIS_INPUT_TOKEN_BUDGET", 0)),
        "reranker_disabled": reranker_disabled,
        "hybrid_route_version": HYBRID_ROUTE_VERSION if hybrid_enabled else "",
        "hybrid_prompt_version": HYBRID_PROMPT_VERSION if hybrid_enabled else "",
    }
    if hybrid_enabled:
        artifact_fields.update(mcq_decoder_constraint_metadata())
    hybrid_policy = None
    if hybrid_enabled:
        hybrid_policy = {
            "route_version": HYBRID_ROUTE_VERSION,
            "prompt_version": HYBRID_PROMPT_VERSION,
            "system_prompt_sha256": hashlib.sha256(
                build_strict_mcq_messages("", "")[0]["content"].encode("utf-8")
            ).hexdigest(),
            "temperature": 0,
            "thinking_enabled": False,
            **mcq_decoder_constraint_metadata(),
            "context_window_tokens": args.context_window_tokens,
            "max_input_tokens": args.max_input_tokens,
            "max_output_tokens": args.max_output_tokens,
            "safety_margin_tokens": (
                args.context_window_tokens - args.max_input_tokens - args.max_output_tokens
            ),
            "executor_tokenizer_model": args.executor_model,
            "child_tokens": args.child_tokens,
            "child_overlap_tokens": args.child_overlap_tokens,
            "child_tokenizer_model": artifact_fields["doc_chunk_tokenizer_model"],
            "candidate_strategy": "all_children_exact_faiss",
            "evidence_order": "faiss_score_descending",
            "reranker_enabled": False,
            "bm25_enabled": False,
            "parent_expansion_enabled": False,
            "knowledge_lookup_enabled": False,
            "consensus_enabled": False,
            "fact_extraction_enabled": False,
            "embedding_contract_version": artifact_fields["embedding_contract_version"],
            "embedding_pooling": "last_token",
            "embedding_padding_side": "left",
            "embedding_normalization": "l2",
        }
    namespace_kwargs = {
        "synthesis_max_chunks": artifact_fields["synthesis_max_chunks"],
        "mcq_prompt_style": artifact_fields["mcq_prompt_style"],
        "mcq_synthesis_max_tokens": artifact_fields["mcq_synthesis_max_tokens"],
        "reranker_disabled": artifact_fields["reranker_disabled"],
        "reranker_relevance_threshold": artifact_fields["reranker_relevance_threshold"],
        "reranker_max_length": artifact_fields["reranker_max_length"],
        "min_reranked_results": artifact_fields["min_reranked_results"],
        "doc_chunk_size": artifact_fields["doc_chunk_size"],
        "doc_chunk_overlap": artifact_fields["doc_chunk_overlap"],
        "doc_chunk_tokens": artifact_fields["doc_chunk_tokens"],
        "doc_chunk_overlap_tokens": artifact_fields["doc_chunk_overlap_tokens"],
        "doc_chunk_tokenizer_model": artifact_fields["doc_chunk_tokenizer_model"],
        "embedding_query_instruction": artifact_fields["embedding_query_instruction"],
        "embedding_max_length": artifact_fields["embedding_max_length"],
        "embedding_contract_version": artifact_fields["embedding_contract_version"],
        "synthesis_input_token_budget": artifact_fields["synthesis_input_token_budget"],
        "search_mode": artifact_fields["search_mode"],
        "iterative_reader_version": artifact_fields["iterative_reader_version"],
        "scan_min_chunk_ratio": artifact_fields["scan_min_chunk_ratio"],
        "scan_max_chunk_ratio": artifact_fields["scan_max_chunk_ratio"],
        "scan_min_chunks": artifact_fields["scan_min_chunks"],
        "scan_max_chunks": artifact_fields["scan_max_chunks"],
        "scan_max_tokens": artifact_fields["scan_max_tokens"],
        "scan_empty_ledger_fallback_ratio": artifact_fields["scan_empty_ledger_fallback_ratio"],
        "iterative_packed_fallback_input_token_budget": artifact_fields[
            "iterative_packed_fallback_input_token_budget"
        ],
        "iterative_memory_max_chars": artifact_fields["iterative_memory_max_chars"],
        "iterative_batch_max_chunks": artifact_fields["iterative_batch_max_chunks"],
        "iterative_batch_input_token_budget": artifact_fields["iterative_batch_input_token_budget"],
        "scan_order": artifact_fields["scan_order"],
        "hybrid_policy": hybrid_policy,
    }
    return {
        "artifact_fields": artifact_fields,
        "namespace_kwargs": namespace_kwargs,
        "search_top_k": 0 if retrieval_knobs_ignored else args.top_k,
        "search_rerank_top": rerank_top_effective or 0,
        "hybrid_policy": hybrid_policy,
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


def normalize_llm_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.api_key_env is None:
        if args.llm_provider == "openrouter":
            args.api_key_env = "OPENROUTER_API_KEY"
        elif args.llm_provider == "openai_compatible":
            args.api_key_env = None
        else:
            args.api_key_env = "ANTHROPIC_API_KEY"
    if args.llm_provider == "openrouter":
        if args.executor_model == "claude-sonnet-4-5":
            args.executor_model = "anthropic/claude-sonnet-4.5"
        if args.evaluator_model == "claude-haiku-4-5":
            args.evaluator_model = "anthropic/claude-haiku-4.5"
    if args.llm_provider == "openai_compatible":
        if args.executor_model == "claude-sonnet-4-5":
            args.executor_model = DEFAULT_OPENAI_COMPAT_EXECUTOR_MODEL
        if args.evaluator_model == "claude-haiku-4-5":
            args.evaluator_model = DEFAULT_OPENAI_COMPAT_EVALUATOR_MODEL
    return args


def _call_hybrid_executor(scs, controller, args: argparse.Namespace, messages: list[dict]) -> str:
    response = scs.create_llm_message(
        model=args.executor_model,
        max_tokens=args.max_output_tokens,
        temperature=0,
        system=messages[0]["content"],
        messages=messages[1:],
        extra_body=build_openai_compatible_mcq_extra_body(),
    )
    controller.metrics.record_call(
        args.executor_model,
        int(getattr(response.usage, "input_tokens", 0) or 0),
        int(getattr(response.usage, "output_tokens", 0) or 0),
    )
    return _response_text(response)


def _validate_hybrid_args(args: argparse.Namespace, embedding_max_length: int | None = None) -> None:
    if args.llm_provider != "openai_compatible":
        raise ValueError("Hybrid LongBench-v2 routing requires --llm-provider openai_compatible")
    if args.mode != "cache":
        raise ValueError("Hybrid LongBench-v2 routing requires --mode cache")
    if args.context_window_tokens <= args.max_output_tokens:
        raise ValueError("--context-window-tokens must exceed --max-output-tokens")
    if args.max_input_tokens <= 0:
        raise ValueError("--max-input-tokens must be positive")
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive")
    if args.max_input_tokens + args.max_output_tokens > args.context_window_tokens:
        raise ValueError(
            "--max-input-tokens plus --max-output-tokens must not exceed "
            "--context-window-tokens"
        )
    if args.child_tokens <= 0:
        raise ValueError("--child-tokens must be positive")
    if args.child_overlap_tokens < 0 or args.child_overlap_tokens >= args.child_tokens:
        raise ValueError("--child-overlap-tokens must be non-negative and less than --child-tokens")
    if embedding_max_length is not None and args.child_tokens >= embedding_max_length:
        raise ValueError("--child-tokens must be less than the embedding input limit")


def run_hybrid_route_audit(
    args: argparse.Namespace,
    selected_rows: list[dict],
    tokenizer_factory: Callable[[str], Any],
) -> Path:
    _validate_hybrid_args(args)
    tokenizer = tokenizer_factory(args.executor_model)
    tokenizer_info = executor_tokenizer_metadata(tokenizer, args.executor_model)
    audited_rows = []
    for row in selected_rows:
        query = build_query(row)
        rendered_tokens = chat_token_count(
            tokenizer,
            build_strict_mcq_messages(row["context"], query),
        )
        audited_rows.append(
            {
                "case_id": row["case_id"],
                "source_id": row["source_id"],
                "source_scope_hash": build_source_scope_hash(row["source_id"], row["context"]),
                "row_type": row["row_type"],
                "rendered_input_tokens": rendered_tokens,
                "route": "direct_fit" if rendered_tokens <= args.max_input_tokens else "dense_child_packed",
            }
        )

    counts = sorted(row["rendered_input_tokens"] for row in audited_rows)
    fit_rows = sorted(
        (row for row in audited_rows if row["route"] == "direct_fit"),
        key=lambda row: row["rendered_input_tokens"],
    )
    overlength_rows = sorted(
        (row for row in audited_rows if row["route"] == "dense_child_packed"),
        key=lambda row: row["rendered_input_tokens"],
    )
    median_idx = (len(counts) - 1) // 2
    p95_idx = min(len(counts) - 1, max(0, int((len(counts) * 0.95) + 0.999999) - 1))
    suggested = {
        "ordinary_direct_fit_source_id": fit_rows[(len(fit_rows) - 1) // 2]["source_id"] if fit_rows else None,
        "largest_direct_fit_source_id": fit_rows[-1]["source_id"] if fit_rows else None,
        "smallest_overlength_source_id": overlength_rows[0]["source_id"] if overlength_rows else None,
    }
    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / "longbench_v2_route_audit" / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    output_path = out_dir / "route_audit.json"
    payload = {
        "run_id": run_id,
        "created_at": started_at.isoformat(),
        "route_version": HYBRID_ROUTE_VERSION,
        "prompt_version": HYBRID_PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(
            build_strict_mcq_messages("", "")[0]["content"].encode("utf-8")
        ).hexdigest(),
        "temperature": 0,
        "thinking_enabled": False,
        **mcq_decoder_constraint_metadata(),
        **tokenizer_info,
        "suite_csv": str(args.suite_csv),
        "suite_csv_sha256": _sha256_file(Path(args.suite_csv)),
        "source_json_path": str(args.source_json_path),
        "source_json_sha256": _sha256_file(Path(args.source_json_path)),
        "row_types_requested": _parse_csv_values(args.row_types),
        "source_ids_requested": _parse_csv_values(args.source_ids),
        "context_window_tokens": args.context_window_tokens,
        "max_input_tokens": args.max_input_tokens,
        "max_output_tokens": args.max_output_tokens,
        "safety_margin_tokens": args.context_window_tokens - args.max_input_tokens - args.max_output_tokens,
        "rows_selected": len(audited_rows),
        "direct_fit_count": len(fit_rows),
        "overlength_count": len(overlength_rows),
        "rendered_input_tokens": {
            "min": counts[0],
            "median": counts[median_idx],
            "p95": counts[p95_idx],
            "max": counts[-1],
        },
        "suggested_smoke_source_ids": suggested,
        "rows": audited_rows,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[LONGBENCH-V2] Route audit: {output_path}")
    print(
        "[LONGBENCH-V2] Routes: "
        f"direct_fit={len(fit_rows)}, dense_child_packed={len(overlength_rows)}"
    )
    return output_path


def run_longbench_benchmark(
    args: argparse.Namespace,
    tokenizer_factory: Callable[[str], Any] | None = None,
) -> None:
    args = normalize_llm_args(args)
    if args.synthesis_max_chunks < 1:
        raise ValueError("--synthesis-max-chunks must be >= 1")
    if args.cache_save_interval < 1:
        raise ValueError("--cache-save-interval must be >= 1")

    suite_csv = Path(args.suite_csv)
    source_json_path = Path(args.source_json_path)
    row_types = _parse_csv_values(args.row_types)
    if not row_types:
        raise ValueError("--row-types must include at least one row type")

    contexts = load_context_by_source_id(source_json_path)
    all_rows = load_suite_rows(suite_csv, contexts)
    source_ids = _parse_csv_values(args.source_ids)
    selected_rows = filter_suite_rows(
        all_rows,
        row_types=row_types,
        max_rows=args.max_rows,
        source_ids=source_ids,
    )
    selected_rows = order_suite_rows(selected_rows, args.row_order)
    if not selected_rows:
        raise ValueError("No LongBench-v2 rows matched the requested filters")

    configured_search_mode = os.getenv("SEMANTIC_CACHE_SEARCH_MODE", "packed").strip().lower() or "packed"
    if configured_search_mode == "hybrid":
        _validate_hybrid_args(args)
    if args.route_audit_only:
        if configured_search_mode != "hybrid":
            raise ValueError("--route-audit-only requires SEMANTIC_CACHE_SEARCH_MODE=hybrid")
        run_hybrid_route_audit(
            args,
            selected_rows,
            tokenizer_factory or _load_executor_tokenizer,
        )
        return

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

    scs = _import_semantic_cache_system()
    scs.configure_llm_provider(
        provider=args.llm_provider,
        api_key_env=args.api_key_env,
        executor_model=args.executor_model,
        evaluator_model=args.evaluator_model,
        openrouter_base_url=args.openrouter_base_url,
        openai_compat_base_url=args.openai_compat_base_url,
        openai_compat_executor_base_url=args.openai_compat_executor_base_url,
        openai_compat_evaluator_base_url=args.openai_compat_evaluator_base_url,
        openai_compat_api_key_env=args.openai_compat_api_key_env,
    )
    openai_compatible_extra_body_config = {}
    if args.llm_provider == "openai_compatible":
        extra_body_config_getter = getattr(scs, "get_openai_compatible_extra_body_config", None)
        if callable(extra_body_config_getter):
            openai_compatible_extra_body_config = extra_body_config_getter(redact=True)
    effective_config = build_effective_config(scs, args)
    effective_fields = effective_config["artifact_fields"]
    hybrid_enabled = effective_fields["search_mode"] == "hybrid"
    executor_tokenizer = None
    if hybrid_enabled:
        _validate_hybrid_args(args, embedding_max_length=effective_fields["embedding_max_length"])
        executor_tokenizer = (tokenizer_factory or _load_executor_tokenizer)(args.executor_model)
        tokenizer_info = executor_tokenizer_metadata(executor_tokenizer, args.executor_model)
        effective_fields.update(tokenizer_info)
        effective_config["hybrid_policy"].update(tokenizer_info)

    if cache_state_enabled:
        cache_namespace, dataset_signature = resolve_cache_namespace(
            suite_csv_sha256=suite_csv_sha256,
            source_json_sha256=source_json_sha256,
            selected_rows=selected_rows,
            executor_model=args.executor_model,
            top_k=args.top_k,
            rerank_top=args.rerank_top,
            row_types=row_types,
            llm_provider=args.llm_provider,
            evaluator_model=args.evaluator_model,
            openai_compatible_extra_body=openai_compatible_extra_body_config,
            **effective_config["namespace_kwargs"],
        )
        cache_state_root = (
            Path(args.cache_state_root)
            if args.cache_state_root
            else Path(args.output_dir) / ARTIFACT_SUBDIR / "cache_state"
        )
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
    print(f"[LONGBENCH-V2] Row order: {args.row_order}")
    print(f"[LONGBENCH-V2] Mode: {args.mode}")
    if effective_fields["search_mode"] == "iterative":
        print(
            "[LONGBENCH-V2] Retrieval config: "
            "search_mode=iterative, "
            f"scan_min_ratio={effective_fields['scan_min_chunk_ratio']}, "
            f"scan_max_ratio={effective_fields['scan_max_chunk_ratio']}, "
            f"scan_min={effective_fields['scan_min_chunks']}, "
            f"scan_max={effective_fields['scan_max_chunks']}, "
            f"scan_max_tokens={effective_fields['scan_max_tokens']}, "
            f"empty_ledger_fallback_ratio={effective_fields['scan_empty_ledger_fallback_ratio']}, "
            f"memory_max_chars={effective_fields['iterative_memory_max_chars']}, "
            f"batch_max_chunks={effective_fields['iterative_batch_max_chunks']}, "
            f"batch_input_token_budget={effective_fields['iterative_batch_input_token_budget']}, "
            f"reader_version={effective_fields['iterative_reader_version']}"
        )
    elif hybrid_enabled:
        print(
            "[LONGBENCH-V2] Retrieval config: "
            "search_mode=hybrid, "
            f"max_input_tokens={args.max_input_tokens}, "
            f"context_window_tokens={args.context_window_tokens}, "
            f"max_output_tokens={args.max_output_tokens}, "
            f"child_tokens={args.child_tokens}, "
            f"child_overlap_tokens={args.child_overlap_tokens}, "
            "candidate_strategy=all_children_exact_faiss, reranker_disabled=True"
        )
    else:
        print(
            "[LONGBENCH-V2] Retrieval config: "
            f"top_k={args.top_k}, rerank_top={args.rerank_top}, "
            f"reranker_disabled={effective_fields['reranker_disabled']}, "
            f"reranker_threshold={effective_fields['reranker_relevance_threshold']}, "
            f"reranker_max_length={effective_fields['reranker_max_length']}, "
            f"synthesis_max_chunks={effective_fields['synthesis_max_chunks']}, "
            f"search_mode={effective_fields['search_mode']}"
        )
    print(f"[LONGBENCH-V2] Output dir: {out_dir}")
    if cache_state_enabled:
        print(f"[LONGBENCH-V2] Cache state root: {cache_state_root}")
        print(f"[LONGBENCH-V2] Cache namespace : {cache_namespace}")
        print(f"[LONGBENCH-V2] Cache state: {'warm start' if cache_state_existed_before_run else 'cold start'}")

    shared_embedder = scs.EmbeddingEngine()
    effective_fields["embedding_device_effective"] = _coerce_text(
        getattr(shared_embedder, "device", effective_fields["embedding_device"])
    )
    effective_fields["embedding_dtype_effective"] = _coerce_text(
        getattr(shared_embedder, "torch_dtype_name", effective_fields["embedding_dtype"])
    )
    shared_reranker = None if effective_fields["reranker_disabled"] else scs.Reranker()

    prediction_rows: list[dict] = []
    bridge_rows: list[dict] = []
    context_doc_cache: dict[str, Path] = {}
    active_source_id = ""
    cache_load_ms = 0.0
    cache_save_count = 0
    cache_final_save_ms = 0.0
    cache_controller = None

    if cache_state_enabled:
        cache_controller = scs.SemanticCacheController(
            metrics=scs.ExecutionMetrics(),
            embedder=shared_embedder,
            reranker=shared_reranker,
            corpus_id=cache_namespace,
            corpus_domain="longbench_v2",
        )
        if cache_state_path is not None:
            cache_load_attempts = 1
            t_load = time.time()
            if cache_state_path.exists() and cache_controller.load(cache_state_path):
                cache_load_successes = 1
            cache_load_ms = (time.time() - t_load) * 1000.0
            # The benchmark runner owns persistence cadence; avoid store() auto-saves.
            cache_controller._persist_path = None
        cache_entries_before_run = cache_controller.get_total_entries()

    for idx, row in enumerate(selected_rows, start=1):
        row_t0 = time.time()
        print(f"[LONGBENCH-V2] Row {idx}/{len(selected_rows)}: {row['case_id']}")
        source_id = _coerce_text(row.get("source_id")) or _coerce_text(row.get("case_id"))
        source_scope_hash = build_source_scope_hash(source_id, row.get("context", ""))
        context_write_ms = 0.0
        docs_dir = context_doc_cache.get(source_id)
        if not hybrid_enabled and docs_dir is None:
            t_write = time.time()
            docs_dir = _write_context_doc(out_dir, row)
            context_write_ms = (time.time() - t_write) * 1000.0
            context_doc_cache[source_id] = docs_dir

        if cache_state_enabled:
            controller = cache_controller
        else:
            controller = scs.SemanticCacheController(
                metrics=scs.ExecutionMetrics(),
                embedder=shared_embedder,
                reranker=shared_reranker,
                corpus_id=f"longbench_v2_{idx}",
                corpus_domain="longbench_v2",
            )

        ingest_ms = 0.0
        ingested_chunks = 0
        query = build_query(row)
        before = _snapshot_metrics(controller.metrics)
        t0 = time.time()
        api_status = "ok"
        api_error = ""
        hybrid_telemetry = {
            "hybrid_route": None,
            "source_scope_hash": source_scope_hash if hybrid_enabled else None,
            "full_rendered_input_tokens": None,
            "final_rendered_input_tokens": None,
            "valid_choice": None,
            "compact_cache_write": False,
            "executor_answer_calls": 0,
            "semantic_verifier_calls": 0,
            "document_embedding_count": 0,
            "document_embedding_ms": 0.0,
            "document_child_count": 0,
            "child_encoded_length_max": 0,
            "faiss_search_ms": 0.0,
            "faiss_candidates": [],
            "exact_packing_ms": 0.0,
            "selected_child_count": 0,
            "dropped_child_count": 0,
            "selected_child_indices": [],
            "selected_evidence_ranges": [],
        }
        if hybrid_enabled:
            try:
                controller.activate_data_scope(source_scope_hash)
                output = controller.lookup_cached_result(query) if cache_state_enabled else None
                lookup_after = _snapshot_metrics(controller.metrics)
                hybrid_telemetry["semantic_verifier_calls"] = int(
                    lookup_after["calls"] - before["calls"]
                )
                if output is not None:
                    hybrid_telemetry["hybrid_route"] = f"{output['cache_type']}_cache"
                    hybrid_telemetry["valid_choice"] = bool(_strict_choice(output.get("answer", "")))
                else:
                    controller.metrics.cache_misses += 1
                    full_messages = build_strict_mcq_messages(row["context"], query)
                    full_rendered_tokens = chat_token_count(executor_tokenizer, full_messages)
                    hybrid_telemetry["full_rendered_input_tokens"] = full_rendered_tokens

                    if full_rendered_tokens <= args.max_input_tokens:
                        hybrid_telemetry["hybrid_route"] = "direct_fit"
                        hybrid_telemetry["final_rendered_input_tokens"] = full_rendered_tokens
                        answer = _call_hybrid_executor(scs, controller, args, full_messages)
                    else:
                        hybrid_telemetry["hybrid_route"] = "dense_child_packed"
                        if docs_dir is None:
                            t_write = time.time()
                            docs_dir = _write_context_doc(out_dir, row)
                            context_write_ms = (time.time() - t_write) * 1000.0
                            context_doc_cache[source_id] = docs_dir
                        if source_id != active_source_id:
                            t_ingest = time.time()
                            ingested_chunks = controller.ingest(
                                docs_dir,
                                data_scope_hash=source_scope_hash,
                                source_id=source_id,
                            )
                            ingest_ms = (time.time() - t_ingest) * 1000.0
                            active_source_id = source_id
                            ingest_info = getattr(controller, "_last_ingest_info", {}) or {}
                            hybrid_telemetry["document_embedding_count"] = ingested_chunks
                            hybrid_telemetry["document_embedding_ms"] = float(
                                ingest_info.get("embedding_ms") or 0.0
                            )
                            hybrid_telemetry["child_encoded_length_max"] = int(
                                ingest_info.get("child_encoded_length_max") or 0
                            )
                            if (
                                hybrid_telemetry["child_encoded_length_max"]
                                >= effective_fields["embedding_max_length"]
                            ):
                                raise RuntimeError(
                                    "Hybrid child reached the embedding input limit; "
                                    "reduce --child-tokens"
                                )
                        total_children = int(getattr(controller.doc_index, "total", 0) or 0)
                        hybrid_telemetry["document_child_count"] = total_children
                        if not hybrid_telemetry["child_encoded_length_max"]:
                            ingest_info = getattr(controller, "_last_ingest_info", {}) or {}
                            hybrid_telemetry["child_encoded_length_max"] = int(
                                ingest_info.get("child_encoded_length_max") or 0
                            )
                        search_started = time.time()
                        results = controller.retrieve(
                            query,
                            top_k=total_children,
                            rerank_top=0,
                            use_reranker=False,
                            query_embedding=getattr(
                                controller,
                                "_last_cache_query_embedding",
                                None,
                            ),
                        )
                        hybrid_telemetry["faiss_search_ms"] = round(
                            (time.time() - search_started) * 1000.0,
                            3,
                        )
                        hybrid_telemetry["faiss_candidates"] = [
                            {
                                "child_index": (result.get("metadata") or {}).get(
                                    "child_index",
                                    (result.get("metadata") or {}).get("chunk_index"),
                                ),
                                "score": round(float(result.get("score") or 0.0), 8),
                            }
                            for result in results
                        ]
                        packing_started = time.time()
                        packed_messages, packing_info = pack_hybrid_children(
                            tokenizer=executor_tokenizer,
                            query=query,
                            results=results,
                            max_input_tokens=args.max_input_tokens,
                        )
                        hybrid_telemetry["exact_packing_ms"] = round(
                            (time.time() - packing_started) * 1000.0,
                            3,
                        )
                        hybrid_telemetry.update(
                            {
                                "faiss_candidate_count": packing_info["faiss_candidate_count"],
                                "final_rendered_input_tokens": packing_info["rendered_input_tokens"],
                                "selected_child_count": packing_info["selected_child_count"],
                                "dropped_child_count": packing_info["dropped_child_count"],
                                "selected_child_indices": packing_info["selected_child_indices"],
                                "selected_evidence_ranges": packing_info["selected_evidence_ranges"],
                            }
                        )
                        full_messages = packed_messages
                        answer = _call_hybrid_executor(scs, controller, args, packed_messages)

                    hybrid_telemetry["executor_answer_calls"] = 1
                    valid_choice = _strict_choice(answer)
                    hybrid_telemetry["valid_choice"] = bool(valid_choice)
                    if valid_choice:
                        controller.store_compact_mcq(
                            query,
                            valid_choice,
                            model_used=args.executor_model,
                            source_id=source_id,
                            route=hybrid_telemetry["hybrid_route"],
                            provenance={
                                "source_scope_hash": source_scope_hash,
                                "full_rendered_input_tokens": full_rendered_tokens,
                                "final_rendered_input_tokens": hybrid_telemetry[
                                    "final_rendered_input_tokens"
                                ],
                                "selected_evidence_ranges": hybrid_telemetry[
                                    "selected_evidence_ranges"
                                ],
                            },
                        )
                        hybrid_telemetry["compact_cache_write"] = True
                    retrieval_info = (
                        getattr(controller, "_last_retrieval_info", {}) or {}
                        if hybrid_telemetry["hybrid_route"] == "dense_child_packed"
                        else {}
                    )
                    output = {
                        "query": query,
                        "answer": answer,
                        "from_cache": False,
                        "retrieval": {
                            **retrieval_info,
                            **hybrid_telemetry,
                        },
                    }
            except Exception as exc:
                api_status = "error"
                api_error = f"{type(exc).__name__}: {exc}"
                output = {
                    "query": query,
                    "answer": "",
                    "from_cache": False,
                    "retrieval": hybrid_telemetry,
                }
        else:
            if not cache_state_enabled or source_id != active_source_id:
                t_ingest = time.time()
                ingested_chunks = controller.ingest(docs_dir)
                ingest_ms = (time.time() - t_ingest) * 1000.0
                active_source_id = source_id
            output = controller.search(
                query,
                top_k=effective_config["search_top_k"],
                rerank_top=effective_config["search_rerank_top"],
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
            "context_chars": len(row.get("context", "")),
            "context_token_estimate": estimate_context_tokens(row),
            **effective_fields,
            "ingested_chunks": ingested_chunks,
            "expected_cache_type": row.get("expected_cache_type", ""),
            "expected_from_cache": row.get("expected_from_cache", ""),
            "from_cache": actual_from_cache,
            "cache_type": actual_cache_type,
            "answer_correct": correct,
            "latency_ms": round(latency_ms, 3),
            "row_wall_ms": round((time.time() - row_t0) * 1000.0, 3),
            "cache_load_ms": round(cache_load_ms, 3) if idx == 1 else 0.0,
            "context_write_ms": round(context_write_ms, 3),
            "ingest_ms": round(ingest_ms, 3),
            "search_ms": round(latency_ms, 3),
            "save_ms": 0.0,
            "delta_calls": after["calls"] - before["calls"],
            "delta_input_tokens": after["input_tokens"] - before["input_tokens"],
            "delta_output_tokens": after["output_tokens"] - before["output_tokens"],
            "delta_cost_usd": round(after["cost"] - before["cost"], 8),
            "faiss_candidate_count": retrieval.get("faiss_candidate_count"),
            "candidate_text_count": retrieval.get("candidate_text_count"),
            "reranker_enabled": retrieval.get("reranker_enabled"),
            "reranker_returned_count": retrieval.get("reranker_returned_count"),
            "reranker_fallback_used": retrieval.get("reranker_fallback_used"),
            "synthesis_input_token_budget": retrieval.get("synthesis_input_token_budget"),
            "synthesis_source_truncated": retrieval.get("synthesis_source_truncated"),
            "synthesis_estimated_input_tokens_before": retrieval.get("synthesis_estimated_input_tokens_before"),
            "synthesis_estimated_input_tokens_after": retrieval.get("synthesis_estimated_input_tokens_after"),
            "synthesis_packed_chunk_count": retrieval.get("synthesis_packed_chunk_count"),
            "synthesis_dropped_chunk_count": retrieval.get("synthesis_dropped_chunk_count"),
            "synthesis_selected_chunk_indices": retrieval.get("synthesis_selected_chunk_indices"),
            "iterative_scan_total_chunks": retrieval.get("iterative_scan_total_chunks"),
            "iterative_scan_early_stop_min_chunks": retrieval.get("iterative_scan_early_stop_min_chunks"),
            "iterative_scan_budget": retrieval.get("iterative_scan_budget"),
            "iterative_scan_empty_ledger_fallback_budget": retrieval.get("iterative_scan_empty_ledger_fallback_budget"),
            "iterative_scan_visited_chunk_count": retrieval.get("iterative_scan_visited_chunk_count"),
            "iterative_scan_faiss_top_n": retrieval.get("iterative_scan_faiss_top_n"),
            "iterative_scan_faiss_result_count": retrieval.get("iterative_scan_faiss_result_count"),
            "iterative_scan_empty_ledger_fallback_used": retrieval.get("iterative_scan_empty_ledger_fallback_used"),
            "iterative_scan_extra_scan_used": retrieval.get("iterative_scan_extra_scan_used"),
            "iterative_scan_extra_scan_reason": retrieval.get("iterative_scan_extra_scan_reason"),
            "iterative_scan_extra_scan_chunk_count": retrieval.get("iterative_scan_extra_scan_chunk_count"),
            "iterative_scan_packed_fallback_used": retrieval.get("iterative_scan_packed_fallback_used"),
            "iterative_scan_early_stop": retrieval.get("iterative_scan_early_stop"),
            "iterative_scan_stop_reason": retrieval.get("iterative_scan_stop_reason"),
            "iterative_scan_selected_chunk_indices": retrieval.get("iterative_scan_selected_chunk_indices"),
            "iterative_scan_supporting_chunk_indices": retrieval.get("iterative_scan_supporting_chunk_indices"),
            "iterative_scan_inspector_call_count": retrieval.get("iterative_scan_inspector_call_count"),
            "iterative_batching_enabled": retrieval.get("iterative_batching_enabled"),
            "iterative_scan_inspector_llm_call_count": retrieval.get("iterative_scan_inspector_llm_call_count"),
            "iterative_scan_batch_count": retrieval.get("iterative_scan_batch_count"),
            "iterative_scan_batch_sizes": retrieval.get("iterative_scan_batch_sizes"),
            "iterative_scan_batch_fallback_count": retrieval.get("iterative_scan_batch_fallback_count"),
            "iterative_scan_final_adjudication_call_count": retrieval.get("iterative_scan_final_adjudication_call_count"),
            "iterative_scan_packed_fallback_call_count": retrieval.get("iterative_scan_packed_fallback_call_count"),
            "iterative_scan_packed_fallback_reason": retrieval.get("iterative_scan_packed_fallback_reason"),
            "iterative_scan_final_answer": retrieval.get("iterative_scan_final_answer"),
            "iterative_scan_final_reason": retrieval.get("iterative_scan_final_reason"),
            "iterative_scan_final_raw_response": retrieval.get("iterative_scan_final_raw_response"),
            "iterative_scan_useful_memory_count": retrieval.get("iterative_scan_useful_memory_count"),
            "iterative_scan_memory_char_count": retrieval.get("iterative_scan_memory_char_count"),
            "iterative_scan_memory_update_count": retrieval.get("iterative_scan_memory_update_count"),
            "iterative_scan_target_fact_count": retrieval.get("iterative_scan_target_fact_count"),
            "iterative_scan_code_mapping_count": retrieval.get("iterative_scan_code_mapping_count"),
            "iterative_scan_open_question_count": retrieval.get("iterative_scan_open_question_count"),
            "iterative_scan_observation_count": retrieval.get("iterative_scan_observation_count"),
            "iterative_scan_rule_count": retrieval.get("iterative_scan_rule_count"),
            "iterative_scan_example_count": retrieval.get("iterative_scan_example_count"),
            "iterative_scan_parse_failure_count": retrieval.get("iterative_scan_parse_failure_count"),
            "iterative_scan_evidence_ledger": retrieval.get("iterative_scan_evidence_ledger"),
            "api_status": api_status,
            "api_error": api_error,
            **hybrid_telemetry,
        }
        bridge_rows.append(bridge_row)

        if (
            cache_state_enabled
            and cache_state_path is not None
            and idx % args.cache_save_interval == 0
        ):
            t_save = time.time()
            controller.save(cache_state_path)
            save_ms = (time.time() - t_save) * 1000.0
            bridge_row["save_ms"] = round(save_ms, 3)
            cache_save_count += 1
            cache_entries_after_run = controller.get_total_entries()
        bridge_row["row_wall_ms"] = round((time.time() - row_t0) * 1000.0, 3)

    if cache_state_enabled and cache_state_path is not None and cache_controller is not None:
        t_final_save = time.time()
        cache_controller.save(cache_state_path)
        cache_final_save_ms = (time.time() - t_final_save) * 1000.0
        cache_save_count += 1
        cache_entries_after_run = cache_controller.get_total_entries()

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
    hybrid_route_counts = dict(
        sorted(Counter(row.get("hybrid_route") for row in bridge_rows if row.get("hybrid_route")).items())
    )
    timing_summary = {
        "cache_load_ms": round(cache_load_ms, 3),
        "cache_final_save_ms": round(cache_final_save_ms, 3),
        "cache_save_count": cache_save_count,
        "context_write_ms": round(sum(float(row.get("context_write_ms") or 0.0) for row in bridge_rows), 3),
        "ingest_ms": round(sum(float(row.get("ingest_ms") or 0.0) for row in bridge_rows), 3),
        "search_ms": round(sum(float(row.get("search_ms") or 0.0) for row in bridge_rows), 3),
        "interval_save_ms": round(sum(float(row.get("save_ms") or 0.0) for row in bridge_rows), 3),
        "row_wall_ms": round(sum(float(row.get("row_wall_ms") or 0.0) for row in bridge_rows), 3),
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
        "llm_provider": args.llm_provider,
        "api_key_env": args.api_key_env,
        "openrouter_base_url": args.openrouter_base_url,
        "openai_compat_base_url": args.openai_compat_base_url,
        "openai_compat_executor_base_url": args.openai_compat_executor_base_url,
        "openai_compat_evaluator_base_url": args.openai_compat_evaluator_base_url,
        "openai_compat_api_key_env": args.openai_compat_api_key_env,
        "openai_compat_extra_body": openai_compatible_extra_body_config,
        "executor_model": args.executor_model,
        "evaluator_model": args.evaluator_model,
        **effective_fields,
        "cache_save_interval": args.cache_save_interval,
        "row_order": args.row_order,
        "row_types_requested": row_types,
        "source_ids_requested": source_ids,
        "max_rows": args.max_rows,
        "rows_selected": len(bridge_rows),
        "row_type_counts": row_type_counts,
        "answer_correct_count": correct_count,
        "answer_accuracy": round(correct_count / len(bridge_rows), 6) if bridge_rows else 0.0,
        "actual_route_counts": actual_route_counts,
        "by_row_type": summarize_rows(bridge_rows, "row_type"),
        "timing_summary": timing_summary,
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
        "full_context_query_baseline_input_tokens": totals["context_token_estimate"],
        "input_token_savings_vs_full_context_query_baseline": totals["context_token_estimate"] - totals["input_tokens"],
        "input_token_savings_percent_vs_full_context_query_baseline": pct_savings(
            totals["context_token_estimate"],
            totals["input_tokens"],
        ),
        "unique_source_context_token_estimate": totals["unique_source_context_token_estimate"],
        "input_token_savings_vs_unique_source_context": (
            totals["unique_source_context_token_estimate"] - totals["input_tokens"]
        ),
        "input_token_savings_percent_vs_unique_source_context": pct_savings(
            totals["unique_source_context_token_estimate"],
            totals["input_tokens"],
        ),
        "total_dataset_context_token_estimate": totals["context_token_estimate"],
        "input_token_savings_vs_context": totals["context_token_estimate"] - totals["input_tokens"],
        "input_token_savings_percent": pct_savings(
            totals["context_token_estimate"],
            totals["input_tokens"],
        ),
        "total_estimated_cost_usd": round(totals["cost"], 8),
    }
    if hybrid_enabled:
        cache_state_bytes = 0
        if cache_state_path and cache_state_path.exists():
            cache_state_bytes = sum(
                path.stat().st_size for path in cache_state_path.rglob("*") if path.is_file()
            )
        api_error_count = sum(1 for row in bridge_rows if row.get("api_status") == "error")
        context_length_error_count = sum(
            1
            for row in bridge_rows
            if "context" in str(row.get("api_error") or "").lower()
            and (
                "length" in str(row.get("api_error") or "").lower()
                or "token" in str(row.get("api_error") or "").lower()
            )
        )
        manifest.update(
            {
                "hybrid_policy": effective_config["hybrid_policy"],
                "context_window_tokens": args.context_window_tokens,
                "max_input_tokens": args.max_input_tokens,
                "max_output_tokens": args.max_output_tokens,
                "context_window_safety_margin_tokens": (
                    args.context_window_tokens - args.max_input_tokens - args.max_output_tokens
                ),
                "child_tokens": args.child_tokens,
                "child_overlap_tokens": args.child_overlap_tokens,
                "hybrid_route_counts": hybrid_route_counts,
                "by_hybrid_route": summarize_rows(bridge_rows, "hybrid_route"),
                "valid_choice_count": sum(bool(row.get("valid_choice")) for row in bridge_rows),
                "invalid_choice_count": sum(
                    row.get("valid_choice") is False for row in bridge_rows
                ),
                "api_error_count": api_error_count,
                "context_length_error_count": context_length_error_count,
                "executor_answer_calls": sum(
                    int(row.get("executor_answer_calls") or 0) for row in bridge_rows
                ),
                "semantic_verifier_calls": sum(
                    int(row.get("semantic_verifier_calls") or 0) for row in bridge_rows
                ),
                "document_embedding_count": sum(
                    int(row.get("document_embedding_count") or 0) for row in bridge_rows
                ),
                "document_embedding_ms": round(
                    sum(float(row.get("document_embedding_ms") or 0.0) for row in bridge_rows),
                    3,
                ),
                "full_rendered_input_tokens_on_misses": sum(
                    int(row.get("full_rendered_input_tokens") or 0) for row in bridge_rows
                ),
                "final_rendered_input_tokens_on_misses": sum(
                    int(row.get("final_rendered_input_tokens") or 0) for row in bridge_rows
                ),
                "cache_state_bytes": cache_state_bytes,
            }
        )
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
            f"entries_after={cache_entries_after_run}, "
            f"saves={cache_save_count}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LongBench-v2 retrieval/cache benchmark runner")
    parser.add_argument("--suite-csv", type=Path, default=DEFAULT_SUITE_CSV)
    parser.add_argument("--source-json-path", type=Path, default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--mode", choices=["baseline", "cache"], default="cache")
    parser.add_argument("--cache-reset", action="store_true")
    parser.add_argument("--cache-state-root", type=Path, default=None)
    parser.add_argument("--row-types", type=str, default=DEFAULT_ROW_TYPES)
    parser.add_argument(
        "--source-ids",
        type=str,
        default="",
        help="Optional comma-separated source IDs to run, preserving suite order.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Cap selected rows after filtering (0 means all)")
    parser.add_argument("--llm-provider", choices=["anthropic", "openrouter", "openai_compatible"], default="anthropic")
    parser.add_argument("--api-key-env", type=str, default=None)
    parser.add_argument("--executor-model", type=str, default="claude-sonnet-4-5")
    parser.add_argument("--evaluator-model", type=str, default="claude-haiku-4-5")
    parser.add_argument("--openrouter-base-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--openai-compat-base-url", type=str, default=os.getenv("OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--openai-compat-executor-base-url", type=str, default=os.getenv("OPENAI_COMPAT_EXECUTOR_BASE_URL", ""))
    parser.add_argument("--openai-compat-evaluator-base-url", type=str, default=os.getenv("OPENAI_COMPAT_EVALUATOR_BASE_URL", ""))
    parser.add_argument("--openai-compat-api-key-env", type=str, default=os.getenv("OPENAI_COMPAT_API_KEY_ENV", ""))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rerank-top", type=int, default=DEFAULT_RERANK_TOP)
    parser.add_argument("--synthesis-max-chunks", type=int, default=DEFAULT_SYNTHESIS_MAX_CHUNKS)
    parser.add_argument(
        "--context-window-tokens",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW_TOKENS,
        help="Served model context limit for hybrid routing.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help="Maximum rendered hybrid request input.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Strict MCQ output-token cap for hybrid executor calls.",
    )
    parser.add_argument("--child-tokens", type=int, default=DEFAULT_CHILD_TOKENS)
    parser.add_argument(
        "--child-overlap-tokens",
        type=int,
        default=DEFAULT_CHILD_OVERLAP_TOKENS,
    )
    parser.add_argument(
        "--route-audit-only",
        action="store_true",
        help="Write exact rendered-token route classifications without loading cache models or calling APIs.",
    )
    parser.add_argument(
        "--row-order",
        choices=["input", "source_grouped"],
        default=DEFAULT_ROW_ORDER,
        help="Row execution order. source_grouped keeps rows with the same source_id adjacent.",
    )
    parser.add_argument(
        "--cache-save-interval",
        type=int,
        default=DEFAULT_CACHE_SAVE_INTERVAL,
        help="Save cache state every N rows in cache mode, plus a final save. Use 1 for per-row saves.",
    )
    parser.add_argument("--disable-reranker", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_artifacts"))
    parser.add_argument("--manifest-note", type=str, default="")
    return parser


def main() -> None:
    start = time.time()
    parser = build_arg_parser()
    args = parser.parse_args()
    args = normalize_llm_args(args)
    run_longbench_benchmark(args)
    print(f"\nTotal elapsed time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
