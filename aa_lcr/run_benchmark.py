"""Run one of the four AA-LCR direct/hybrid benchmark cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aa_lcr.api import Completion, call_chat_completion, call_with_retries  # noqa: E402
from aa_lcr.dataset import (  # noqa: E402
    ARCHIVE_SHA256,
    DATASET_RELEASES,
    DEFAULT_DATASET_MANIFEST,
    DEFAULT_DOCUMENTS_ROOT,
    DEFAULT_QUESTIONS_CSV,
    Question,
    build_scope_hash,
    dataset_signature,
    document_paths,
    group_questions,
    load_dataset_manifest,
    load_documents,
    load_questions,
    sha256_file,
    validate_dataset,
)
from aa_lcr.grading import Grader, add_grader_arguments  # noqa: E402
from aa_lcr.prompting import (  # noqa: E402
    build_messages,
    chat_token_count,
    non_thinking_extra_body,
    pack_retrieved_children,
    prepare_direct_messages,
    prompt_contract_metadata,
)


DEFAULT_EXECUTOR_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_EVALUATOR_MODEL = "Qwen/Qwen3.5-35B-A3B"
DEFAULT_EXECUTOR_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_EVALUATOR_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_MAX_OUTPUT_TOKENS = 16384
DEFAULT_CHILD_TOKENS = 7500
DEFAULT_CHILD_OVERLAP_TOKENS = 750
REPORT_FILENAME = "aa_lcr_equality_eval_report.json"


@dataclass(frozen=True)
class Experiment:
    name: str
    mode: str
    context_window_tokens: int
    max_input_tokens: int
    allow_direct_truncation: bool


EXPERIMENTS = {
    "direct_262k": Experiment("direct_262k", "direct", 262144, 240000, False),
    "hybrid_262k": Experiment("hybrid_262k", "hybrid", 262144, 240000, False),
    "direct_64k": Experiment("direct_64k", "direct", 65536, 60000, True),
    "hybrid_64k": Experiment("hybrid_64k", "hybrid", 65536, 60000, False),
}


def _load_tokenizer(model: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Transformers is required for AA-LCR token accounting"
        ) from exc
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def _import_semantic_cache_system():
    import semantic_cache_system as scs

    return scs


def _csv_cell(value: object) -> object:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})


def _parse_values(raw: str) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def select_questions(
    questions: list[Question],
    *,
    question_ids: list[str],
    document_set_ids: list[str],
    max_rows: int,
) -> list[Question]:
    question_filter = set(question_ids)
    set_filter = set(document_set_ids)
    selected = [
        question
        for question in questions
        if (not question_filter or question.question_id in question_filter)
        and (not set_filter or question.document_set_id in set_filter)
    ]
    selected = [question for group in group_questions(selected) for question in group]
    if max_rows > 0:
        selected = selected[:max_rows]
    return selected


def _tokenizer_metadata(tokenizer: Any, model: str) -> dict:
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    if not isinstance(chat_template, str):
        chat_template = json.dumps(chat_template, sort_keys=True, default=str)
    return {
        "executor_tokenizer_model": model,
        "executor_tokenizer_name_or_path": str(
            getattr(tokenizer, "name_or_path", model) or model
        ),
        "executor_tokenizer_class": type(tokenizer).__name__,
        "executor_chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
    }


def _snapshot_metrics(metrics: Any) -> dict[str, float]:
    totals = metrics.get_totals() if hasattr(metrics, "get_totals") else {}
    return {
        "calls": float(totals.get("calls", getattr(metrics, "calls", 0)) or 0),
        "input_tokens": float(
            totals.get("input_tokens", getattr(metrics, "input_tokens", 0)) or 0
        ),
        "output_tokens": float(
            totals.get("output_tokens", getattr(metrics, "output_tokens", 0)) or 0
        ),
    }


def _validate_dataset_manifest(
    manifest: dict,
    questions_csv: Path,
    questions: list[Question],
    documents_root: Path,
) -> dict:
    validation = validate_dataset(questions, documents_root)
    actual_questions_hash = sha256_file(questions_csv)
    recorded_questions_hash = str(manifest.get("questions_sha256") or "")
    if recorded_questions_hash and recorded_questions_hash != actual_questions_hash:
        raise ValueError("Question CSV hash does not match dataset_manifest.json")
    archive_path = Path(str(manifest.get("archive_path") or ""))
    recorded_archive_hash = str(manifest.get("archive_sha256") or "")
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Dataset archive recorded in the manifest is missing: {archive_path}"
        )
    actual_archive_hash = sha256_file(archive_path)
    if recorded_archive_hash and recorded_archive_hash != actual_archive_hash:
        raise ValueError(
            "Extracted-text archive hash does not match dataset_manifest.json"
        )
    if int(manifest.get("question_count") or 0) not in {0, len(questions)}:
        raise ValueError("Question count does not match dataset_manifest.json")
    dataset_version = "custom"
    for version, release in DATASET_RELEASES.items():
        if manifest.get("dataset_revision") == release["revision"]:
            if (
                actual_questions_hash != release["questions_sha256"]
                or actual_archive_hash != ARCHIVE_SHA256
            ):
                raise ValueError("Dataset files do not match the pinned release")
            dataset_version = version
            break
    if manifest.get("dataset_version", dataset_version) != dataset_version:
        raise ValueError("Dataset version does not match its revision")
    return {
        **validation,
        "dataset_version": dataset_version,
        "questions_sha256": actual_questions_hash,
        "archive_sha256": actual_archive_hash,
        "archive_path": str(archive_path),
    }


def preflight_questions(
    questions: list[Question],
    documents_root: Path,
    tokenizer: Any,
    experiment: Experiment,
) -> dict:
    document_cache: dict[str, list[tuple[str, str]]] = {}
    rows = []
    for question in questions:
        if question.document_set_id not in document_cache:
            document_cache[question.document_set_id] = load_documents(
                question, documents_root
            )
        documents = document_cache[question.document_set_id]
        messages = build_messages([text for _, text in documents], question.question)
        tokens = chat_token_count(tokenizer, messages)
        if (
            experiment.mode == "direct"
            and not experiment.allow_direct_truncation
            and tokens > experiment.max_input_tokens
        ):
            raise ValueError(
                f"{question.case_id} uses {tokens} tokens, exceeding the "
                f"{experiment.max_input_tokens}-token direct_262k budget"
            )
        final_tokens: int | None = (
            tokens if tokens <= experiment.max_input_tokens else None
        )
        prompt_truncated = False
        if experiment.mode == "direct" and tokens > experiment.max_input_tokens:
            prepared_messages, truncation = prepare_direct_messages(
                [text for _, text in documents],
                question.question,
                tokenizer,
                experiment.max_input_tokens,
                allow_truncation=experiment.allow_direct_truncation,
            )
            final_tokens = truncation["prompt_tokens_after_truncation"]
            prompt_truncated = bool(truncation["prompt_truncated"])
            if question.question not in prepared_messages[0]["content"]:
                raise ValueError(
                    f"Direct truncation removed the question for {question.case_id}"
                )
        rows.append(
            {
                "case_id": question.case_id,
                "question_id": question.question_id,
                "document_set_id": question.document_set_id,
                "document_category": question.document_category,
                "full_rendered_input_tokens": tokens,
                "final_rendered_input_tokens": final_tokens,
                "prompt_truncated": prompt_truncated,
                "route": (
                    "direct_fit"
                    if tokens <= experiment.max_input_tokens
                    else (
                        "middle_truncated"
                        if experiment.mode == "direct"
                        else "dense_child_packed"
                    )
                ),
            }
        )
    token_counts = sorted(row["full_rendered_input_tokens"] for row in rows)
    return {
        "experiment": experiment.name,
        "rows": rows,
        "row_count": len(rows),
        "route_counts": dict(sorted(Counter(row["route"] for row in rows).items())),
        "rendered_input_tokens": {
            "min": token_counts[0],
            "median": token_counts[(len(token_counts) - 1) // 2],
            "max": token_counts[-1],
        },
    }


def _call_model(
    completion_caller: Callable[..., Completion],
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    extra_body: dict,
    api_key: str,
    timeout_seconds: int,
    max_retries: int,
) -> tuple[Completion, int]:
    return call_with_retries(
        lambda: completion_caller(
            base_url=base_url,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body=extra_body,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        ),
        max_retries=max_retries,
    )


def _summarize(rows: list[dict], field: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(field) or "unlabeled"), []).append(row)
    result = {}
    for label, bucket in sorted(grouped.items()):
        valid = [row for row in bucket if row.get("grade_valid")]
        correct = sum(row.get("answer_correct") is True for row in valid)
        cached = sum(bool(row.get("from_cache")) for row in bucket)
        result[label] = {
            "rows": len(bucket),
            "valid_grades": len(valid),
            "accuracy": round(correct / len(valid), 6) if valid else None,
            "cache_hit_rate": round(cached / len(bucket), 6) if bucket else 0.0,
        }
    return result


def _build_report(run_dir: Path, rows: list[dict], manifest: dict) -> dict:
    return {
        "run_dir": str(run_dir),
        "benchmark_target": "aa_lcr_reasoning",
        "experiment": manifest["experiment"],
        "dataset_version": manifest["dataset_version"],
        "dataset_revision": manifest["dataset_revision"],
        "evaluator_model": manifest["evaluator_model"],
        "grader_prompt_version": manifest["grader_prompt_version"],
        "max_output_tokens": manifest["max_output_tokens"],
        "grader_api_style": manifest["grader_api_style"],
        "grader_reasoning_effort": manifest["grader_reasoning_effort"],
        "output_length_terminated_count": manifest["output_length_terminated_count"],
        "grader_length_terminated_count": manifest["grader_length_terminated_count"],
        "executor_finish_reason_counts": manifest["executor_finish_reason_counts"],
        "scored_rows": len(rows),
        "answer_accuracy": manifest["answer_accuracy"],
        "valid_grade_accuracy": manifest["valid_grade_accuracy"],
        "invalid_grade_count": manifest["invalid_grade_count"],
        "api_error_count": manifest["api_error_count"],
        "by_document_category": manifest["by_document_category"],
        "by_route": manifest["by_route"],
        "incorrect_rows": [
            {
                "case_id": row["case_id"],
                "question_id": row["question_id"],
                "document_set_id": row["document_set_id"],
                "official_answer": row["official_answer"],
                "candidate_answer": row["candidate_answer"],
                "grade": row["grade"],
            }
            for row in rows
            if row.get("grade") == "INCORRECT"
        ],
        "invalid_grade_rows": [
            {
                "case_id": row["case_id"],
                "executor_status": row["executor_status"],
                "executor_error": row["executor_error"],
                "grader_status": row["grader_status"],
                "grader_error": row["grader_error"],
                "grader_raw_output": row["grader_raw_output"],
            }
            for row in rows
            if not row.get("grade_valid")
        ],
        "semantic_hit_audit": [
            {
                "case_id": row["case_id"],
                "document_set_id": row["document_set_id"],
                "question": row["question"],
                "candidate_answer": row["candidate_answer"],
                "answer_correct": row["answer_correct"],
                "cache_provenance": row["cache_provenance"],
            }
            for row in rows
            if row.get("cache_type") == "semantic"
        ],
    }


def run_benchmark(
    args: argparse.Namespace,
    *,
    tokenizer_factory: Callable[[str], Any] | None = None,
    completion_caller: Callable[..., Completion] = call_chat_completion,
    scs_module: Any | None = None,
) -> Path | None:
    experiment = EXPERIMENTS[args.experiment]
    if args.max_output_tokens is None:
        args.max_output_tokens = (
            DEFAULT_MAX_OUTPUT_TOKENS
            if experiment.context_window_tokens == 262144
            else 512
        )
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive")
    if args.repeat_id < 1:
        raise ValueError("--repeat-id must be positive")
    if experiment.mode == "hybrid" and args.grader_api_style != "vllm":
        raise ValueError(
            "Hybrid cache verification requires a local vLLM evaluator; "
            "regrade saved answers for hosted grading"
        )
    if (
        experiment.max_input_tokens + args.max_output_tokens
        > experiment.context_window_tokens
    ):
        raise ValueError(
            "Input and output budgets exceed the experiment context window"
        )
    if args.child_tokens <= 0:
        raise ValueError("--child-tokens must be positive")
    if args.child_overlap_tokens < 0 or args.child_overlap_tokens >= args.child_tokens:
        raise ValueError("--child-overlap-tokens must be smaller than --child-tokens")
    if experiment.mode == "hybrid" and (
        args.child_tokens != DEFAULT_CHILD_TOKENS
        or args.child_overlap_tokens != DEFAULT_CHILD_OVERLAP_TOKENS
    ):
        raise ValueError(
            "AA-LCR hybrid experiments require 7500-token children with 750-token overlap"
        )

    questions_csv = Path(args.questions_csv)
    documents_root = Path(args.documents_root)
    dataset_manifest_path = Path(args.dataset_manifest)
    questions = load_questions(questions_csv)
    dataset_manifest = load_dataset_manifest(dataset_manifest_path)
    dataset_info = _validate_dataset_manifest(
        dataset_manifest,
        questions_csv,
        questions,
        documents_root,
    )
    selected = select_questions(
        questions,
        question_ids=_parse_values(args.question_ids),
        document_set_ids=_parse_values(args.document_set_ids),
        max_rows=args.max_rows,
    )
    if not selected:
        raise ValueError("No AA-LCR questions matched the requested filters")

    tokenizer = (tokenizer_factory or _load_tokenizer)(args.executor_model)
    preflight = preflight_questions(selected, documents_root, tokenizer, experiment)
    if args.preflight_only:
        payload = {
            **preflight,
            "dataset_version": dataset_info["dataset_version"],
            "dataset_revision": str(dataset_manifest.get("dataset_revision") or ""),
            **_tokenizer_metadata(tokenizer, args.executor_model),
            "context_window_tokens": experiment.context_window_tokens,
            "max_input_tokens": experiment.max_input_tokens,
            "max_output_tokens": args.max_output_tokens,
        }
        rendered = json.dumps(payload, indent=2)
        if args.preflight_output:
            output = Path(args.preflight_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
        print(rendered)
        return None

    grader = Grader(
        args,
        tokenizer_factory=tokenizer_factory or _load_tokenizer,
        completion_caller=completion_caller,
    )
    serving_metadata = {}
    if args.serving_metadata:
        serving_metadata = json.loads(Path(args.serving_metadata).read_text())
        if not isinstance(serving_metadata, dict):
            raise ValueError("--serving-metadata must contain a JSON object")
    started_at = datetime.now(timezone.utc)
    run_id = args.run_id or (
        f"v{dataset_info['dataset_version']}_out{args.max_output_tokens}_"
        f"r{args.repeat_id}_{started_at.strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    run_dir = Path(args.output_root) / experiment.name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = run_dir / "predictions.jsonl"
    bridge_path = run_dir / "bridge_rows.jsonl"
    bridge_csv_path = run_dir / "bridge_rows.csv"
    manifest_path = run_dir / "manifest.json"
    report_path = run_dir / REPORT_FILENAME

    print(f"[AA-LCR] Experiment: {experiment.name}")
    print(f"[AA-LCR] Questions: {len(selected)}")
    print(f"[AA-LCR] Output: {run_dir}")

    api_key = os.getenv(args.api_key_env, "") if args.api_key_env else ""
    document_cache: dict[str, list[tuple[str, str]]] = {}
    controller = None
    active_ingested_set = ""
    cache_state_path = run_dir / "cache_state"
    if experiment.mode == "hybrid":
        scs = scs_module or _import_semantic_cache_system()
        scs.configure_llm_provider(
            provider="openai_compatible",
            executor_model=args.executor_model,
            evaluator_model=args.evaluator_model,
            openai_compat_executor_base_url=args.executor_base_url,
            openai_compat_evaluator_base_url=args.evaluator_base_url,
            openai_compat_api_key_env=args.api_key_env,
        )
        scs.DOCUMENT_CHUNK_TOKENS = args.child_tokens
        scs.DOCUMENT_CHUNK_OVERLAP_TOKENS = args.child_overlap_tokens
        scs.DOCUMENT_CHUNK_TOKENIZER_MODEL = str(getattr(scs, "EMBEDDING_MODEL", ""))
        embedder = scs.EmbeddingEngine()
        if args.child_tokens >= int(getattr(scs, "EMBEDDING_MAX_LENGTH", 8192)):
            raise ValueError("--child-tokens must be below the embedding input limit")
        controller = scs.SemanticCacheController(
            metrics=scs.ExecutionMetrics(),
            embedder=embedder,
            reranker=None,
            corpus_id=f"aa_lcr_{run_id}",
            corpus_domain="aa_lcr",
        )

    prediction_rows: list[dict] = []
    bridge_rows: list[dict] = []
    for index, question in enumerate(selected, start=1):
        row_started = time.time()
        print(f"[AA-LCR] {index}/{len(selected)} {question.case_id}")
        if question.document_set_id not in document_cache:
            document_cache[question.document_set_id] = load_documents(
                question, documents_root
            )
        documents = document_cache[question.document_set_id]
        scope_hash = build_scope_hash(documents)
        document_texts = [text for _, text in documents]
        full_messages = build_messages(document_texts, question.question)
        full_tokens = chat_token_count(tokenizer, full_messages)

        route = "direct"
        from_cache = False
        cache_type = "miss"
        cache_provenance: dict = {}
        selected_evidence_ranges: list[dict] = []
        final_tokens = full_tokens
        prompt_truncated = False
        prompt_tokens_removed = 0
        ingested_chunks = 0
        ingest_ms = 0.0
        document_embedding_ms = 0.0
        faiss_search_ms = 0.0
        packing_ms = 0.0
        semantic_verifier_calls = 0
        verifier_input_tokens = 0
        verifier_output_tokens = 0
        executor_status = "ok"
        executor_error = ""
        executor_attempts = 0
        executor_usage = Completion("", 0, 0, {})
        candidate_answer = ""

        try:
            if experiment.mode == "direct":
                request_messages, truncation = prepare_direct_messages(
                    document_texts,
                    question.question,
                    tokenizer,
                    experiment.max_input_tokens,
                    allow_truncation=experiment.allow_direct_truncation,
                )
                prompt_truncated = truncation["prompt_truncated"]
                final_tokens = truncation["prompt_tokens_after_truncation"]
                prompt_tokens_removed = truncation["prompt_tokens_removed"]
                route = "middle_truncated" if prompt_truncated else "direct_fit"
            else:
                assert controller is not None
                controller.activate_data_scope(scope_hash)
                before = _snapshot_metrics(controller.metrics)
                cached = controller.lookup_cached_result(question.question)
                after = _snapshot_metrics(controller.metrics)
                semantic_verifier_calls = int(after["calls"] - before["calls"])
                verifier_input_tokens = int(
                    after["input_tokens"] - before["input_tokens"]
                )
                verifier_output_tokens = int(
                    after["output_tokens"] - before["output_tokens"]
                )
                if cached is not None:
                    from_cache = True
                    cache_type = str(cached.get("cache_type") or "unknown")
                    cache_provenance = cached.get("cache_provenance") or {}
                    route = f"{cache_type}_cache"
                    candidate_answer = str(cached.get("answer") or "").strip()
                    request_messages = []
                    final_tokens = 0
                else:
                    controller.metrics.cache_misses += 1
                    if full_tokens <= experiment.max_input_tokens:
                        route = "direct_fit"
                        request_messages = full_messages
                    else:
                        route = "dense_child_packed"
                        if active_ingested_set != question.document_set_id:
                            ingest_started = time.time()
                            ingested_chunks = controller.ingest(
                                document_paths(question, documents_root)[0].parent,
                                data_scope_hash=scope_hash,
                                source_id=question.document_set_id,
                                ordered_filenames=list(question.data_source_filenames),
                            )
                            ingest_ms = (time.time() - ingest_started) * 1000.0
                            active_ingested_set = question.document_set_id
                            ingest_info = (
                                getattr(controller, "_last_ingest_info", {}) or {}
                            )
                            document_embedding_ms = float(
                                ingest_info.get("embedding_ms") or 0.0
                            )
                            encoded_max = int(
                                ingest_info.get("child_encoded_length_max") or 0
                            )
                            if encoded_max >= int(
                                getattr(scs, "EMBEDDING_MAX_LENGTH", 8192)
                            ):
                                raise RuntimeError(
                                    "A child reached the embedding input limit; reduce --child-tokens"
                                )
                        search_started = time.time()
                        results = controller.retrieve(
                            question.question,
                            top_k=int(getattr(controller.doc_index, "total", 0) or 0),
                            rerank_top=0,
                            use_reranker=False,
                            query_embedding=getattr(
                                controller, "_last_cache_query_embedding", None
                            ),
                        )
                        faiss_search_ms = (time.time() - search_started) * 1000.0
                        packing_started = time.time()
                        request_messages, packing = pack_retrieved_children(
                            tokenizer,
                            question.question,
                            results,
                            experiment.max_input_tokens,
                        )
                        packing_ms = (time.time() - packing_started) * 1000.0
                        final_tokens = packing["rendered_input_tokens"]
                        selected_evidence_ranges = packing["selected_evidence_ranges"]

            if not from_cache:
                executor_usage, executor_attempts = _call_model(
                    completion_caller,
                    base_url=args.executor_base_url,
                    model=args.executor_model,
                    messages=request_messages,
                    max_tokens=args.max_output_tokens,
                    extra_body=non_thinking_extra_body(),
                    api_key=api_key,
                    timeout_seconds=args.request_timeout_seconds,
                    max_retries=args.max_retries,
                )
                candidate_answer = executor_usage.text
                if experiment.mode == "hybrid" and candidate_answer:
                    assert controller is not None
                    controller.store_compact_answer(
                        question.question,
                        candidate_answer,
                        model_used=args.executor_model,
                        source_id=question.document_set_id,
                        route=route,
                        provenance={
                            "source_scope_hash": scope_hash,
                            "full_rendered_input_tokens": full_tokens,
                            "final_rendered_input_tokens": final_tokens,
                            "selected_evidence_ranges": selected_evidence_ranges,
                        },
                    )
        except Exception as exc:
            executor_attempts = int(getattr(exc, "attempts", executor_attempts) or 0)
            executor_status = "error"
            executor_error = f"{type(exc).__name__}: {exc}"
            if args.fail_fast:
                raise

        grader_status = "not_run"
        grader_error = ""
        grader_attempts = 0
        grader_usage = Completion("", 0, 0, {})
        grade = ""
        if executor_status == "ok":
            try:
                grader_usage, grader_attempts, grade = grader.grade(
                    question.question, question.answer, candidate_answer
                )
                grader_status = "ok" if grade else "invalid_output"
                if not grade:
                    grader_error = (
                        "Evaluator returned an invalid verdict or an incomplete response"
                    )
            except Exception as exc:
                grader_attempts = int(getattr(exc, "attempts", grader_attempts) or 0)
                grader_status = "error"
                grader_error = f"{type(exc).__name__}: {exc}"
                if args.fail_fast:
                    raise

        grade_valid = bool(grade)
        answer_correct = grade == "CORRECT" if grade_valid else None
        delta_calls = (
            (1 if executor_status == "ok" and not from_cache else 0)
            + semantic_verifier_calls
            + (1 if grader_status in {"ok", "invalid_output"} else 0)
        )
        prediction = {
            "id": question.case_id,
            "sample_id": question.case_id,
            "case_id": question.case_id,
            "question_id": question.question_id,
            "document_set_id": question.document_set_id,
            "document_category": question.document_category,
            "question": question.question,
            "generation": candidate_answer,
            "candidate_answer": candidate_answer,
            "answer": question.answer,
            "official_answer": question.answer,
            "grade": grade,
            "data_source_filenames": list(question.data_source_filenames),
            "source_scope_hash": scope_hash,
            "executor_status": executor_status,
            "executor_finish_reason": executor_usage.finish_reason,
            "executor_output_tokens": executor_usage.output_tokens,
            "max_output_tokens": args.max_output_tokens,
        }
        prediction_rows.append(prediction)
        bridge_rows.append(
            {
                **prediction,
                "reported_input_tokens": question.input_tokens,
                "mode": experiment.mode,
                "experiment": experiment.name,
                "route": route,
                "from_cache": from_cache,
                "cache_type": cache_type,
                "cache_provenance": cache_provenance,
                "answer_correct": answer_correct,
                "grade_valid": grade_valid,
                "grader_raw_output": grader_usage.text,
                "grader_finish_reason": grader_usage.finish_reason,
                "executor_error": executor_error,
                "executor_attempts": executor_attempts,
                "grader_status": grader_status,
                "grader_error": grader_error,
                "grader_attempts": grader_attempts,
                "full_rendered_input_tokens": full_tokens,
                "final_rendered_input_tokens": final_tokens,
                "prompt_truncated": prompt_truncated,
                "prompt_tokens_removed": prompt_tokens_removed,
                "context_window_tokens": experiment.context_window_tokens,
                "max_input_tokens": experiment.max_input_tokens,
                "selected_evidence_ranges": selected_evidence_ranges,
                "ingested_chunks": ingested_chunks,
                "ingest_ms": round(ingest_ms, 3),
                "document_embedding_ms": round(document_embedding_ms, 3),
                "faiss_search_ms": round(faiss_search_ms, 3),
                "packing_ms": round(packing_ms, 3),
                "semantic_verifier_calls": semantic_verifier_calls,
                "executor_calls": 0 if from_cache or executor_status != "ok" else 1,
                "grader_calls": 1 if grader_status in {"ok", "invalid_output"} else 0,
                "delta_calls": delta_calls,
                "delta_input_tokens": (
                    verifier_input_tokens
                    + executor_usage.input_tokens
                    + grader_usage.input_tokens
                ),
                "delta_output_tokens": (
                    verifier_output_tokens
                    + executor_usage.output_tokens
                    + grader_usage.output_tokens
                ),
                "request_attempts": (
                    executor_attempts + grader_attempts + semantic_verifier_calls
                ),
                "executor_usage": executor_usage.raw_usage,
                "grader_usage": grader_usage.raw_usage,
                "latency_ms": round((time.time() - row_started) * 1000.0, 3),
            }
        )

    if controller is not None:
        controller.save(cache_state_path)

    _write_jsonl(predictions_path, prediction_rows)
    _write_jsonl(bridge_path, bridge_rows)
    _write_csv(bridge_csv_path, bridge_rows)

    finished_at = datetime.now(timezone.utc)
    valid_rows = [row for row in bridge_rows if row["grade_valid"]]
    correct_count = sum(row["answer_correct"] is True for row in valid_rows)
    invalid_grade_count = len(bridge_rows) - len(valid_rows)
    api_error_count = sum(
        row["executor_status"] == "error" or row["grader_status"] == "error"
        for row in bridge_rows
    )
    route_counts = dict(sorted(Counter(row["route"] for row in bridge_rows).items()))
    cache_hit_count = sum(bool(row["from_cache"]) for row in bridge_rows)
    cache_hit_rows = [
        row for row in bridge_rows if row["from_cache"] and row["grade_valid"]
    ]
    cache_hit_correct = sum(row["answer_correct"] is True for row in cache_hit_rows)
    over_budget_count = sum(
        row["full_rendered_input_tokens"] > experiment.max_input_tokens
        for row in bridge_rows
    )
    over_budget_final_count = sum(
        int(row["final_rendered_input_tokens"] or 0) > experiment.max_input_tokens
        for row in bridge_rows
    )
    full_run = len(selected) == 100 and len(questions) == 100
    accepted_full_run = bool(
        full_run
        and api_error_count == 0
        and invalid_grade_count == 0
        and over_budget_final_count == 0
        and (experiment.context_window_tokens != 262144 or over_budget_count == 0)
    )
    archive_sha = dataset_info["archive_sha256"]
    signature = dataset_signature(
        dataset_info["questions_sha256"], archive_sha, questions
    )
    total_input_tokens = sum(int(row["delta_input_tokens"]) for row in bridge_rows)
    total_output_tokens = sum(int(row["delta_output_tokens"]) for row in bridge_rows)
    manifest = {
        "run_id": run_id,
        "artifact_schema_version": 2,
        "repeat_id": args.repeat_id,
        "executor_serving_metadata": serving_metadata,
        "executor_serving_metadata_sha256": (
            sha256_file(Path(args.serving_metadata)) if args.serving_metadata else None
        ),
        "created_at": finished_at.isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round((finished_at - started_at).total_seconds(), 3),
        "benchmark_target": "aa_lcr_reasoning",
        "experiment": experiment.name,
        "mode": experiment.mode,
        "context_window_tokens": experiment.context_window_tokens,
        "max_input_tokens": experiment.max_input_tokens,
        "max_output_tokens": args.max_output_tokens,
        "context_window_safety_margin_tokens": (
            experiment.context_window_tokens
            - experiment.max_input_tokens
            - args.max_output_tokens
        ),
        "direct_truncation_policy": (
            "middle_keep_first_last"
            if experiment.allow_direct_truncation
            else "disabled"
        ),
        "executor_model": args.executor_model,
        "evaluator_model": args.evaluator_model,
        "executor_base_url": args.executor_base_url,
        "evaluator_base_url": args.evaluator_base_url,
        "temperature": 0,
        "thinking_enabled": False,
        "grader_method": "llm_equality_checker",
        "benchmark_label_field": "answer",
        "benchmark_label_type": "open_answer_string",
        "grader_labels": ["CORRECT", "INCORRECT"],
        **prompt_contract_metadata(args.grader_prompt_version),
        **grader.metadata(),
        **_tokenizer_metadata(tokenizer, args.executor_model),
        "questions_csv": str(questions_csv),
        "questions_sha256": dataset_info["questions_sha256"],
        "documents_root": str(documents_root),
        "archive_path": dataset_info["archive_path"],
        "archive_sha256": archive_sha,
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_version": dataset_info["dataset_version"],
        "dataset_revision": str(dataset_manifest.get("dataset_revision") or ""),
        "dataset_signature": signature,
        "document_order_policy": "data_source_filenames_csv_order",
        "dataset_question_count": len(questions),
        "dataset_document_set_count": dataset_info["document_set_count"],
        "dataset_referenced_document_count": dataset_info["referenced_document_count"],
        "question_ids": [question.question_id for question in selected],
        "max_rows": args.max_rows,
        "rows_selected": len(bridge_rows),
        "full_run": full_run,
        "accepted_full_run": accepted_full_run,
        "valid_grade_count": len(valid_rows),
        "invalid_grade_count": invalid_grade_count,
        "answer_correct_count": correct_count,
        "answer_accuracy": (
            round(correct_count / len(bridge_rows), 6)
            if bridge_rows and invalid_grade_count == 0
            else None
        ),
        "valid_grade_accuracy": (
            round(correct_count / len(valid_rows), 6) if valid_rows else None
        ),
        "api_error_count": api_error_count,
        "route_counts": route_counts,
        "by_document_category": _summarize(bridge_rows, "document_category"),
        "by_route": _summarize(bridge_rows, "route"),
        "timing_summary_ms": {
            "row_latency": round(
                sum(float(row["latency_ms"]) for row in bridge_rows), 3
            ),
            "ingest": round(sum(float(row["ingest_ms"]) for row in bridge_rows), 3),
            "document_embedding": round(
                sum(float(row["document_embedding_ms"]) for row in bridge_rows), 3
            ),
            "faiss_search": round(
                sum(float(row["faiss_search_ms"]) for row in bridge_rows), 3
            ),
            "packing": round(sum(float(row["packing_ms"]) for row in bridge_rows), 3),
        },
        "executor_finish_reason_counts": dict(
            Counter(
                row["executor_finish_reason"] or "unknown"
                for row in bridge_rows if row["executor_calls"]
            )
        ),
        "output_length_terminated_count": sum(
            row["executor_finish_reason"] == "length" for row in bridge_rows
        ),
        "grader_length_terminated_count": sum(
            row["grader_finish_reason"] == "length" for row in bridge_rows
        ),
        "truncated_row_count": sum(
            bool(row["prompt_truncated"]) for row in bridge_rows
        ),
        "over_budget_full_prompt_count": over_budget_count,
        "over_budget_final_prompt_count": over_budget_final_count,
        "cache_hit_count": cache_hit_count,
        "cache_hit_rate": round(cache_hit_count / len(bridge_rows), 6),
        "cache_hit_accuracy": (
            round(cache_hit_correct / len(cache_hit_rows), 6)
            if cache_hit_rows
            else None
        ),
        "semantic_hit_count": sum(
            row["cache_type"] == "semantic" for row in bridge_rows
        ),
        "exact_hit_count": sum(row["cache_type"] == "exact" for row in bridge_rows),
        "semantic_verifier_calls": sum(
            int(row["semantic_verifier_calls"]) for row in bridge_rows
        ),
        "executor_calls": sum(int(row["executor_calls"]) for row in bridge_rows),
        "grader_calls": sum(int(row["grader_calls"]) for row in bridge_rows),
        "total_api_calls": sum(int(row["delta_calls"]) for row in bridge_rows),
        "total_request_attempts": sum(
            int(row["request_attempts"]) for row in bridge_rows
        ),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "child_tokens": args.child_tokens if experiment.mode == "hybrid" else None,
        "child_overlap_tokens": (
            args.child_overlap_tokens if experiment.mode == "hybrid" else None
        ),
        "fresh_cache_state": experiment.mode == "hybrid",
        "cache_entries_after_run": (
            controller.get_total_entries() if controller is not None else 0
        ),
        "cache_state_path": str(cache_state_path) if controller is not None else "",
        "preflight_route_counts": preflight["route_counts"],
        "artifacts": {
            "predictions": str(predictions_path),
            "bridge_rows": str(bridge_path),
            "bridge_rows_csv": str(bridge_csv_path),
            "eval_report": str(report_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path.write_text(
        json.dumps(_build_report(run_dir, bridge_rows, manifest), indent=2),
        encoding="utf-8",
    )
    print(f"[AA-LCR] Accuracy: {manifest['answer_accuracy']}")
    print(f"[AA-LCR] Manifest: {manifest_path}")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one AA-LCR reasoning experiment")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--questions-csv", type=Path, default=DEFAULT_QUESTIONS_CSV)
    parser.add_argument("--documents-root", type=Path, default=DEFAULT_DOCUMENTS_ROOT)
    parser.add_argument(
        "--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST
    )
    parser.add_argument("--question-ids", default="")
    parser.add_argument("--document-set-ids", default="")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--executor-model", default=DEFAULT_EXECUTOR_MODEL)
    parser.add_argument("--evaluator-model", default=DEFAULT_EVALUATOR_MODEL)
    parser.add_argument(
        "--executor-base-url",
        default=(
            os.getenv("OPENAI_COMPAT_EXECUTOR_BASE_URL")
            or os.getenv("OPENAI_COMPAT_BASE_URL")
            or DEFAULT_EXECUTOR_BASE_URL
        ),
    )
    parser.add_argument(
        "--evaluator-base-url",
        default=(
            os.getenv("OPENAI_COMPAT_EVALUATOR_BASE_URL")
            or os.getenv("OPENAI_COMPAT_BASE_URL")
            or DEFAULT_EVALUATOR_BASE_URL
        ),
    )
    parser.add_argument(
        "--api-key-env", default=os.getenv("OPENAI_COMPAT_API_KEY_ENV", "")
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Defaults to 16384 for 262K cells, 512 for 64K cells",
    )
    parser.add_argument("--child-tokens", type=int, default=DEFAULT_CHILD_TOKENS)
    parser.add_argument(
        "--child-overlap-tokens", type=int, default=DEFAULT_CHILD_OVERLAP_TOKENS
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--request-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--output-root", type=Path, default=Path("benchmark_artifacts/aa_lcr")
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--repeat-id", type=int, default=1)
    parser.add_argument(
        "--serving-metadata", type=Path,
        help="JSON of actual executor weights revision, vLLM version and settings",
    )
    add_grader_arguments(parser)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path, default=None)
    return parser


def main() -> None:
    run_benchmark(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
