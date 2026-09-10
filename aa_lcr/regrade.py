"""Regrade saved AA-LCR answers without regenerating or replacing them."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from aa_lcr.api import Completion, call_chat_completion
from aa_lcr.dataset import (
    DATASET_RELEASES,
    DEFAULT_DATASET_MANIFEST,
    DEFAULT_DOCUMENTS_ROOT,
    DEFAULT_QUESTIONS_CSV,
    build_scope_hash,
    dataset_signature,
    load_dataset_manifest,
    load_documents,
    load_questions,
    sha256_file,
)
from aa_lcr.grading import Grader, add_grader_arguments
from aa_lcr.run_benchmark import (
    DEFAULT_EVALUATOR_BASE_URL,
    DEFAULT_EVALUATOR_MODEL,
    REPORT_FILENAME,
    _build_report,
    _load_tokenizer,
    _summarize,
    _validate_dataset_manifest,
    _write_csv,
    _write_jsonl,
)


def read_rows(path):
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    ids = [row["question_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicate question IDs in {path}")
    return {row["question_id"]: row for row in rows}


def validate_source(source_dir, questions, documents_root, dataset_info):
    source_dir = Path(source_dir)
    manifest = json.loads((source_dir / "manifest.json").read_text())
    predictions = read_rows(source_dir / "predictions.jsonl")
    bridge = read_rows(source_dir / "bridge_rows.jsonl")
    ids = manifest["question_ids"]
    if (
        len(ids) != len(set(ids))
        or set(ids) != set(predictions)
        or set(ids) != set(bridge)
        or manifest["rows_selected"] != len(ids)
    ):
        raise ValueError("Source manifest and prediction/bridge question IDs differ")
    if manifest["archive_sha256"] != dataset_info["archive_sha256"]:
        raise ValueError("Source and grading document archives differ")
    target = {question.question_id: question for question in questions}
    scopes = {}
    rows = []
    for question_id in ids:
        if question_id not in target:
            raise ValueError(f"Missing grading question: {question_id}")
        question = target[question_id]
        if question.document_set_id not in scopes:
            scopes[question.document_set_id] = build_scope_hash(
                load_documents(question, documents_root)
            )
        prediction, row = predictions[question_id], bridge[question_id]
        for item in (prediction, row):
            if any(
                item.get(key) != getattr(question, key)
                for key in ("question", "document_set_id", "document_category")
            ):
                raise ValueError(f"Question/document identity mismatch: {question_id}")
        if (
            row["candidate_answer"] != prediction["candidate_answer"]
            or row["data_source_filenames"] != list(question.data_source_filenames)
            or row["source_scope_hash"] != scopes[question.document_set_id]
        ):
            raise ValueError(f"Answer/document content mismatch: {question_id}")
        rows.append(row)
    return manifest, rows


def run_regrade(
    args, *, tokenizer_factory=None, completion_caller=call_chat_completion
):
    source = Path(args.source_run).resolve()
    output = Path(args.output_dir).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Regrading output must be separate from its source run")
    if output.exists():
        raise FileExistsError(f"Regrading output already exists: {output}")
    questions = load_questions(args.questions_csv)
    dataset = load_dataset_manifest(args.dataset_manifest)
    info = _validate_dataset_manifest(
        dataset, Path(args.questions_csv), questions, Path(args.documents_root)
    )
    source_manifest, source_rows = validate_source(
        source, questions, Path(args.documents_root), info
    )
    selected_ids = [
        value.strip() for value in args.question_ids.split(",") if value.strip()
    ]
    if selected_ids:
        if len(selected_ids) != len(set(selected_ids)) or not set(selected_ids) <= {
            row["question_id"] for row in source_rows
        }:
            raise ValueError("Unknown or duplicate selected question IDs")
        source_rows = [row for row in source_rows if row["question_id"] in selected_ids]
    if args.validate_only:
        print(
            json.dumps(
                {
                    "rows": len(source_rows),
                    "dataset_version": info["dataset_version"],
                    "source_run": str(source),
                },
                indent=2,
            )
        )
        return None
    grader = Grader(
        args,
        tokenizer_factory=tokenizer_factory or _load_tokenizer,
        completion_caller=completion_caller,
    )
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    target = {question.question_id: question for question in questions}
    rows = []
    for source_row in source_rows:
        row = dict(source_row)
        question = target[row["question_id"]]
        completion, attempts, grade = Completion("", 0, 0, {}), 0, ""
        status, error = "not_run", ""
        row_started = time.monotonic()
        if row["executor_status"] == "ok":
            try:
                completion, attempts, grade = grader.grade(
                    question.question, question.answer, row["candidate_answer"]
                )
                status = "ok" if grade else "invalid_output"
                if not grade:
                    error = "Evaluator returned an invalid verdict or an incomplete response"
            except Exception as exc:
                attempts = int(getattr(exc, "attempts", 0))
                status, error = "error", f"{type(exc).__name__}: {exc}"
        row.update(
            source_grade=source_row["grade"],
            source_official_answer=source_row["official_answer"],
            official_answer=question.answer,
            answer=question.answer,
            grade=grade,
            grade_valid=bool(grade),
            answer_correct=(grade == "CORRECT" if grade else None),
            grader_status=status,
            grader_error=error,
            grader_attempts=attempts,
            grader_raw_output=completion.text,
            grader_usage=completion.raw_usage,
            grader_finish_reason=completion.finish_reason,
            regrade_latency_ms=round((time.monotonic() - row_started) * 1000, 3),
            grader_calls=int(status in {"ok", "invalid_output"}),
            delta_input_tokens=completion.input_tokens,
            delta_output_tokens=completion.output_tokens,
            delta_calls=int(status in {"ok", "invalid_output"}),
            request_attempts=attempts,
        )
        rows.append(row)
        print(f"[AA-LCR regrade] {row['question_id']}: {grade or status}")
    finished = datetime.now(timezone.utc)
    valid = [row for row in rows if row["grade_valid"]]
    correct = sum(row["answer_correct"] is True for row in valid)
    errors = sum(
        row["executor_status"] == "error" or row["grader_status"] == "error"
        for row in rows
    )
    manifest = {
        **{
            key: value
            for key, value in source_manifest.items()
            if key.startswith("executor_")
            or key
            in {
                "benchmark_target",
                "experiment",
                "mode",
                "context_window_tokens",
                "max_input_tokens",
                "max_output_tokens",
                "temperature",
                "thinking_enabled",
                "repeat_id",
                "child_tokens",
                "child_overlap_tokens",
                "prompt_version",
                "prompt_template_sha256",
            }
        },
        **grader.metadata(),
        "artifact_schema_version": 2,
        "artifact_type": "regrade",
        "run_id": output.name,
        "source_run": str(source),
        "generation_source_run": source_manifest.get(
            "generation_source_run", str(source)
        ),
        "source_question_ids": source_manifest["question_ids"],
        "question_ids": [row["question_id"] for row in rows],
        "rows_selected": len(rows),
        "full_run": len(rows) == 100 and len(questions) == 100,
        "source_predictions_sha256": sha256_file(source / "predictions.jsonl"),
        "source_bridge_sha256": sha256_file(source / "bridge_rows.jsonl"),
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "source_dataset_version": source_manifest.get("dataset_version")
        or next(
            (
                version
                for version, release in DATASET_RELEASES.items()
                if release["revision"] == source_manifest["dataset_revision"]
            ),
            "custom",
        ),
        "source_dataset_revision": source_manifest["dataset_revision"],
        "dataset_version": info["dataset_version"],
        "dataset_revision": dataset["dataset_revision"],
        "dataset_manifest": str(args.dataset_manifest),
        "questions_csv": str(args.questions_csv),
        "documents_root": str(args.documents_root),
        "questions_sha256": info["questions_sha256"],
        "archive_path": info["archive_path"],
        "archive_sha256": info["archive_sha256"],
        "dataset_signature": dataset_signature(
            info["questions_sha256"], info["archive_sha256"], questions
        ),
        "dataset_question_count": len(questions),
        "dataset_document_set_count": info["document_set_count"],
        "dataset_referenced_document_count": info["referenced_document_count"],
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "created_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "usage_scope": "regrading_only",
        "executor_calls": 0,
        "grader_calls": sum(r["grader_calls"] for r in rows),
        "semantic_verifier_calls": 0,
        "timing_summary_ms": {"regrading": sum(r["regrade_latency_ms"] for r in rows)},
        "total_input_tokens": sum(r["delta_input_tokens"] for r in rows),
        "total_output_tokens": sum(r["delta_output_tokens"] for r in rows),
        "total_api_calls": sum(r["delta_calls"] for r in rows),
        "total_request_attempts": sum(r["request_attempts"] for r in rows),
        "valid_grade_count": len(valid),
        "invalid_grade_count": len(rows) - len(valid),
        "answer_correct_count": correct,
        "api_error_count": errors,
        "answer_accuracy": correct / len(rows) if len(valid) == len(rows) else None,
        "valid_grade_accuracy": correct / len(valid) if valid else None,
        "accepted_full_run": len(rows) == 100 and len(valid) == 100 and errors == 0,
        "by_document_category": _summarize(rows, "document_category"),
        "by_route": _summarize(rows, "route"),
        "route_counts": dict(Counter(row["route"] for row in rows)),
        "output_length_terminated_count": sum(
            r.get("executor_finish_reason") == "length" for r in rows
        ),
        "executor_finish_reason_counts": dict(
            Counter(
                row.get("executor_finish_reason") or "unknown"
                for row in rows
                if row["executor_status"] == "ok" and not row.get("from_cache")
            )
        ),
        "grader_length_terminated_count": sum(
            r["grader_finish_reason"] == "length" for r in rows
        ),
        "artifacts": {
            name: str(output / filename)
            for name, filename in (
                ("predictions", "predictions.jsonl"),
                ("bridge_rows", "bridge_rows.jsonl"),
                ("bridge_rows_csv", "bridge_rows.csv"),
                ("eval_report", REPORT_FILENAME),
            )
        },
    }
    manifest["total_tokens"] = (
        manifest["total_input_tokens"] + manifest["total_output_tokens"]
    )
    _write_jsonl(output / "predictions.jsonl", rows)
    _write_jsonl(output / "bridge_rows.jsonl", rows)
    _write_csv(output / "bridge_rows.csv", rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / REPORT_FILENAME).write_text(
        json.dumps(_build_report(output, rows, manifest), indent=2)
    )
    return output


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--questions-csv", type=Path, default=DEFAULT_QUESTIONS_CSV)
    parser.add_argument("--documents-root", type=Path, default=DEFAULT_DOCUMENTS_ROOT)
    parser.add_argument(
        "--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST
    )
    parser.add_argument("--evaluator-model", default=DEFAULT_EVALUATOR_MODEL)
    parser.add_argument(
        "--evaluator-base-url",
        default=os.getenv("OPENAI_COMPAT_EVALUATOR_BASE_URL")
        or DEFAULT_EVALUATOR_BASE_URL,
    )
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--request-timeout-seconds", type=int, default=1800)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--question-ids", default="", help="Optional subset for a grader sanity check"
    )
    add_grader_arguments(parser)
    return parser


if __name__ == "__main__":
    run_regrade(build_arg_parser().parse_args())
