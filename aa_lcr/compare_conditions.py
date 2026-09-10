"""Compare deliberately different AA-LCR conditions, optionally across repeats."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

from aa_lcr.compare_runs import _percentile, load_run


CONDITION_FIELDS = (
    "dataset_revision",
    "questions_sha256",
    "archive_sha256",
    "experiment",
    "executor_model",
    "max_input_tokens",
    "max_output_tokens",
    "temperature",
    "thinking_enabled",
    "prompt_template_sha256",
    "executor_chat_template_sha256",
    "executor_serving_metadata",
    "evaluator_model",
    "evaluator_base_url",
    "grader_prompt_version",
    "grader_prompt_template_sha256",
    "grader_api_style",
    "grader_reasoning_effort",
    "grader_temperature",
    "grader_max_output_tokens",
    "grader_context_window",
    "grader_chat_template_sha256",
)


def load_condition(paths):
    if len({Path(path).resolve() for path in paths}) != len(paths):
        raise ValueError("A run cannot be counted twice as a repeat")
    runs = [load_run(path) for path in paths]
    manifests = [manifest for manifest, _ in runs]
    settings = {field: manifests[0].get(field) for field in CONDITION_FIELDS}
    if any(
        {field: m.get(field) for field in CONDITION_FIELDS} != settings
        for m in manifests
    ):
        raise ValueError("Repeats within a condition have different settings")
    origins = [
        str(
            Path(
                m.get("generation_source_run") or m.get("source_run") or path
            ).resolve()
        )
        for path, m in zip(paths, manifests)
    ]
    if len(origins) != len(set(origins)):
        raise ValueError("Regrades of the same generation are not independent repeats")
    grouped = {}
    all_rows = []
    for manifest, rows in runs:
        ids = [row["question_id"] for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(manifest["question_ids"]):
            raise ValueError("Duplicate or missing question IDs")
        if grouped and set(ids) != set(grouped):
            raise ValueError("Question IDs differ between repeats")
        for row in rows:
            if (
                not row.get("grade_valid")
                or type(row.get("answer_correct")) is not bool
            ):
                raise ValueError("Comparison requires valid grades for every answer")
            if (
                row.get("executor_status") == "error"
                or row.get("grader_status") == "error"
            ):
                raise ValueError("Comparison contains failed requests")
            grouped.setdefault(row["question_id"], []).append(row)
        all_rows.extend(rows)
    tokens = [
        row.get("executor_usage", {}).get(
            "completion_tokens", row.get("executor_output_tokens")
        )
        for row in all_rows
    ]
    known_tokens = [count for count in tokens if count is not None]
    summary = {
        "runs": [str(path) for path in paths],
        "repeats": len(paths),
        "questions": len(grouped),
        "responses": len(all_rows),
        "accuracy": mean(row["answer_correct"] for row in all_rows),
        "settings": settings,
        "output_tokens": {
            "known_count": len(known_tokens),
            "mean": mean(known_tokens) if known_tokens else None,
            "max": max(known_tokens) if known_tokens else None,
        },
        "executor_finish_reasons": dict(
            Counter(row.get("executor_finish_reason") or "unknown" for row in all_rows)
        ),
        "output_budget_hit_count": sum(
            count is not None
            and count == row.get("max_output_tokens", settings["max_output_tokens"])
            for count, row in zip(tokens, all_rows)
        ),
        "elapsed_seconds": sum(m.get("elapsed_seconds", 0) for m in manifests),
        "usage_scopes": sorted(
            {m.get("usage_scope", "generation_and_grading") for m in manifests}
        ),
        "api_error_count": sum(m.get("api_error_count", 0) for m in manifests),
        "invalid_grade_count": sum(m.get("invalid_grade_count", 0) for m in manifests),
    }
    return summary, grouped


def compare_conditions(
    baseline_paths, candidate_paths, *, replicates=20000, seed=20260908
):
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    baseline, left = load_condition(baseline_paths)
    candidate, right = load_condition(candidate_paths)
    if set(left) != set(right):
        raise ValueError("Question IDs differ between conditions")
    if (
        baseline["settings"]["archive_sha256"]
        != candidate["settings"]["archive_sha256"]
    ):
        raise ValueError("Document archives differ between conditions")
    pairs = []
    clusters = {}
    for question_id in left:
        rows = left[question_id] + right[question_id]
        first = rows[0]
        for row in rows:
            if any(
                row.get(key) != first.get(key)
                for key in (
                    "question",
                    "document_set_id",
                    "document_category",
                    "source_scope_hash",
                )
            ):
                raise ValueError(f"Question/document identity mismatch: {question_id}")
            if not row.get("source_scope_hash"):
                raise ValueError(f"Missing document identity: {question_id}")
        a, b = (
            mean(r["answer_correct"] for r in left[question_id]),
            mean(r["answer_correct"] for r in right[question_id]),
        )
        pair = {
            "question_id": question_id,
            "document_set_id": first["document_set_id"],
            "baseline_pass_rate": a,
            "candidate_pass_rate": b,
            "delta": b - a,
            "baseline_answers": [r["candidate_answer"] for r in left[question_id]],
            "candidate_answers": [r["candidate_answer"] for r in right[question_id]],
            "baseline_reference": first["official_answer"],
            "candidate_reference": right[question_id][0]["official_answer"],
        }
        pairs.append(pair)
        clusters.setdefault(first["document_set_id"], []).append(pair)
    rng = random.Random(seed)
    names = list(clusters)
    samples = {"baseline_accuracy": [], "candidate_accuracy": [], "delta": []}
    for _ in range(replicates):
        sample = [row for _ in names for row in clusters[rng.choice(names)]]
        for metric, field in (
            ("baseline_accuracy", "baseline_pass_rate"),
            ("candidate_accuracy", "candidate_pass_rate"),
            ("delta", "delta"),
        ):
            samples[metric].append(mean(row[field] for row in sample))
    return {
        "baseline": baseline,
        "candidate": candidate,
        "settings_differences": {
            field: {
                "baseline": baseline["settings"][field],
                "candidate": candidate["settings"][field],
            }
            for field in CONDITION_FIELDS
            if baseline["settings"][field] != candidate["settings"][field]
        },
        "delta": candidate["accuracy"] - baseline["accuracy"],
        "improved_question_ids": [p["question_id"] for p in pairs if p["delta"] > 0],
        "regressed_question_ids": [p["question_id"] for p in pairs if p["delta"] < 0],
        "confidence_intervals_95": {
            metric: {
                "lower": _percentile(values, 0.025),
                "upper": _percentile(values, 0.975),
            }
            for metric, values in samples.items()
        },
        "bootstrap": {"replicates": replicates, "seed": seed, "clusters": len(names)},
        "serving_metadata_available": bool(
            baseline["settings"]["executor_serving_metadata"]
            and candidate["settings"]["executor_serving_metadata"]
        ),
        "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260908)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = compare_conditions(
        args.baseline,
        args.candidate,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(f"Paired accuracy delta: {report['delta']:+.2%}; report: {args.output}")


if __name__ == "__main__":
    main()
