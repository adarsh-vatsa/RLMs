"""Validate and compare the four AA-LCR experiment artifacts."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


EXPERIMENT_ORDER = (
    "direct_262k",
    "hybrid_262k",
    "direct_64k",
    "hybrid_64k",
)
EXPECTED_BUDGETS = {
    "direct_262k": ("direct", 262144, 240000),
    "hybrid_262k": ("hybrid", 262144, 240000),
    "direct_64k": ("direct", 65536, 60000),
    "hybrid_64k": ("hybrid", 65536, 60000),
}
DEFAULT_BOOTSTRAP_REPLICATES = 20000
DEFAULT_BOOTSTRAP_SEED = 20260814


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(payload)
    return rows


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    run_dir = Path(run_dir)
    manifest = _load_json(run_dir / "manifest.json")
    rows = _load_jsonl(run_dir / "bridge_rows.jsonl")
    if int(manifest.get("rows_selected") or 0) != len(rows):
        raise ValueError(f"Row count does not reconcile for {run_dir}")
    return manifest, rows


def _shared_contract(manifest: dict) -> tuple:
    return tuple(
        manifest.get(field)
        for field in (
            "dataset_revision",
            "dataset_signature",
            "questions_sha256",
            "archive_sha256",
            "question_ids",
            "executor_model",
            "evaluator_model",
            "max_output_tokens",
            "temperature",
            "thinking_enabled",
            "prompt_version",
            "prompt_template_sha256",
            "grader_prompt_version",
            "grader_prompt_template_sha256",
            "executor_tokenizer_model",
            "executor_chat_template_sha256",
        )
    )


def validate_runs(
    runs: dict[str, tuple[dict, list[dict]]],
    *,
    require_full_runs: bool = True,
) -> None:
    if set(runs) != set(EXPERIMENT_ORDER):
        raise ValueError("Exactly the four AA-LCR experiment cells are required")
    baseline_contract = _shared_contract(runs[EXPERIMENT_ORDER[0]][0])
    baseline_ids = [row.get("question_id") for row in runs[EXPERIMENT_ORDER[0]][1]]
    for experiment in EXPERIMENT_ORDER:
        manifest, rows = runs[experiment]
        if manifest.get("experiment") != experiment:
            raise ValueError(f"Expected {experiment}, got {manifest.get('experiment')}")
        expected_mode, expected_window, expected_input = EXPECTED_BUDGETS[experiment]
        actual = (
            manifest.get("mode"),
            manifest.get("context_window_tokens"),
            manifest.get("max_input_tokens"),
        )
        if actual != (expected_mode, expected_window, expected_input):
            raise ValueError(
                f"Unexpected mode/window/input budget for {experiment}: {actual}"
            )
        if _shared_contract(manifest) != baseline_contract:
            raise ValueError(
                f"Dataset, model, tokenizer, or prompt mismatch in {experiment}"
            )
        if [row.get("question_id") for row in rows] != baseline_ids:
            raise ValueError(f"Question IDs or order differ in {experiment}")
        if int(manifest.get("api_error_count") or 0):
            raise ValueError(f"{experiment} contains API errors")
        if int(manifest.get("invalid_grade_count") or 0):
            raise ValueError(f"{experiment} contains invalid equality grades")
        if any(not row.get("grade_valid") for row in rows):
            raise ValueError(f"{experiment} contains an invalid row grade")
        if require_full_runs and not manifest.get("accepted_full_run"):
            raise ValueError(f"{experiment} is not an accepted 100-question full run")
    for experiment in ("hybrid_262k", "hybrid_64k"):
        manifest = runs[experiment][0]
        if (manifest.get("child_tokens"), manifest.get("child_overlap_tokens")) != (
            7500,
            750,
        ):
            raise ValueError(
                f"Unexpected hybrid chunking configuration in {experiment}"
            )


def _accuracy(rows: list[dict]) -> float:
    return sum(row.get("answer_correct") is True for row in rows) / len(rows)


def _paired_delta(left: list[dict], right: list[dict]) -> float:
    return _accuracy(left) - _accuracy(right)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of an empty list")
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(len(ordered) - 1, lower + 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def clustered_bootstrap_intervals(
    rows_by_run: dict[str, list[dict]],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, dict]:
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    cluster_order = []
    clustered: dict[str, dict[str, list[dict]]] = {}
    for experiment, rows in rows_by_run.items():
        clustered[experiment] = {}
        for row in rows:
            cluster = str(row["document_set_id"])
            if experiment == EXPERIMENT_ORDER[0] and cluster not in cluster_order:
                cluster_order.append(cluster)
            clustered[experiment].setdefault(cluster, []).append(row)
    expected_clusters = set(cluster_order)
    if any(set(mapping) != expected_clusters for mapping in clustered.values()):
        raise ValueError("Document-set clusters differ across runs")

    def sample_rows(experiment: str, sample: list[str]) -> list[dict]:
        return [row for cluster in sample for row in clustered[experiment][cluster]]

    metrics: dict[str, Callable[[dict[str, list[dict]]], float]] = {
        **{
            f"{experiment}_accuracy": (
                lambda sampled, experiment=experiment: _accuracy(sampled[experiment])
            )
            for experiment in EXPERIMENT_ORDER
        },
        "hybrid_minus_direct_262k": lambda sampled: _paired_delta(
            sampled["hybrid_262k"], sampled["direct_262k"]
        ),
        "hybrid_minus_direct_64k": lambda sampled: _paired_delta(
            sampled["hybrid_64k"], sampled["direct_64k"]
        ),
        "direct_64k_minus_262k": lambda sampled: _paired_delta(
            sampled["direct_64k"], sampled["direct_262k"]
        ),
        "hybrid_64k_minus_262k": lambda sampled: _paired_delta(
            sampled["hybrid_64k"], sampled["hybrid_262k"]
        ),
        "hybrid_reasoning_retention": lambda sampled: (
            _accuracy(sampled["hybrid_64k"]) / _accuracy(sampled["hybrid_262k"])
            if _accuracy(sampled["hybrid_262k"])
            else 0.0
        ),
    }
    rng = random.Random(seed)
    values = {name: [] for name in metrics}
    for _ in range(replicates):
        sampled_clusters = [rng.choice(cluster_order) for _ in cluster_order]
        sampled = {
            experiment: sample_rows(experiment, sampled_clusters)
            for experiment in EXPERIMENT_ORDER
        }
        for name, metric in metrics.items():
            values[name].append(metric(sampled))
    return {
        name: {
            "lower": round(_percentile(samples, 0.025), 6),
            "upper": round(_percentile(samples, 0.975), 6),
        }
        for name, samples in values.items()
    }


def _breakdown(rows: list[dict], field: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(field) or "unlabeled"), []).append(row)
    return {
        label: {
            "rows": len(bucket),
            "accuracy": round(_accuracy(bucket), 6),
        }
        for label, bucket in sorted(grouped.items())
    }


def compare_runs(
    run_dirs: dict[str, Path],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    require_full_runs: bool = True,
) -> dict:
    runs = {experiment: load_run(path) for experiment, path in run_dirs.items()}
    validate_runs(runs, require_full_runs=require_full_runs)
    manifests = {experiment: runs[experiment][0] for experiment in EXPERIMENT_ORDER}
    rows = {experiment: runs[experiment][1] for experiment in EXPERIMENT_ORDER}
    accuracies = {
        experiment: _accuracy(rows[experiment]) for experiment in EXPERIMENT_ORDER
    }
    semantic_hits = [
        {
            "experiment": experiment,
            "case_id": row["case_id"],
            "question_id": row["question_id"],
            "document_set_id": row["document_set_id"],
            "question": row["question"],
            "candidate_answer": row["candidate_answer"],
            "answer_correct": row["answer_correct"],
            "cache_provenance": row.get("cache_provenance") or {},
        }
        for experiment in ("hybrid_262k", "hybrid_64k")
        for row in rows[experiment]
        if row.get("cache_type") == "semantic"
    ]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_target": "aa_lcr_reasoning_four_run_comparison",
        "dataset_revision": manifests["direct_262k"]["dataset_revision"],
        "dataset_signature": manifests["direct_262k"]["dataset_signature"],
        "question_count": len(rows["direct_262k"]),
        "bootstrap": {
            "unit": "document_set_id",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "accuracy": {name: round(value, 6) for name, value in accuracies.items()},
        "paired_deltas": {
            "hybrid_minus_direct_262k": round(
                accuracies["hybrid_262k"] - accuracies["direct_262k"], 6
            ),
            "hybrid_minus_direct_64k": round(
                accuracies["hybrid_64k"] - accuracies["direct_64k"], 6
            ),
            "direct_64k_minus_262k": round(
                accuracies["direct_64k"] - accuracies["direct_262k"], 6
            ),
            "hybrid_64k_minus_262k": round(
                accuracies["hybrid_64k"] - accuracies["hybrid_262k"], 6
            ),
            "direct_reasoning_loss_262k_minus_64k": round(
                accuracies["direct_262k"] - accuracies["direct_64k"], 6
            ),
            "hybrid_reasoning_loss_262k_minus_64k": round(
                accuracies["hybrid_262k"] - accuracies["hybrid_64k"], 6
            ),
            "hybrid_reasoning_retention": (
                round(accuracies["hybrid_64k"] / accuracies["hybrid_262k"], 6)
                if accuracies["hybrid_262k"]
                else None
            ),
        },
        "confidence_intervals_95": clustered_bootstrap_intervals(
            rows,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "by_document_category": {
            experiment: _breakdown(rows[experiment], "document_category")
            for experiment in EXPERIMENT_ORDER
        },
        "by_route": {
            experiment: _breakdown(rows[experiment], "route")
            for experiment in EXPERIMENT_ORDER
        },
        "operations": {
            experiment: {
                field: manifests[experiment].get(field)
                for field in (
                    "elapsed_seconds",
                    "total_api_calls",
                    "total_request_attempts",
                    "total_input_tokens",
                    "total_output_tokens",
                    "truncated_row_count",
                    "cache_hit_count",
                    "cache_hit_rate",
                    "cache_hit_accuracy",
                    "semantic_hit_count",
                    "semantic_verifier_calls",
                    "executor_calls",
                    "grader_calls",
                )
            }
            for experiment in EXPERIMENT_ORDER
        },
        "route_counts": {
            experiment: dict(
                sorted(Counter(row["route"] for row in rows[experiment]).items())
            )
            for experiment in EXPERIMENT_ORDER
        },
        "semantic_hit_audit": semantic_hits,
        "run_dirs": {
            experiment: str(run_dirs[experiment]) for experiment in EXPERIMENT_ORDER
        },
    }
    return report


def _markdown_report(report: dict) -> str:
    lines = [
        "# AA-LCR Four-Run Comparison",
        "",
        f"Questions: {report['question_count']}",
        "",
        "| Experiment | Accuracy | Calls | Input tokens | Cache hits |",
        "|---|---:|---:|---:|---:|",
    ]
    for experiment in EXPERIMENT_ORDER:
        operations = report["operations"][experiment]
        lines.append(
            f"| {experiment} | {report['accuracy'][experiment]:.3f} | "
            f"{operations['total_api_calls']} | {operations['total_input_tokens']} | "
            f"{operations['cache_hit_count']} |"
        )
    lines.extend(
        [
            "",
            "## Paired deltas",
            "",
            *[f"- {name}: {value}" for name, value in report["paired_deltas"].items()],
            "",
            f"Semantic hits requiring audit: {len(report['semantic_hit_audit'])}",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare four AA-LCR runs")
    for experiment in EXPERIMENT_ORDER:
        parser.add_argument(
            f"--{experiment.replace('_', '-')}", type=Path, required=True
        )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dirs = {
        experiment: getattr(args, experiment) for experiment in EXPERIMENT_ORDER
    }
    report = compare_runs(
        run_dirs,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir or Path("benchmark_artifacts/aa_lcr/comparisons") / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "comparison.json"
    markdown_path = output_dir / "comparison.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    print(f"[AA-LCR] Comparison JSON: {json_path}")
    print(f"[AA-LCR] Comparison Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
