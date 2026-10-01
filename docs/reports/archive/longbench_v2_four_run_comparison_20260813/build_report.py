#!/usr/bin/env python3
"""Build the four-run LongBench-v2 comparison data and report artifact."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
SUITE_PATH = REPO_ROOT / "benchmark_data/long_bench_v2/data_cache_suite.csv"
RUN_PATHS = {
    "full_hybrid": REPO_ROOT / "benchmark_artifacts/longbench_v2/20260810T054951Z",
    "no_exact_hybrid": REPO_ROOT / "benchmark_artifacts/longbench_v2/20260812T014012Z",
    "full_direct": REPO_ROOT / "benchmark_artifacts/longbench_v2_api/20260812T035929Z",
    "no_exact_direct": REPO_ROOT / "benchmark_artifacts/longbench_v2_api/20260812T152758Z",
}
RUN_LABELS = {
    "full_hybrid": "Cache-enabled retrieval — all traffic types",
    "no_exact_hybrid": "Cache-enabled retrieval — original + paraphrase",
    "full_direct": "Uncached model — all traffic types",
    "no_exact_direct": "Uncached model — original + paraphrase",
}
CACHE_ENABLED_LABEL = "Cache-enabled retrieval"
UNCACHED_LABEL = "Uncached model"
ALL_TRAFFIC_LABEL = "All traffic types (1,509 rows)"
ORIGINAL_PARAPHRASE_LABEL = "Original + paraphrase only (1,006 rows)"
ROW_TYPE_LABELS = {
    "original": "Original question",
    "exact": "Exact repetition",
    "semantic": "Meaning-preserving paraphrase",
}
EXPECTED_ROW_TYPES = {
    "full_hybrid": {"exact": 503, "original": 503, "semantic": 503},
    "full_direct": {"exact": 503, "original": 503, "semantic": 503},
    "no_exact_hybrid": {"original": 503, "semantic": 503},
    "no_exact_direct": {"original": 503, "semantic": 503},
}


def load_runs() -> dict[str, dict]:
    runs = {}
    for name, run_dir in RUN_PATHS.items():
        manifest = json.loads((run_dir / "manifest.json").read_text())
        rows = [
            json.loads(line)
            for line in (run_dir / "bridge_rows.jsonl").read_text().splitlines()
        ]
        runs[name] = {
            "manifest": manifest,
            "rows": rows,
            "by_case": {row["case_id"]: row for row in rows},
        }
    return runs


def validate_sources(runs: dict[str, dict]) -> dict:
    suite_hashes = {run["manifest"]["suite_csv_sha256"] for run in runs.values()}
    source_hashes = {run["manifest"]["source_json_sha256"] for run in runs.values()}
    assert len(suite_hashes) == len(source_hashes) == 1

    checks = []
    for name, run in runs.items():
        manifest = run["manifest"]
        rows = run["rows"]
        row_type_counts = dict(sorted(Counter(row["row_type"] for row in rows).items()))
        assert len(rows) == manifest["rows_selected"]
        assert len({row["case_id"] for row in rows}) == len(rows)
        assert row_type_counts == EXPECTED_ROW_TYPES[name]
        assert manifest["row_type_counts"] == EXPECTED_ROW_TYPES[name]
        assert manifest["answer_correct_count"] == sum(
            bool(row["answer_correct"]) for row in rows
        )
        assert manifest["valid_choice_count"] == len(rows)
        assert manifest["invalid_choice_count"] == 0
        assert manifest["api_error_count"] == 0
        assert all(row["api_status"] == "ok" for row in rows)
        assert all(row["valid_choice"] for row in rows)
        assert manifest["mcq_decoder_constraint_version"] == (
            "vllm_structured_choice_abcd_v1"
        )
        assert manifest["mcq_allowed_choices"] == ["A", "B", "C", "D"]
        checks.append(
            {
                "run": RUN_LABELS[name],
                "rows": len(rows),
                "unique_case_ids": len({row["case_id"] for row in rows}),
                "api_errors": manifest["api_error_count"],
                "invalid_choices": manifest["invalid_choice_count"],
                "suite_hash_match": True,
                "source_hash_match": True,
                "status": "passed",
            }
        )

    full_ids = {
        "hybrid": set(runs["full_hybrid"]["by_case"]),
        "direct": set(runs["full_direct"]["by_case"]),
    }
    no_exact_ids = {
        "hybrid": set(runs["no_exact_hybrid"]["by_case"]),
        "direct": set(runs["no_exact_direct"]["by_case"]),
    }
    assert full_ids["hybrid"] == full_ids["direct"]
    assert no_exact_ids["hybrid"] == no_exact_ids["direct"]
    assert no_exact_ids["hybrid"] <= full_ids["hybrid"]
    assert len(full_ids["hybrid"] - no_exact_ids["hybrid"]) == 503
    return {
        "suite_csv_sha256": next(iter(suite_hashes)),
        "source_json_sha256": next(iter(source_hashes)),
        "checks": checks,
    }


def accuracy(rows: list[dict]) -> float:
    return sum(bool(row["answer_correct"]) for row in rows) / len(rows)


def cluster_bootstrap_delta(
    hybrid: dict[str, dict],
    direct: dict[str, dict],
    case_ids: list[str],
    *,
    seed: int,
    replicates: int = 20_000,
) -> dict:
    source_deltas: dict[str, list[int]] = defaultdict(list)
    for case_id in case_ids:
        hybrid_row = hybrid[case_id]
        direct_row = direct[case_id]
        source_deltas[hybrid_row["source_id"]].append(
            int(bool(hybrid_row["answer_correct"]))
            - int(bool(direct_row["answer_correct"]))
        )

    source_ids = sorted(source_deltas)
    numerator = sum(sum(source_deltas[source_id]) for source_id in source_ids)
    denominator = sum(len(source_deltas[source_id]) for source_id in source_ids)
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(replicates):
        sampled = [source_ids[rng.randrange(len(source_ids))] for _ in source_ids]
        sample_numerator = sum(sum(source_deltas[source_id]) for source_id in sampled)
        sample_denominator = sum(len(source_deltas[source_id]) for source_id in sampled)
        bootstrap.append(sample_numerator / sample_denominator)
    bootstrap.sort()
    return {
        "delta": numerator / denominator,
        "ci_low": bootstrap[int(0.025 * replicates)],
        "ci_high": bootstrap[int(0.975 * replicates) - 1],
        "source_groups": len(source_ids),
        "replicates": replicates,
        "seed": seed,
    }


def paired_result(
    runs: dict[str, dict],
    population: str,
    hybrid_name: str,
    direct_name: str,
    seed: int,
) -> dict:
    hybrid = runs[hybrid_name]["by_case"]
    direct = runs[direct_name]["by_case"]
    case_ids = sorted(set(hybrid) & set(direct))
    hybrid_correct = sum(bool(hybrid[case_id]["answer_correct"]) for case_id in case_ids)
    direct_correct = sum(bool(direct[case_id]["answer_correct"]) for case_id in case_ids)
    hybrid_only = sum(
        bool(hybrid[case_id]["answer_correct"])
        and not bool(direct[case_id]["answer_correct"])
        for case_id in case_ids
    )
    direct_only = sum(
        bool(direct[case_id]["answer_correct"])
        and not bool(hybrid[case_id]["answer_correct"])
        for case_id in case_ids
    )
    interval = cluster_bootstrap_delta(hybrid, direct, case_ids, seed=seed)
    return {
        "population": population,
        "rows": len(case_ids),
        "source_groups": interval["source_groups"],
        "hybrid_correct": hybrid_correct,
        "direct_correct": direct_correct,
        "hybrid_accuracy": hybrid_correct / len(case_ids),
        "direct_accuracy": direct_correct / len(case_ids),
        "delta_pp": interval["delta"] * 100,
        "ci_low_pp": interval["ci_low"] * 100,
        "ci_high_pp": interval["ci_high"] * 100,
        "hybrid_correct_direct_wrong": hybrid_only,
        "direct_correct_hybrid_wrong": direct_only,
        "prediction_agreement": sum(
            hybrid[case_id]["prediction"] == direct[case_id]["prediction"]
            for case_id in case_ids
        )
        / len(case_ids),
        "bootstrap_replicates": interval["replicates"],
        "bootstrap_seed": interval["seed"],
    }


def row_type_rows(run: dict) -> list[dict]:
    output = []
    for row_type in sorted({row["row_type"] for row in run["rows"]}):
        rows = [row for row in run["rows"] if row["row_type"] == row_type]
        output.append(
            {
                "row_type": ROW_TYPE_LABELS[row_type],
                "rows": len(rows),
                "correct": sum(bool(row["answer_correct"]) for row in rows),
                "accuracy": accuracy(rows),
                "api_calls": sum(int(row.get("delta_calls") or 0) for row in rows),
                "input_tokens": sum(
                    int(row.get("delta_input_tokens") or 0) for row in rows
                ),
                "output_tokens": sum(
                    int(row.get("delta_output_tokens") or 0) for row in rows
                ),
            }
        )
    return output


def build_comparison_data(runs: dict[str, dict], validation: dict) -> dict:
    suite_meta = {
        row["case_id"]: row
        for row in csv.DictReader(SUITE_PATH.open(encoding="utf-8", newline=""))
    }
    full_pair = paired_result(
        runs, ALL_TRAFFIC_LABEL, "full_hybrid", "full_direct", 20260813
    )
    no_exact_pair = paired_result(
        runs,
        ORIGINAL_PARAPHRASE_LABEL,
        "no_exact_hybrid",
        "no_exact_direct",
        20260814,
    )

    run_summary = []
    for name in ("no_exact_hybrid", "no_exact_direct", "full_hybrid", "full_direct"):
        manifest = runs[name]["manifest"]
        is_hybrid = "hybrid" in name
        population = (
            ALL_TRAFFIC_LABEL
            if name.startswith("full_")
            else ORIGINAL_PARAPHRASE_LABEL
        )
        run_summary.append(
            {
                "run": RUN_LABELS[name],
                "run_id": manifest["run_id"],
                "architecture": CACHE_ENABLED_LABEL if is_hybrid else UNCACHED_LABEL,
                "population": population,
                "included_row_types": ", ".join(manifest["row_types_requested"]),
                "rows": manifest["rows_selected"],
                "correct": manifest["answer_correct_count"],
                "accuracy": manifest["answer_accuracy"],
                "elapsed_hours": manifest["elapsed_seconds"] / 3600,
                "api_calls": manifest["total_api_calls"],
                "input_tokens": manifest["total_input_tokens"],
                "output_tokens": manifest["total_output_tokens"],
                "total_tokens": manifest["total_tokens"],
                "tokens_per_row": manifest["total_tokens"] / manifest["rows_selected"],
                "calls_per_row": manifest["total_api_calls"] / manifest["rows_selected"],
                "truncated_rows": manifest.get("truncated_row_count"),
                "api_errors": manifest["api_error_count"],
                "invalid_choices": manifest["invalid_choice_count"],
            }
        )

    pair_accuracy = []
    for pair in (no_exact_pair, full_pair):
        pair_accuracy.extend(
            [
                {
                    "population": pair["population"],
                    "architecture": CACHE_ENABLED_LABEL,
                    "accuracy": pair["hybrid_accuracy"],
                    "correct": pair["hybrid_correct"],
                    "rows": pair["rows"],
                    "delta_pp": pair["delta_pp"],
                },
                {
                    "population": pair["population"],
                    "architecture": UNCACHED_LABEL,
                    "accuracy": pair["direct_accuracy"],
                    "correct": pair["direct_correct"],
                    "rows": pair["rows"],
                    "delta_pp": 0.0,
                },
            ]
        )

    efficiency_reduction = []
    pair_names = [
        (ORIGINAL_PARAPHRASE_LABEL, "no_exact_hybrid", "no_exact_direct"),
        (ALL_TRAFFIC_LABEL, "full_hybrid", "full_direct"),
    ]
    fields = [
        ("Total tokens", "total_tokens"),
        ("Runtime", "elapsed_seconds"),
        ("API calls", "total_api_calls"),
    ]
    for population, hybrid_name, direct_name in pair_names:
        hybrid_manifest = runs[hybrid_name]["manifest"]
        direct_manifest = runs[direct_name]["manifest"]
        for metric, field in fields:
            hybrid_value = hybrid_manifest[field]
            direct_value = direct_manifest[field]
            efficiency_reduction.append(
                {
                    "comparison": f"{population} · {metric}",
                    "population": population,
                    "metric": metric,
                    "reduction": 1 - hybrid_value / direct_value,
                    "hybrid_value": hybrid_value,
                    "direct_value": direct_value,
                    "direct_to_hybrid_ratio": direct_value / hybrid_value,
                }
            )

    full_hybrid = runs["full_hybrid"]["by_case"]
    full_direct = runs["full_direct"]["by_case"]
    common_ids = sorted(set(full_hybrid) & set(full_direct))
    truncation_accuracy = []
    truncation_delta = []
    for label, is_truncated in (("Not truncated", False), ("Truncated", True)):
        case_ids = [
            case_id
            for case_id in common_ids
            if bool(full_direct[case_id]["prompt_truncated"]) is is_truncated
        ]
        hybrid_rows = [full_hybrid[case_id] for case_id in case_ids]
        direct_rows = [full_direct[case_id] for case_id in case_ids]
        hybrid_accuracy = accuracy(hybrid_rows)
        direct_accuracy = accuracy(direct_rows)
        truncation_accuracy.extend(
            [
                {
                    "direct_input_group": label,
                    "architecture": CACHE_ENABLED_LABEL,
                    "accuracy": hybrid_accuracy,
                    "correct": sum(bool(row["answer_correct"]) for row in hybrid_rows),
                    "rows": len(case_ids),
                },
                {
                    "direct_input_group": label,
                    "architecture": UNCACHED_LABEL,
                    "accuracy": direct_accuracy,
                    "correct": sum(bool(row["answer_correct"]) for row in direct_rows),
                    "rows": len(case_ids),
                },
            ]
        )
        truncation_delta.append(
            {
                "direct_input_group": label,
                "rows": len(case_ids),
                "hybrid_accuracy": hybrid_accuracy,
                "direct_accuracy": direct_accuracy,
                "delta_pp": (hybrid_accuracy - direct_accuracy) * 100,
            }
        )

    domain_accuracy = []
    original_ids = [
        case_id
        for case_id in common_ids
        if full_hybrid[case_id]["row_type"] == "original"
    ]
    domains: dict[str, list[str]] = defaultdict(list)
    for case_id in original_ids:
        domains[suite_meta[case_id]["domain"]].append(case_id)
    for domain, case_ids in sorted(domains.items()):
        hybrid_accuracy = sum(
            bool(full_hybrid[case_id]["answer_correct"]) for case_id in case_ids
        ) / len(case_ids)
        direct_accuracy = sum(
            bool(full_direct[case_id]["answer_correct"]) for case_id in case_ids
        ) / len(case_ids)
        domain_accuracy.append(
            {
                "domain": domain,
                "rows": len(case_ids),
                "hybrid_accuracy": hybrid_accuracy,
                "direct_accuracy": direct_accuracy,
                "delta_pp": (hybrid_accuracy - direct_accuracy) * 100,
            }
        )

    stability = []
    for architecture, full_name, no_exact_name in (
        (CACHE_ENABLED_LABEL, "full_hybrid", "no_exact_hybrid"),
        (UNCACHED_LABEL, "full_direct", "no_exact_direct"),
    ):
        full = runs[full_name]["by_case"]
        no_exact = runs[no_exact_name]["by_case"]
        case_ids = sorted(set(full) & set(no_exact))
        full_correct = sum(bool(full[case_id]["answer_correct"]) for case_id in case_ids)
        no_exact_correct = sum(
            bool(no_exact[case_id]["answer_correct"]) for case_id in case_ids
        )
        stability.append(
            {
                "architecture": architecture,
                "shared_rows": len(case_ids),
                "prediction_agreement_rows": sum(
                    full[case_id]["prediction"] == no_exact[case_id]["prediction"]
                    for case_id in case_ids
                ),
                "prediction_agreement": sum(
                    full[case_id]["prediction"] == no_exact[case_id]["prediction"]
                    for case_id in case_ids
                )
                / len(case_ids),
                "full_subset_accuracy": full_correct / len(case_ids),
                "no_exact_accuracy": no_exact_correct / len(case_ids),
                "accuracy_shift_pp": (no_exact_correct - full_correct)
                / len(case_ids)
                * 100,
            }
        )

    hybrid_full = runs["full_hybrid"]["manifest"]
    hybrid_no_exact = runs["no_exact_hybrid"]["manifest"]
    direct_full = runs["full_direct"]["manifest"]
    direct_no_exact = runs["no_exact_direct"]["manifest"]
    exact_removal_effect = [
        {
            "architecture": CACHE_ENABLED_LABEL,
            "removed_rows": hybrid_full["rows_selected"] - hybrid_no_exact["rows_selected"],
            "api_call_change": hybrid_no_exact["total_api_calls"] - hybrid_full["total_api_calls"],
            "token_change": hybrid_no_exact["total_tokens"] - hybrid_full["total_tokens"],
            "runtime_change_percent": (
                hybrid_no_exact["elapsed_seconds"] / hybrid_full["elapsed_seconds"] - 1
            ),
        },
        {
            "architecture": UNCACHED_LABEL,
            "removed_rows": direct_full["rows_selected"] - direct_no_exact["rows_selected"],
            "api_call_change": direct_no_exact["total_api_calls"] - direct_full["total_api_calls"],
            "token_change": direct_no_exact["total_tokens"] - direct_full["total_tokens"],
            "runtime_change_percent": (
                direct_no_exact["elapsed_seconds"] / direct_full["elapsed_seconds"] - 1
            ),
        },
    ]

    for name, run in runs.items():
        for row in row_type_rows(run):
            row["run"] = RUN_LABELS[name]

    headline = {
        "full_accuracy_delta_pp": full_pair["delta_pp"],
        "full_hybrid_accuracy": full_pair["hybrid_accuracy"],
        "full_direct_accuracy": full_pair["direct_accuracy"],
        "no_exact_accuracy_delta_pp": no_exact_pair["delta_pp"],
        "no_exact_hybrid_accuracy": no_exact_pair["hybrid_accuracy"],
        "no_exact_direct_accuracy": no_exact_pair["direct_accuracy"],
        "full_token_reduction": 1
        - hybrid_full["total_tokens"] / direct_full["total_tokens"],
        "full_speedup_x": direct_full["elapsed_seconds"] / hybrid_full["elapsed_seconds"],
        "no_exact_token_reduction": 1
        - hybrid_no_exact["total_tokens"] / direct_no_exact["total_tokens"],
        "no_exact_speedup_x": direct_no_exact["elapsed_seconds"]
        / hybrid_no_exact["elapsed_seconds"],
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question": (
            "How does cache-enabled retrieval compare with the uncached model on "
            "the 1,006 original-question and paraphrase rows that exclude exact "
            "repetitions, and secondarily across all 1,509 traffic rows?"
        ),
        "validation": validation,
        "headline": headline,
        "run_summary": run_summary,
        "paired_comparisons": [no_exact_pair, full_pair],
        "pair_accuracy": pair_accuracy,
        "efficiency_reduction": efficiency_reduction,
        "truncation_accuracy": truncation_accuracy,
        "truncation_delta": truncation_delta,
        "domain_accuracy": domain_accuracy,
        "stability": stability,
        "exact_removal_effect": exact_removal_effect,
        "row_type_detail": [
            {"run": RUN_LABELS[name], **row}
            for name in (
                "no_exact_hybrid",
                "no_exact_direct",
                "full_hybrid",
                "full_direct",
            )
            for row in row_type_rows(runs[name])
        ],
        "methods": {
            "accuracy": "answer_correct_count divided by rows_selected",
            "resource_reduction": (
                "1 - cache-enabled retrieval manifest value / matched uncached-model "
                "manifest value"
            ),
            "confidence_interval": (
                "Percentile bootstrap of paired correctness differences with source_id "
                "as the resampling cluster; 20,000 replicates and fixed seeds."
            ),
            "population_note": (
                "The primary 1,006-row comparison contains original questions and "
                "meaning-preserving paraphrases, with exact repetitions excluded. The "
                "secondary 1,509-row traffic comparison adds one exact repetition per "
                "source group. Repetitions and paraphrases are linked to the original "
                "and are not independent benchmark questions."
            ),
        },
        "chart_map": [
            {
                "section": "Resource use",
                "question": "How much runtime, call, and token volume does cache-enabled retrieval avoid?",
                "family": "Comparison & Ranking",
                "type": "bar",
                "fields": ["comparison", "reduction"],
                "palette": "single blue root with neutral zero reference",
            },
            {
                "section": "Matched quality",
                "question": "Does cache-enabled retrieval preserve or improve accuracy in both populations?",
                "family": "Comparison & Ranking",
                "type": "grouped bar",
                "fields": ["population", "architecture", "accuracy"],
                "palette": "hard two-root cap: blue for cache-enabled retrieval, orange for the uncached model",
            },
            {
                "section": "Long-context mechanism",
                "question": "Is the quality gain concentrated where the uncached model truncates its input?",
                "family": "Comparison & Ranking",
                "type": "grouped bar",
                "fields": ["direct_input_group", "architecture", "accuracy"],
                "palette": "hard two-root cap: blue for cache-enabled retrieval, orange for the uncached model",
            },
            {
                "section": "Domain robustness",
                "question": "Where does original-row accuracy improve or decline?",
                "family": "Uncertainty & Benchmark",
                "type": "diverging bar",
                "fields": ["domain", "delta_pp"],
                "palette": "single blue root with neutral zero reference and signed labels",
            },
        ],
    }


def source_metadata(generated_at: str) -> dict:
    tables_used = []
    for run_path in RUN_PATHS.values():
        tables_used.extend(
            [
                str((run_path / "manifest.json").relative_to(REPO_ROOT)),
                str((run_path / "bridge_rows.jsonl").relative_to(REPO_ROOT)),
            ]
        )
    return {
        "id": "four_run_comparison",
        "label": "Four LongBench-v2 benchmark artifacts and row-level comparison",
        "path": "comparison_data.json",
        "description": (
            "Generated by build_report.py from the four listed manifest.json and "
            "bridge_rows.jsonl artifacts plus the prepared benchmark suite. "
            f"Inputs: {', '.join(tables_used + [str(SUITE_PATH.relative_to(REPO_ROOT))])}. "
            f"Generated at {generated_at}."
        ),
        "query": {
            "engine": "DuckDB",
            "language": "sql",
            "sql": (
                "SELECT * FROM read_json_auto(["
                + ", ".join(f"'{path}'" for path in tables_used)
                + "], union_by_name = true);"
            ),
            "description": (
                "Canonical source extraction for the four manifests and bridge-row "
                "files. build_report.py performs the reviewed paired calculations "
                "and writes comparison_data.json."
            ),
            "executed_at": generated_at,
            "tables_used": tables_used + [
                str(SUITE_PATH.relative_to(REPO_ROOT)),
            ],
            "filters": [
                "Original + paraphrase only (1,006 rows): exact repetitions excluded",
                "All traffic types (1,509 rows): original questions, exact repetitions, and meaning-preserving paraphrases",
                "All rows; no sampling",
                "Paired comparisons use identical case_id populations",
            ],
            "metric_definitions": [
                "Accuracy = correct rows / selected rows",
                "Runtime = manifest elapsed_seconds",
                "Token and API-call totals = manifest-reported usage",
                "Cache-enabled reduction = 1 - cache-enabled retrieval value / matched uncached-model value",
                "Accuracy uncertainty = source_id-clustered percentile bootstrap",
            ],
        },
    }


def build_artifact(data: dict) -> dict:
    generated_at = data["generated_at"]
    source = source_metadata(generated_at)
    headline = data["headline"]
    no_exact_pair, full_pair = data["paired_comparisons"]
    full_hybrid = next(
        row for row in data["run_summary"] if row["run"] == RUN_LABELS["full_hybrid"]
    )
    full_direct = next(
        row for row in data["run_summary"] if row["run"] == RUN_LABELS["full_direct"]
    )
    no_exact_hybrid = next(
        row
        for row in data["run_summary"]
        if row["run"] == RUN_LABELS["no_exact_hybrid"]
    )
    no_exact_direct = next(
        row
        for row in data["run_summary"]
        if row["run"] == RUN_LABELS["no_exact_direct"]
    )
    stability = {row["architecture"]: row for row in data["stability"]}

    cards = [
        {
            "id": "token_card",
            "description": "Total input plus output tokens versus matched uncached-model runs.",
            "dataset": "headline",
            "sourceId": source["id"],
            "metrics": [
                {
                    "label": "Original + paraphrase reduction",
                    "field": "no_exact_token_reduction",
                    "format": "percent",
                },
                {
                    "label": "All-traffic reduction",
                    "field": "full_token_reduction",
                    "format": "percent",
                },
            ],
        },
        {
            "id": "no_exact_accuracy_card",
            "description": "Paired accuracy across 1,006 original-question and paraphrase rows; exact repetitions excluded.",
            "dataset": "headline",
            "sourceId": source["id"],
            "metrics": [
                {
                    "label": "Cache-enabled",
                    "field": "no_exact_hybrid_accuracy",
                    "format": "percent",
                },
                {
                    "label": "Uncached",
                    "field": "no_exact_direct_accuracy",
                    "format": "percent",
                },
                {
                    "label": "Cache-enabled gain",
                    "field": "no_exact_accuracy_delta_pp",
                    "format": "number",
                    "unit": "pp",
                    "signed": True,
                },
            ],
        },
        {
            "id": "runtime_card",
            "description": "Uncached-model elapsed time divided by cache-enabled retrieval elapsed time.",
            "dataset": "headline",
            "sourceId": source["id"],
            "metrics": [
                {
                    "label": "Original + paraphrase speedup",
                    "field": "no_exact_speedup_x",
                    "format": "number",
                    "unit": "x",
                },
                {
                    "label": "All-traffic speedup",
                    "field": "full_speedup_x",
                    "format": "number",
                    "unit": "x",
                },
            ],
        },
        {
            "id": "full_accuracy_card",
            "description": "Paired accuracy difference across all 1,509 traffic rows.",
            "dataset": "headline",
            "sourceId": source["id"],
            "metrics": [
                {"label": "Cache-enabled", "field": "full_hybrid_accuracy", "format": "percent"},
                {"label": "Uncached", "field": "full_direct_accuracy", "format": "percent"},
                {
                    "label": "Cache-enabled gain",
                    "field": "full_accuracy_delta_pp",
                    "format": "number",
                    "unit": "pp",
                    "signed": True,
                },
            ],
        },
    ]

    charts = [
        {
            "id": "pair_accuracy_chart",
            "title": "Accuracy by experiment pair",
            "subtitle": "Original + paraphrase n=1,006 first; all traffic types n=1,509 second; correct rows divided by selected rows",
            "intent": "comparison",
            "question": "Does cache-enabled retrieval preserve or improve accuracy in both experiment populations?",
            "rationale": "Grouped bars preserve the two matched populations and architecture pairing.",
            "type": "bar",
            "dataset": "pair_accuracy",
            "sourceId": source["id"],
            "encodings": {
                "x": {"field": "population", "type": "nominal", "label": "Population"},
                "y": {
                    "field": "accuracy",
                    "type": "quantitative",
                    "format": "percent",
                    "label": "Accuracy",
                },
                "color": {
                    "field": "architecture",
                    "type": "nominal",
                    "label": "Architecture",
                },
                "tooltip": [
                    {"field": "correct", "type": "quantitative", "label": "Correct"},
                    {"field": "rows", "type": "quantitative", "label": "Rows"},
                    {
                        "field": "delta_pp",
                        "type": "quantitative",
                        "label": "Cache-enabled delta (pp)",
                    },
                ],
            },
            "combinationRationale": "Color distinguishes cache-enabled retrieval from the uncached model within each population.",
            "valueFormat": "percent",
            "layout": "full",
            "maxRows": 4,
        },
        {
            "id": "efficiency_reduction_chart",
            "title": "Resource reduction from cache-enabled retrieval",
            "subtitle": "The exact-repetition-free 1,006-row comparison is shown first; positive values indicate less time, calls, or tokens",
            "intent": "comparison",
            "question": "How much workload does cache-enabled retrieval avoid versus the uncached model?",
            "rationale": "Six same-direction relative reductions share one zero-based percentage scale.",
            "type": "bar",
            "dataset": "efficiency_reduction",
            "sourceId": source["id"],
            "encodings": {
                "x": {
                    "field": "comparison",
                    "type": "nominal",
                    "label": "Experiment and metric",
                },
                "y": {
                    "field": "reduction",
                    "type": "quantitative",
                    "format": "percent",
                    "label": "Reduction",
                },
                "tooltip": [
                    {"field": "hybrid_value", "type": "quantitative", "label": "Cache-enabled retrieval"},
                    {"field": "direct_value", "type": "quantitative", "label": "Uncached model"},
                    {
                        "field": "direct_to_hybrid_ratio",
                        "type": "quantitative",
                        "label": "Uncached / cache-enabled",
                    },
                ],
            },
            "valueFormat": "percent",
            "layout": "full",
            "maxRows": 6,
        },
        {
            "id": "truncation_accuracy_chart",
            "title": "Accuracy by uncached-model input handling",
            "subtitle": "All-traffic 1,509-row pair; the uncached model middle-truncated 321 rows and left 1,188 rows intact",
            "intent": "comparison",
            "question": "Is the cache-enabled quality gain concentrated where the uncached model truncates its input?",
            "rationale": "Grouped bars compare architectures on the same predefined truncation cohorts.",
            "type": "bar",
            "dataset": "truncation_accuracy",
            "sourceId": source["id"],
            "encodings": {
                "x": {
                    "field": "direct_input_group",
                    "type": "nominal",
                    "label": "Uncached-model input handling",
                },
                "y": {
                    "field": "accuracy",
                    "type": "quantitative",
                    "format": "percent",
                    "label": "Accuracy",
                },
                "color": {
                    "field": "architecture",
                    "type": "nominal",
                    "label": "Architecture",
                },
                "tooltip": [
                    {"field": "correct", "type": "quantitative", "label": "Correct"},
                    {"field": "rows", "type": "quantitative", "label": "Rows"},
                ],
            },
            "combinationRationale": "Color distinguishes the two architectures within each input cohort.",
            "valueFormat": "percent",
            "layout": "full",
            "maxRows": 4,
        },
        {
            "id": "domain_delta_chart",
            "title": "Original-row accuracy delta by domain",
            "subtitle": "Cache-enabled retrieval minus the uncached model on 503 original questions; percentage points",
            "intent": "comparison",
            "question": "Across which domains is the cache-enabled quality difference positive or negative?",
            "rationale": "Signed bars expose heterogeneity around a meaningful zero reference.",
            "type": "bar",
            "dataset": "domain_accuracy",
            "sourceId": source["id"],
            "encodings": {
                "x": {"field": "domain", "type": "nominal", "label": "Domain"},
                "y": {
                    "field": "delta_pp",
                    "type": "quantitative",
                    "label": "Cache-enabled delta (pp)",
                },
                "tooltip": [
                    {"field": "rows", "type": "quantitative", "label": "Original questions"},
                    {
                        "field": "hybrid_accuracy",
                        "type": "quantitative",
                        "format": "percent",
                        "label": "Cache-enabled accuracy",
                    },
                    {
                        "field": "direct_accuracy",
                        "type": "quantitative",
                        "format": "percent",
                        "label": "Uncached accuracy",
                    },
                ],
            },
            "valueFormat": "number",
            "layout": "full",
            "maxRows": 6,
        },
    ]

    tables = [
        {
            "id": "run_summary_table",
            "title": "Experiment totals",
            "subtitle": "Manifest totals for the four requested runs",
            "dataset": "run_summary",
            "sourceId": source["id"],
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "run", "label": "Run", "type": "text"},
                {"field": "rows", "label": "Rows", "format": "number"},
                {"field": "correct", "label": "Correct", "format": "number"},
                {"field": "accuracy", "label": "Accuracy", "format": "percent"},
                {"field": "elapsed_hours", "label": "Runtime (h)", "format": "number"},
                {"field": "api_calls", "label": "API calls", "format": "number"},
                {"field": "total_tokens", "label": "Total tokens", "format": "number"},
                {"field": "tokens_per_row", "label": "Tokens / row", "format": "number"},
            ],
        },
        {
            "id": "paired_table",
            "title": "Paired accuracy comparisons",
            "subtitle": "95% percentile intervals resample the 503 source groups; 20,000 replicates",
            "dataset": "paired_comparisons",
            "sourceId": source["id"],
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "population", "label": "Population", "type": "text"},
                {"field": "rows", "label": "Rows", "format": "number"},
                {"field": "hybrid_correct", "label": "Cache-enabled correct", "format": "number"},
                {"field": "direct_correct", "label": "Uncached correct", "format": "number"},
                {
                    "field": "delta_pp",
                    "label": "Cache-enabled delta (pp)",
                    "format": "number",
                    "movement": True,
                },
                {"field": "ci_low_pp", "label": "95% CI low", "format": "number"},
                {"field": "ci_high_pp", "label": "95% CI high", "format": "number"},
                {
                    "field": "prediction_agreement",
                    "label": "Prediction agreement",
                    "format": "percent",
                },
            ],
        },
        {
            "id": "row_type_table",
            "title": "Accuracy by row type",
            "subtitle": "Original questions are independent; exact repetitions and paraphrases are linked traffic fixtures",
            "dataset": "row_type_detail",
            "sourceId": source["id"],
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "run", "label": "Run", "type": "text"},
                {"field": "row_type", "label": "Row type", "type": "text"},
                {"field": "rows", "label": "Rows", "format": "number"},
                {"field": "correct", "label": "Correct", "format": "number"},
                {"field": "accuracy", "label": "Accuracy", "format": "percent"},
                {"field": "api_calls", "label": "API calls", "format": "number"},
                {"field": "input_tokens", "label": "Input tokens", "format": "number"},
            ],
        },
        {
            "id": "stability_table",
            "title": "Retained-row rerun stability",
            "subtitle": "The same 1,006 original-question and paraphrase case IDs in both dataset variants",
            "dataset": "stability",
            "sourceId": source["id"],
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "architecture", "label": "Architecture", "type": "text"},
                {"field": "shared_rows", "label": "Shared rows", "format": "number"},
                {
                    "field": "prediction_agreement_rows",
                    "label": "Matching predictions",
                    "format": "number",
                },
                {
                    "field": "prediction_agreement",
                    "label": "Agreement",
                    "format": "percent",
                },
                {
                    "field": "full_subset_accuracy",
                    "label": "All-traffic subset accuracy",
                    "format": "percent",
                },
                {
                    "field": "no_exact_accuracy",
                    "label": "Original + paraphrase accuracy",
                    "format": "percent",
                },
                {
                    "field": "accuracy_shift_pp",
                    "label": "Rerun shift (pp)",
                    "format": "number",
                    "movement": True,
                },
            ],
        },
        {
            "id": "validation_table",
            "title": "Artifact integrity checks",
            "subtitle": "Row counts, uniqueness, request validity, and dataset identity",
            "dataset": "validation_checks",
            "sourceId": source["id"],
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "run", "label": "Run", "type": "text"},
                {"field": "rows", "label": "Rows", "format": "number"},
                {"field": "unique_case_ids", "label": "Unique case IDs", "format": "number"},
                {"field": "api_errors", "label": "API errors", "format": "number"},
                {"field": "invalid_choices", "label": "Invalid choices", "format": "number"},
                {"field": "status", "label": "Check status", "type": "text"},
            ],
        },
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "layout": "full",
            "body": "# Token-Efficient Long-Context Inference: Cache-Enabled Retrieval vs Uncached Model",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "layout": "full",
            "sourceId": source["id"],
            "body": (
                "## Technical summary\n\n"
                f"- **Primary result: cache-enabled retrieval used {headline['no_exact_token_reduction']:.1%} fewer tokens without sacrificing accuracy.** "
                f"In the more conservative 1,006-row comparison, which excludes exact repetitions, it processed "
                f"{no_exact_hybrid['total_tokens']/1_000_000:.1f}M tokens versus {no_exact_direct['total_tokens']/1_000_000:.1f}M for the uncached model. "
                f"Both made {no_exact_hybrid['api_calls']:,} API calls. Accuracy was {no_exact_pair['hybrid_accuracy']:.2%} versus "
                f"{no_exact_pair['direct_accuracy']:.2%}, a **+{no_exact_pair['delta_pp']:.2f} percentage-point** difference "
                f"(95% source-cluster bootstrap CI {no_exact_pair['ci_low_pp']:.2f} to {no_exact_pair['ci_high_pp']:.2f}).\n"
                f"- **The same comparison completed {headline['no_exact_speedup_x']:.2f}× faster.** Runtime is less controlled than token volume because "
                "the experiments ran as separate HPC jobs, but the direction agrees with the token reduction.\n"
                f"- **The all-traffic experiment shows the additional operational value of exact-cache reuse.** Across 1,509 rows, including 503 exact "
                f"repetitions, cache-enabled retrieval used {headline['full_token_reduction']:.1%} fewer tokens, 33.3% fewer API calls, and ran "
                f"{headline['full_speedup_x']:.2f}× faster. Accuracy was also {full_pair['delta_pp']:+.2f} points higher.\n"
                "- **The quality advantage is largest on overlength inputs.** Cache-enabled retrieval led by "
                f"{data['truncation_delta'][1]['delta_pp']:.2f} points on the 321 rows whose uncached prompt was middle-truncated, "
                f"versus {data['truncation_delta'][0]['delta_pp']:.2f} points on the 1,188 untruncated rows.\n"
                f"- **Treat the accuracy difference between dataset variants as rerun variation, not an effect of removing exact repetitions.** The uncached-model "
                f"outputs were identical on all {stability[UNCACHED_LABEL]['shared_rows']:,} retained rows, but cache-enabled retrieval "
                f"matched on {stability[CACHE_ENABLED_LABEL]['prediction_agreement_rows']:,}/{stability[CACHE_ENABLED_LABEL]['shared_rows']:,} "
                f"predictions and shifted {stability[CACHE_ENABLED_LABEL]['accuracy_shift_pp']:.2f} points despite identical routes, "
                "calls, and token totals."
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "layout": "full",
            "cardIds": [card["id"] for card in cards],
        },
        {
            "id": "experiment_definitions",
            "type": "markdown",
            "layout": "full",
            "sourceId": source["id"],
            "body": (
                "## How to read the four experiments\n\n"
                "- **Cache-enabled retrieval:** the production-style path, which can answer through retrieval, exact-cache reuse, or semantic-cache reuse.\n"
                "- **Uncached model:** the comparison path, where every selected row is sent independently to the same Qwen model without retrieval or cache reuse.\n"
                "- **All traffic types (1,509 rows):** 503 original benchmark questions, 503 exact repetitions of those questions, and 503 meaning-preserving paraphrases.\n"
                "- **Original + paraphrase only (1,006 rows):** the same original questions and paraphrases, with the 503 exact repetitions excluded."
            ),
        },
        {
            "id": "resource_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": source["id"],
            "body": (
                "## Excluding exact repetitions, cache-enabled retrieval halves token use while preserving accuracy\n\n"
                f"This 1,006-row comparison is the primary evaluation because neither system receives credit for verbatim repetition. "
                f"Cache-enabled retrieval processed {no_exact_hybrid['total_tokens']/1_000_000:.1f}M total tokens, compared with "
                f"{no_exact_direct['total_tokens']/1_000_000:.1f}M for the uncached model—a {headline['no_exact_token_reduction']:.1%} reduction. "
                f"Both systems made {no_exact_hybrid['api_calls']:,} API calls, so the reduction comes from processing less context per request rather "
                f"than making fewer requests. Cache-enabled retrieval was {headline['no_exact_speedup_x']:.2f}× faster and scored "
                f"{no_exact_pair['hybrid_accuracy']:.2%} accuracy versus {no_exact_pair['direct_accuracy']:.2%}; the evidence therefore supports "
                "a strong efficiency claim with no observed quality tradeoff."
            ),
        },
        {
            "id": "efficiency_chart",
            "type": "chart",
            "layout": "full",
            "chartId": "efficiency_reduction_chart",
        },
        {
            "id": "matched_quality",
            "type": "markdown",
            "layout": "full",
            "sourceId": source["id"],
            "body": (
                "## Accuracy is preserved—and modestly improved—in the exact-repetition-free comparison\n\n"
                "Both paired differences are positive after resampling at the 503-source level, which avoids treating "
                "linked rows as independent questions. Excluding exact repetitions, cache-enabled retrieval answered 34 more of 1,006 rows correctly, "
                f"a +{no_exact_pair['delta_pp']:.2f}-point difference with a 95% interval of {no_exact_pair['ci_low_pp']:.2f} to "
                f"{no_exact_pair['ci_high_pp']:.2f}. The all-traffic experiment, reported second, shows a 63-row or +{full_pair['delta_pp']:.2f}-point gain. "
                "The difference between these estimates is consistent with "
                "the observed cache-enabled rerun variability and should not be interpreted as proof that exact repetitions change reasoning quality."
            ),
        },
        {"id": "pair_accuracy", "type": "chart", "layout": "full", "chartId": "pair_accuracy_chart"},
        {
            "id": "paired_evidence",
            "type": "table",
            "layout": "full",
            "tableId": "paired_table",
        },
        {
            "id": "original_quality",
            "type": "markdown",
            "layout": "full",
            "sourceId": source["id"],
            "body": (
                "## Original-question quality shows the same pattern with visible rerun variation\n\n"
                "In the experiment excluding exact repetitions, cache-enabled retrieval answered 241 of 503 independent original questions correctly "
                "(47.91%), versus 225 (44.73%) for the uncached model, a +3.18-point difference. In the all-traffic experiment, the same original-question "
                "slice scored 246 (48.91%) versus 225 (44.73%), a +4.17-point difference. Exact-repetition rows "
                "repeat the original questions verbatim, while semantic-paraphrase rows preserve their meaning, so original-question "
                "accuracy should lead advisor-facing claims about question-solving quality."
            ),
        },
        {
            "id": "row_type_evidence",
            "type": "table",
            "layout": "full",
            "tableId": "row_type_table",
        },
        {
            "id": "exact_traffic_extension",
            "type": "markdown",
            "layout": "full",
            "sourceId": source["id"],
            "body": (
                "## Exact repetitions increase the operational savings in repeated traffic\n\n"
                "The 1,509-row experiment is a secondary traffic scenario rather than the primary quality comparison: it adds one verbatim repetition "
                "for every original question. Those rows represent a favorable cache opportunity, so their benefit should not be generalized to workloads "
                "without repeated questions. Within that stated scope, the result is operationally meaningful. "
                f"Across all traffic types, cache-enabled retrieval used {full_hybrid['api_calls']:,} calls and {full_hybrid['total_tokens']/1_000_000:.1f}M "
                f"tokens versus {full_direct['api_calls']:,} calls and {full_direct['total_tokens']/1_000_000:.1f}M tokens "
                "for the uncached model. Excluding 503 exact repetitions removed 503 uncached-model calls and about 60.0M uncached-model tokens. "
                "Cache-enabled calls and tokens did not change because all 503 exact repetitions were answered as zero-call exact-cache hits; "
                f"the cache-enabled run without those rows still used {no_exact_hybrid['api_calls']:,} calls and {no_exact_hybrid['total_tokens']/1_000_000:.1f}M tokens. "
                "This explains why the all-traffic result reaches a 66.2% token reduction and 33.3% fewer API calls, beyond the already substantial "
                "token reduction observed when exact repetitions are excluded."
            ),
        },
        {
            "id": "run_summary",
            "type": "table",
            "layout": "full",
            "tableId": "run_summary_table",
        },
        {
            "id": "truncation_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": source["id"],
            "body": (
                "## Retrieval packing matters most where the uncached prompt is truncated\n\n"
                f"On the 321 rows whose uncached prompt was truncated, cache-enabled retrieval achieved {data['truncation_delta'][1]['hybrid_accuracy']:.2%} "
                f"accuracy versus {data['truncation_delta'][1]['direct_accuracy']:.2%} for the uncached model, a "
                f"{data['truncation_delta'][1]['delta_pp']:.2f}-point gap. On the 1,188 rows that fit the uncached-model input "
                f"budget, the gap was only {data['truncation_delta'][0]['delta_pp']:.2f} points. This descriptive split "
                "supports a long-context mechanism: dense child retrieval and exact token-budget packing preserve useful "
                "evidence that middle truncation can discard. It does not by itself isolate retrieval from every other "
                "execution difference."
            ),
        },
        {
            "id": "truncation_chart",
            "type": "chart",
            "layout": "full",
            "chartId": "truncation_accuracy_chart",
        },
        {
            "id": "domain_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": source["id"],
            "body": (
                "## Original-row gains are heterogeneous across domains\n\n"
                "The all-traffic cache-enabled run led most strongly in Code Repository Understanding (+14.0 points) and Long "
                "In-context Learning (+9.88 points). It was essentially tied in Single-Document QA (-0.57 points). "
                "These domain cuts are descriptive and vary in size from 33 to 175 originals, so they identify where to "
                "inspect examples rather than establish domain-specific significance."
            ),
        },
        {"id": "domain_chart", "type": "chart", "layout": "full", "chartId": "domain_delta_chart"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "layout": "full",
            "sourceId": source["id"],
            "body": (
                "## Scope, data, and metric definitions\n\n"
                "The primary 1,006-row dataset contains 503 original questions and 503 meaning-preserving paraphrases; exact repetitions are excluded. "
                "The secondary all-traffic dataset adds one exact repetition per source group, for 1,509 rows. Accuracy is correct rows divided by selected rows; "
                "runtime is manifest wall time; API calls and tokens are server-reported manifest totals. Repetition and paraphrase rows are "
                "linked traffic fixtures, not 1,006 additional independent benchmark questions. Original-only accuracy is therefore the "
                "cleanest measure of question-solving quality, while all-row accuracy is an end-to-end traffic measure."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "layout": "full",
            "sourceId": source["id"],
            "body": (
                "## Methodology and validation\n\n"
                "All four artifacts use identical suite and source JSON hashes, Qwen3.6-35B-A3B execution, the same 262,144-token "
                "context profile, and the same structured A–D decoder contract. Case IDs match exactly within each paired population. "
                "Every row completed successfully with a valid choice and zero API errors. Headline accuracy intervals use a paired "
                "percentile bootstrap with `source_id` as the cluster, 20,000 replicates, and fixed seeds. Resource comparisons use the "
                "matched manifest totals without normalizing away the intentionally different row counts."
            ),
        },
        {
            "id": "validation_evidence",
            "type": "table",
            "layout": "full",
            "tableId": "validation_table",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "layout": "full",
            "sourceId": source["id"],
            "body": (
                "## Limitations, uncertainty, and robustness\n\n"
                "**Cache-enabled rerun variability is material but smaller than the paired advantage.** The cache-enabled run excluding exact repetitions "
                f"changed {stability[CACHE_ENABLED_LABEL]['shared_rows'] - stability[CACHE_ENABLED_LABEL]['prediction_agreement_rows']} of "
                f"{stability[CACHE_ENABLED_LABEL]['shared_rows']:,} retained predictions and scored {abs(stability[CACHE_ENABLED_LABEL]['accuracy_shift_pp']):.2f} "
                "points lower than the same-row subset of the all-traffic cache-enabled run. The uncached-model rerun reproduced all retained predictions. "
                "A single run per condition cannot estimate cache-enabled execution variance.\n\n"
                "**Runtime is operational, not a controlled latency estimate.** Runs occurred in separate HPC jobs, so node load, service "
                "queueing, and filesystem behavior may contribute to elapsed time. Calls and token totals are more portable workload measures.\n\n"
                "**The truncation and domain cuts are descriptive.** They were computed after the runs and are not adjusted for multiple "
                "comparisons. The report makes no causal claim that retrieval alone produces the observed accuracy difference."
            ),
        },
        {
            "id": "stability_evidence",
            "type": "table",
            "layout": "full",
            "tableId": "stability_table",
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Recommended next steps\n\n"
                "1. **Lead with the 1,006-row result:** 49.3% fewer tokens with no observed accuracy tradeoff after exact repetitions are excluded.\n"
                "2. **Present the 1,509-row result as a repeated-traffic scenario** that quantifies the additional operational value of exact-cache hits.\n"
                "3. **Repeat the cache-enabled retrieval condition at least three times** with the same service build and record seed, node, and vLLM settings to quantify execution variance.\n"
                "4. **Audit the 107 overlength original sources** because they account for the clearest accuracy separation and directly test the retrieval-packing hypothesis."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Further questions\n\n"
                "- What causes the 5.2% cache-enabled prediction disagreement across nominally matched reruns despite temperature zero?\n"
                "- Do repeated cache-enabled runs preserve the large advantage on rows whose uncached prompts are truncated?\n"
                "- Which evidence ranges are selected when cache-enabled retrieval is correct but the uncached model is wrong on overlength examples, and are those ranges removed by middle truncation?"
            ),
        },
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": "Token-Efficient Long-Context Inference: Cache-Enabled Retrieval vs Uncached Model",
        "description": (
            "Technical comparison led by the exact-repetition-free 1,006-row evaluation, "
            "with the 1,509-row repeated-traffic experiment presented secondarily."
        ),
        "generatedAt": generated_at,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": [source],
        "blocks": blocks,
        "notes": [
            "Audience: technical advisor.",
            "Required technical-report sections are present in the prescribed order.",
            "All charts use comparison bars because the evidence consists of discrete experiment, cohort, and domain contrasts rather than time series.",
            "The source file comparison_data.json preserves validation checks, chart map, and calculation details.",
        ],
    }
    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "headline": [data["headline"]],
            "run_summary": data["run_summary"],
            "paired_comparisons": data["paired_comparisons"],
            "pair_accuracy": data["pair_accuracy"],
            "efficiency_reduction": data["efficiency_reduction"],
            "truncation_accuracy": data["truncation_accuracy"],
            "domain_accuracy": data["domain_accuracy"],
            "stability": data["stability"],
            "row_type_detail": data["row_type_detail"],
            "validation_checks": data["validation"]["checks"],
        },
        "accessIssues": [],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": [source],
    }


def main() -> None:
    runs = load_runs()
    validation = validate_sources(runs)
    comparison_data = build_comparison_data(runs, validation)
    artifact = build_artifact(comparison_data)
    (OUT_DIR / "comparison_data.json").write_text(
        json.dumps(comparison_data, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "artifact.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT_DIR / 'comparison_data.json'}")
    print(f"wrote {OUT_DIR / 'artifact.json'}")


if __name__ == "__main__":
    main()
