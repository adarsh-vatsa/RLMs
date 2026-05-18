"""Summarize DP memo benchmark manifests for quick cold/warm comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def resolve_manifest(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    candidate = path / "manifest.json"
    if candidate.exists():
        return candidate
    manifests = sorted(path.glob("*/manifest.json"))
    if manifests:
        return manifests[-1]
    raise FileNotFoundError(f"No manifest.json found for {path}")


def summarize_manifest(path: Path) -> dict:
    manifest_path = resolve_manifest(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    totals = dict(manifest.get("totals", {}))
    settings = dict(manifest.get("settings", {}))
    memo_stats = dict(manifest.get("memo_stats", {}))

    def first_present(*keys: str):
        for key in keys:
            if key in totals:
                return totals[key]
        return None

    return {
        "manifest": str(manifest_path),
        "benchmark": manifest.get("benchmark"),
        "data_kind": manifest.get("data_kind"),
        "corpus_id": manifest.get("corpus_id"),
        "model": manifest.get("model"),
        "lengths": settings.get("lengths"),
        "solver_mode": settings.get("solver_mode"),
        "samples": first_present("samples", "questions"),
        "accuracy_contains": totals.get("accuracy_contains"),
        "model_calls": totals.get("model_calls"),
        "aggregate_calls": totals.get("aggregate_calls"),
        "avg_latency_ms": totals.get("avg_latency_ms"),
        "initial_coverage_ratio": first_present(
            "initial_coverage_ratio",
            "fact_initial_coverage_ratio",
        ),
        "final_coverage_ratio": first_present(
            "final_coverage_ratio",
            "fact_final_coverage_ratio",
        ),
        "exact_replay_checks": first_present("exact_replay_checks", "exact_replays"),
        "memo_entries": manifest.get("memo_entries", memo_stats.get("entry_count")),
        "fragment_mix": memo_stats.get("by_fragment_kind", {}),
        "dependency_edge_count": memo_stats.get("dependency_edge_count"),
        "evidence_span_count": memo_stats.get("evidence_span_count"),
        "v1_model_calls": totals.get("v1_model_calls"),
        "invalidated_entries": totals.get("invalidated_entries"),
        "v2_model_calls": totals.get("v2_model_calls"),
        "v2_initial_coverage_ratio": totals.get("v2_initial_coverage_ratio"),
        "v2_reused_windows": totals.get("v2_reused_windows"),
        "v2_missing_windows": totals.get("v2_missing_windows"),
        "warm_model_calls": totals.get("warm_model_calls"),
    }


def summarize(paths: Iterable[Path]) -> list[dict]:
    return [summarize_manifest(path) for path in paths]


def compact_nulls(row: dict) -> dict:
    return {key: value for key, value in row.items() if value is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DP memo benchmark manifests.")
    parser.add_argument("paths", nargs="+", help="Manifest files or run directories")
    parser.add_argument("--compact", action="store_true", help="Omit null-valued fields from output rows.")
    args = parser.parse_args()

    rows = summarize(Path(path) for path in args.paths)
    if args.compact:
        rows = [compact_nulls(row) for row in rows]
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
