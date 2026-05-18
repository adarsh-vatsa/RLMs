"""Concatenate LongBench-v2 bridge_rows.csv files across experiment runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Sequence


LONG_BENCH_ARTIFACT_DIRS = ("longbench_v2", "longbench_v2_rlm", "longbench_v2_api")
DEFAULT_OUTPUT_PATH = Path("benchmark_artifacts/longbench_v2_combined_bridge_rows.csv")
METADATA_COLUMNS = [
    "experiment_namespace",
    "run_id",
    "benchmark_target",
    "baseline_type",
    "mode",
    "llm_provider",
    "executor_model",
    "evaluator_model",
    "api_provider",
    "api_model",
    "rlm_backend",
    "rlm_model",
    "manifest_note",
    "run_elapsed_seconds",
    "source_bridge_rows_csv",
]


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _iter_bridge_csv_paths(artifact_root: Path, namespaces: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for namespace in namespaces:
        namespace_dir = artifact_root / namespace
        if not namespace_dir.exists():
            continue
        paths.extend(sorted(namespace_dir.glob("*/bridge_rows.csv")))
    return sorted(paths)


def _load_rows(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def _metadata_for_path(path: Path) -> dict:
    run_dir = path.parent
    namespace = run_dir.parent.name
    manifest = _read_manifest(run_dir / "manifest.json")
    return {
        "experiment_namespace": namespace,
        "run_id": _coerce_text(manifest.get("run_id") or run_dir.name),
        "benchmark_target": _coerce_text(manifest.get("benchmark_target") or manifest.get("official_target")),
        "baseline_type": _coerce_text(manifest.get("baseline_type")),
        "mode": _coerce_text(manifest.get("mode")),
        "llm_provider": _coerce_text(manifest.get("llm_provider")),
        "executor_model": _coerce_text(manifest.get("executor_model")),
        "evaluator_model": _coerce_text(manifest.get("evaluator_model")),
        "api_provider": _coerce_text(manifest.get("api_provider")),
        "api_model": _coerce_text(manifest.get("api_model")),
        "rlm_backend": _coerce_text(manifest.get("rlm_backend")),
        "rlm_model": _coerce_text(manifest.get("rlm_model")),
        "manifest_note": _coerce_text(manifest.get("note")),
        "run_elapsed_seconds": _coerce_text(manifest.get("elapsed_seconds")),
        "source_bridge_rows_csv": str(path),
    }


def concatenate_bridge_rows(paths: Sequence[Path]) -> tuple[list[dict], list[str]]:
    all_rows: list[dict] = []
    data_columns: list[str] = []
    seen_columns: set[str] = set()

    for path in paths:
        rows, columns = _load_rows(path)
        for column in columns:
            if column not in seen_columns and column not in METADATA_COLUMNS:
                seen_columns.add(column)
                data_columns.append(column)
        metadata = _metadata_for_path(path)
        for row in rows:
            all_rows.append({**metadata, **row})

    return all_rows, [*METADATA_COLUMNS, *data_columns]


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict], paths: Sequence[Path], output_path: Path) -> None:
    print(f"[LONGBENCH-V2] Bridge CSV files: {len(paths)}")
    print(f"[LONGBENCH-V2] Combined rows: {len(rows)}")
    print(f"[LONGBENCH-V2] Output: {output_path}")

    for label, key in [
        ("namespace", "experiment_namespace"),
        ("benchmark_target", "benchmark_target"),
        ("row_type", "row_type"),
    ]:
        counts = Counter(row.get(key, "") or "<blank>" for row in rows)
        if not counts:
            continue
        print(f"[LONGBENCH-V2] Counts by {label}:")
        for value, count in sorted(counts.items()):
            print(f"  {value}: {count}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Concatenate LongBench-v2 bridge_rows.csv artifacts")
    parser.add_argument("--artifact-root", type=Path, default=Path("benchmark_artifacts"))
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--namespaces",
        default=",".join(LONG_BENCH_ARTIFACT_DIRS),
        help="Comma-separated LongBench artifact namespaces to scan. Default: longbench_v2,longbench_v2_rlm,longbench_v2_api",
    )
    args = parser.parse_args(argv)

    namespaces = tuple(part.strip() for part in args.namespaces.split(",") if part.strip())
    invalid = [namespace for namespace in namespaces if namespace not in LONG_BENCH_ARTIFACT_DIRS]
    if invalid:
        raise ValueError(
            "Only LongBench-v2 namespaces are supported: "
            f"{', '.join(LONG_BENCH_ARTIFACT_DIRS)}. Invalid: {', '.join(invalid)}"
        )

    paths = _iter_bridge_csv_paths(args.artifact_root, namespaces)
    if not paths:
        raise FileNotFoundError(
            f"No LongBench-v2 bridge_rows.csv files found under {args.artifact_root} "
            f"for namespaces: {', '.join(namespaces)}"
        )

    rows, columns = concatenate_bridge_rows(paths)
    _write_csv(args.output_path, rows, columns)
    _print_summary(rows, paths, args.output_path)


if __name__ == "__main__":
    main()
