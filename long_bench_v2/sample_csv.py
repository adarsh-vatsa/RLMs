"""Sample a balanced LongBench-v2 CSV suite for cheap test runs."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from long_bench_v2.export_csv import CSV_COLUMNS


DEFAULT_INPUT_PATH = Path("benchmark_data/long_bench_v2/data_cache_suite.csv")
DEFAULT_OUTPUT_PATH = Path("benchmark_data/long_bench_v2/data_cache_suite_sample.csv")
DEFAULT_ROW_TYPES = ("original", "exact", "semantic")


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = [column for column in CSV_COLUMNS if column not in (reader.fieldnames or [])]
        required_missing = [column for column in missing_columns if column != "token_count"]
        if required_missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(required_missing)}")
        return [{column: row.get(column, "") for column in CSV_COLUMNS} for row in reader]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_row_types(value: str) -> tuple[str, ...]:
    row_types = tuple(part.strip() for part in value.split(",") if part.strip())
    if not row_types:
        raise ValueError("At least one row type is required")
    return row_types


def sample_rows(rows: list[dict], sample_size: int, row_types: Sequence[str], seed: int) -> list[dict]:
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")

    row_type_set = set(row_types)
    source_ids_by_type: dict[str, set[str]] = {row_type: set() for row_type in row_types}
    for row in rows:
        row_type = row.get("row_type", "")
        source_id = row.get("source_id", "")
        if row_type in row_type_set and source_id:
            source_ids_by_type[row_type].add(source_id)

    missing_row_types = [row_type for row_type, source_ids in source_ids_by_type.items() if not source_ids]
    if missing_row_types:
        raise ValueError(f"Input CSV has no rows for row types: {', '.join(missing_row_types)}")

    eligible_source_ids = set.intersection(*source_ids_by_type.values()) if source_ids_by_type else set()
    if sample_size > len(eligible_source_ids):
        raise ValueError(
            f"Requested {sample_size} source-linked samples, but only {len(eligible_source_ids)} "
            f"source_ids have all requested row types: {', '.join(row_types)}"
        )

    rng = random.Random(seed)
    selected_source_ids = set(rng.sample(sorted(eligible_source_ids), sample_size))
    sampled_rows = [
        row
        for row in rows
        if row.get("row_type", "") in row_type_set and row.get("source_id", "") in selected_source_ids
    ]

    counts = Counter(row.get("row_type", "") for row in sampled_rows)
    bad_counts = {
        row_type: counts.get(row_type, 0)
        for row_type in row_types
        if counts.get(row_type, 0) != sample_size
    }
    if bad_counts:
        details = ", ".join(f"{row_type}={count}" for row_type, count in sorted(bad_counts.items()))
        raise ValueError(f"Sample is not balanced at {sample_size} rows per type: {details}")

    return sampled_rows


def print_row_type_summary(rows: list[dict]) -> None:
    counts = Counter(row.get("row_type", "") or "<blank>" for row in rows)
    if not counts:
        print("[LONGBENCH-V2] Row type summary: <empty>")
        return

    summary_rows = sorted(counts.items())
    row_type_width = max(len("row_type"), *(len(row_type) for row_type, _ in summary_rows))
    count_width = max(len("count"), *(len(str(count)) for _, count in summary_rows))
    print("[LONGBENCH-V2] Row type summary:")
    print(f"{'row_type'.ljust(row_type_width)}  {'count'.rjust(count_width)}")
    print(f"{'-' * row_type_width}  {'-' * count_width}")
    for row_type, count in summary_rows:
        print(f"{row_type.ljust(row_type_width)}  {str(count).rjust(count_width)}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sample a balanced LongBench-v2 CSV suite")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--sample-size", type=int, default=10, help="Rows to keep per row type. Default: 10")
    parser.add_argument(
        "--row-types",
        default=",".join(DEFAULT_ROW_TYPES),
        help="Comma-separated row types to sample together by source_id. Default: original,exact,semantic",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling. Default: 0")
    args = parser.parse_args(argv)

    row_types = parse_row_types(args.row_types)
    rows = _read_rows(args.input_path)
    sampled_rows = sample_rows(rows, sample_size=args.sample_size, row_types=row_types, seed=args.seed)
    _write_rows(args.output_path, sampled_rows)
    print(f"[LONGBENCH-V2] Wrote {len(sampled_rows)} rows to {args.output_path}")
    print_row_type_summary(sampled_rows)


if __name__ == "__main__":
    main()
