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
DEFAULT_SELECTION_STRATEGY = "random"
DEFAULT_TOKEN_BUCKETS = (
    ("short", 0, 75000),
    ("medium", 75001, 150000),
    ("long", 150001, 300000),
)


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


def _token_count(row: dict) -> int | None:
    raw = (row.get("token_count") or "").strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def parse_token_buckets(value: str) -> tuple[tuple[str, int, int], ...]:
    buckets: list[tuple[str, int, int]] = []
    for raw_bucket in (value or "").split(","):
        raw_bucket = raw_bucket.strip()
        if not raw_bucket:
            continue
        parts = [part.strip() for part in raw_bucket.split(":")]
        if len(parts) != 3:
            raise ValueError("Token buckets must use name:min:max format")
        name, raw_min, raw_max = parts
        if not name:
            raise ValueError("Token bucket names cannot be blank")
        try:
            min_tokens = int(raw_min)
            max_tokens = int(raw_max)
        except ValueError as exc:
            raise ValueError(f"Token bucket {name!r} has non-integer bounds") from exc
        if min_tokens < 0 or max_tokens < min_tokens:
            raise ValueError(f"Token bucket {name!r} must satisfy 0 <= min <= max")
        buckets.append((name, min_tokens, max_tokens))
    if not buckets:
        raise ValueError("At least one token bucket is required")
    return tuple(buckets)


def _format_token_buckets(buckets: Sequence[tuple[str, int, int]]) -> str:
    return ",".join(f"{name}:{min_tokens}:{max_tokens}" for name, min_tokens, max_tokens in buckets)


def _bucket_for_token_count(token_count: int, buckets: Sequence[tuple[str, int, int]]) -> str:
    for name, min_tokens, max_tokens in buckets:
        if min_tokens <= token_count <= max_tokens:
            return name
    return ""


def sample_rows(
    rows: list[dict],
    sample_size: int,
    row_types: Sequence[str],
    seed: int,
    max_token_count: int = 0,
    selection_strategy: str = DEFAULT_SELECTION_STRATEGY,
    token_buckets: Sequence[tuple[str, int, int]] = DEFAULT_TOKEN_BUCKETS,
) -> list[dict]:
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if max_token_count < 0:
        raise ValueError("max_token_count must be non-negative")
    if selection_strategy not in {"random", "shortest", "token_stratified"}:
        raise ValueError("selection_strategy must be random, shortest, or token_stratified")

    row_type_set = set(row_types)
    source_ids_by_type: dict[str, set[str]] = {row_type: set() for row_type in row_types}
    rows_by_source: dict[str, list[dict]] = {}
    for row in rows:
        row_type = row.get("row_type", "")
        source_id = row.get("source_id", "")
        if row_type in row_type_set and source_id:
            source_ids_by_type[row_type].add(source_id)
            rows_by_source.setdefault(source_id, []).append(row)

    missing_row_types = [row_type for row_type, source_ids in source_ids_by_type.items() if not source_ids]
    if missing_row_types:
        raise ValueError(f"Input CSV has no rows for row types: {', '.join(missing_row_types)}")

    eligible_source_ids = set.intersection(*source_ids_by_type.values()) if source_ids_by_type else set()
    source_token_counts: dict[str, int] = {}
    if max_token_count > 0 or selection_strategy in {"shortest", "token_stratified"}:
        for source_id in sorted(eligible_source_ids):
            token_counts = [
                token_count
                for row in rows_by_source.get(source_id, [])
                if row.get("row_type", "") in row_type_set
                for token_count in [_token_count(row)]
                if token_count is not None
            ]
            if not token_counts:
                eligible_source_ids.discard(source_id)
                continue
            source_token_counts[source_id] = max(token_counts)

    if max_token_count > 0:
        eligible_source_ids = {
            source_id
            for source_id in eligible_source_ids
            if source_token_counts.get(source_id, max_token_count + 1) <= max_token_count
        }

    if sample_size > len(eligible_source_ids):
        raise ValueError(
            f"Requested {sample_size} source-linked samples, but only {len(eligible_source_ids)} "
            f"source_ids have all requested row types: {', '.join(row_types)}"
        )

    if selection_strategy == "token_stratified":
        buckets = tuple(token_buckets or DEFAULT_TOKEN_BUCKETS)
        bucket_source_ids: dict[str, list[str]] = {name: [] for name, _, _ in buckets}
        for source_id in sorted(eligible_source_ids):
            bucket_name = _bucket_for_token_count(source_token_counts.get(source_id, -1), buckets)
            if bucket_name:
                bucket_source_ids[bucket_name].append(source_id)

        base_count = sample_size // len(buckets)
        remainder = sample_size % len(buckets)
        selected_source_ids = set()
        rng = random.Random(seed)
        for idx, (bucket_name, min_tokens, max_tokens) in enumerate(buckets):
            requested = base_count + (1 if idx < remainder else 0)
            if requested == 0:
                continue
            available = bucket_source_ids.get(bucket_name, [])
            if requested > len(available):
                raise ValueError(
                    f"Token bucket {bucket_name!r} ({min_tokens}-{max_tokens}) requested {requested} "
                    f"source-linked samples, but only {len(available)} are eligible"
                )
            selected_source_ids.update(rng.sample(available, requested))
    elif selection_strategy == "shortest":
        ordered_source_ids = sorted(
            eligible_source_ids,
            key=lambda source_id: (source_token_counts.get(source_id, 0), source_id),
        )
        selected_source_ids = set(ordered_source_ids[:sample_size])
    else:
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


def print_token_bucket_summary(rows: list[dict], buckets: Sequence[tuple[str, int, int]]) -> None:
    source_token_counts: dict[str, int] = {}
    for row in rows:
        source_id = row.get("source_id", "")
        token_count = _token_count(row)
        if source_id and token_count is not None:
            source_token_counts[source_id] = max(token_count, source_token_counts.get(source_id, 0))

    if not source_token_counts:
        print("[LONGBENCH-V2] Token bucket summary: <no token_count values>")
        return

    bucket_values: dict[str, list[int]] = {name: [] for name, _, _ in buckets}
    for token_count in source_token_counts.values():
        bucket_name = _bucket_for_token_count(token_count, buckets)
        if bucket_name:
            bucket_values[bucket_name].append(token_count)

    print("[LONGBENCH-V2] Token bucket summary:")
    print("bucket  sources  min_tokens  max_tokens")
    print("------  -------  ----------  ----------")
    for name, _, _ in buckets:
        values = bucket_values.get(name, [])
        if values:
            print(f"{name.ljust(6)}  {str(len(values)).rjust(7)}  {str(min(values)).rjust(10)}  {str(max(values)).rjust(10)}")
        else:
            print(f"{name.ljust(6)}  {str(0).rjust(7)}  {'-'.rjust(10)}  {'-'.rjust(10)}")


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
        "--max-token-count",
        type=int,
        default=0,
        help="Only sample source_ids whose requested rows have token_count at or below this value. 0 disables the cap.",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=["random", "shortest", "token_stratified"],
        default=DEFAULT_SELECTION_STRATEGY,
        help="Choose eligible sources randomly, shortest-first, or stratified by token_count. Default: random.",
    )
    parser.add_argument(
        "--token-buckets",
        default=_format_token_buckets(DEFAULT_TOKEN_BUCKETS),
        help=(
            "Comma-separated token buckets for token_stratified sampling as name:min:max. "
            f"Default: {_format_token_buckets(DEFAULT_TOKEN_BUCKETS)}"
        ),
    )
    parser.add_argument(
        "--row-types",
        default=",".join(DEFAULT_ROW_TYPES),
        help="Comma-separated row types to sample together by source_id. Default: original,exact,semantic",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling. Default: 0")
    args = parser.parse_args(argv)

    row_types = parse_row_types(args.row_types)
    token_buckets = parse_token_buckets(args.token_buckets)
    rows = _read_rows(args.input_path)
    sampled_rows = sample_rows(
        rows,
        sample_size=args.sample_size,
        row_types=row_types,
        seed=args.seed,
        max_token_count=args.max_token_count,
        selection_strategy=args.selection_strategy,
        token_buckets=token_buckets,
    )
    _write_rows(args.output_path, sampled_rows)
    print(f"[LONGBENCH-V2] Wrote {len(sampled_rows)} rows to {args.output_path}")
    print_row_type_summary(sampled_rows)
    if args.selection_strategy == "token_stratified":
        print_token_bucket_summary(sampled_rows, token_buckets)


if __name__ == "__main__":
    main()
