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


def parse_csv_values(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _token_count(row: dict) -> int | None:
    raw = (row.get("token_count") or "").strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def sample_rows(
    rows: list[dict],
    sample_size: int,
    row_types: Sequence[str],
    seed: int,
    min_token_count: int = 0,
    max_token_count: int = 0,
    selection_strategy: str = DEFAULT_SELECTION_STRATEGY,
    domains: Sequence[str] = (),
    samples_per_domain: int = 0,
) -> list[dict]:
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if samples_per_domain < 0:
        raise ValueError("samples_per_domain must be non-negative")
    if min_token_count < 0:
        raise ValueError("min_token_count must be non-negative")
    if max_token_count < 0:
        raise ValueError("max_token_count must be non-negative")
    if max_token_count > 0 and min_token_count > max_token_count:
        raise ValueError("min_token_count must be less than or equal to max_token_count")
    if selection_strategy not in {"random", "shortest", "longest"}:
        raise ValueError("selection_strategy must be random, shortest, or longest")

    row_type_set = set(row_types)
    domain_filter = {domain.strip() for domain in domains if domain.strip()}
    source_ids_by_type: dict[str, set[str]] = {row_type: set() for row_type in row_types}
    rows_by_source: dict[str, list[dict]] = {}
    source_domain_sets: dict[str, set[str]] = {}
    for row in rows:
        row_type = row.get("row_type", "")
        source_id = row.get("source_id", "")
        if row_type in row_type_set and source_id:
            source_ids_by_type[row_type].add(source_id)
            rows_by_source.setdefault(source_id, []).append(row)
            domain = row.get("domain", "").strip()
            if domain:
                source_domain_sets.setdefault(source_id, set()).add(domain)

    missing_row_types = [row_type for row_type, source_ids in source_ids_by_type.items() if not source_ids]
    if missing_row_types:
        raise ValueError(f"Input CSV has no rows for row types: {', '.join(missing_row_types)}")

    eligible_source_ids = set.intersection(*source_ids_by_type.values()) if source_ids_by_type else set()
    source_domains: dict[str, str] = {}
    for source_id in sorted(eligible_source_ids):
        source_domain_set = source_domain_sets.get(source_id, set())
        if len(source_domain_set) > 1:
            details = ", ".join(sorted(source_domain_set))
            raise ValueError(f"source_id {source_id} spans multiple domains: {details}")
        source_domains[source_id] = next(iter(source_domain_set), "")
    if domain_filter:
        eligible_source_ids = {
            source_id
            for source_id in eligible_source_ids
            if source_domains.get(source_id, "") in domain_filter
        }

    source_token_counts: dict[str, int] = {}
    if min_token_count > 0 or max_token_count > 0 or selection_strategy in {"shortest", "longest"}:
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

    if min_token_count > 0:
        eligible_source_ids = {
            source_id
            for source_id in eligible_source_ids
            if source_token_counts.get(source_id, min_token_count - 1) >= min_token_count
        }

    if max_token_count > 0:
        eligible_source_ids = {
            source_id
            for source_id in eligible_source_ids
            if source_token_counts.get(source_id, max_token_count + 1) <= max_token_count
        }

    def choose_source_ids(source_ids: list[str], count: int, rng: random.Random) -> list[str]:
        if selection_strategy == "shortest":
            ordered_source_ids = sorted(
                source_ids,
                key=lambda source_id: (source_token_counts.get(source_id, 0), source_id),
            )
            return ordered_source_ids[:count]
        if selection_strategy == "longest":
            ordered_source_ids = sorted(
                source_ids,
                key=lambda source_id: (-source_token_counts.get(source_id, 0), source_id),
            )
            return ordered_source_ids[:count]
        return rng.sample(sorted(source_ids), count)

    rng = random.Random(seed)
    if samples_per_domain > 0:
        source_ids_by_domain: dict[str, list[str]] = {}
        for source_id in sorted(eligible_source_ids):
            domain = source_domains.get(source_id, "")
            if domain:
                source_ids_by_domain.setdefault(domain, []).append(source_id)
        if not source_ids_by_domain:
            raise ValueError("No eligible source_ids remain after domain filtering")
        undersized_domains = {
            domain: len(source_ids)
            for domain, source_ids in source_ids_by_domain.items()
            if len(source_ids) < samples_per_domain
        }
        if undersized_domains:
            details = ", ".join(
                f"{domain}={count}" for domain, count in sorted(undersized_domains.items())
            )
            raise ValueError(
                f"Requested {samples_per_domain} source-linked samples per domain, "
                f"but these domains have fewer eligible source_ids: {details}"
            )
        selected_source_ids = {
            source_id
            for domain in sorted(source_ids_by_domain)
            for source_id in choose_source_ids(
                source_ids_by_domain[domain],
                samples_per_domain,
                rng,
            )
        }
    elif sample_size > len(eligible_source_ids):
        raise ValueError(
            f"Requested {sample_size} source-linked samples, but only {len(eligible_source_ids)} "
            f"source_ids have all requested row types: {', '.join(row_types)}"
        )
    else:
        selected_source_ids = set(choose_source_ids(sorted(eligible_source_ids), sample_size, rng))

    sampled_rows = [
        row
        for row in rows
        if row.get("row_type", "") in row_type_set and row.get("source_id", "") in selected_source_ids
    ]

    selected_source_count = len(selected_source_ids)
    counts = Counter(row.get("row_type", "") for row in sampled_rows)
    bad_counts = {
        row_type: counts.get(row_type, 0)
        for row_type in row_types
        if counts.get(row_type, 0) != selected_source_count
    }
    if bad_counts:
        details = ", ".join(f"{row_type}={count}" for row_type, count in sorted(bad_counts.items()))
        raise ValueError(f"Sample is not balanced at {selected_source_count} rows per type: {details}")

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


def print_domain_summary(rows: list[dict]) -> None:
    counts = Counter(row.get("domain", "") or "<blank>" for row in rows)
    if not counts:
        print("[LONGBENCH-V2] Domain summary: <empty>")
        return

    summary_rows = sorted(counts.items())
    domain_width = max(len("domain"), *(len(domain) for domain, _ in summary_rows))
    count_width = max(len("count"), *(len(str(count)) for _, count in summary_rows))
    print("[LONGBENCH-V2] Domain summary:")
    print(f"{'domain'.ljust(domain_width)}  {'count'.rjust(count_width)}")
    print(f"{'-' * domain_width}  {'-' * count_width}")
    for domain, count in summary_rows:
        print(f"{domain.ljust(domain_width)}  {str(count).rjust(count_width)}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sample a balanced LongBench-v2 CSV suite")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--sample-size", type=int, default=10, help="Source groups to sample. Default: 10")
    parser.add_argument(
        "--min-token-count",
        type=int,
        default=0,
        help="Only sample source_ids whose requested rows have token_count at or above this value. 0 disables the floor.",
    )
    parser.add_argument(
        "--max-token-count",
        type=int,
        default=0,
        help="Only sample source_ids whose requested rows have token_count at or below this value. 0 disables the cap.",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=["random", "shortest", "longest"],
        default=DEFAULT_SELECTION_STRATEGY,
        help="Choose eligible sources randomly, by shortest token_count first, or by longest token_count first. Default: random.",
    )
    parser.add_argument(
        "--row-types",
        default=",".join(DEFAULT_ROW_TYPES),
        help="Comma-separated row types to sample together by source_id. Default: original,exact,semantic",
    )
    parser.add_argument(
        "--domains",
        default="",
        help="Comma-separated domain values to sample from. Empty means all domains.",
    )
    parser.add_argument(
        "--samples-per-domain",
        type=int,
        default=0,
        help=(
            "When greater than 0, sample this many source groups from each eligible domain. "
            "This overrides --sample-size."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling. Default: 0")
    args = parser.parse_args(argv)

    row_types = parse_row_types(args.row_types)
    domains = parse_csv_values(args.domains)
    rows = _read_rows(args.input_path)
    sampled_rows = sample_rows(
        rows,
        sample_size=args.sample_size,
        row_types=row_types,
        seed=args.seed,
        min_token_count=args.min_token_count,
        max_token_count=args.max_token_count,
        selection_strategy=args.selection_strategy,
        domains=domains,
        samples_per_domain=args.samples_per_domain,
    )
    _write_rows(args.output_path, sampled_rows)
    print(f"[LONGBENCH-V2] Wrote {len(sampled_rows)} rows to {args.output_path}")
    print_row_type_summary(sampled_rows)
    print_domain_summary(sampled_rows)


if __name__ == "__main__":
    main()
