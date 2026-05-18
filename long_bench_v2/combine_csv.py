"""Combine LongBench-v2 original/exact rows with generated workload rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from long_bench_v2.export_csv import CSV_COLUMNS, _context_id


DEFAULT_BASE_CSV_PATH = Path("benchmark_data/long_bench_v2/data.csv")
DEFAULT_SOURCE_JSON_PATH = Path("benchmark_data/long_bench_v2/data.json")
DEFAULT_OUTPUT_PATH = Path("benchmark_data/long_bench_v2/data_cache_suite.csv")


LAST_ROW_TYPE_COUNTS: Counter[str] = Counter()


def _normalize_row(row: dict) -> dict:
    normalized = {column: row.get(column, "") for column in CSV_COLUMNS}
    row_type = normalized["row_type"]
    if not normalized["is_scored"]:
        normalized["is_scored"] = "false" if row_type == "setup" else "true"
    return normalized


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = [column for column in CSV_COLUMNS if column not in (reader.fieldnames or [])]
        required_missing = [
            column
            for column in missing_columns
            if column not in {"is_scored", "setup_case_id", "token_count"}
        ]
        if required_missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(required_missing)}")
        return [_normalize_row(row) for row in reader]


def _estimated_token_count(text: str) -> str:
    return str(math.ceil(len(text.split()) * 1.33))


def _load_token_count_maps(source_json_path: Path | None) -> tuple[dict[str, str], dict[str, str]]:
    if not source_json_path or not source_json_path.exists():
        return {}, {}

    raw = json.loads(source_json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected {source_json_path} to contain a JSON list")

    by_source_id: dict[str, str] = {}
    by_context_id: dict[str, str] = {}
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Row {idx} in {source_json_path} is not a JSON object")
        source_id = str(item.get("_id") or "").strip()
        context = str(item.get("context") or "")
        token_count = _estimated_token_count(context)
        if source_id:
            by_source_id[source_id] = token_count
        context_id = str(item.get("context_id") or "").strip()
        if not context_id:
            context_id = _context_id(context)
        by_context_id[context_id] = token_count

    return by_source_id, by_context_id


def _fill_token_counts(rows: list[dict], source_json_path: Path | None) -> None:
    by_source_id, by_context_id = _load_token_count_maps(source_json_path)
    if not by_source_id and not by_context_id:
        return

    for row in rows:
        if row.get("token_count"):
            continue
        source_id = row.get("source_id", "")
        context_id = row.get("context_id", "")
        if source_id in by_source_id:
            row["token_count"] = by_source_id[source_id]
        elif context_id in by_context_id:
            row["token_count"] = by_context_id[context_id]


def combine_csv(
    base_csv_path: Path,
    output_path: Path,
    semantic_csv_path: Path | None = None,
    knowledge_csv_path: Path | None = None,
    source_json_path: Path | None = DEFAULT_SOURCE_JSON_PATH,
) -> int:
    global LAST_ROW_TYPE_COUNTS
    base_rows = _read_rows(base_csv_path)
    semantic_rows = _read_rows(semantic_csv_path) if semantic_csv_path else []
    knowledge_rows = _read_rows(knowledge_csv_path) if knowledge_csv_path and knowledge_csv_path.exists() else []
    combined_rows = [*base_rows, *semantic_rows, *knowledge_rows]
    _fill_token_counts(combined_rows, source_json_path)

    seen_case_ids = set()
    for row in combined_rows:
        case_id = row.get("case_id", "")
        if not case_id:
            raise ValueError("Found row with empty case_id")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate case_id found: {case_id}")
        seen_case_ids.add(case_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined_rows)
    LAST_ROW_TYPE_COUNTS = Counter(row.get("row_type", "") or "<blank>" for row in combined_rows)
    return len(combined_rows)


def print_row_type_summary(row_type_counts: Counter[str]) -> None:
    if not row_type_counts:
        return
    rows = sorted(row_type_counts.items())
    row_type_width = max(len("row_type"), *(len(row_type) for row_type, _ in rows))
    count_width = max(len("count"), *(len(str(count)) for _, count in rows))
    print("[LONGBENCH-V2] Row type summary:")
    print(f"{'row_type'.ljust(row_type_width)}  {'count'.rjust(count_width)}")
    print(f"{'-' * row_type_width}  {'-' * count_width}")
    for row_type, count in rows:
        print(f"{row_type.ljust(row_type_width)}  {str(count).rjust(count_width)}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Combine LongBench-v2 base, semantic, and knowledge CSV rows")
    parser.add_argument("--base-csv-path", type=Path, default=DEFAULT_BASE_CSV_PATH)
    parser.add_argument("--semantic-csv-path", type=Path, default=None)
    parser.add_argument("--knowledge-csv-path", type=Path, default=None)
    parser.add_argument(
        "--source-json-path",
        type=Path,
        default=DEFAULT_SOURCE_JSON_PATH,
        help="Original LongBench-v2 JSON used to backfill estimated context token_count values",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)

    row_count = combine_csv(
        args.base_csv_path,
        args.output_path,
        semantic_csv_path=args.semantic_csv_path,
        knowledge_csv_path=args.knowledge_csv_path,
        source_json_path=args.source_json_path,
    )
    print(f"[LONGBENCH-V2] Wrote {row_count} rows to {args.output_path}")
    print_row_type_summary(LAST_ROW_TYPE_COUNTS)


if __name__ == "__main__":
    main()
