"""Export the original LongBench-v2 JSON dataset to CSV for inspection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence


DEFAULT_INPUT_PATH = Path("benchmark_data/long_bench_v2/data.json")
DEFAULT_OUTPUT_PATH = Path("benchmark_data/long_bench_v2/data.csv")

CSV_COLUMNS = [
    "case_id",
    "source_id",
    "row_type",
    "is_scored",
    "setup_case_id",
    "context_id",
    "expected_cache_type",
    "expected_from_cache",
    "depends_on_case_id",
    "domain",
    "sub_domain",
    "difficulty",
    "length",
    "question",
    "choice_A",
    "choice_B",
    "choice_C",
    "choice_D",
    "answer",
]


def _coerce_text(value: object) -> str:
    return str(value) if value is not None else ""


def _context_id(context: str) -> str:
    return hashlib.sha256(context.encode("utf-8")).hexdigest()[:12]


def _base_row(item: dict, source_id: str) -> dict:
    return {
        "source_id": source_id,
        "context_id": _context_id(_coerce_text(item.get("context"))),
        "domain": _coerce_text(item.get("domain")),
        "sub_domain": _coerce_text(item.get("sub_domain")),
        "difficulty": _coerce_text(item.get("difficulty")),
        "length": _coerce_text(item.get("length")),
        "question": _coerce_text(item.get("question")),
        "choice_A": _coerce_text(item.get("choice_A")),
        "choice_B": _coerce_text(item.get("choice_B")),
        "choice_C": _coerce_text(item.get("choice_C")),
        "choice_D": _coerce_text(item.get("choice_D")),
        "answer": _coerce_text(item.get("answer")),
    }


def load_rows(input_path: Path, include_exact: bool = True) -> list[dict]:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected {input_path} to contain a JSON list")

    rows = []
    seen_ids = set()
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Row {idx} is not a JSON object")
        source_id = _coerce_text(item.get("_id")).strip()
        if not source_id:
            raise ValueError(f"Row {idx} is missing _id")
        if source_id in seen_ids:
            raise ValueError(f"Duplicate _id found: {source_id}")
        seen_ids.add(source_id)

        base = _base_row(item, source_id)
        original_case_id = f"{source_id}__original"
        rows.append({
            "case_id": original_case_id,
            "row_type": "original",
            "is_scored": "true",
            "setup_case_id": "",
            "expected_cache_type": "miss",
            "expected_from_cache": "false",
            "depends_on_case_id": "",
            **base,
        })
        if include_exact:
            rows.append({
                "case_id": f"{source_id}__exact",
                "row_type": "exact",
                "is_scored": "true",
                "setup_case_id": "",
                "expected_cache_type": "exact",
                "expected_from_cache": "true",
                "depends_on_case_id": original_case_id,
                **base,
            })
    return rows


def export_csv(input_path: Path, output_path: Path, include_exact: bool = True) -> int:
    rows = load_rows(input_path, include_exact=include_exact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export LongBench-v2 JSON data to CSV")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--include-exact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include one exact-match duplicate row per original row. Enabled by default.",
    )
    args = parser.parse_args(argv)

    row_count = export_csv(args.input_path, args.output_path, include_exact=args.include_exact)
    print(f"[LONGBENCH-V2] Wrote {row_count} rows to {args.output_path}")


if __name__ == "__main__":
    main()
