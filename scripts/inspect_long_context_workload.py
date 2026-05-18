"""Inspect NoLiMa-style long-context workload size before running models."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nolima.parity_bridge import BookHaystack, list_haystack_assets, load_needle_set_cases  # noqa: E402


def parse_lengths(raw: str) -> list[int]:
    lengths: list[int] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token.endswith("k"):
            lengths.append(int(float(token[:-1]) * 1000))
        else:
            lengths.append(int(token))
    return [length for length in lengths if length > 0]


def estimate_length_row(
    *,
    length: int,
    case_count: int,
    haystack_count: int,
    depth_intervals: int,
    chunk_words: int,
    chunk_size: int,
    agent_window_tokens: int = 0,
) -> dict:
    chunk_count = int(math.ceil(length / float(max(1, chunk_words))))
    solver_windows = int(math.ceil(chunk_count / float(max(1, chunk_size))))
    samples = case_count * haystack_count * depth_intervals
    row = {
        "length": length,
        "samples": samples,
        "estimated_chunks_per_sample": chunk_count,
        "estimated_solver_windows_per_sample": solver_windows,
        "estimated_aggregate_calls_per_sample": 1,
        "estimated_model_calls_per_cold_sample": solver_windows + 1,
        "estimated_model_calls_cold_total": samples * (solver_windows + 1),
    }
    if agent_window_tokens > 0:
        row["agent_window_tokens"] = agent_window_tokens
        row["context_window_multiple"] = round(length / float(agent_window_tokens), 3)
        row["exceeds_agent_window"] = length > agent_window_tokens
    return row


def estimate_workload(
    *,
    needle_set_path: Path,
    haystack_dir: Path,
    lengths: Iterable[int],
    depth_intervals: int,
    chunk_words: int,
    chunk_size: int,
    max_cases: int = 0,
    max_haystacks: int = 0,
    agent_window_tokens: int = 0,
) -> dict:
    cases, needle_hash = load_needle_set_cases(needle_set_path, max_cases=max_cases)
    haystacks = list_haystack_assets(haystack_dir, max_haystacks=max_haystacks)
    haystack_token_counts = []
    for asset in haystacks:
        haystack_token_counts.append(BookHaystack(asset.path).total_tokens)

    length_values = list(lengths)
    rows = [
        estimate_length_row(
            length=length,
            case_count=len(cases),
            haystack_count=len(haystacks),
            depth_intervals=depth_intervals,
            chunk_words=chunk_words,
            chunk_size=chunk_size,
            agent_window_tokens=agent_window_tokens,
        )
        for length in length_values
    ]

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "needle_set_path": str(needle_set_path),
        "needle_set_hash": needle_hash,
        "haystack_dir": str(haystack_dir),
        "case_count": len(cases),
        "haystack_count": len(haystacks),
        "haystack_token_counts": {
            "min": min(haystack_token_counts),
            "max": max(haystack_token_counts),
            "total": sum(haystack_token_counts),
        },
        "lengths": length_values,
        "depth_intervals": depth_intervals,
        "chunk_words": chunk_words,
        "chunk_size": chunk_size,
        "agent_window_tokens": agent_window_tokens,
        "by_length": rows,
        "totals": {
            "samples": sum(row["samples"] for row in rows),
            "estimated_model_calls_cold_total": sum(
                row["estimated_model_calls_cold_total"] for row in rows
            ),
        },
    }


def missing_data_summary(needle_set_path: Path, haystack_dir: Path) -> dict:
    return {
        "data_present": False,
        "needle_set_path_exists": needle_set_path.exists(),
        "haystack_dir_exists": haystack_dir.exists(),
        "fetch_command": "bash nolima/scripts/fetch_dataset.sh --target-root benchmark_data/nolima --with-long",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a NoLiMa long-context workload plan.")
    parser.add_argument("--needle-set-path", default="benchmark_data/nolima/needlesets/needle_set.json")
    parser.add_argument("--haystack-dir", default="benchmark_data/nolima/haystack/rand_shuffle_long")
    parser.add_argument("--lengths", default="32K,64K,128K,256K")
    parser.add_argument("--depth-intervals", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-haystacks", type=int, default=0)
    parser.add_argument("--chunk-words", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument(
        "--agent-window-tokens",
        type=int,
        default=0,
        help="Optional comparison window; 0 disables window-multiple reporting.",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    needle_set_path = Path(args.needle_set_path)
    haystack_dir = Path(args.haystack_dir)
    if not needle_set_path.exists() or not haystack_dir.exists():
        summary = missing_data_summary(needle_set_path, haystack_dir)
    else:
        summary = {
            "data_present": True,
            **estimate_workload(
                needle_set_path=needle_set_path,
                haystack_dir=haystack_dir,
                lengths=parse_lengths(args.lengths),
                depth_intervals=args.depth_intervals,
                chunk_words=args.chunk_words,
                chunk_size=args.chunk_size,
                max_cases=args.max_cases,
                max_haystacks=args.max_haystacks,
                agent_window_tokens=args.agent_window_tokens,
            ),
        }

    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
