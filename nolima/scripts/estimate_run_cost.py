"""Estimate benchmark spend from an existing NoLiMa run directory.

Usage:
  python nolima/scripts/estimate_run_cost.py \
    --run-dir benchmark_artifacts/official_nolima/<run_id> \
    --project-samples 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate NoLiMa run spend from bridge rows")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--project-samples", type=int, default=1000)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    bridge_path = run_dir / "bridge_rows.jsonl"
    if not bridge_path.exists():
        raise FileNotFoundError(f"bridge_rows.jsonl not found: {bridge_path}")

    rows = load_jsonl(bridge_path)
    if not rows:
        raise ValueError(f"No bridge rows found in: {bridge_path}")

    total_samples = len(rows)
    total_cost = sum(float(r.get("delta_cost_usd", 0.0)) for r in rows)
    total_calls = sum(float(r.get("delta_calls", 0.0)) for r in rows)
    avg_latency_ms = sum(float(r.get("latency_ms", 0.0)) for r in rows) / total_samples
    cache_hits = sum(1 for r in rows if bool(r.get("from_cache")))

    avg_cost_per_sample = total_cost / total_samples
    projected_cost = avg_cost_per_sample * max(0, args.project_samples)

    summary = {
        "run_dir": str(run_dir),
        "samples": total_samples,
        "total_cost_usd": round(total_cost, 8),
        "avg_cost_usd_per_sample": round(avg_cost_per_sample, 8),
        "project_samples": int(args.project_samples),
        "projected_cost_usd": round(projected_cost, 4),
        "total_delta_calls": round(total_calls, 4),
        "avg_latency_ms": round(avg_latency_ms, 3),
        "cache_hit_rate": round(cache_hits / total_samples, 6),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
