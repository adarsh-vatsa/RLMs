"""High-parity NoLiMa benchmark bridge for this repository.

This runner expands NoLiMa needle-set definitions into depth-swept placements,
executes the repository's SemanticCache system as the system under test, and
writes benchmark artifacts under benchmark_artifacts/official_nolima.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Ensure repository root is importable when executing via
# `python nolima/run_benchmark.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import semantic_cache_system as scs
from nolima.parity_bridge import (
    build_dataset_signature,
    iter_nolima_samples,
    list_haystack_assets,
    load_needle_set_cases,
)


# Work around duplicate OpenMP runtime loading on macOS when combining
# FAISS and Torch-backed model components in a single process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    return ""


def _parse_lengths(raw: str) -> List[int]:
    values: List[int] = []
    for part in (raw or "").split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token.endswith("k"):
            base = token[:-1].strip()
            if not base:
                raise ValueError(f"Invalid length token: {part}")
            values.append(int(float(base) * 1000))
        else:
            values.append(int(token))

    deduped: List[int] = []
    seen = set()
    for value in values:
        if value <= 0:
            continue
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _snapshot_metrics(metrics: scs.ExecutionMetrics) -> Dict[str, float]:
    total_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    for data in metrics.stats.values():
        total_calls += data["calls"]
        total_input_tokens += data["input_tokens"]
        total_output_tokens += data["output_tokens"]
        total_cost += data["cost"]
    return {
        "calls": total_calls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cost": total_cost,
    }


def _sanitize_path_segment(raw: str, max_len: int = 180) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", _coerce_text(raw)).strip("._-")
    if not value:
        value = "default"
    return value[:max_len]


def _build_controller_with_components(
    corpus_id: str,
    corpus_domain: str,
    embedder: scs.EmbeddingEngine,
    reranker: scs.Reranker,
    executor_model: str,
) -> scs.SemanticCacheController:
    scs.EXECUTOR_MODEL = executor_model
    metrics = scs.ExecutionMetrics()
    return scs.SemanticCacheController(
        metrics=metrics,
        embedder=embedder,
        reranker=reranker,
        corpus_id=corpus_id,
        corpus_domain=corpus_domain,
    )


def run_official_nolima_benchmark(args: argparse.Namespace) -> None:
    lengths = _parse_lengths(args.lengths)
    if not lengths:
        raise ValueError("--lengths must include at least one positive value")
    if args.depth_intervals <= 0:
        raise ValueError("--depth-intervals must be > 0")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / "official_nolima" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cases, needle_set_hash = load_needle_set_cases(
        Path(args.needle_set_path),
        max_cases=args.max_cases,
    )
    haystacks = list_haystack_assets(
        Path(args.haystack_dir),
        max_haystacks=args.max_haystacks,
    )

    dataset_signature = build_dataset_signature(
        needle_set_hash=needle_set_hash,
        cases=cases,
        haystacks=haystacks,
        lengths=lengths,
        depth_intervals=args.depth_intervals,
        seed=args.seed,
    )

    lengths_sig = "-".join(str(x) for x in lengths)
    cache_namespace = _sanitize_path_segment(
        f"{args.corpus_id}__nolima__L{lengths_sig}__D{args.depth_intervals}__"
        f"S{args.seed}__{dataset_signature}"
    )

    cache_state_enabled = args.mode == "cache"
    cache_state_root = (
        Path(args.cache_state_root)
        if args.cache_state_root
        else Path(args.output_dir) / "official_nolima" / "cache_state"
    )
    cache_state_path = cache_state_root / cache_namespace

    cache_state_existed_before_reset = cache_state_path.exists()
    if cache_state_enabled and args.cache_reset and cache_state_existed_before_reset:
        shutil.rmtree(cache_state_path)

    if cache_state_enabled:
        cache_state_root.mkdir(parents=True, exist_ok=True)
    cache_state_existed_before_run = cache_state_path.exists()

    predictions_path = out_dir / "predictions.jsonl"
    bridge_rows_path = out_dir / "bridge_rows.jsonl"
    manifest_path = out_dir / "manifest.json"

    print(f"\n[NoLiMa] Run id: {run_id}")
    print(f"[NoLiMa] Cases: {len(cases)}")
    print(f"[NoLiMa] Haystacks: {len(haystacks)}")
    print(f"[NoLiMa] Lengths: {lengths}")
    print(f"[NoLiMa] Depth intervals: {args.depth_intervals}")
    print(f"[NoLiMa] Output dir: {out_dir}")

    if cache_state_enabled:
        print(f"[NoLiMa] Cache state root: {cache_state_root}")
        print(f"[NoLiMa] Cache namespace : {cache_namespace}")
        print(f"[NoLiMa] Dataset signature: {dataset_signature}")
        if cache_state_existed_before_run:
            print("[NoLiMa] Cache state: warm start (existing state found)")
        else:
            print("[NoLiMa] Cache state: cold start (no prior state)")

    shared_embedder = scs.EmbeddingEngine()
    shared_reranker = scs.Reranker()

    total_samples = 0
    cache_load_attempts = 0
    cache_load_successes = 0
    cache_hits = 0
    cache_entries_before_run = 0
    cache_entries_after_run = 0

    with predictions_path.open("w", encoding="utf-8") as predictions_file, bridge_rows_path.open(
        "w", encoding="utf-8"
    ) as bridge_file:
        sample_iter = iter_nolima_samples(
            cases=cases,
            haystacks=haystacks,
            lengths=lengths,
            depth_intervals=args.depth_intervals,
            seed=args.seed,
            shift=args.shift,
            static_depth=args.static_depth,
        )

        for idx, sample in enumerate(sample_iter, start=1):
            if args.max_samples > 0 and idx > args.max_samples:
                break

            total_samples += 1
            safe_sample_id = re.sub(r"[^a-zA-Z0-9._-]", "_", _coerce_text(sample["id"]))
            sample_docs_dir = out_dir / "sample_docs" / safe_sample_id
            sample_docs_dir.mkdir(parents=True, exist_ok=True)
            context_path = sample_docs_dir / "context.txt"
            context_path.write_text(_coerce_text(sample["context"]).rstrip() + "\n", encoding="utf-8")

            controller_corpus_id = cache_namespace if cache_state_enabled else f"{args.corpus_id}_{idx}"
            controller = _build_controller_with_components(
                corpus_id=controller_corpus_id,
                corpus_domain=args.domain,
                embedder=shared_embedder,
                reranker=shared_reranker,
                executor_model=args.executor_model,
            )

            if cache_state_enabled:
                cache_load_attempts += 1
                if cache_state_path.exists() and controller.load(cache_state_path):
                    cache_load_successes += 1
                if idx == 1:
                    cache_entries_before_run = controller.get_total_entries()

            controller.ingest(sample_docs_dir)

            before = _snapshot_metrics(controller.metrics)
            t0 = time.time()
            output = controller.search(
                _coerce_text(sample["question"]),
                top_k=args.top_k,
                rerank_top=args.rerank_top,
                synthesize=True,
                cache_read=cache_state_enabled,
            )
            latency_ms = (time.time() - t0) * 1000.0
            after = _snapshot_metrics(controller.metrics)

            answer = _coerce_text(output.get("answer"))
            prediction_row = {
                "id": sample["id"],
                "sample_id": sample["id"],
                "task": sample["task"],
                "test_name": sample["test_name"],
                "length": sample["length"],
                "depth_percent": sample["depth_percent"],
                "depth_index": sample["depth_index"],
                "question": sample["question"],
                "generation": answer,
                "prediction": answer,
                "answer": answer,
                "expected_answer": sample["expected_answer"],
                "gold_answers": sample["gold_answers"],
                "haystack_name": sample["haystack_name"],
                "haystack_hash": sample["haystack_hash"],
                "needle": sample["needle"],
                "question_type": sample["question_type"],
                "selected_character": sample["selected_character"],
            }
            predictions_file.write(json.dumps(prediction_row, ensure_ascii=False) + "\n")

            bridge_row = {
                "sample_id": sample["id"],
                "task": sample["task"],
                "test_name": sample["test_name"],
                "length": sample["length"],
                "depth_percent": sample["depth_percent"],
                "source_file": sample["haystack_path"],
                "latency_ms": round(latency_ms, 3),
                "delta_calls": after["calls"] - before["calls"],
                "delta_input_tokens": after["input_tokens"] - before["input_tokens"],
                "delta_output_tokens": after["output_tokens"] - before["output_tokens"],
                "delta_cost_usd": round(after["cost"] - before["cost"], 8),
                "from_cache": bool(output.get("from_cache")),
                "cache_type": _coerce_text(output.get("cache_type")),
                "placement_metadata": sample.get("placement_metadata", {}),
            }
            bridge_file.write(json.dumps(bridge_row, ensure_ascii=False) + "\n")

            if output.get("from_cache"):
                cache_hits += 1

            if cache_state_enabled:
                controller.save(cache_state_path)
                cache_entries_after_run = controller.get_total_entries()

            if idx % 50 == 0:
                print(f"[NoLiMa] Processed samples: {idx}")

    if total_samples == 0:
        raise ValueError("No samples selected; verify needle-set, haystack, and filters")

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_target": "nolima",
        "mode": args.mode,
        "domain": args.domain,
        "corpus_id": args.corpus_id,
        "executor_model": args.executor_model,
        "top_k": args.top_k,
        "rerank_top": args.rerank_top,
        "seed": args.seed,
        "shift": args.shift,
        "static_depth": args.static_depth,
        "depth_intervals": args.depth_intervals,
        "lengths_requested": lengths,
        "samples_selected": total_samples,
        "needle_set_path": str(Path(args.needle_set_path)),
        "needle_set_hash": needle_set_hash,
        "haystack_dir": str(Path(args.haystack_dir)),
        "haystack_count": len(haystacks),
        "dataset_signature": dataset_signature,
        "artifacts": {
            "predictions": str(predictions_path),
            "bridge_rows": str(bridge_rows_path),
        },
        "nolima_parity": {
            "metric_default": "contains",
            "depth_interval_count": args.depth_intervals,
            "lengths": lengths,
            "needle_set_variant": Path(args.needle_set_path).name,
            "system_under_test": "semantic_cache_system",
        },
    }

    if cache_state_enabled:
        manifest["cache_reuse"] = {
            "enabled": True,
            "cache_namespace": cache_namespace,
            "cache_state_root": str(cache_state_root),
            "cache_state_path": str(cache_state_path),
            "cache_state_existed_before_reset": cache_state_existed_before_reset,
            "cache_state_existed_before_run": cache_state_existed_before_run,
            "cache_reset_requested": bool(args.cache_reset),
            "cache_load_attempts": cache_load_attempts,
            "cache_load_successes": cache_load_successes,
            "cache_entries_before_run": cache_entries_before_run,
            "cache_entries_after_run": cache_entries_after_run,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / total_samples, 6),
        }
    else:
        manifest["cache_reuse"] = {
            "enabled": False,
            "reason": "mode is baseline",
        }

    if args.manifest_note:
        manifest["note"] = args.manifest_note

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n[NoLiMa] Completed.")
    print(f"[NoLiMa] Predictions : {predictions_path}")
    print(f"[NoLiMa] Bridge rows : {bridge_rows_path}")
    print(f"[NoLiMa] Manifest    : {manifest_path}")
    if cache_state_enabled:
        print(
            "[NoLiMa] Cache reuse summary: "
            f"loads={cache_load_successes}/{cache_load_attempts}, "
            f"hits={cache_hits}/{total_samples}, "
            f"entries_before={cache_entries_before_run}, "
            f"entries_after={cache_entries_after_run}"
        )


def main() -> None:
    start = time.time()
    parser = argparse.ArgumentParser(
        description="High-parity NoLiMa benchmark bridge for semantic cache architecture"
    )
    parser.add_argument(
        "--needle-set-path",
        type=str,
        default="benchmark_data/nolima/needlesets/needle_set.json",
        help=(
            "Path to NoLiMa needle-set JSON file "
            "(smoke fixture: benchmark_fixtures/nolima/needlesets/needle_set.json)"
        ),
    )
    parser.add_argument(
        "--haystack-dir",
        type=str,
        default="benchmark_data/nolima/haystack/rand_shuffle",
        help=(
            "Directory containing NoLiMa haystack .txt files "
            "(smoke fixture: benchmark_fixtures/nolima/haystack/rand_shuffle)"
        ),
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="250,500,1K,2K,4K,8K,16K,32K",
        help="Comma-separated context lengths (supports K suffix)",
    )
    parser.add_argument(
        "--depth-intervals",
        type=int,
        default=26,
        help="Number of evenly spaced depth placements between 0 and 100",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for deterministic placeholder/distractor behavior",
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=0,
        help="Shift applied to haystack placement start",
    )
    parser.add_argument(
        "--static-depth",
        type=float,
        default=-1.0,
        help="Static placement depth in [0,1]; -1 keeps depth sweep mode",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional cap on expanded needle-set cases (0 means all)",
    )
    parser.add_argument(
        "--max-haystacks",
        type=int,
        default=0,
        help="Optional cap on haystack files (0 means all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on total generated samples (0 means all)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "cache"],
        default="cache",
        help="Architecture execution mode during prediction generation",
    )
    parser.add_argument("--corpus-id", type=str, default="official_nolima")
    parser.add_argument("--domain", type=str, default="general")
    parser.add_argument(
        "--executor-model",
        type=str,
        default="claude-sonnet-4-5",
        help="Model name assigned to semantic_cache_system.EXECUTOR_MODEL",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--rerank-top", type=int, default=5)
    parser.add_argument(
        "--cache-state-root",
        type=str,
        default="",
        help=(
            "Root directory for persistent cache state in cache mode. "
            "If omitted, defaults to <output-dir>/official_nolima/cache_state"
        ),
    )
    parser.add_argument(
        "--cache-reset",
        action="store_true",
        help="Reset the resolved cache namespace before run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_artifacts",
        help="Root output directory for official run artifacts",
    )
    parser.add_argument(
        "--manifest-note",
        type=str,
        default="",
        help="Optional free-text note stored in manifest",
    )
    args = parser.parse_args()

    run_official_nolima_benchmark(args)
    elapsed = time.time() - start
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
