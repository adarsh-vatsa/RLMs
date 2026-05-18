"""Run a deterministic mutable-workspace DP memoization workload."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

if "anthropic" not in sys.modules:
    anthropic_stub = types.ModuleType("anthropic")

    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.messages = None

    anthropic_stub.Anthropic = FakeAnthropic
    sys.modules["anthropic"] = anthropic_stub

import semantic_cache_system as scs  # noqa: E402
from language_memoization import DuckDBMemoStore, EvidenceSpan  # noqa: E402


class FakeEmbedder:
    def encode_query(self, query):
        return np.array([1.0, 0.0], dtype="float32")

    def encode_single(self, text):
        return np.array([1.0, 0.0], dtype="float32")

    def encode_documents(self, documents):
        return np.array([[1.0, 0.0] for _ in documents], dtype="float32")


V1_CHUNKS = [
    "Launch codeword is ORCHID-57.",
    "Backup channel is north-bridge-9.",
    "Invoice INV-44 is due Friday.",
    "Legacy codeword TULIP-12 is retired.",
]

V2_CHUNKS = [
    "Launch codeword is ORCHID-57.",
    "Backup channel is east-relay-3.",
    "Invoice INV-44 is due Friday.",
    "Legacy codeword TULIP-12 is retired.",
]

QUERY = "Extract reusable operational facts."


def deterministic_fact_solver(controller: scs.SemanticCacheController, chunks: list[str]):
    def solver(task, scope):
        text = chunks[scope.start]
        return {
            "result": f"fact:{text}",
            "confidence": 1.0,
            "evidence": [
                EvidenceSpan(
                    document_id=scope.document_id,
                    start=scope.start,
                    end=scope.end,
                    unit=scope.unit,
                    text=text,
                )
            ],
        }

    return solver


def run_workload(*, duckdb_path: Path, output_dir: Path, reset_db: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if reset_db and duckdb_path.exists():
        duckdb_path.unlink()
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    store = DuckDBMemoStore(duckdb_path)
    controller = scs.SemanticCacheController(
        metrics=scs.ExecutionMetrics(),
        embedder=FakeEmbedder(),
        reranker=None,
        corpus_id="dp_memo_mutable_workload",
        memo_store=store,
    )

    started = time.time()
    first = controller.solve_with_memo(
        QUERY,
        V1_CHUNKS,
        deterministic_fact_solver(controller, V1_CHUNKS),
        document_id="ops-runbook",
        chunk_size=1,
        content_hash="",
    )

    changed_scope = controller._memo_scope("ops-runbook", 1, 2, unit="chunk", content_hash="")
    rejected = store.invalidate_scope(changed_scope, reason="chunk 1 updated in workload v2")

    second = controller.solve_with_memo(
        QUERY,
        V2_CHUNKS,
        deterministic_fact_solver(controller, V2_CHUNKS),
        document_id="ops-runbook",
        chunk_size=1,
        content_hash="",
    )
    store.close()

    reopened = DuckDBMemoStore(duckdb_path)
    replay_controller = scs.SemanticCacheController(
        metrics=scs.ExecutionMetrics(),
        embedder=FakeEmbedder(),
        reranker=None,
        corpus_id="dp_memo_mutable_workload",
        memo_store=reopened,
    )
    warm = replay_controller.solve_with_memo(
        QUERY,
        V2_CHUNKS,
        deterministic_fact_solver(replay_controller, V2_CHUNKS),
        document_id="ops-runbook",
        chunk_size=1,
        content_hash="",
    )
    stats = reopened.stats()
    reopened.close()

    manifest = {
        "benchmark": "dp_memo_mutable_workload",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_id": "dp_memo_mutable_workload",
        "duckdb_path": str(duckdb_path),
        "query": QUERY,
        "changed_scope": changed_scope.to_dict(),
        "totals": {
            "v1_model_calls": first["model_calls"],
            "v1_aggregate_calls": first["aggregate_calls"],
            "invalidated_entries": len(rejected),
            "v2_model_calls": second["model_calls"],
            "v2_aggregate_calls": second["aggregate_calls"],
            "v2_initial_coverage_ratio": second["initial_memo_telemetry"]["coverage_ratio"],
            "v2_reused_windows": second["reused_window_count"],
            "v2_missing_windows": second["missing_window_count"],
            "warm_model_calls": warm["model_calls"],
            "warm_aggregate_calls": warm["aggregate_calls"],
            "latency_ms": round((time.time() - started) * 1000.0, 3),
        },
        "answers": {
            "v1": first["answer"],
            "v2": second["answer"],
            "warm": warm["answer"],
        },
        "rejected_entry_ids": [entry.entry_id for entry in rejected],
        "memo_entries": stats["entry_count"],
        "memo_stats": stats,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a mutable-workspace DP memo workload.")
    parser.add_argument("--duckdb-path", default="benchmark_artifacts/dp_memo_mutable_workload/memo.duckdb")
    parser.add_argument("--output-dir", default="benchmark_artifacts/dp_memo_mutable_workload")
    parser.add_argument("--reset-db", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = run_workload(
        duckdb_path=Path(args.duckdb_path),
        output_dir=out_dir,
        reset_db=args.reset_db,
    )
    print(json.dumps(manifest["totals"], indent=2))
    print(f"Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
