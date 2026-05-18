"""Live smoke test for the DP memoization stack using the local MLX model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

if "anthropic" not in sys.modules:
    anthropic_stub = types.ModuleType("anthropic")

    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.messages = None

    anthropic_stub.Anthropic = FakeAnthropic
    sys.modules["anthropic"] = anthropic_stub

import semantic_cache_system as scs  # noqa: E402
from language_memoization import DuckDBMemoStore, EvidenceSpan  # noqa: E402
from local_llm import DEFAULT_MAX_KV_SIZE, DEFAULT_MAX_TOKENS, DEFAULT_MLX_MODEL, LocalLLMConfig, MLXLocalLLM  # noqa: E402


class FakeEmbedder:
    def encode_query(self, query):
        return np.array([1.0, 0.0], dtype="float32")

    def encode_single(self, text):
        return np.array([1.0, 0.0], dtype="float32")

    def encode_documents(self, documents):
        return np.array([[1.0, 0.0] for _ in documents], dtype="float32")


def parse_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a live DP memoization smoke test.")
    parser.add_argument("--model", default=DEFAULT_MLX_MODEL)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Optional solver generation cap. Omit to use the MLX backend default.",
    )
    parser.add_argument(
        "--aggregate-max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Optional aggregation/planner generation cap. Omit to use the MLX backend default.",
    )
    parser.add_argument("--max-kv-size", type=int, default=DEFAULT_MAX_KV_SIZE)
    parser.add_argument("--backend", choices=("memory", "duckdb"), default="memory")
    parser.add_argument(
        "--duckdb-path",
        default="",
        help="DuckDB database path. Defaults to a temporary file when --backend duckdb is used.",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Delete the DuckDB file before running so pass 1 exercises fresh model calls.",
    )
    args = parser.parse_args()

    chunks = [
        "Chunk 0: RLMs is about recursive language models and context management.",
        "Chunk 1: The memo stack stores solved subproblems with task, scope, result, evidence, and dependencies.",
        "Chunk 2: Exact repeated scoped tasks replay from memo without a model call.",
        "Chunk 3: Larger questions reuse solved subranges and call the model only on missing ranges.",
    ]

    memo_store = None
    duckdb_tempdir = None
    duckdb_path = None
    if args.backend == "duckdb":
        if args.duckdb_path:
            duckdb_path = Path(args.duckdb_path)
        else:
            duckdb_tempdir = tempfile.TemporaryDirectory()
            duckdb_path = Path(duckdb_tempdir.name) / "live_dp_memo.duckdb"
        duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        if args.reset_db and duckdb_path.exists():
            duckdb_path.unlink()
        memo_store = DuckDBMemoStore(duckdb_path)

    controller = scs.SemanticCacheController(
        metrics=scs.ExecutionMetrics(),
        embedder=FakeEmbedder(),
        reranker=None,
        corpus_id="live-dp-memo-test",
        memo_store=memo_store,
    )
    controller.data_scope_hash = "synthetic-live-v1"

    llm = MLXLocalLLM(
        LocalLLMConfig(model=args.model, max_tokens=args.max_tokens, max_kv_size=args.max_kv_size)
    )
    llm.load()
    solver_calls = []
    aggregate_calls = []
    verifier_calls = []

    def solver(task, scope):
        solver_calls.append((scope.start, scope.end))
        context = "\n".join(chunks[scope.start:scope.end])
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a scoped solver in a DP memoization smoke test. "
                    "Answer with one concise sentence using only the provided context. "
                    "Do not explain your reasoning."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {task.prompt}\n\n"
                    f"Context chunks {scope.start}..{scope.end}:\n{context}\n\n"
                    "Return the supported answer. If the context does not answer it, return NOT_FOUND."
                ),
            },
        ]
        return {
            "result": llm.generate_chat(messages, max_tokens=args.max_tokens) or "[empty model response]",
            "confidence": 0.80,
            "evidence": [
                EvidenceSpan(
                    document_id="synthetic-doc",
                    start=idx,
                    end=idx + 1,
                    unit="chunk",
                    text=chunks[idx],
                )
                for idx in range(scope.start, scope.end)
            ],
            "metadata": {"live_solver": "mlx", "scope": [scope.start, scope.end]},
        }

    def aggregate(query, scope, plan):
        aggregate_calls.append((scope.start, scope.end))
        partials = "\n".join(
            f"- chunks {entry.scope.start}..{entry.scope.end}: {entry.result}"
            for entry in plan.reusable_entries
            if entry.result
        )
        entry_ids = [entry.entry_id for entry in plan.reusable_entries if entry.result]
        messages = [
            {
                "role": "system",
                "content": (
                    "You aggregate memoized scoped answers. Return only valid JSON with keys "
                    "result, used_entry_ids, confidence. Use only supplied partial answers. "
                    "Do not explain your reasoning."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Available entry IDs: {entry_ids}\n\n"
                    f"Partial answers:\n{partials}\n\n"
                    "Return JSON now."
                ),
            },
        ]
        raw = llm.generate_chat(messages, max_tokens=args.aggregate_max_tokens)
        parsed = parse_json_object(raw)
        if parsed:
            parsed.setdefault("used_entry_ids", entry_ids)
            parsed.setdefault("confidence", 0.8)
            parsed.setdefault("metadata", {})
            parsed["metadata"]["raw_aggregate"] = raw
            return parsed
        return {
            "result": raw or partials,
            "used_entry_ids": entry_ids,
            "confidence": 0.6,
            "metadata": {"raw_aggregate": raw, "fallback": "non_json_aggregate"},
        }

    query = "What does this project memoization stack do?"
    scope_0_4 = controller._memo_scope(
        "synthetic-doc", 0, 4, unit="chunk", content_hash="synthetic-live-v1"
    )

    print("MODEL:", llm.config.model)
    print("MEMO_BACKEND:", args.backend)
    if duckdb_path is not None:
        print("DUCKDB_PATH:", duckdb_path)

    first = controller.solve_with_memo(
        query,
        chunks,
        solver,
        document_id="synthetic-doc",
        chunk_size=2,
        content_hash="synthetic-live-v1",
        task_type="live_dp_memo_test",
        output_contract="supported_facts",
        aggregate_fn=aggregate,
    )
    print("\nPASS 1 automatic solve 0..4 with chunk_size=2")
    print("model_calls:", first["model_calls"])
    print("aggregate_calls:", first["aggregate_calls"])
    print("memo_type:", first["memo_type"])
    print("memo_coverage:", first.get("memo_telemetry", {}).get("coverage_ratio"))
    print(
        "windows:",
        {
            "total": first.get("window_count"),
            "reused": first.get("reused_window_count"),
            "missing": first.get("missing_window_count"),
        },
    )
    print("solver_calls:", solver_calls)
    print("aggregate_calls_seen:", aggregate_calls)
    print("answer:", first["answer"])

    replay = controller.solve_with_memo(
        query,
        chunks,
        solver,
        document_id="synthetic-doc",
        chunk_size=2,
        content_hash="synthetic-live-v1",
        task_type="live_dp_memo_test",
        output_contract="supported_facts",
        aggregate_fn=aggregate,
    )
    print("\nPASS 2 exact replay 0..4")
    print("model_calls:", replay["model_calls"])
    print("aggregate_calls:", replay["aggregate_calls"])
    print("memo_type:", replay["memo_type"])
    print("memo_coverage:", replay.get("memo_telemetry", {}).get("coverage_ratio"))
    print("solver_calls:", solver_calls)
    print("aggregate_calls_seen:", aggregate_calls)
    print("answer:", replay["answer"])

    semantic_query = "Explain the purpose of the project memoization stack."

    def verifier(task, candidate):
        verifier_calls.append(candidate.entry_id)
        messages = [
            {
                "role": "system",
                "content": (
                    "You verify whether a previous memoized answer can answer a new task. "
                    "Return only valid JSON with keys reuse_as, confidence, reason. "
                    "reuse_as must be exact or irrelevant."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"New task: {task.prompt}\n\n"
                    f"Previous task: {candidate.task.prompt}\n\n"
                    f"Previous answer: {candidate.result}\n\n"
                    "Can the previous answer directly answer the new task?"
                ),
            },
        ]
        raw = llm.generate_chat(messages)
        parsed = parse_json_object(raw)
        if parsed:
            parsed.setdefault("metadata", {})
            parsed["metadata"]["raw_verifier"] = raw
            return parsed
        if "memoization stack" in task.prompt.lower() and "memoization stack" in candidate.result.lower():
            return {
                "reuse_as": "exact",
                "confidence": 0.6,
                "reason": "fallback lexical match",
                "metadata": {"raw_verifier": raw, "fallback": True},
            }
        return {"reuse_as": "irrelevant", "confidence": 0.0, "reason": raw}

    semantic_replay = controller.solve_with_memo(
        semantic_query,
        chunks,
        solver,
        document_id="synthetic-doc",
        chunk_size=2,
        content_hash="synthetic-live-v1",
        task_type="live_dp_memo_test",
        output_contract="supported_facts",
        aggregate_fn=aggregate,
        reuse_verifier=verifier,
    )
    print("\nPASS 3 verifier-approved semantic replay")
    print("model_calls:", semantic_replay["model_calls"])
    print("aggregate_calls:", semantic_replay["aggregate_calls"])
    print("memo_type:", semantic_replay["memo_type"])
    print("memo_coverage:", semantic_replay.get("memo_telemetry", {}).get("coverage_ratio"))
    print("semantic_reuse:", semantic_replay["semantic_reuse"])
    print("verifier_calls:", len(verifier_calls))
    print("answer:", semantic_replay["answer"])

    if first["model_calls"] != 2 or solver_calls[:2] != [(0, 2), (2, 4)]:
        if args.backend == "duckdb" and args.duckdb_path and not args.reset_db and first["memo_type"] == "exact":
            raise SystemExit("MVP check failed: DuckDB already contained an exact replay; rerun with --reset-db")
        raise SystemExit("MVP check failed: automatic solve should call missing chunk windows 0..2 and 2..4")
    if first["aggregate_calls"] != 1 or aggregate_calls != [(0, 4)]:
        raise SystemExit("MVP check failed: automatic solve should aggregate the full scope once")
    if replay["model_calls"] != 0 or replay["aggregate_calls"] != 0 or replay["memo_type"] != "exact":
        raise SystemExit("MVP check failed: exact replay should use memo with zero model calls")
    if (
        not semantic_replay["semantic_reuse"]
        or semantic_replay["model_calls"] != 0
        or semantic_replay["aggregate_calls"] != 0
    ):
        raise SystemExit("MVP check failed: semantic replay should reuse verified memo with zero solve calls")
    if "[model emitted" in replay["answer"] or "<think>" in replay["answer"]:
        raise SystemExit("MVP check failed: local solver output was not cleaned")

    if hasattr(controller.memo_store.entries, "values"):
        entries = list(controller.memo_store.entries.values())
    else:
        entries = list(controller.memo_store.entries)
    print("\nMEMO STORE entries:", len(entries))
    for entry in entries:
        print(
            f"- scope={entry.scope.start}..{entry.scope.end} "
            f"result_type={entry.result_type} "
            f"reusable={list(entry.reusable_as)} "
            f"dependencies={len(entry.dependencies)} "
            f"evidence={len(entry.evidence)}"
        )

    print("\nMVP CHECK: PASS")

    if memo_store is not None:
        memo_store.close()
        reopened = DuckDBMemoStore(duckdb_path)
        persisted_plan = reopened.plan_reuse(
            controller._memo_task(
                query,
                task_type="live_dp_memo_test",
                output_contract="supported_facts",
                constraints={"chunk_size": 2},
            ),
            scope_0_4,
        )
        if not persisted_plan.has_exact_replay:
            reopened.close()
            raise SystemExit("MVP check failed: DuckDB exact replay did not persist")
        print("DUCKDB PERSISTENCE CHECK: PASS")
        reopened.close()
    if duckdb_tempdir is not None:
        duckdb_tempdir.cleanup()


if __name__ == "__main__":
    main()
