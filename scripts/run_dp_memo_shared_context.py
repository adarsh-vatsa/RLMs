"""Run DP memoization on a shared-context multi-question fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import types
from datetime import datetime, timezone
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


def chunk_sentences(text: str, sentences_per_chunk: int) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    chunks = []
    for start in range(0, len(sentences), sentences_per_chunk):
        chunks.append(" ".join(sentences[start : start + sentences_per_chunk]))
    return chunks or [text]


def parse_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def planner_response(text: str, fallback_entry_ids: list[str]) -> dict:
    """Parse Q3 planner output, including common partial-JSON truncations."""
    parsed = parse_json_object(text)
    if parsed:
        parsed.setdefault("action", "answer")
        parsed.setdefault("used_entry_ids", fallback_entry_ids)
        parsed.setdefault("confidence", 0.75)
        parsed.setdefault("reason", "model returned structured plan")
        return parsed

    answer_match = re.search(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.DOTALL)
    if answer_match:
        try:
            answer = json.loads(f'"{answer_match.group(1)}"')
        except json.JSONDecodeError:
            answer = answer_match.group(1)

        used_entry_ids = re.findall(r'"([a-f0-9]{24})"', text)
        confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        confidence = 0.6
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
            except ValueError:
                confidence = 0.6
        return {
            "action": "answer",
            "answer": answer,
            "used_entry_ids": used_entry_ids or fallback_entry_ids,
            "confidence": confidence,
            "reason": "model returned partial JSON; extracted answer field",
        }

    return {
        "action": "answer",
        "answer": text,
        "used_entry_ids": fallback_entry_ids,
        "confidence": 0.5,
        "reason": "model returned non-json answer",
    }


def expected_values(answer) -> list[str]:
    if isinstance(answer, list):
        return [str(item) for item in answer]
    return [str(answer)]


def answer_matches(answer: str, expected) -> bool:
    lower = answer.lower()
    return all(value.lower() in lower for value in expected_values(expected))


def main() -> None:
    parser = argparse.ArgumentParser(description="DP memo benchmark for shared-context QA.")
    parser.add_argument("--fixture", default="benchmark_fixtures/dp_memo/shared_context_qa.json")
    parser.add_argument("--duckdb-path", default="benchmark_artifacts/dp_memo_shared_context/memo.duckdb")
    parser.add_argument("--output-dir", default="benchmark_artifacts/dp_memo_shared_context")
    parser.add_argument("--reset-db", action="store_true")
    parser.add_argument("--sentences-per-chunk", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MLX_MODEL)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Optional fact-extraction generation cap. Omit to use the MLX backend default.",
    )
    parser.add_argument(
        "--aggregate-max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Optional memo-planner generation cap. Omit to use the MLX backend default.",
    )
    parser.add_argument("--max-kv-size", type=int, default=DEFAULT_MAX_KV_SIZE)
    args = parser.parse_args()

    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.duckdb_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if args.reset_db and db_path.exists():
        db_path.unlink()

    memo_store = DuckDBMemoStore(db_path)
    controller = scs.SemanticCacheController(
        metrics=scs.ExecutionMetrics(),
        embedder=FakeEmbedder(),
        reranker=None,
        corpus_id=str(fixture.get("dataset", "dp-memo-shared-context")),
        memo_store=memo_store,
    )
    llm = MLXLocalLLM(
        LocalLLMConfig(model=args.model, max_tokens=args.max_tokens, max_kv_size=args.max_kv_size)
    )

    predictions_path = out_dir / "predictions.jsonl"
    rows_path = out_dir / "bridge_rows.jsonl"
    totals = {
        "questions": 0,
        "correct": 0,
        "model_calls": 0,
        "aggregate_calls": 0,
        "answer_calls": 0,
        "answer_replays": 0,
        "semantic_reuse": 0,
        "verifier_calls": 0,
        "exact_replays": 0,
        "fact_extraction_model_calls": 0,
        "fact_extraction_aggregate_calls": 0,
        "fact_initial_covered_length": 0,
        "fact_initial_missing_length": 0,
        "fact_final_covered_length": 0,
        "fact_final_missing_length": 0,
        "fact_windows": 0,
        "fact_reused_windows": 0,
        "fact_missing_windows": 0,
        "latency_ms": 0.0,
    }

    with predictions_path.open("w", encoding="utf-8") as pred_file, rows_path.open("w", encoding="utf-8") as row_file:
        for document in fixture.get("documents", []):
            document_id = str(document["id"])
            context = str(document["context"])
            chunks = chunk_sentences(context, args.sentences_per_chunk)
            content_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]
            fact_solver_calls = []

            def fact_solver(task, scope):
                fact_solver_calls.append((scope.start, scope.end))
                scoped_context = "\n".join(chunks[scope.start : scope.end])
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Extract reusable factual claims from this context slice. "
                            "Keep exact names, codes, channels, invoices, and active/obsolete distinctions. "
                            "Return concise bullet lines only."
                        ),
                    },
                    {"role": "user", "content": scoped_context},
                ]
                facts = llm.generate_chat(messages, max_tokens=args.max_tokens)
                return {
                    "result": facts,
                    "not_found": not facts.strip(),
                    "confidence": 0.8,
                    "evidence": [
                        EvidenceSpan(
                            document_id=document_id,
                            start=idx,
                            end=idx + 1,
                            unit="chunk",
                            text=chunks[idx],
                        )
                        for idx in range(scope.start, scope.end)
                    ],
                }

            def fact_aggregate(query, scope, plan):
                entries = [entry for entry in plan.reusable_entries if entry.result]
                return {
                    "result": "\n".join(entry.result for entry in entries),
                    "used_entry_ids": [entry.entry_id for entry in entries],
                    "confidence": 0.8,
                }

            facts_result = controller.solve_with_memo(
                "Extract all reusable factual claims from this shared context.",
                chunks,
                fact_solver,
                document_id=document_id,
                chunk_size=args.chunk_size,
                content_hash=content_hash,
                task_type="shared_context_fact_extract",
                output_contract="fact_bullets",
                aggregate_fn=fact_aggregate,
            )
            fact_scope = controller._memo_scope(document_id, 0, len(chunks), content_hash=content_hash)
            fact_task = controller._memo_task(
                "Extract all reusable factual claims from this shared context.",
                task_type="shared_context_fact_extract",
                output_contract="fact_bullets",
                constraints={"chunk_size": args.chunk_size},
            )
            fact_plan = controller.memo_store.plan_reuse(fact_task, fact_scope)
            totals["fact_extraction_model_calls"] += int(facts_result["model_calls"])
            totals["fact_extraction_aggregate_calls"] += int(facts_result["aggregate_calls"])
            fact_initial_telemetry = dict(facts_result.get("initial_memo_telemetry") or {})
            fact_final_telemetry = dict(facts_result.get("memo_telemetry") or {})
            totals["fact_initial_covered_length"] += int(fact_initial_telemetry.get("covered_length", 0))
            totals["fact_initial_missing_length"] += int(fact_initial_telemetry.get("missing_length", 0))
            totals["fact_final_covered_length"] += int(fact_final_telemetry.get("covered_length", 0))
            totals["fact_final_missing_length"] += int(fact_final_telemetry.get("missing_length", 0))
            totals["fact_windows"] += int(facts_result.get("window_count", 0))
            totals["fact_reused_windows"] += int(facts_result.get("reused_window_count", 0))
            totals["fact_missing_windows"] += int(facts_result.get("missing_window_count", 0))

            for question_row in document.get("questions", []):
                question_id = str(question_row["id"])
                question = str(question_row["question"])
                expected = question_row["answer"]

                started = time.time()

                def memo_planner(query, candidates):
                    compact_candidates = [
                        {
                            "entry_id": candidate["entry_id"],
                            "task": candidate["task"],
                            "scope": candidate["scope"],
                            "result": candidate["result"],
                            "reusable_as": candidate["reusable_as"],
                            "confidence": candidate["confidence"],
                            "evidence": candidate["evidence"],
                        }
                        for candidate in candidates
                    ]
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a memo-aware planner. Answer the question using only the memo "
                                "candidate entries. Return only valid compact JSON with keys action, "
                                "answer, used_entry_ids, confidence. action must be answer when sufficient. "
                                "The answer must be a concise plain string, not another JSON object."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Question: {query}\n\n"
                                f"Memo candidates:\n{json.dumps(compact_candidates, ensure_ascii=False)}"
                            ),
                        },
                    ]
                    raw = llm.generate_chat(messages, max_tokens=args.aggregate_max_tokens)
                    return planner_response(raw, [c["entry_id"] for c in candidates])

                answer_result = controller.answer_from_memo_context(
                    question,
                    memo_planner,
                    fact_scope,
                    task_type="shared_context_question_answer",
                    output_contract="short_answer",
                    candidate_limit=12,
                )
                answer = str(answer_result["answer"]).strip()
                latency_ms = (time.time() - started) * 1000.0

                correct = answer_matches(answer, expected)
                totals["questions"] += 1
                totals["correct"] += int(correct)
                totals["answer_calls"] += int(answer_result["planner_calls"])
                totals["latency_ms"] += latency_ms
                answer_replay = controller.answer_from_memo_context(
                    question,
                    memo_planner,
                    fact_scope,
                    task_type="shared_context_question_answer",
                    output_contract="short_answer",
                    candidate_limit=12,
                )
                totals["answer_replays"] += int(answer_replay["planner_calls"] == 0)
                fact_replay = controller.solve_with_memo(
                    "Extract all reusable factual claims from this shared context.",
                    chunks,
                    fact_solver,
                    document_id=document_id,
                    chunk_size=args.chunk_size,
                    content_hash=content_hash,
                    task_type="shared_context_fact_extract",
                    output_contract="fact_bullets",
                    aggregate_fn=fact_aggregate,
                )
                totals["exact_replays"] += int(
                    fact_replay["model_calls"] == 0 and fact_replay["aggregate_calls"] == 0
                )

                pred_file.write(
                    json.dumps(
                        {
                            "document_id": document_id,
                            "id": question_id,
                            "question": question,
                            "generation": answer,
                            "expected_answer": expected,
                            "correct_contains": correct,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                row_file.write(
                    json.dumps(
                        {
                            "document_id": document_id,
                            "question_id": question_id,
                            "correct_contains": correct,
                            "latency_ms": round(latency_ms, 3),
                            "chunk_count": len(chunks),
                            "answer_planner_calls": answer_result["planner_calls"],
                            "answer_replay_planner_calls": answer_replay["planner_calls"],
                            "memo_candidate_count": answer_result.get("candidate_count", 0),
                            "answer_dependencies": answer_result.get("dependencies", []),
                            "fact_model_calls": facts_result["model_calls"],
                            "fact_aggregate_calls": facts_result["aggregate_calls"],
                            "fact_window_count": facts_result.get("window_count", 0),
                            "fact_reused_window_count": facts_result.get("reused_window_count", 0),
                            "fact_missing_window_count": facts_result.get("missing_window_count", 0),
                            "fact_initial_memo_telemetry": fact_initial_telemetry,
                            "fact_memo_telemetry": fact_final_telemetry,
                            "fact_window_memo_telemetry": facts_result.get("window_memo_telemetry", []),
                            "fact_solver_scopes": fact_solver_calls,
                            "fact_replay_model_calls": fact_replay["model_calls"],
                            "fact_replay_aggregate_calls": fact_replay["aggregate_calls"],
                            "fact_replay_memo_telemetry": fact_replay.get("memo_telemetry", {}),
                            "answer": answer,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                print(
                    f"[{question_id}] correct={correct} planner_calls={answer_result['planner_calls']} "
                    f"answer_replay={answer_replay['planner_calls']} "
                    f"fact_replay=({fact_replay['model_calls']},{fact_replay['aggregate_calls']}) "
                    f"answer={answer!r}"
                )

    manifest = {
        "benchmark": "dp_memo_shared_context",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "fixture": args.fixture,
        "duckdb_path": str(db_path),
        "predictions": str(predictions_path),
        "bridge_rows": str(rows_path),
        "settings": {
            "sentences_per_chunk": args.sentences_per_chunk,
            "chunk_size": args.chunk_size,
        },
        "totals": {
            **totals,
            "accuracy_contains": round(totals["correct"] / max(totals["questions"], 1), 6),
            "fact_initial_coverage_ratio": round(
                totals["fact_initial_covered_length"]
                / max(totals["fact_initial_covered_length"] + totals["fact_initial_missing_length"], 1),
                6,
            ),
            "fact_final_coverage_ratio": round(
                totals["fact_final_covered_length"]
                / max(totals["fact_final_covered_length"] + totals["fact_final_missing_length"], 1),
                6,
            ),
            "fact_reused_window_ratio": round(
                totals["fact_reused_windows"] / max(totals["fact_windows"], 1),
                6,
            ),
            "avg_latency_ms": round(totals["latency_ms"] / max(totals["questions"], 1), 3),
        },
        "memo_entries": len(controller.memo_store.entries),
        "memo_stats": controller.memo_store.stats(),
    }
    if hasattr(controller.memo_store, "context_chunk_count"):
        manifest["context_chunks"] = controller.memo_store.context_chunk_count(controller.corpus_id)
    if hasattr(controller.memo_store, "graph_edges"):
        edges = controller.memo_store.graph_edges()
        manifest["memo_graph"] = {
            "edge_count": len(edges),
            "edges": edges[:20],
        }
        exact_entries = [
            entry
            for entry in controller.memo_store.entries.values()
            if entry.dependencies
        ]
        if exact_entries:
            manifest["memo_graph"]["sample_lineage"] = controller.memo_store.lineage(
                exact_entries[-1].entry_id
            )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    memo_store.close()

    print("\n[DP Memo Shared Context] Completed")
    print(json.dumps(manifest["totals"], indent=2))
    print(f"Predictions: {predictions_path}")
    print(f"Rows       : {rows_path}")
    print(f"Manifest   : {manifest_path}")
    print(f"DuckDB     : {db_path}")


if __name__ == "__main__":
    main()
