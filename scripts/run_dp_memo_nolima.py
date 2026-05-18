"""Run the DP memoization stack on NoLiMa fixture or official data."""

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
from language_memoization import (  # noqa: E402
    DuckDBMemoStore,
    EvidenceSpan,
    REUSE_AGGREGATION_COMPONENT,
    REUSE_SUPPORTING_FACT,
)
from local_llm import DEFAULT_MAX_KV_SIZE, DEFAULT_MAX_TOKENS, DEFAULT_MLX_MODEL, LocalLLMConfig, MLXLocalLLM  # noqa: E402
from nolima.parity_bridge import (  # noqa: E402
    build_dataset_signature,
    iter_nolima_samples,
    list_haystack_assets,
    load_needle_set_cases,
)


class FakeEmbedder:
    def encode_query(self, query):
        return np.array([1.0, 0.0], dtype="float32")

    def encode_single(self, text):
        return np.array([1.0, 0.0], dtype="float32")

    def encode_documents(self, documents):
        return np.array([[1.0, 0.0] for _ in documents], dtype="float32")


def parse_lengths(raw: str) -> list[int]:
    lengths = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token.endswith("k"):
            lengths.append(int(float(token[:-1]) * 1000))
        else:
            lengths.append(int(token))
    return lengths


def chunk_text(text: str, words_per_chunk: int) -> list[str]:
    words = re.findall(r"\S+", text)
    chunks = []
    for start in range(0, len(words), words_per_chunk):
        part = " ".join(words[start : start + words_per_chunk])
        if part:
            chunks.append(part)
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


def answer_matches(answer: str, gold_answers: list[str]) -> bool:
    normalized = answer.lower()
    return any(str(gold).lower() in normalized for gold in gold_answers)


def data_kind(needle_set_path: str, haystack_dir: str) -> str:
    joined = f"{needle_set_path} {haystack_dir}"
    if "benchmark_fixtures" in joined:
        return "fixture"
    if "benchmark_data" in joined:
        return "official_or_downloaded"
    return "custom"


def document_id_for_sample(sample: dict, mode: str) -> str:
    if mode == "stable":
        return "__".join(
            [
                str(sample["test_name"]),
                str(sample["haystack_name"]),
                f"D{int(sample['depth_index']):02d}",
            ]
        )
    return str(sample["id"])


def content_hash_for_sample(context: str, sample: dict, mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "source":
        payload = {
            "test_name": sample.get("test_name"),
            "haystack_hash": sample.get("haystack_hash"),
            "needle": sample.get("needle"),
            "depth_index": sample.get("depth_index"),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]


def stable_corpus_id(needle_hash: str, haystacks: list, seed: int) -> str:
    payload = {
        "needle_hash": needle_hash,
        "haystacks": [
            {"name": haystack.name, "sha256": haystack.sha256}
            for haystack in haystacks
        ],
        "seed": seed,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"dp_memo_nolima_stable_{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(description="DP memoization benchmark on NoLiMa fixtures.")
    parser.add_argument("--needle-set-path", default="benchmark_fixtures/nolima/needlesets/needle_set.json")
    parser.add_argument("--haystack-dir", default="benchmark_fixtures/nolima/haystack/rand_shuffle")
    parser.add_argument("--lengths", default="250")
    parser.add_argument("--depth-intervals", type=int, default=2)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Limit expanded NoLiMa cases before sample generation; 0 uses all cases.",
    )
    parser.add_argument(
        "--max-haystacks",
        type=int,
        default=0,
        help="Limit haystack files before sample generation; 0 uses all haystacks.",
    )
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Skip this many generated samples before applying --max-samples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-words", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument(
        "--solver-mode",
        choices=("answer", "evidence"),
        default="answer",
        help=(
            "answer asks each scope for the final answer; evidence asks each scope "
            "for reusable candidate evidence before aggregation."
        ),
    )
    parser.add_argument(
        "--document-id-mode",
        choices=("sample", "stable"),
        default="sample",
        help=(
            "sample keeps each NoLiMa sample isolated; stable removes length from "
            "the document id so overlapping length sweeps can reuse prefix work."
        ),
    )
    parser.add_argument(
        "--content-hash-mode",
        choices=("sample", "source", "none"),
        default="sample",
        help=(
            "sample hashes the whole generated context; source hashes stable source "
            "metadata; none disables content-hash isolation for experiments."
        ),
    )
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
        help="Optional aggregation generation cap. Omit to use the MLX backend default.",
    )
    parser.add_argument("--max-kv-size", type=int, default=DEFAULT_MAX_KV_SIZE)
    parser.add_argument("--duckdb-path", default="benchmark_artifacts/dp_memo_nolima/memo.duckdb")
    parser.add_argument("--output-dir", default="benchmark_artifacts/dp_memo_nolima")
    parser.add_argument("--reset-db", action="store_true")
    parser.add_argument(
        "--corpus-id-mode",
        choices=("dataset", "stable"),
        default="dataset",
        help=(
            "dataset namespaces by full dataset signature including lengths; stable "
            "excludes lengths so overlapping length sweeps can reuse memo work."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print a scoped solver progress line every N model calls; 0 disables progress.",
    )
    args = parser.parse_args()

    lengths = parse_lengths(args.lengths)
    cases, needle_hash = load_needle_set_cases(Path(args.needle_set_path), max_cases=args.max_cases)
    haystacks = list_haystack_assets(Path(args.haystack_dir), max_haystacks=args.max_haystacks)
    dataset_signature = build_dataset_signature(
        needle_set_hash=needle_hash,
        cases=cases,
        haystacks=haystacks,
        lengths=lengths,
        depth_intervals=args.depth_intervals,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.duckdb_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if args.reset_db and db_path.exists():
        db_path.unlink()

    corpus_id = (
        stable_corpus_id(needle_hash, haystacks, args.seed)
        if args.corpus_id_mode == "stable"
        else f"dp_memo_nolima_{dataset_signature}"
    )
    memo_store = DuckDBMemoStore(db_path)
    controller = scs.SemanticCacheController(
        metrics=scs.ExecutionMetrics(),
        embedder=FakeEmbedder(),
        reranker=None,
        corpus_id=corpus_id,
        memo_store=memo_store,
    )
    llm = MLXLocalLLM(
        LocalLLMConfig(model=args.model, max_tokens=args.max_tokens, max_kv_size=args.max_kv_size)
    )

    predictions_path = out_dir / "predictions.jsonl"
    rows_path = out_dir / "bridge_rows.jsonl"
    manifest_path = out_dir / "manifest.json"
    progress_path = out_dir / "progress.jsonl"

    totals = {
        "samples": 0,
        "correct": 0,
        "model_calls": 0,
        "aggregate_calls": 0,
        "semantic_reuse": 0,
        "exact_replay_checks": 0,
        "initial_covered_length": 0,
        "initial_missing_length": 0,
        "final_covered_length": 0,
        "final_missing_length": 0,
        "windows": 0,
        "reused_windows": 0,
        "missing_windows": 0,
        "latency_ms": 0.0,
    }

    def write_progress(progress_file, event: dict) -> None:
        progress_file.write(
            json.dumps({"time": datetime.now(timezone.utc).isoformat(), **event}, ensure_ascii=False)
            + "\n"
        )
        progress_file.flush()

    with (
        predictions_path.open("w", encoding="utf-8") as pred_file,
        rows_path.open("w", encoding="utf-8") as row_file,
        progress_path.open("w", encoding="utf-8") as progress_file,
    ):
        for source_idx, sample in enumerate(
            iter_nolima_samples(
                cases=cases,
                haystacks=haystacks,
                lengths=lengths,
                depth_intervals=args.depth_intervals,
                seed=args.seed,
            ),
            start=1,
        ):
            if args.sample_offset and source_idx <= args.sample_offset:
                continue
            idx = source_idx - args.sample_offset
            if args.max_samples and idx > args.max_samples:
                break

            context = str(sample["context"])
            chunks = chunk_text(context, args.chunk_words)
            question = str(sample["question"])
            document_id = document_id_for_sample(sample, args.document_id_mode)
            sample_hash = content_hash_for_sample(context, sample, args.content_hash_mode)
            solver_scopes = []
            aggregate_calls_seen = []
            write_progress(
                progress_file,
                {
                    "event": "sample_start",
                    "sample_index": idx,
                    "sample_id": sample["id"],
                    "corpus_id": controller.corpus_id,
                    "document_id": document_id,
                    "length": sample["length"],
                    "depth_percent": sample["depth_percent"],
                    "chunk_count": len(chunks),
                    "chunk_words": args.chunk_words,
                    "chunk_size": args.chunk_size,
                },
            )

            def solver(task, scope):
                solver_scopes.append((scope.start, scope.end))
                if args.progress_every > 0 and len(solver_scopes) % args.progress_every == 0:
                    print(
                        f"[{idx}] solver_call={len(solver_scopes)} "
                        f"scope={scope.start}..{scope.end}/{len(chunks)}",
                        flush=True,
                    )
                write_progress(
                    progress_file,
                    {
                        "event": "solver_call",
                        "sample_index": idx,
                        "sample_id": sample["id"],
                        "corpus_id": controller.corpus_id,
                        "solver_call": len(solver_scopes),
                        "scope_start": scope.start,
                        "scope_end": scope.end,
                        "chunk_count": len(chunks),
                    },
                )
                scoped_context = "\n".join(chunks[scope.start : scope.end])
                if args.solver_mode == "evidence":
                    system_prompt = (
                        "Extract compact standalone facts from this context slice. "
                        "Always include named characters, places, landmarks, aliases, and "
                        "relationships. Do not over-filter by the question; the aggregator will "
                        "decide relevance later. If the slice has no named-entity or relationship "
                        "facts, return exactly NOT_FOUND. If the slice contains any named person, "
                        "place, landmark, title, code, or relationship, do not return NOT_FOUND. "
                        "Return only concise evidence lines."
                    )
                else:
                    system_prompt = (
                        "Answer the question using this context slice. "
                        "Use explicitly stated facts and strong logical or common-knowledge "
                        "inferences when the slice supports them. "
                        "If the slice provides no explicit or inferential support, return exactly NOT_FOUND. "
                        "Return only the answer."
                    )
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Downstream question, do not use this to filter out facts: {task.prompt}\n\n"
                            f"Context:\n{scoped_context}"
                        ),
                    },
                ]
                answer = llm.generate_chat(messages, max_tokens=args.max_tokens)
                return {
                    "result": answer,
                    "not_found": answer.strip().upper() == "NOT_FOUND",
                    "confidence": 0.75,
                    "evidence": [
                        EvidenceSpan(
                            document_id=document_id,
                            start=chunk_idx,
                            end=chunk_idx + 1,
                            unit="chunk",
                            text=chunks[chunk_idx],
                        )
                        for chunk_idx in range(scope.start, scope.end)
                    ],
                    "metadata": {"benchmark": "nolima", "solver_mode": args.solver_mode},
                    "reusable_as": (
                        (REUSE_SUPPORTING_FACT, REUSE_AGGREGATION_COMPONENT)
                        if args.solver_mode == "evidence"
                        else (REUSE_AGGREGATION_COMPONENT,)
                    ),
                }

            def aggregate(query, scope, plan):
                aggregate_calls_seen.append((scope.start, scope.end))
                entries = [entry for entry in plan.reusable_entries if entry.result and "NOT_FOUND" not in entry.result]
                partials = "\n".join(
                    f"- id={entry.entry_id} chunks={entry.scope.start}..{entry.scope.end}: {entry.result}"
                    for entry in entries
                )
                entry_ids = [entry.entry_id for entry in entries]
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You answer NoLiMa retrieval questions from scoped partial answers or evidence. "
                            "Use explicitly stated facts and strong logical/common-knowledge inferences. "
                            "For two-hop location questions, connect landmarks, cities, regions, and countries "
                            "using common geographic knowledge when the evidence names a relevant place. "
                            "Return only JSON with result, used_entry_ids, confidence. "
                            "The result should be the shortest answer string."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Question: {query}\n\nPartial answers:\n{partials}\n\nReturn JSON.",
                    },
                ]
                raw = llm.generate_chat(messages, max_tokens=args.aggregate_max_tokens)
                parsed = parse_json_object(raw)
                if parsed:
                    parsed.setdefault("used_entry_ids", entry_ids)
                    parsed.setdefault("confidence", 0.75)
                    return parsed
                return {"result": raw, "used_entry_ids": entry_ids, "confidence": 0.5}

            started = time.time()
            result = controller.solve_with_memo(
                question,
                chunks,
                solver,
                document_id=document_id,
                chunk_size=args.chunk_size,
                content_hash=sample_hash,
                task_type="nolima_fixture",
                output_contract="short_answer",
                aggregate_fn=aggregate,
            )
            latency_ms = (time.time() - started) * 1000.0

            replay = controller.solve_with_memo(
                question,
                chunks,
                solver,
                document_id=document_id,
                chunk_size=args.chunk_size,
                content_hash=sample_hash,
                task_type="nolima_fixture",
                output_contract="short_answer",
                aggregate_fn=aggregate,
            )
            if replay["model_calls"] == 0 and replay["aggregate_calls"] == 0:
                totals["exact_replay_checks"] += 1

            answer = str(result["answer"]).strip()
            gold = list(sample["gold_answers"])
            correct = answer_matches(answer, gold)
            totals["samples"] += 1
            totals["correct"] += int(correct)
            totals["model_calls"] += int(result["model_calls"])
            totals["aggregate_calls"] += int(result["aggregate_calls"])
            totals["semantic_reuse"] += int(bool(result.get("semantic_reuse")))
            initial_telemetry = dict(result.get("initial_memo_telemetry") or {})
            final_telemetry = dict(result.get("memo_telemetry") or {})
            totals["initial_covered_length"] += int(initial_telemetry.get("covered_length", 0))
            totals["initial_missing_length"] += int(initial_telemetry.get("missing_length", 0))
            totals["final_covered_length"] += int(final_telemetry.get("covered_length", 0))
            totals["final_missing_length"] += int(final_telemetry.get("missing_length", 0))
            totals["windows"] += int(result.get("window_count", 0))
            totals["reused_windows"] += int(result.get("reused_window_count", 0))
            totals["missing_windows"] += int(result.get("missing_window_count", 0))
            totals["latency_ms"] += latency_ms

            pred_file.write(
                json.dumps(
                    {
                        "id": sample["id"],
                        "question": question,
                        "generation": answer,
                        "prediction": answer,
                        "expected_answer": sample["expected_answer"],
                        "gold_answers": gold,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            row_file.write(
                json.dumps(
                    {
                        "sample_id": sample["id"],
                        "corpus_id": controller.corpus_id,
                        "document_id": document_id,
                        "length": sample["length"],
                        "depth_percent": sample["depth_percent"],
                        "correct_contains": correct,
                        "latency_ms": round(latency_ms, 3),
                        "context_chars": len(context),
                        "context_words": len(re.findall(r"\S+", context)),
                        "chunk_count": len(chunks),
                        "chunk_size": args.chunk_size,
                        "chunk_words": args.chunk_words,
                        "model_calls": result["model_calls"],
                        "aggregate_calls": result["aggregate_calls"],
                        "window_count": result.get("window_count", 0),
                        "reused_window_count": result.get("reused_window_count", 0),
                        "missing_window_count": result.get("missing_window_count", 0),
                        "solver_scopes": solver_scopes,
                        "initial_memo_telemetry": initial_telemetry,
                        "memo_telemetry": final_telemetry,
                        "window_memo_telemetry": result.get("window_memo_telemetry", []),
                        "replay_model_calls": replay["model_calls"],
                        "replay_aggregate_calls": replay["aggregate_calls"],
                        "replay_memo_telemetry": replay.get("memo_telemetry", {}),
                        "memo_entry_id": result.get("memo_entry_id"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            print(
                f"[{idx}] correct={correct} model_calls={result['model_calls']} "
                f"aggregate_calls={result['aggregate_calls']} replay=({replay['model_calls']},"
                f"{replay['aggregate_calls']}) answer={answer!r} gold={gold}"
            )
            write_progress(
                progress_file,
                {
                    "event": "sample_done",
                    "sample_index": idx,
                    "sample_id": sample["id"],
                    "corpus_id": controller.corpus_id,
                    "correct_contains": correct,
                    "model_calls": result["model_calls"],
                    "aggregate_calls": result["aggregate_calls"],
                    "replay_model_calls": replay["model_calls"],
                    "replay_aggregate_calls": replay["aggregate_calls"],
                    "memo_entry_id": result.get("memo_entry_id"),
                },
            )

    manifest = {
        "benchmark": "dp_memo_nolima",
        "data_kind": data_kind(args.needle_set_path, args.haystack_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_signature": dataset_signature,
        "corpus_id": controller.corpus_id,
        "model": args.model,
        "duckdb_path": str(db_path),
        "predictions": str(predictions_path),
        "bridge_rows": str(rows_path),
        "progress": str(progress_path),
        "settings": {
            "lengths": lengths,
            "depth_intervals": args.depth_intervals,
            "max_cases": args.max_cases,
            "max_haystacks": args.max_haystacks,
            "max_samples": args.max_samples,
            "sample_offset": args.sample_offset,
            "chunk_words": args.chunk_words,
            "chunk_size": args.chunk_size,
            "solver_mode": args.solver_mode,
            "corpus_id_mode": args.corpus_id_mode,
            "document_id_mode": args.document_id_mode,
            "content_hash_mode": args.content_hash_mode,
            "progress_every": args.progress_every,
        },
        "totals": {
            **totals,
            "accuracy_contains": round(totals["correct"] / max(totals["samples"], 1), 6),
            "initial_coverage_ratio": round(
                totals["initial_covered_length"]
                / max(totals["initial_covered_length"] + totals["initial_missing_length"], 1),
                6,
            ),
            "final_coverage_ratio": round(
                totals["final_covered_length"]
                / max(totals["final_covered_length"] + totals["final_missing_length"], 1),
                6,
            ),
            "reused_window_ratio": round(totals["reused_windows"] / max(totals["windows"], 1), 6),
            "avg_latency_ms": round(totals["latency_ms"] / max(totals["samples"], 1), 3),
        },
        "memo_entries": len(controller.memo_store.entries),
        "memo_stats": controller.memo_store.stats(),
    }
    if hasattr(controller.memo_store, "context_chunk_count"):
        manifest["context_chunks"] = controller.memo_store.context_chunk_count(controller.corpus_id)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    memo_store.close()

    print("\n[DP Memo NoLiMa] Completed")
    print(json.dumps(manifest["totals"], indent=2))
    print(f"Predictions: {predictions_path}")
    print(f"Rows       : {rows_path}")
    print(f"Manifest   : {manifest_path}")
    print(f"DuckDB     : {db_path}")


if __name__ == "__main__":
    main()
