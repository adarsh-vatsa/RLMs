"""Official RULER v2 benchmark bridge for the semantic cache architecture.

This script is intentionally official-benchmark-only. It takes prepared RULER-style
samples, runs predictions through this architecture, and optionally executes the
official evaluator command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import semantic_cache_system as scs


# Work around duplicate OpenMP runtime loading on macOS when combining
# FAISS and Torch-backed model components in a single process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, list):
        for item in value:
            text = _coerce_text(item)
            if text:
                return text
    return ""


def _coerce_expected_answer(value):
    """Preserve list-valued expected answers for multi-value RULER tasks."""
    if value is None:
        return ""
    if isinstance(value, list):
        coerced = []
        for item in value:
            text = _coerce_text(item)
            if text:
                coerced.append(text)
        return coerced if coerced else ""
    return _coerce_text(value)


def _has_expected_answer(value) -> bool:
    if isinstance(value, list):
        return len(value) > 0
    return bool(_coerce_text(value))


def _parse_csv_values(raw: str) -> List[str]:
    values = [part.strip() for part in (raw or "").split(",")]
    return [part for part in values if part]


def _infer_task_from_source_file(source_file: str) -> str:
    if not source_file:
        return ""
    path = Path(source_file)
    parent = path.parent.name.strip()
    if parent:
        return parent
    return ""


def _infer_length_from_source_file(source_file: str) -> str:
    if not source_file:
        return ""
    setup_name = Path(source_file).parent.parent.name.strip()
    if not setup_name:
        return ""
    match = re.search(r"_(\d+)$", setup_name)
    if match:
        return match.group(1)
    return ""


def _split_prompt_like_record(prompt: str) -> tuple[str, str]:
    text = (prompt or "").strip()
    if not text:
        return "", ""

    # QA-style RULER2 records use "Text:" followed by a short query block.
    if "\nText:" in text:
        parts = text.rsplit("\nText:", 1)
        if len(parts) == 2:
            context = parts[0].strip()
            query = ("Text:" + parts[1]).strip()
            if context and query:
                return context, query

    # NIAH-style records usually append the question as the last line.
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        last = lines[-1].strip()
        if last.endswith("?") or last.lower().startswith(("what ", "which ", "who ", "where ", "when ", "how ")):
            context = "\n".join(lines[:-1]).strip()
            if context:
                return context, last

    # Fallback: use full prompt as context and a generic retrieval question.
    return text, "Answer the question from the provided context."


def _format_command(template: str, placeholders: Dict[str, str]) -> str:
    command = template
    for key, value in placeholders.items():
        command = command.replace("{" + key + "}", value)
    return command


def _run_command(command: str, cwd: Path) -> None:
    print(f"[OFFICIAL] Running command: {command}")
    completed = subprocess.run(command, shell=True, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {command}"
        )


def _load_official_samples(prepared_path: Path) -> List[dict]:
    if not prepared_path.exists():
        raise FileNotFoundError(f"Prepared data path not found: {prepared_path}")

    candidate_files: List[Path] = []
    if prepared_path.is_file():
        candidate_files = [prepared_path]
    else:
        for pattern in ("*.jsonl", "*.json"):
            candidate_files.extend(sorted(prepared_path.rglob(pattern)))

    samples: List[dict] = []
    for file_path in candidate_files:
        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            for line in file_path.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if not text:
                    continue
                obj = json.loads(text)
                if isinstance(obj, dict):
                    obj.setdefault("__source_file", str(file_path))
                    samples.append(obj)
        elif suffix == ".json":
            obj = json.loads(file_path.read_text(encoding="utf-8"))
            rows: List[dict] = []
            if isinstance(obj, list):
                rows = [x for x in obj if isinstance(x, dict)]
            elif isinstance(obj, dict):
                for key in ("samples", "examples", "data", "items", "instances"):
                    value = obj.get(key)
                    if isinstance(value, list):
                        rows = [x for x in value if isinstance(x, dict)]
                        break
                if not rows:
                    rows = [obj]
            for row in rows:
                row.setdefault("__source_file", str(file_path))
            samples.extend(rows)

    if not samples:
        raise ValueError(
            f"No JSON/JSONL records found under prepared data path: {prepared_path}"
        )
    return samples


def _normalize_official_sample(
    item: dict,
    idx: int,
    fallback_length: str,
) -> Optional[dict]:
    question = (
        _coerce_text(item.get("question"))
        or _coerce_text(item.get("query"))
        or _coerce_text(item.get("prompt"))
        or _coerce_text(item.get("instruction"))
    )
    context = (
        _coerce_text(item.get("context"))
        or _coerce_text(item.get("input"))
        or _coerce_text(item.get("document"))
        or _coerce_text(item.get("text"))
        or _coerce_text(item.get("content"))
    )
    task = (
        _coerce_text(item.get("task"))
        or _coerce_text(item.get("subtask"))
        or _coerce_text(item.get("category"))
    )
    length = (
        _coerce_text(item.get("length") or item.get("context_length") or item.get("seq_len"))
        or fallback_length
    )
    expected = ""
    for key in ("answer", "expected_answer", "expected", "target", "output"):
        candidate = _coerce_expected_answer(item.get(key))
        if _has_expected_answer(candidate):
            expected = candidate
            break
    sample_id = (
        _coerce_text(item.get("id"))
        or _coerce_text(item.get("sample_id"))
        or _coerce_text(item.get("query_id"))
        or f"sample_{idx:06d}"
    )

    if not question or not context:
        prompt_text = question or context
        if prompt_text:
            context, question = _split_prompt_like_record(prompt_text)

    source_file = _coerce_text(item.get("__source_file"))

    if not task:
        task = _infer_task_from_source_file(source_file)

    inferred_length = _infer_length_from_source_file(source_file)
    if inferred_length:
        length = inferred_length

    if not question or not context:
        return None

    return {
        "id": sample_id,
        "task": task,
        "length": length,
        "question": question,
        "context": context,
        "expected_answer": expected,
        "source_file": _coerce_text(item.get("__source_file")),
    }


def _filter_official_samples(
    samples: List[dict],
    selected_tasks: List[str],
    selected_lengths: List[str],
    max_samples_per_task: int,
) -> List[dict]:
    task_filter = {task.lower() for task in selected_tasks}
    length_filter = {length for length in selected_lengths}
    counts: Dict[str, int] = {}
    filtered: List[dict] = []

    for sample in samples:
        task_name = (sample.get("task") or "").lower()
        length_name = sample.get("length") or ""

        if not task_name or not length_name:
            continue
        if task_filter and task_name not in task_filter:
            continue
        if length_filter and length_name not in length_filter:
            continue

        key = sample.get("task") or "unlabeled"
        if max_samples_per_task > 0:
            used = counts.get(key, 0)
            if used >= max_samples_per_task:
                continue
            counts[key] = used + 1

        filtered.append(sample)

    return filtered


def _build_dataset_content_signature(samples: List[dict]) -> str:
    """Build a stable content signature independent of parent folder naming."""
    hasher = hashlib.sha256()
    ordered = sorted(
        samples,
        key=lambda row: (
            _coerce_text(row.get("task")),
            _coerce_text(row.get("length")),
            _coerce_text(row.get("id")),
        ),
    )
    for sample in ordered:
        payload = {
            "id": _coerce_text(sample.get("id")),
            "task": _coerce_text(sample.get("task")),
            "length": _coerce_text(sample.get("length")),
            "question": _coerce_text(sample.get("question")),
            "context": _coerce_text(sample.get("context")),
            "expected_answer": _coerce_expected_answer(sample.get("expected_answer")),
        }
        hasher.update(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        )
        hasher.update(b"\n")
    return hasher.hexdigest()[:16]


def _sanitize_path_segment(raw: str, max_len: int = 180) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", _coerce_text(raw)).strip("._-")
    if not value:
        value = "default"
    return value[:max_len]


def _resolve_cache_namespace(
    corpus_id: str,
    selected_tasks: List[str],
    selected_lengths: List[str],
    selected_samples: List[dict],
) -> tuple[str, str]:
    dataset_sig = _build_dataset_content_signature(selected_samples)
    task_sig = "-".join(sorted({task.lower() for task in selected_tasks}))
    length_sig = "-".join(sorted({length for length in selected_lengths}))
    base = f"{corpus_id}__{task_sig}__{length_sig}__{dataset_sig}"
    return _sanitize_path_segment(base), dataset_sig


def _snapshot_metrics(metrics: scs.ExecutionMetrics) -> dict:
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


def _build_controller_with_components(
    corpus_id: str,
    corpus_domain: str,
    embedder: scs.EmbeddingEngine,
    reranker: scs.Reranker,
) -> scs.SemanticCacheController:
    scs.EXECUTOR_MODEL = "claude-sonnet-4-5"
    metrics = scs.ExecutionMetrics()
    return scs.SemanticCacheController(
        metrics=metrics,
        embedder=embedder,
        reranker=reranker,
        corpus_id=corpus_id,
        corpus_domain=corpus_domain,
    )


def run_official_benchmark(args: argparse.Namespace) -> None:
    """Run official benchmark bridge generation and optional official scoring.

    Correctness is intentionally delegated to the official evaluator command.
    This function generates prediction artifacts and telemetry only; it does not
    compute custom correctness metrics inside this repository.
    """
    selected_tasks = _parse_csv_values(args.official_tasks)
    selected_lengths = _parse_csv_values(args.official_lengths)
    if not selected_tasks:
        raise ValueError("--official-tasks must include at least one task")
    if not selected_lengths:
        raise ValueError("--official-lengths must include at least one length")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / "official_ruler_v2" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared_path = Path(args.official_prepared_data)
    placeholders = {
        "tasks_csv": ",".join(selected_tasks),
        "lengths_csv": ",".join(selected_lengths),
        "prepared_data": str(prepared_path),
        "results_dir": str(out_dir),
    }

    if args.official_prep_command:
        prep_command = _format_command(args.official_prep_command, placeholders)
        _run_command(prep_command, cwd=Path.cwd())

    raw_samples = _load_official_samples(prepared_path)
    normalized_samples: List[dict] = []
    fallback_length = selected_lengths[0] if len(selected_lengths) == 1 else ""
    for idx, item in enumerate(raw_samples, start=1):
        normalized = _normalize_official_sample(item, idx, fallback_length)
        if normalized is not None:
            normalized_samples.append(normalized)

    selected = _filter_official_samples(
        normalized_samples,
        selected_tasks=selected_tasks,
        selected_lengths=selected_lengths,
        max_samples_per_task=args.official_max_samples_per_task,
    )
    if not selected:
        raise ValueError(
            "No prepared samples matched requested tasks/lengths with required fields"
        )

    seen_ids = set()
    for sample in selected:
        sample_id = sample["id"]
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample id in selected dataset: {sample_id}")
        seen_ids.add(sample_id)

    predictions_path = out_dir / "predictions.jsonl"
    bridge_rows_path = out_dir / "bridge_rows.jsonl"
    manifest_path = out_dir / "manifest.json"

    print(f"\n[OFFICIAL] Run id: {run_id}")
    print(f"[OFFICIAL] Selected samples: {len(selected)}")
    print(f"[OFFICIAL] Tasks: {selected_tasks}")
    print(f"[OFFICIAL] Lengths: {selected_lengths}")
    print(f"[OFFICIAL] Output dir: {out_dir}")

    cache_state_enabled = args.mode == "cache"
    cache_namespace = ""
    dataset_signature = ""
    cache_state_root: Optional[Path] = None
    cache_state_path: Optional[Path] = None
    cache_state_existed_before_reset = False
    cache_state_existed_before_run = False
    cache_entries_before_run = 0
    cache_entries_after_run = 0
    cache_load_attempts = 0
    cache_load_successes = 0
    cache_hits = 0

    if cache_state_enabled:
        cache_namespace, dataset_signature = _resolve_cache_namespace(
            corpus_id=args.corpus_id,
            selected_tasks=selected_tasks,
            selected_lengths=selected_lengths,
            selected_samples=selected,
        )
        if args.official_cache_state_root:
            cache_state_root = Path(args.official_cache_state_root)
        else:
            cache_state_root = Path(args.output_dir) / "official_ruler_v2" / "cache_state"
        cache_state_path = cache_state_root / cache_namespace

        cache_state_existed_before_reset = cache_state_path.exists()
        if args.official_cache_reset and cache_state_existed_before_reset:
            shutil.rmtree(cache_state_path)
            print(f"[OFFICIAL] Cache reset requested. Removed state at: {cache_state_path}")

        cache_state_root.mkdir(parents=True, exist_ok=True)
        cache_state_existed_before_run = cache_state_path.exists()
        print(f"[OFFICIAL] Cache state root: {cache_state_root}")
        print(f"[OFFICIAL] Cache namespace : {cache_namespace}")
        print(f"[OFFICIAL] Dataset signature: {dataset_signature}")
        if cache_state_existed_before_run:
            print("[OFFICIAL] Cache state: warm start (existing state found)")
        else:
            print("[OFFICIAL] Cache state: cold start (no prior state)")

    shared_embedder = scs.EmbeddingEngine()
    shared_reranker = scs.Reranker()

    prediction_rows: List[dict] = []
    bridge_rows: List[dict] = []

    for idx, sample in enumerate(selected, start=1):
        safe_sample_id = re.sub(r"[^a-zA-Z0-9._-]", "_", sample["id"])
        sample_docs_dir = out_dir / "sample_docs" / safe_sample_id
        sample_docs_dir.mkdir(parents=True, exist_ok=True)
        doc_path = sample_docs_dir / "context.txt"
        doc_path.write_text(sample["context"].rstrip() + "\n", encoding="utf-8")

        controller_corpus_id = (
            cache_namespace if cache_state_enabled else f"{args.corpus_id}_{idx}"
        )
        controller = _build_controller_with_components(
            corpus_id=controller_corpus_id,
            corpus_domain=args.domain,
            embedder=shared_embedder,
            reranker=shared_reranker,
        )

        if cache_state_enabled and cache_state_path is not None:
            cache_load_attempts += 1
            if cache_state_path.exists() and controller.load(cache_state_path):
                cache_load_successes += 1
            loaded_entries = controller.get_total_entries()
            if idx == 1:
                cache_entries_before_run = loaded_entries

        controller.ingest(sample_docs_dir)

        before = _snapshot_metrics(controller.metrics)
        t0 = time.time()
        output = controller.search(
            sample["question"],
            top_k=args.top_k,
            rerank_top=args.rerank_top,
            synthesize=True,
            cache_read=args.mode == "cache",
        )
        latency_ms = (time.time() - t0) * 1000.0
        after = _snapshot_metrics(controller.metrics)

        answer = _coerce_text(output.get("answer"))
        prediction_rows.append(
            {
                "id": sample["id"],
                "sample_id": sample["id"],
                "task": sample["task"],
                "length": sample["length"],
                "question": sample["question"],
                "generation": answer,
                "prediction": answer,
                "answer": answer,
                "expected_answer": sample.get("expected_answer", ""),
            }
        )
        bridge_rows.append(
            {
                "sample_id": sample["id"],
                "task": sample["task"],
                "length": sample["length"],
                "source_file": sample.get("source_file", ""),
                "latency_ms": round(latency_ms, 3),
                "delta_calls": after["calls"] - before["calls"],
                "delta_input_tokens": after["input_tokens"] - before["input_tokens"],
                "delta_output_tokens": after["output_tokens"] - before["output_tokens"],
                "delta_cost_usd": round(after["cost"] - before["cost"], 8),
                "from_cache": bool(output.get("from_cache")),
                "cache_type": _coerce_text(output.get("cache_type")),
            }
        )

        if output.get("from_cache"):
            cache_hits += 1

        if cache_state_enabled and cache_state_path is not None:
            controller.save(cache_state_path)
            cache_entries_after_run = controller.get_total_entries()

    with open(predictions_path, "w", encoding="utf-8") as f:
        for row in prediction_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(bridge_rows_path, "w", encoding="utf-8") as f:
        for row in bridge_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    evaluator_command = ""
    if args.official_eval_command:
        # Official correctness/scoring happens here via external evaluator.
        placeholders.update(
            {
                "predictions": str(predictions_path),
                "bridge_rows": str(bridge_rows_path),
            }
        )
        evaluator_command = _format_command(args.official_eval_command, placeholders)
        _run_command(evaluator_command, cwd=Path.cwd())

    implemented_tasks = sorted({row["task"] for row in bridge_rows})
    baseline_total = 13
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_target": "ruler_v2",
        "tasks_requested": selected_tasks,
        "lengths_requested": selected_lengths,
        "samples_selected": len(bridge_rows),
        "mode": args.mode,
        "top_k": args.top_k,
        "rerank_top": args.rerank_top,
        "official_prepared_data": str(prepared_path),
        "official_prep_command": args.official_prep_command,
        "official_eval_command": evaluator_command,
        "artifacts": {
            "predictions": str(predictions_path),
            "bridge_rows": str(bridge_rows_path),
        },
        "baseline13_progress": {
            "baseline_total_tasks": baseline_total,
            "implemented_task_count": len(implemented_tasks),
            "pending_task_count": max(0, baseline_total - len(implemented_tasks)),
            "implemented_tasks": implemented_tasks,
        },
        "official_score_status": (
            "official evaluator command executed"
            if evaluator_command
            else "evaluator skipped (no --official-eval-command provided)"
        ),
    }
    if cache_state_enabled:
        manifest["cache_reuse"] = {
            "enabled": True,
            "cache_namespace": cache_namespace,
            "dataset_signature": dataset_signature,
            "cache_state_root": str(cache_state_root) if cache_state_root else "",
            "cache_state_path": str(cache_state_path) if cache_state_path else "",
            "cache_state_existed_before_reset": cache_state_existed_before_reset,
            "cache_state_existed_before_run": cache_state_existed_before_run,
            "cache_reset_requested": bool(args.official_cache_reset),
            "cache_load_attempts": cache_load_attempts,
            "cache_load_successes": cache_load_successes,
            "cache_entries_before_run": cache_entries_before_run,
            "cache_entries_after_run": cache_entries_after_run,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / len(bridge_rows), 6) if bridge_rows else 0.0,
        }
    else:
        manifest["cache_reuse"] = {
            "enabled": False,
            "reason": "mode is baseline",
        }
    if args.manifest_note:
        manifest["note"] = args.manifest_note

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n[OFFICIAL] Completed.")
    print(f"[OFFICIAL] Predictions : {predictions_path}")
    print(f"[OFFICIAL] Bridge rows : {bridge_rows_path}")
    print(f"[OFFICIAL] Manifest    : {manifest_path}")
    if cache_state_enabled:
        print(
            "[OFFICIAL] Cache reuse summary: "
            f"loads={cache_load_successes}/{cache_load_attempts}, "
            f"hits={cache_hits}/{len(bridge_rows)}, "
            f"entries_before={cache_entries_before_run}, "
            f"entries_after={cache_entries_after_run}"
        )
    if not evaluator_command:
        print(
            "[OFFICIAL] Evaluator command not provided. "
            "Set --official-eval-command for strict official metrics."
        )


def main() -> None:
    start = time.time()
    parser = argparse.ArgumentParser(
        description="Official RULER v2 benchmark bridge for semantic cache architecture"
    )
    parser.add_argument(
        "--official-prepared-data",
        type=str,
        required=True,
        help="Path to official prepared JSON/JSONL file or directory",
    )
    parser.add_argument(
        "--official-tasks",
        type=str,
        default="mk_niah_basic,mv_niah_basic,qa_basic",
        help="Comma-separated RULER v2 task names",
    )
    parser.add_argument(
        "--official-lengths",
        type=str,
        default="8192,32768",
        help="Comma-separated context lengths",
    )
    parser.add_argument(
        "--official-max-samples-per-task",
        type=int,
        default=0,
        help="Cap samples per task (0 means all)",
    )
    parser.add_argument(
        "--official-prep-command",
        type=str,
        default="",
        help=(
            "Optional shell command for official prep. "
            "Placeholders: {tasks_csv}, {lengths_csv}, {prepared_data}, {results_dir}"
        ),
    )
    parser.add_argument(
        "--official-eval-command",
        type=str,
        default="",
        help=(
            "Optional shell command for official evaluator. "
            "When provided, this is the source of correctness/scoring. "
            "Placeholders: {predictions}, {bridge_rows}, {prepared_data}, "
            "{results_dir}, {tasks_csv}, {lengths_csv}"
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "cache"],
        default="cache",
        help="Architecture execution mode during prediction generation",
    )
    parser.add_argument("--corpus-id", type=str, default="official_ruler")
    parser.add_argument(
        "--official-cache-state-root",
        type=str,
        default="",
        help=(
            "Root directory for persistent cache state in cache mode. "
            "If omitted, defaults to <output-dir>/official_ruler_v2/cache_state"
        ),
    )
    parser.add_argument(
        "--official-cache-reset",
        action="store_true",
        help=(
            "Reset only the resolved persistent cache namespace before run. "
            "No automatic cache deletion happens unless this flag is set."
        ),
    )
    parser.add_argument("--domain", type=str, default="general")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--rerank-top", type=int, default=5)
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
    run_official_benchmark(args)
    end = time.time()
    elapsed = end - start
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")
    

if __name__ == "__main__":
    main()
