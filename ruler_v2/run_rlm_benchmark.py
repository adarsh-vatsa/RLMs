"""Uncached RLM baseline runner for prepared RULER v2 samples."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")

ARTIFACT_SUBDIR = "official_ruler_v2_rlm"
DEFAULT_RLM_BACKEND = "anthropic"
DEFAULT_RLM_MODEL = "claude-haiku-4-5-20251001"
RLM_ROOT_PROMPT = "Answer the question using the provided context. Return only the final answer."

# Mirrors semantic_cache_system.MODEL_FAMILY_PRICING_USD_PER_1K without importing
# the cache stack. The RLM runner is intentionally cache-free and dependency-light.
MODEL_FAMILY_PRICING_USD_PER_1K = {
    "sonnet": {"input": 0.003, "output": 0.015},
    "haiku": {"input": 0.001, "output": 0.005},
}


def _coerce_text(value: Any) -> str:
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


def _coerce_expected_answer(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, list):
        coerced = [_coerce_text(item) for item in value]
        coerced = [item for item in coerced if item]
        return coerced if coerced else ""
    return _coerce_text(value)


def _has_expected_answer(value: Any) -> bool:
    if isinstance(value, list):
        return len(value) > 0
    return bool(_coerce_text(value))


def _parse_csv_values(raw: str) -> List[str]:
    values = [part.strip() for part in (raw or "").split(",")]
    return [part for part in values if part]


def _infer_task_from_source_file(source_file: str) -> str:
    if not source_file:
        return ""
    parent = Path(source_file).parent.name.strip()
    return parent or ""


def _infer_length_from_source_file(source_file: str) -> str:
    if not source_file:
        return ""
    setup_name = Path(source_file).parent.parent.name.strip()
    match = re.search(r"_(\d+)$", setup_name)
    return match.group(1) if match else ""


def _split_prompt_like_record(prompt: str) -> tuple[str, str]:
    text = (prompt or "").strip()
    if not text:
        return "", ""

    if "\nText:" in text:
        parts = text.rsplit("\nText:", 1)
        if len(parts) == 2:
            context = parts[0].strip()
            query = ("Text:" + parts[1]).strip()
            if context and query:
                return context, query

    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        last = lines[-1].strip()
        if last.endswith("?") or last.lower().startswith(
            ("what ", "which ", "who ", "where ", "when ", "how ")
        ):
            context = "\n".join(lines[:-1]).strip()
            if context:
                return context, last

    return text, "Answer the question from the provided context."


def _format_command(template: str, placeholders: Dict[str, str]) -> str:
    command = template
    for key, value in placeholders.items():
        command = command.replace("{" + key + "}", value)
    return command


def _run_command(command: str, cwd: Path) -> None:
    print(f"[RLM-RULER] Running command: {command}")
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
        "source_file": source_file,
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


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return repr(value)


def _get_nested_number(value: Any, key_options: set[str]) -> float:
    safe = _json_safe(value)
    if isinstance(safe, dict):
        for key, item in safe.items():
            if key in key_options and isinstance(item, (int, float)):
                return float(item)
        total = 0.0
        for item in safe.values():
            total += _get_nested_number(item, key_options)
        return total
    if isinstance(safe, list):
        return sum(_get_nested_number(item, key_options) for item in safe)
    return 0.0


def _model_family(model: str) -> str:
    model_lower = model.lower()
    if "haiku" in model_lower:
        return "haiku"
    if "sonnet" in model_lower:
        return "sonnet"
    return ""


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    family = _model_family(model)
    if not family:
        return 0.0
    rates = MODEL_FAMILY_PRICING_USD_PER_1K[family]
    return (input_tokens / 1000.0 * rates["input"]) + (
        output_tokens / 1000.0 * rates["output"]
    )


def parse_rlm_usage_summary(usage_summary: Any, model: str) -> dict:
    safe_usage = _json_safe(usage_summary)
    input_tokens = int(
        _get_nested_number(
            safe_usage,
            {"input_tokens", "prompt_tokens", "total_input_tokens"},
        )
    )
    output_tokens = int(
        _get_nested_number(
            safe_usage,
            {"output_tokens", "completion_tokens", "total_output_tokens"},
        )
    )
    total_tokens = int(
        _get_nested_number(safe_usage, {"total_tokens", "tokens_total"})
    )
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    calls = int(
        _get_nested_number(
            safe_usage,
            {"calls", "api_calls", "request_count", "requests", "num_calls", "total_calls"},
        )
    )

    if usage_summary is None:
        status = "missing_usage_summary"
    elif input_tokens or output_tokens or total_tokens or calls:
        status = "parsed"
    else:
        status = "no_supported_usage_fields"

    return {
        "raw": safe_usage,
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(_estimate_cost_usd(model, input_tokens, output_tokens), 8),
        "usage_parse_status": status,
    }


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def _write_csv_rows(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})


def _aggregate_bridge_row_totals(rows: List[dict]) -> Dict[str, float]:
    total_calls = sum(int(row.get("delta_calls", 0) or 0) for row in rows)
    total_input_tokens = sum(int(row.get("delta_input_tokens", 0) or 0) for row in rows)
    total_output_tokens = sum(int(row.get("delta_output_tokens", 0) or 0) for row in rows)
    total_cost = sum(float(row.get("delta_cost_usd", 0.0) or 0.0) for row in rows)
    return {
        "calls": total_calls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "cost": total_cost,
    }


def _build_default_rlm_factory(args: argparse.Namespace, out_dir: Path) -> Callable[[], Any]:
    try:
        from rlm import RLM
        from rlm.logger import RLMLogger
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RLM package is not installed. Install the upstream package first, "
            "for example: `pip install rlms` or install the local checkout of "
            "https://github.com/alexzhang13/rlm."
        ) from exc

    args = normalize_rlm_args(args)
    backend_kwargs = {"model_name": args.rlm_model}
    api_key = os.getenv(args.rlm_api_key_env) if args.rlm_api_key_env else ""
    if api_key:
        backend_kwargs["api_key"] = api_key
    base_url = getattr(args, "rlm_base_url", "")
    if base_url:
        backend_kwargs["base_url"] = base_url

    logger = None
    if args.rlm_log_trajectories:
        log_dir = out_dir / "rlm_trajectories"
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = RLMLogger(log_dir=str(log_dir))

    def factory():
        return RLM(
            backend=args.rlm_backend,
            backend_kwargs=dict(backend_kwargs),
            environment=args.rlm_environment,
            max_depth=args.rlm_max_depth,
            max_iterations=args.rlm_max_iterations,
            logger=logger,
            verbose=bool(args.rlm_verbose),
        )

    return factory


def normalize_rlm_args(args: argparse.Namespace) -> argparse.Namespace:
    api_key_env = getattr(args, "rlm_api_key_env", None)
    base_url = (getattr(args, "rlm_base_url", "") or "").lower()
    if api_key_env is None:
        args.rlm_api_key_env = (
            "OPENROUTER_API_KEY" if "openrouter.ai" in base_url else "ANTHROPIC_API_KEY"
        )
    return args


def run_rlm_benchmark(
    args: argparse.Namespace,
    rlm_factory: Optional[Callable[[], Any]] = None,
) -> None:
    args = normalize_rlm_args(args)
    selected_tasks = _parse_csv_values(args.official_tasks)
    selected_lengths = _parse_csv_values(args.official_lengths)
    if not selected_tasks:
        raise ValueError("--official-tasks must include at least one task")
    if not selected_lengths:
        raise ValueError("--official-lengths must include at least one length")

    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / ARTIFACT_SUBDIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared_path = Path(args.official_prepared_data)
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
    bridge_rows_csv_path = out_dir / "bridge_rows.csv"
    manifest_path = out_dir / "manifest.json"

    print(f"\n[RLM-RULER] Run id: {run_id}")
    print(f"[RLM-RULER] Selected samples: {len(selected)}")
    print(f"[RLM-RULER] Tasks: {selected_tasks}")
    print(f"[RLM-RULER] Lengths: {selected_lengths}")
    print(f"[RLM-RULER] Backend/model: {args.rlm_backend}/{args.rlm_model}")
    print(f"[RLM-RULER] Output dir: {out_dir}")

    if rlm_factory is None:
        rlm_factory = _build_default_rlm_factory(args, out_dir)

    prediction_rows: List[dict] = []
    bridge_rows: List[dict] = []

    for idx, sample in enumerate(selected, start=1):
        print(f"[RLM-RULER] Sample {idx}/{len(selected)}: {sample['id']}")
        prompt = {"context": sample["context"], "question": sample["question"]}
        rlm_client = rlm_factory()

        t0 = time.time()
        result = rlm_client.completion(prompt=prompt, root_prompt=RLM_ROOT_PROMPT)
        latency_ms = (time.time() - t0) * 1000.0

        answer = _coerce_text(getattr(result, "response", result))
        root_model = _coerce_text(getattr(result, "root_model", "")) or args.rlm_model
        execution_time = getattr(result, "execution_time", None)
        usage = parse_rlm_usage_summary(
            getattr(result, "usage_summary", None),
            model=root_model or args.rlm_model,
        )

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
                "delta_calls": usage["calls"],
                "delta_input_tokens": usage["input_tokens"],
                "delta_output_tokens": usage["output_tokens"],
                "delta_cost_usd": usage["cost_usd"],
                "rlm_backend": args.rlm_backend,
                "rlm_model": args.rlm_model,
                "rlm_root_model": root_model,
                "rlm_execution_time": execution_time,
                "rlm_usage_summary": usage["raw"],
                "usage_parse_status": usage["usage_parse_status"],
            }
        )

    with predictions_path.open("w", encoding="utf-8") as predictions_file:
        for row in prediction_rows:
            predictions_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with bridge_rows_path.open("w", encoding="utf-8") as bridge_file:
        for row in bridge_rows:
            bridge_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_csv_rows(bridge_rows_csv_path, bridge_rows)

    placeholders = {
        "tasks_csv": ",".join(selected_tasks),
        "lengths_csv": ",".join(selected_lengths),
        "prepared_data": str(prepared_path),
        "results_dir": str(out_dir),
        "predictions": str(predictions_path),
        "bridge_rows": str(bridge_rows_path),
    }
    evaluator_command = ""
    if args.official_eval_command:
        evaluator_command = _format_command(args.official_eval_command, placeholders)
        _run_command(evaluator_command, cwd=Path.cwd())

    totals = _aggregate_bridge_row_totals(bridge_rows)
    finished_at = datetime.now(timezone.utc)
    elapsed_seconds = round((finished_at - started_at).total_seconds(), 3)
    manifest = {
        "run_id": run_id,
        "created_at": finished_at.isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "official_target": "ruler_v2_rlm",
        "baseline_type": "rlm_uncached",
        "tasks_requested": selected_tasks,
        "lengths_requested": selected_lengths,
        "samples_selected": len(bridge_rows),
        "official_prepared_data": str(prepared_path),
        "official_eval_command": evaluator_command,
        "rlm_backend": args.rlm_backend,
        "rlm_model": args.rlm_model,
        "rlm_environment": args.rlm_environment,
        "rlm_max_iterations": args.rlm_max_iterations,
        "rlm_max_depth": args.rlm_max_depth,
        "rlm_log_trajectories": bool(args.rlm_log_trajectories),
        "artifacts": {
            "predictions": str(predictions_path),
            "bridge_rows": str(bridge_rows_path),
            "bridge_rows_csv": str(bridge_rows_csv_path),
        },
        "total_api_calls": totals["calls"],
        "total_input_tokens": totals["input_tokens"],
        "total_output_tokens": totals["output_tokens"],
        "total_tokens": totals["total_tokens"],
        "total_estimated_cost_usd": round(totals["cost"], 8),
        "official_score_status": (
            "official evaluator command executed"
            if evaluator_command
            else "evaluator skipped (no --official-eval-command provided)"
        ),
        "cache_reuse": {
            "enabled": False,
            "reason": "RLM baseline is intentionally uncached",
        },
    }
    if args.rlm_log_trajectories:
        manifest["artifacts"]["rlm_trajectories"] = str(out_dir / "rlm_trajectories")
    if args.manifest_note:
        manifest["note"] = args.manifest_note

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n[RLM-RULER] Completed.")
    print(f"[RLM-RULER] Predictions : {predictions_path}")
    print(f"[RLM-RULER] Bridge rows : {bridge_rows_path}")
    print(f"[RLM-RULER] Bridge CSV  : {bridge_rows_csv_path}")
    print(f"[RLM-RULER] Manifest    : {manifest_path}")
    if not evaluator_command:
        print(
            "[RLM-RULER] Evaluator command not provided. "
            "Set --official-eval-command for strict official metrics."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Uncached RLM baseline runner for prepared RULER v2 samples"
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
        "--official-eval-command",
        type=str,
        default="",
        help=(
            "Optional shell command for official evaluator. "
            "Placeholders: {predictions}, {bridge_rows}, {prepared_data}, "
            "{results_dir}, {tasks_csv}, {lengths_csv}"
        ),
    )
    parser.add_argument("--rlm-backend", type=str, default=DEFAULT_RLM_BACKEND)
    parser.add_argument("--rlm-model", type=str, default=DEFAULT_RLM_MODEL)
    parser.add_argument("--rlm-environment", type=str, default="local")
    parser.add_argument("--rlm-max-iterations", type=int, default=30)
    parser.add_argument("--rlm-max-depth", type=int, default=1)
    parser.add_argument(
        "--rlm-base-url",
        type=str,
        default="",
        help="Optional OpenAI-compatible backend base URL, e.g. https://openrouter.ai/api/v1",
    )
    parser.add_argument(
        "--rlm-api-key-env",
        type=str,
        default=None,
        help=(
            "Environment variable used as backend_kwargs['api_key'] when present. "
            "Defaults to ANTHROPIC_API_KEY, or OPENROUTER_API_KEY when --rlm-base-url uses openrouter.ai."
        ),
    )
    parser.add_argument("--rlm-verbose", action="store_true")
    parser.add_argument("--rlm-log-trajectories", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_artifacts",
        help="Root output directory for RLM run artifacts",
    )
    parser.add_argument(
        "--manifest-note",
        type=str,
        default="",
        help="Optional free-text note stored in manifest",
    )
    return parser


def main() -> None:
    start = time.time()
    parser = build_arg_parser()
    args = normalize_rlm_args(parser.parse_args())
    run_rlm_benchmark(args)
    elapsed = time.time() - start
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
