"""Cache-mode benchmark runner for exact, semantic, and knowledge hits.

This runner uses a small labeled synthetic suite to evaluate whether the
semantic cache takes the expected reuse path for a follow-up query:

- exact
- semantic
- knowledge
- miss

Unlike RULER/NoLiMa, this suite is not an official benchmark. It exists to
measure cache-route correctness and false-positive resistance directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


VALID_CACHE_TYPES = {"exact", "semantic", "knowledge", "miss"}
VALID_ANSWER_MODES = {"exact", "contains", "lastline_exact", "lastline_contains"}


@dataclass(frozen=True)
class FixtureDocument:
    filename: str
    text: str


@dataclass(frozen=True)
class FixtureCorpus:
    corpus_id: str
    docs: List[FixtureDocument]


@dataclass(frozen=True)
class FixtureCase:
    case_id: str
    title: str
    scenario: str
    corpus_id: str
    seed_queries: List[str]
    eval_query: str
    expected_answers: List[str]
    expected_answer_mode: str
    expected_cache_type: str
    expected_from_cache: bool


@dataclass(frozen=True)
class FixtureSuite:
    suite_name: str
    corpora: Dict[str, FixtureCorpus]
    cases: List[FixtureCase]


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    return ""


def _coerce_text_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            text = _coerce_text(item)
            if text:
                out.append(text)
        return out
    text = _coerce_text(value)
    return [text] if text else []


def _parse_csv_values(raw: str) -> List[str]:
    values = [part.strip() for part in (raw or "").split(",")]
    return [part for part in values if part]


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", _coerce_text(text)).strip().lower()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sanitize_path_segment(raw: str, max_len: int = 180) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", _coerce_text(raw)).strip("._-")
    if not value:
        value = "default"
    return value[:max_len]


def _load_fixture_suite(path: Path) -> FixtureSuite:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fixture suite not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Cache benchmark fixture must be a JSON object")

    suite_name = _coerce_text(raw.get("suite_name")) or path.stem

    raw_corpora = raw.get("corpora")
    raw_cases = raw.get("cases")
    if not isinstance(raw_corpora, list) or not isinstance(raw_cases, list):
        raise ValueError("Fixture suite must include 'corpora' and 'cases' lists")

    corpora: Dict[str, FixtureCorpus] = {}
    for corpus_obj in raw_corpora:
        if not isinstance(corpus_obj, dict):
            continue
        corpus_id = _coerce_text(corpus_obj.get("corpus_id"))
        raw_docs = corpus_obj.get("docs")
        if not corpus_id or not isinstance(raw_docs, list):
            continue

        docs: List[FixtureDocument] = []
        for idx, doc_obj in enumerate(raw_docs, start=1):
            if not isinstance(doc_obj, dict):
                continue
            filename = _coerce_text(doc_obj.get("filename")) or f"doc_{idx:02d}.txt"
            text = _coerce_text(doc_obj.get("text"))
            if text:
                docs.append(FixtureDocument(filename=filename, text=text))

        if not docs:
            raise ValueError(f"Corpus '{corpus_id}' has no valid documents")
        if corpus_id in corpora:
            raise ValueError(f"Duplicate corpus_id in fixture suite: {corpus_id}")
        corpora[corpus_id] = FixtureCorpus(corpus_id=corpus_id, docs=docs)

    cases: List[FixtureCase] = []
    seen_case_ids = set()
    for case_obj in raw_cases:
        if not isinstance(case_obj, dict):
            continue
        case_id = _coerce_text(case_obj.get("case_id"))
        title = _coerce_text(case_obj.get("title")) or case_id
        scenario = _coerce_text(case_obj.get("scenario")) or "unlabeled"
        corpus_id = _coerce_text(case_obj.get("corpus_id"))
        seed_queries = _coerce_text_list(case_obj.get("seed_queries"))
        eval_query = _coerce_text(case_obj.get("eval_query"))
        expected_answers = _coerce_text_list(case_obj.get("expected_answers"))
        expected_answer_mode = _coerce_text(case_obj.get("expected_answer_mode")) or "contains"
        expected_cache_type = _coerce_text(case_obj.get("expected_cache_type")) or "miss"
        expected_from_cache = bool(case_obj.get("expected_from_cache", expected_cache_type != "miss"))

        if not case_id:
            raise ValueError("Every case must define case_id")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate case_id in fixture suite: {case_id}")
        seen_case_ids.add(case_id)

        if corpus_id not in corpora:
            raise ValueError(f"Case '{case_id}' references unknown corpus_id '{corpus_id}'")
        if not eval_query:
            raise ValueError(f"Case '{case_id}' is missing eval_query")
        if expected_cache_type not in VALID_CACHE_TYPES:
            raise ValueError(
                f"Case '{case_id}' has invalid expected_cache_type '{expected_cache_type}'"
            )
        if expected_answer_mode not in VALID_ANSWER_MODES:
            raise ValueError(
                f"Case '{case_id}' has invalid expected_answer_mode '{expected_answer_mode}'"
            )
        if not expected_answers:
            raise ValueError(f"Case '{case_id}' must define at least one expected answer")

        cases.append(
            FixtureCase(
                case_id=case_id,
                title=title,
                scenario=scenario,
                corpus_id=corpus_id,
                seed_queries=seed_queries,
                eval_query=eval_query,
                expected_answers=expected_answers,
                expected_answer_mode=expected_answer_mode,
                expected_cache_type=expected_cache_type,
                expected_from_cache=expected_from_cache,
            )
        )

    if not cases:
        raise ValueError("Fixture suite has no valid cases")

    return FixtureSuite(suite_name=suite_name, corpora=corpora, cases=cases)


def _filter_cases(
    cases: Sequence[FixtureCase],
    case_ids: List[str],
    max_cases: int,
) -> List[FixtureCase]:
    selected = list(cases)
    if case_ids:
        selected_map = {case.case_id: case for case in selected}
        missing = [case_id for case_id in case_ids if case_id not in selected_map]
        if missing:
            raise ValueError(f"Unknown case_ids requested: {', '.join(missing)}")
        selected = [selected_map[case_id] for case_id in case_ids]
    if max_cases > 0:
        selected = selected[:max_cases]
    if not selected:
        raise ValueError("No benchmark cases selected")
    return selected


def _answer_matches(prediction: str, expected_answers: Sequence[str], mode: str) -> bool:
    raw_prediction = _coerce_text(prediction)
    normalized_prediction = _normalize_for_match(raw_prediction)
    normalized_expected = [_normalize_for_match(answer) for answer in expected_answers if _coerce_text(answer)]
    if not normalized_expected:
        return False

    if mode == "exact":
        return any(normalized_prediction == expected for expected in normalized_expected)
    if mode == "contains":
        return any(expected in normalized_prediction for expected in normalized_expected)
    if mode == "lastline_exact":
        last_line = _normalize_for_match(raw_prediction.splitlines()[-1] if raw_prediction.splitlines() else raw_prediction)
        return any(last_line == expected for expected in normalized_expected)
    if mode == "lastline_contains":
        last_line = _normalize_for_match(raw_prediction.splitlines()[-1] if raw_prediction.splitlines() else raw_prediction)
        return any(expected in last_line for expected in normalized_expected)
    raise ValueError(f"Unsupported answer mode: {mode}")


def _normalize_cache_type(output: dict) -> str:
    if output.get("from_cache"):
        cache_type = _coerce_text(output.get("cache_type"))
        return cache_type if cache_type in VALID_CACHE_TYPES else "unknown"
    return "miss"


def _csv_cell(value):
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


def _summarize_rows(rows: List[dict], key: str) -> Dict[str, dict]:
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        label = _coerce_text(row.get(key)) or "unlabeled"
        grouped.setdefault(label, []).append(row)

    summary: Dict[str, dict] = {}
    for label, bucket in sorted(grouped.items()):
        total = len(bucket)
        answer_correct = sum(1 for row in bucket if row.get("answer_correct"))
        route_match = sum(1 for row in bucket if row.get("cache_type_match"))
        full_pass = sum(1 for row in bucket if row.get("overall_pass"))
        summary[label] = {
            "cases": total,
            "answer_accuracy": round(answer_correct / total, 6) if total else 0.0,
            "route_match_rate": round(route_match / total, 6) if total else 0.0,
            "overall_pass_rate": round(full_pass / total, 6) if total else 0.0,
        }
    return summary


def _aggregate_eval_totals(rows: List[dict]) -> Dict[str, float]:
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


def _resolve_cache_namespace(
    suite_name: str,
    suite_sha256: str,
    selected_cases: Sequence[FixtureCase],
    executor_model: str,
    top_k: int,
    rerank_top: int,
) -> str:
    case_sig = "-".join(case.case_id for case in selected_cases)
    digest = hashlib.sha256(
        f"{suite_sha256}\n{case_sig}\n{executor_model}\n{top_k}\n{rerank_top}".encode("utf-8")
    ).hexdigest()[:16]
    return _sanitize_path_segment(f"{suite_name}__{digest}")


def _build_eval_report(run_dir: Path, suite_name: str, suite_path: Path, bridge_rows: List[dict], manifest: dict) -> dict:
    failing_rows = [
        {
            "case_id": row["case_id"],
            "title": row["title"],
            "scenario": row["scenario"],
            "expected_cache_type": row["expected_cache_type"],
            "actual_cache_type": row["actual_cache_type"],
            "expected_from_cache": row["expected_from_cache"],
            "actual_from_cache": row["actual_from_cache"],
            "answer_correct": row["answer_correct"],
            "cache_type_match": row["cache_type_match"],
            "from_cache_match": row["from_cache_match"],
            "prediction": row["prediction"],
        }
        for row in bridge_rows
        if not row.get("overall_pass")
    ]
    return {
        "run_dir": str(run_dir),
        "suite_name": suite_name,
        "suite_path": str(suite_path),
        "scored_cases": len(bridge_rows),
        "answer_accuracy": manifest["answer_accuracy"],
        "cache_type_match_rate": manifest["cache_type_match_rate"],
        "overall_pass_rate": manifest["overall_pass_rate"],
        "expected_route_counts": manifest["expected_route_counts"],
        "actual_route_counts": manifest["actual_route_counts"],
        "by_scenario": manifest["by_scenario"],
        "by_expected_cache_type": manifest["by_expected_cache_type"],
        "by_actual_cache_type": manifest["by_actual_cache_type"],
        "failing_cases": failing_rows,
    }


def _import_semantic_cache_system():
    import semantic_cache_system as scs  # Local import keeps fixture tests lightweight.

    return scs


def _snapshot_metrics(metrics) -> Dict[str, float]:
    return metrics.get_totals()


def _write_corpus_docs(out_dir: Path, corpus: FixtureCorpus) -> Path:
    corpus_dir = out_dir / "corpora" / corpus.corpus_id
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for doc in corpus.docs:
        (corpus_dir / doc.filename).write_text(doc.text.rstrip() + "\n", encoding="utf-8")
    return corpus_dir


def run_cache_mode_benchmark(args: argparse.Namespace) -> None:
    suite_path = Path(args.suite_path)
    suite = _load_fixture_suite(suite_path)
    suite_sha256 = _sha256_file(suite_path)
    selected_cases = _filter_cases(
        suite.cases,
        case_ids=_parse_csv_values(args.case_ids),
        max_cases=args.max_cases,
    )

    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / "cache_mode_suite" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = out_dir / "predictions.jsonl"
    bridge_rows_path = out_dir / "bridge_rows.jsonl"
    bridge_rows_csv_path = out_dir / "bridge_rows.csv"
    manifest_path = out_dir / "manifest.json"
    report_path = out_dir / "official_cache_mode_eval_report.json"

    cache_state_enabled = args.mode == "cache"
    cache_namespace = ""
    cache_state_root: Path | None = None
    cache_state_path: Path | None = None
    cache_state_existed_before_reset = False
    cache_state_existed_before_run = False
    cache_load_attempts = 0
    cache_load_successes = 0
    cache_hits = 0
    cache_entries_before_run = 0
    cache_entries_after_run = 0

    if cache_state_enabled:
        cache_namespace = _resolve_cache_namespace(
            suite_name=suite.suite_name,
            suite_sha256=suite_sha256,
            selected_cases=selected_cases,
            executor_model=args.executor_model,
            top_k=args.top_k,
            rerank_top=args.rerank_top,
        )
        cache_state_root = (
            Path(args.cache_state_root)
            if args.cache_state_root
            else Path(args.output_dir) / "cache_mode_suite" / "cache_state"
        )
        cache_state_path = cache_state_root / cache_namespace
        cache_state_existed_before_reset = cache_state_path.exists()
        if args.cache_reset and cache_state_existed_before_reset:
            import shutil

            shutil.rmtree(cache_state_path)
        cache_state_root.mkdir(parents=True, exist_ok=True)
        cache_state_existed_before_run = cache_state_path.exists()

    print(f"\n[CACHE-BENCH] Run id: {run_id}")
    print(f"[CACHE-BENCH] Suite: {suite.suite_name}")
    print(f"[CACHE-BENCH] Cases: {len(selected_cases)}")
    print(f"[CACHE-BENCH] Mode: {args.mode}")
    print(f"[CACHE-BENCH] Output dir: {out_dir}")
    if cache_state_enabled and cache_state_root is not None and cache_state_path is not None:
        print(f"[CACHE-BENCH] Cache state root: {cache_state_root}")
        print(f"[CACHE-BENCH] Cache namespace : {cache_namespace}")
        if cache_state_existed_before_run:
            print("[CACHE-BENCH] Cache state: warm start (existing state found)")
        else:
            print("[CACHE-BENCH] Cache state: cold start (no prior state)")

    scs = _import_semantic_cache_system()
    scs.EXECUTOR_MODEL = args.executor_model
    shared_embedder = scs.EmbeddingEngine()
    shared_reranker = scs.Reranker()

    corpus_dirs: Dict[str, Path] = {}
    for case in selected_cases:
        if case.corpus_id not in corpus_dirs:
            corpus_dirs[case.corpus_id] = _write_corpus_docs(out_dir, suite.corpora[case.corpus_id])

    predictions_rows: List[dict] = []
    bridge_rows: List[dict] = []
    total_setup_calls = 0
    total_setup_input_tokens = 0
    total_setup_output_tokens = 0
    total_setup_cost = 0.0

    for idx, case in enumerate(selected_cases, start=1):
        controller = scs.SemanticCacheController(
            metrics=scs.ExecutionMetrics(),
            embedder=shared_embedder,
            reranker=shared_reranker,
            corpus_id=cache_namespace if cache_state_enabled else f"{suite.suite_name}_{case.case_id}",
            corpus_domain="cache_bench",
        )

        if cache_state_enabled and cache_state_path is not None:
            cache_load_attempts += 1
            if cache_state_path.exists() and controller.load(cache_state_path):
                cache_load_successes += 1
            if idx == 1:
                cache_entries_before_run = controller.get_total_entries()
        controller.ingest(corpus_dirs[case.corpus_id])

        for seed_query in case.seed_queries:
            controller.search(
                seed_query,
                top_k=args.top_k,
                rerank_top=args.rerank_top,
                synthesize=True,
                cache_read=cache_state_enabled,
            )

        setup_totals = _snapshot_metrics(controller.metrics)
        total_setup_calls += int(setup_totals["calls"])
        total_setup_input_tokens += int(setup_totals["input_tokens"])
        total_setup_output_tokens += int(setup_totals["output_tokens"])
        total_setup_cost += float(setup_totals["cost"])

        before_eval = dict(setup_totals)
        t0 = time.time()
        output = controller.search(
            case.eval_query,
            top_k=args.top_k,
            rerank_top=args.rerank_top,
            synthesize=True,
            cache_read=cache_state_enabled,
        )
        latency_ms = (time.time() - t0) * 1000.0
        after_eval = _snapshot_metrics(controller.metrics)

        prediction = _coerce_text(output.get("answer"))
        actual_cache_type = _normalize_cache_type(output)
        actual_from_cache = bool(output.get("from_cache"))
        if actual_from_cache:
            cache_hits += 1
        answer_correct = _answer_matches(
            prediction,
            case.expected_answers,
            case.expected_answer_mode,
        )
        cache_type_match = actual_cache_type == case.expected_cache_type
        from_cache_match = actual_from_cache == case.expected_from_cache
        overall_pass = answer_correct and cache_type_match and from_cache_match

        prediction_row = {
            "id": case.case_id,
            "case_id": case.case_id,
            "title": case.title,
                "scenario": case.scenario,
                "corpus_id": case.corpus_id,
                "mode": args.mode,
                "query": case.eval_query,
                "generation": prediction,
                "prediction": prediction,
            "answer": prediction,
            "expected_answers": case.expected_answers,
            "expected_answer_mode": case.expected_answer_mode,
            "expected_cache_type": case.expected_cache_type,
            "actual_cache_type": actual_cache_type,
            "expected_from_cache": case.expected_from_cache,
            "actual_from_cache": actual_from_cache,
            "answer_correct": answer_correct,
            "cache_type_match": cache_type_match,
            "from_cache_match": from_cache_match,
            "overall_pass": overall_pass,
        }
        predictions_rows.append(prediction_row)

        delta_calls = int(after_eval["calls"] - before_eval["calls"])
        delta_input_tokens = int(after_eval["input_tokens"] - before_eval["input_tokens"])
        delta_output_tokens = int(after_eval["output_tokens"] - before_eval["output_tokens"])
        delta_cost = float(after_eval["cost"] - before_eval["cost"])

        bridge_rows.append(
            {
                "case_id": case.case_id,
                "title": case.title,
                "scenario": case.scenario,
                "corpus_id": case.corpus_id,
                "mode": args.mode,
                "case_index": idx,
                "seed_query_count": len(case.seed_queries),
                "query": case.eval_query,
                "expected_answers": case.expected_answers,
                "expected_answer_mode": case.expected_answer_mode,
                "expected_cache_type": case.expected_cache_type,
                "actual_cache_type": actual_cache_type,
                "expected_from_cache": case.expected_from_cache,
                "actual_from_cache": actual_from_cache,
                "answer_correct": answer_correct,
                "cache_type_match": cache_type_match,
                "from_cache_match": from_cache_match,
                "overall_pass": overall_pass,
                "prediction": prediction,
                "latency_ms": round(latency_ms, 3),
                "setup_calls": int(setup_totals["calls"]),
                "setup_input_tokens": int(setup_totals["input_tokens"]),
                "setup_output_tokens": int(setup_totals["output_tokens"]),
                "setup_cost_usd": round(float(setup_totals["cost"]), 8),
                "delta_calls": delta_calls,
                "delta_input_tokens": delta_input_tokens,
                "delta_output_tokens": delta_output_tokens,
                "delta_cost_usd": round(delta_cost, 8),
                "knowledge_top_score": output.get("knowledge_top_score"),
                "knowledge_margin": output.get("knowledge_margin"),
                "knowledge_verifier_called": output.get("knowledge_verifier_called"),
                "knowledge_verifier_allow": output.get("knowledge_verifier_allow"),
                "knowledge_verifier_reason": output.get("knowledge_verifier_reason"),
                "knowledge_verifier_confidence": output.get("knowledge_verifier_confidence"),
                "knowledge_verifier_trigger_reasons": output.get("knowledge_verifier_trigger_reasons"),
                "knowledge_lexical_support": output.get("knowledge_lexical_support"),
            }
        )

        if cache_state_enabled and cache_state_path is not None:
            controller.save(cache_state_path)
            cache_entries_after_run = controller.get_total_entries()

    with predictions_path.open("w", encoding="utf-8") as predictions_file:
        for row in predictions_rows:
            predictions_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with bridge_rows_path.open("w", encoding="utf-8") as bridge_file:
        for row in bridge_rows:
            bridge_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_csv_rows(bridge_rows_csv_path, bridge_rows)

    eval_totals = _aggregate_eval_totals(bridge_rows)
    total_calls = total_setup_calls + int(eval_totals["calls"])
    total_input_tokens = total_setup_input_tokens + int(eval_totals["input_tokens"])
    total_output_tokens = total_setup_output_tokens + int(eval_totals["output_tokens"])
    total_cost = total_setup_cost + float(eval_totals["cost"])
    actual_route_counts = {route: sum(1 for row in bridge_rows if row["actual_cache_type"] == route) for route in VALID_CACHE_TYPES}
    expected_route_counts = {route: sum(1 for row in bridge_rows if row["expected_cache_type"] == route) for route in VALID_CACHE_TYPES}
    answer_correct_count = sum(1 for row in bridge_rows if row["answer_correct"])
    route_match_count = sum(1 for row in bridge_rows if row["cache_type_match"])
    overall_pass_count = sum(1 for row in bridge_rows if row["overall_pass"])

    finished_at = datetime.now(timezone.utc)
    elapsed_seconds = round((finished_at - started_at).total_seconds(), 3)
    manifest = {
        "run_id": run_id,
        "created_at": finished_at.isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "benchmark_target": "cache_mode_suite",
        "suite_name": suite.suite_name,
        "suite_path": str(suite_path),
        "suite_sha256": suite_sha256,
        "cases_selected": len(selected_cases),
        "mode": args.mode,
        "executor_model": args.executor_model,
        "top_k": args.top_k,
        "rerank_top": args.rerank_top,
        "artifacts": {
            "predictions": str(predictions_path),
            "bridge_rows": str(bridge_rows_path),
            "bridge_rows_csv": str(bridge_rows_csv_path),
            "eval_report": str(report_path),
        },
        "total_api_calls": total_calls,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_estimated_cost_usd": round(total_cost, 8),
        "setup_api_calls": total_setup_calls,
        "setup_input_tokens": total_setup_input_tokens,
        "setup_output_tokens": total_setup_output_tokens,
        "setup_total_tokens": total_setup_input_tokens + total_setup_output_tokens,
        "setup_estimated_cost_usd": round(total_setup_cost, 8),
        "evaluation_api_calls": int(eval_totals["calls"]),
        "evaluation_input_tokens": int(eval_totals["input_tokens"]),
        "evaluation_output_tokens": int(eval_totals["output_tokens"]),
        "evaluation_total_tokens": int(eval_totals["total_tokens"]),
        "evaluation_estimated_cost_usd": round(float(eval_totals["cost"]), 8),
        "answer_accuracy": round(answer_correct_count / len(bridge_rows), 6) if bridge_rows else 0.0,
        "cache_type_match_rate": round(route_match_count / len(bridge_rows), 6) if bridge_rows else 0.0,
        "overall_pass_rate": round(overall_pass_count / len(bridge_rows), 6) if bridge_rows else 0.0,
        "expected_route_counts": expected_route_counts,
        "actual_route_counts": actual_route_counts,
        "by_scenario": _summarize_rows(bridge_rows, "scenario"),
        "by_expected_cache_type": _summarize_rows(bridge_rows, "expected_cache_type"),
        "by_actual_cache_type": _summarize_rows(bridge_rows, "actual_cache_type"),
    }
    if cache_state_enabled:
        manifest["cache_reuse"] = {
            "enabled": True,
            "cache_namespace": cache_namespace,
            "cache_state_root": str(cache_state_root) if cache_state_root else "",
            "cache_state_path": str(cache_state_path) if cache_state_path else "",
            "cache_state_existed_before_reset": cache_state_existed_before_reset,
            "cache_state_existed_before_run": cache_state_existed_before_run,
            "cache_reset_requested": bool(args.cache_reset),
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

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = _build_eval_report(out_dir, suite.suite_name, suite_path, bridge_rows, manifest)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[CACHE-BENCH] Completed.")
    print(f"[CACHE-BENCH] Predictions : {predictions_path}")
    print(f"[CACHE-BENCH] Bridge rows : {bridge_rows_path}")
    print(f"[CACHE-BENCH] Bridge CSV  : {bridge_rows_csv_path}")
    print(f"[CACHE-BENCH] Manifest    : {manifest_path}")
    print(f"[CACHE-BENCH] Eval report : {report_path}")
    print(
        "[CACHE-BENCH] Summary: "
        f"answer_accuracy={manifest['answer_accuracy']:.3f}, "
        f"cache_type_match_rate={manifest['cache_type_match_rate']:.3f}, "
        f"overall_pass_rate={manifest['overall_pass_rate']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache-mode benchmark runner for exact, semantic, and knowledge reuse"
    )
    parser.add_argument(
        "--suite-path",
        type=str,
        default="benchmark_fixtures/cache_bench/cache_mode_suite_v1.json",
        help="Path to the labeled cache-mode benchmark fixture JSON file",
    )
    parser.add_argument(
        "--case-ids",
        type=str,
        default="",
        help="Optional comma-separated case_ids to run in fixture order",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional cap on selected cases (0 means all)",
    )
    parser.add_argument(
        "--executor-model",
        type=str,
        default="claude-sonnet-4-5",
        help="Model name assigned to semantic_cache_system.EXECUTOR_MODEL",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "cache"],
        default="cache",
        help="Run without cache (`baseline`) or with persistent cache reuse (`cache`)",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--rerank-top", type=int, default=5)
    parser.add_argument(
        "--cache-state-root",
        type=str,
        default="",
        help=(
            "Root directory for persistent cache state in cache mode. "
            "If omitted, defaults to <output-dir>/cache_mode_suite/cache_state"
        ),
    )
    parser.add_argument(
        "--cache-reset",
        action="store_true",
        help="Reset the resolved cache namespace before the run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_artifacts",
        help="Root output directory for benchmark artifacts",
    )
    parser.add_argument(
        "--manifest-note",
        type=str,
        default="",
        help="Optional free-text note stored in manifest",
    )
    args = parser.parse_args()
    run_cache_mode_benchmark(args)


if __name__ == "__main__":
    main()
