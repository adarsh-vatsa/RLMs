"""Score existing NoLiMa benchmark predictions.

This evaluates predictions.jsonl produced by nolima/run_benchmark.py without
re-running generation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    return ""


def _coerce_expected_answers(value: object) -> List[str]:
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


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def _metric_em(prediction: str, expected: List[str]) -> float:
    if not expected:
        return 0.0
    return 1.0 if prediction in expected else 0.0


def _metric_contains(prediction: str, expected: List[str]) -> float:
    if not expected:
        return 0.0
    return 1.0 if any(gold in prediction for gold in expected) else 0.0


def _metric_lastline_em(prediction: str, expected: List[str]) -> float:
    if not expected:
        return 0.0
    last_line = prediction.strip().split("\n")[-1].strip()
    return 1.0 if last_line in expected else 0.0


def _metric_lastline_contains(prediction: str, expected: List[str]) -> float:
    if not expected:
        return 0.0
    last_line = prediction.strip().split("\n")[-1].strip()
    return 1.0 if any(gold in last_line for gold in expected) else 0.0


def _metric_fns() -> Dict[str, Callable[[str, List[str]], float]]:
    return {
        "EM": _metric_em,
        "contains": _metric_contains,
        "lastline_EM": _metric_lastline_em,
        "lastline_contains": _metric_lastline_contains,
    }


def build_report(scored_rows: List[dict], run_dir: Path, metric: str) -> dict:
    by_length: Dict[str, List[float]] = defaultdict(list)
    by_task: Dict[str, List[float]] = defaultdict(list)
    by_bucket: Dict[str, List[float]] = defaultdict(list)

    for row in scored_rows:
        length = _coerce_text(row.get("length"))
        task = _coerce_text(row.get("task"))
        score = float(row.get("is_correct", 0.0))
        by_length[length].append(score)
        by_task[task].append(score)
        by_bucket[f"{task}|{length}"].append(score)

    def _summary(values: List[float]) -> dict:
        if not values:
            return {"samples": 0, "accuracy": 0.0}
        return {
            "samples": len(values),
            "accuracy": sum(values) / len(values),
        }

    by_length_summary = {k: _summary(v) for k, v in sorted(by_length.items())}
    by_task_summary = {k: _summary(v) for k, v in sorted(by_task.items())}
    by_bucket_summary = {k: _summary(v) for k, v in sorted(by_bucket.items())}

    all_scores: List[float] = []
    for values in by_bucket.values():
        all_scores.extend(values)

    overall = (sum(all_scores) / len(all_scores)) if all_scores else 0.0

    return {
        "run_dir": str(run_dir),
        "metric": metric,
        "scored_samples": len(all_scores),
        "overall_accuracy": overall,
        "by_length": by_length_summary,
        "by_task": by_task_summary,
        "by_task_length": by_bucket_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score NoLiMa predictions")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to benchmark_artifacts/official_nolima/<run_id>",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["EM", "contains", "lastline_EM", "lastline_contains"],
        default="contains",
        help="NoLiMa-compatible scoring metric",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions.jsonl not found: {predictions_path}")

    metric_fn = _metric_fns()[args.metric]
    rows = _load_jsonl(predictions_path)

    scored_rows: List[dict] = []
    for row in rows:
        prediction = _coerce_text(
            row.get("generation")
            or row.get("prediction")
            or row.get("answer")
        )
        expected = _coerce_expected_answers(row.get("expected_answer"))
        is_correct = metric_fn(prediction, expected)

        scored = dict(row)
        scored["expected_answer_list"] = expected
        scored["is_correct"] = is_correct
        scored_rows.append(scored)

    scored_path = run_dir / "scored_predictions.jsonl"
    with scored_path.open("w", encoding="utf-8") as f:
        for row in scored_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = build_report(scored_rows, run_dir=run_dir, metric=args.metric)
    report_path = run_dir / "official_nolima_eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Wrote {report_path}")
    print(f"Wrote {scored_path}")


if __name__ == "__main__":
    main()
