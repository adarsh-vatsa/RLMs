"""Score existing run_benchmark predictions with NeMo RULER2 evaluator logic.

This script evaluates an existing predictions.jsonl produced by run_benchmark.py
without re-running model generation.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from nemo_skills.evaluation.evaluator.ruler import eval_ruler2


MATCH_TYPE_BY_TASK = {
    "mk_niah_basic": "all",
    "mv_niah_basic": "all",
    "qa_basic": "part",
}


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def _score_task_rows(rows: list[dict], match_type: str) -> list[dict]:
    # NeMo evaluator mutates rows in-place inside an input_file,
    # so we score task slices via a temporary jsonl file.
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False, encoding="utf-8") as tf:
        tmp_path = tf.name
        for row in rows:
            tf.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        eval_ruler2({"input_file": tmp_path, "match_type": match_type, "parse_func": "default"})
        return _load_jsonl(Path(tmp_path))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def build_report(scored_rows: list[dict], run_dir: Path) -> dict:
    by_task_length: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in scored_rows:
        task = str(row.get("task", ""))
        length = str(row.get("length", ""))
        by_task_length[(task, length)].append(float(row.get("is_correct", 0.0)))

    per_bucket: dict[str, dict] = {}
    all_scores: list[float] = []
    for (task, length), vals in sorted(by_task_length.items()):
        accuracy = (sum(vals) / len(vals)) if vals else 0.0
        per_bucket[f"{task}|{length}"] = {
            "samples": len(vals),
            "accuracy": accuracy,
        }
        all_scores.extend(vals)

    overall = (sum(all_scores) / len(all_scores)) if all_scores else 0.0
    return {
        "run_dir": str(run_dir),
        "scored_samples": len(all_scores),
        "overall_accuracy": overall,
        "by_task_length": per_bucket,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score run_benchmark predictions with NeMo RULER2 logic")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to benchmark_artifacts/official_ruler_v2/<run_id>",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions.jsonl not found: {predictions_path}")

    rows = _load_jsonl(predictions_path)
    scored_rows: list[dict] = []
    for task, match_type in MATCH_TYPE_BY_TASK.items():
        task_rows = [row for row in rows if row.get("task") == task]
        if not task_rows:
            continue
        scored_rows.extend(_score_task_rows(task_rows, match_type))

    report = build_report(scored_rows, run_dir)
    report_path = run_dir / "official_ruler2_eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
