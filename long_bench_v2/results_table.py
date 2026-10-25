"""Summarize every run of a LongBench-v2 suite in one table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from execution.results_table import gained_lost, mean, policy, read_rows, write_table


CHOICES = {"A", "B", "C", "D"}


def summarize(suite):
    # The direct and hybrid runners name their folders differently; both write the shared
    # pipeline's execution_manifest.json, execution.jsonl and evaluation.jsonl.
    runs = [path.parent for folder in ("longbench_v2_api", "longbench_v2")
            for path in (suite / folder).glob("*/manifest.json")]
    manifests = {run: json.loads((run / "manifest.json").read_text()) for run in runs}
    configs = {run: json.loads((run / "execution_manifest.json").read_text()) for run in runs}
    runs.sort(key=lambda run: manifests[run]["started_at"])
    correct = {run: {row["case_id"]: bool(row["scores"].get("accuracy"))
                     for row in read_rows(run / "evaluation.jsonl")} for run in runs}
    middle = {configs[run]["context_window_tokens"]: correct[run] for run in runs
              if configs[run]["mode"] == "direct" and configs[run]["direct_overflow"] == "middle"}
    rows = []
    for run in runs:
        config, records, current = configs[run], read_rows(run / "execution.jsonl"), correct[run]
        hybrid = config["mode"] == "hybrid"
        reference = middle.get(config["context_window_tokens"])
        embedding_ms = sum(record["document_embedding_ms"] for record in records)
        rows.append({
            "condition": manifests[run]["run_id"].removeprefix(f"{suite.name}_"),
            "policy": policy(config),
            "window": config["context_window_tokens"],
            "questions": len(records),
            "prompt kept": mean(r["final_rendered_input_tokens"] / r["full_rendered_input_tokens"] for r in records),
            "input tokens": mean(r["final_rendered_input_tokens"] for r in records),
            "accuracy": mean(current.values()),
            "correct": f"{sum(current.values())}/{len(current)}",
            **(gained_lost(current, reference) if reference is not None and reference is not current else {}),
            "invalid answers": sum(r["prediction"].strip() not in CHOICES for r in records),
            "retried requests": sum((r.get("attempts") or 1) > 1 for r in records),
            "generation s/question": mean(r["generation_ms"] for r in records) / 1000,
            "retrieval + packing s/question": mean((r["retrieval_ms"] + r["packing_ms"]) / 1000 for r in records)
                                              if hybrid else "",
            "embedding minutes": round(embedding_ms / 60000) if embedding_ms else "",
            "run minutes": round(manifests[run]["elapsed_seconds"] / 60),
        })
    columns = ["condition", "policy", "window", "questions", "prompt kept", "input tokens", "accuracy", "correct",
               "gained vs middle", "lost vs middle", "p vs middle", "invalid answers", "retried requests",
               "generation s/question", "retrieval + packing s/question", "embedding minutes", "run minutes"]
    return columns, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="benchmark_artifacts/longbench_v2/suites/<suite-id>")
    args = parser.parse_args(argv)
    columns, rows = summarize(args.suite)
    write_table(args.suite, "results", f"LongBench-v2 suite {args.suite.name}", columns, rows)


if __name__ == "__main__":
    main()
