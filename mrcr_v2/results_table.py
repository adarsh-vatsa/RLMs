"""Summarize every run of an MRCR v2 suite in one table."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from execution.results_table import gained_lost, mean, policy, read_rows, write_table


SUITE_CHUNK_TOKENS = 3500


def size_label(tokens):
    return f"{tokens / 1e6:.2f}M" if tokens >= 1e6 else f"{tokens / 1e3:.0f}K"


def summarize(suite):
    runs = [path.parent for mode in ("direct", "hybrid") for path in (suite / mode).glob("*/manifest.json")]
    manifests = {run: json.loads((run / "manifest.json").read_text()) for run in runs}
    runs.sort(key=lambda run: manifests[run]["created_at"])
    records = {run: read_rows(run / "predictions.jsonl") if (run / "predictions.jsonl").exists() else []
               for run in runs}
    complete = [run for run in runs if manifests[run]["status"] == "complete"]
    sizes = {}
    for run in complete:
        for record in records[run]:
            sizes[record["source_id"]] = min(sizes.get(record["source_id"], record["full_rendered_input_tokens"]),
                                             record["full_rendered_input_tokens"])
    sources = sorted(sizes, key=sizes.get)
    matches = {run: {record["case_id"]: bool(record["exact_match"]) for record in records[run]} for run in complete}
    middle = {manifests[run]["context_window_tokens"]: matches[run] for run in complete
              if manifests[run]["execution"]["mode"] == "direct"
              and manifests[run]["execution"]["direct_overflow"] == "middle"}
    rows = []
    for run in runs:
        manifest, execution, answers = manifests[run], manifests[run]["execution"], records[run]
        label = policy(execution, SUITE_CHUNK_TOKENS)
        if execution["mode"] == "hybrid" and manifest.get("retrieval_query") == "question":
            label += ", full-question query"
        row = {
            "condition": manifest["run_id"].removeprefix(f"{suite.name}_"),
            "policy": label,
            "window": manifest["context_window_tokens"],
            "questions": f"{len(answers)}/{len(manifest['selected_example_ids'])}",
        }
        if run not in complete:
            rows.append({**row, "exact match": manifest["status"]})
            continue
        hybrid = execution["mode"] == "hybrid"
        current = matches[run]
        reference = middle.get(manifest["context_window_tokens"])
        row.update({
            "prompt kept": mean(a["final_rendered_input_tokens"] / a["full_rendered_input_tokens"] for a in answers),
            "input tokens": mean(a["final_rendered_input_tokens"] for a in answers),
            "exact match": mean(current.values()),
            "MRCR score": mean(a["mrcr_score"] or 0 for a in answers),
            "correct": f"{sum(current.values())}/{len(current)}",
            **(gained_lost(current, reference) if reference is not None and reference is not current else {}),
            **{f"exact match at {size_label(sizes[source])}":
               mean(bool(a["exact_match"]) for a in answers if a["source_id"] == source) for source in sources},
            "prefix compliance": mean(bool(a["prefix_compliant"]) for a in answers),
            "output tokens": mean(a["output_tokens"] for a in answers),
            "output limit hit": sum(a.get("finish_reason") == "length" for a in answers),
            "retried requests": sum((a.get("attempts") or 1) > 1 for a in answers),
            "generation s/question": mean(a["generation_ms"] for a in answers) / 1000,
            "retrieval + packing s/question": mean((a["retrieval_ms"] + a["packing_ms"]) / 1000 for a in answers)
                                              if hybrid else "",
            "embedding minutes": round(sum(a["document_embedding_ms"] for a in answers) / 60000) if hybrid else "",
            "run minutes": round((datetime.fromisoformat(manifest["completed_at"])
                                  - datetime.fromisoformat(manifest["created_at"])).total_seconds() / 60),
        })
        rows.append(row)
    columns = ["condition", "policy", "window", "questions", "prompt kept", "input tokens", "exact match",
               "MRCR score", "correct", "gained vs middle", "lost vs middle", "p vs middle",
               *[f"exact match at {size_label(sizes[source])}" for source in sources], "prefix compliance",
               "output tokens", "output limit hit", "retried requests", "generation s/question",
               "retrieval + packing s/question", "embedding minutes", "run minutes"]
    return columns, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="benchmark_artifacts/mrcr_v2/suites/<suite-id>")
    args = parser.parse_args(argv)
    columns, rows = summarize(args.suite)
    write_table(args.suite, "results", f"MRCR v2 suite {args.suite.name}", columns, rows)


if __name__ == "__main__":
    main()
