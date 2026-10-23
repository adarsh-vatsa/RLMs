"""Summarize every run of an AA-LCR suite in one table, from one grader's grades."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


# Display formats for the markdown table; the CSV keeps the raw numbers.
FORMATS = {"window": "{:,}", "prompt kept": "{:.0%}", "input tokens": "{:,.0f}", "documents covered": "{:.0%}",
           "accuracy": "{:.1%}", "output tokens": "{:,.0f}", "generation s/question": "{:.1f}",
           "retrieval + packing s/question": "{:.1f}"}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0


def policy(execution):
    if execution["mode"] == "direct":
        overflow = execution["direct_overflow"]
        return "full prompt" if overflow == "unsupported" else f"truncation: {overflow}"
    parts = [execution.get("evidence_selection", "dense")]
    if parts[0] == "random":
        parts[0] += f" seed {execution['selection_seed']}"
    if execution["child_tokens"] != 7500:
        parts.append(f"{execution['child_tokens']}-token chunks")
    if execution.get("neighbor_children"):
        count = execution["neighbor_children"]
        parts.append(f"{count} neighbour{'s' if count > 1 else ''} each side")
    if execution.get("packing", "stop") != "stop":
        parts.append(f"{execution['packing']} packing")
    if execution.get("document_floor"):
        parts.append("document floor")
    return ", ".join(parts)


def verdicts(suite, grader, run):
    """Question ID -> correct, for validly graded answers; None when the run is not graded."""
    grades = suite / "regrades" / grader / run.parent.name / run.name
    if not (grades / "manifest.json").exists():
        return None
    return {row["question_id"]: row["answer_correct"] for row in read_rows(grades / "bridge_rows.jsonl")
            if row["grade_valid"]}


def summarize(suite, grader):
    runs = [path.parent for mode in ("direct", "hybrid") for path in (suite / mode).glob("*/manifest.json")]
    manifests = {run: json.loads((run / "manifest.json").read_text()) for run in runs}
    runs.sort(key=lambda run: manifests[run]["started_at"])
    others = sorted(path.name for path in (suite / "regrades").glob("*") if path.is_dir() and path.name != grader)
    graded = {run: verdicts(suite, grader, run) for run in runs}
    middle = {manifests[run]["context_window_tokens"]: graded[run] for run in runs
              if manifests[run]["execution"]["mode"] == "direct"
              and manifests[run]["execution"]["direct_overflow"] == "middle" and graded[run] is not None}
    rows = []
    for run in runs:
        manifest, execution = manifests[run], manifests[run]["execution"]
        answers = read_rows(run / "bridge_rows.jsonl")
        records = read_rows(run / "execution.jsonl")
        hybrid = execution["mode"] == "hybrid"
        coverage = [len({item["document_id"] for item in answer["selected_evidence_ranges"]})
                    / len(answer["data_source_filenames"]) for answer in answers] if hybrid else []
        embedding_ms = sum(answer.get("document_embedding_ms") or 0 for answer in answers)
        packing_s = mean((record["retrieval_ms"] + record["packing_ms"]) / 1000 for record in records)
        row = {
            "condition": manifest["run_id"].removeprefix(f"{suite.name}_"),
            "policy": policy(execution),
            "window": manifest["context_window_tokens"],
            "prompt kept": mean(a["final_rendered_input_tokens"] / a["full_rendered_input_tokens"] for a in answers),
            "input tokens": mean(a["final_rendered_input_tokens"] for a in answers),
            "documents covered": mean(coverage) if hybrid else "",
            "questions missing a document": sum(share < 1 for share in coverage) if hybrid else "",
            "accuracy": "not graded",
        }
        current = graded[run]
        if current is not None:
            correct = sum(current.values())
            invalid = len(answers) - len(current)
            reference = middle.get(manifest["context_window_tokens"])
            row["accuracy"] = correct / len(current) if current else "no valid grades"
            row["correct"] = f"{correct}/{len(current)}" + (f" ({invalid} invalid)" if invalid else "")
            if reference is not None and reference is not current:
                shared = current.keys() & reference.keys()
                row["gained vs middle"] = sum(current[q] and not reference[q] for q in shared)
                row["lost vs middle"] = sum(reference[q] and not current[q] for q in shared)
            for other in others:
                theirs = verdicts(suite, other, run)
                shared = current.keys() & (theirs or {}).keys()
                if shared:
                    row[f"agreement with {other}"] = mean(current[q] == theirs[q] for q in shared)
        row.update({
            "output tokens": mean(a["executor_output_tokens"] for a in answers),
            "output limit hit": sum(a.get("executor_finish_reason") == "length" for a in answers),
            "retried requests": sum((a.get("executor_attempts") or 1) > 1 for a in answers),
            "generation s/question": mean(r["generation_ms"] for r in records) / 1000,
            "retrieval + packing s/question": packing_s if hybrid else "",
            "embedding minutes": round(embedding_ms / 60000) if embedding_ms else "",
            "run minutes": round(manifest["elapsed_seconds"] / 60),
        })
        rows.append(row)
    columns = ["condition", "policy", "window", "prompt kept", "input tokens", "documents covered",
               "questions missing a document", "accuracy", "correct", "gained vs middle", "lost vs middle",
               *[f"agreement with {other}" for other in others], "output tokens", "output limit hit",
               "retried requests", "generation s/question", "retrieval + packing s/question",
               "embedding minutes", "run minutes"]
    return columns, rows


def markdown(columns, rows):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    def cell(column, value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return FORMATS.get(column, "{:.0%}" if column.startswith("agreement") else "{}").format(value)
        return str(value)

    lines += ["| " + " | ".join(cell(column, row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="benchmark_artifacts/aa_lcr/suites/<suite-id>")
    parser.add_argument("--grader", default="Qwen3.6-35B-A3B", help="Folder name under the suite's regrades/")
    args = parser.parse_args(argv)
    columns, rows = summarize(args.suite, args.grader)
    table = markdown(columns, rows)
    (args.suite / f"results-{args.grader}.md").write_text(
        f"# AA-LCR suite {args.suite.name}, graded by {args.grader}\n\n{table}")
    with (args.suite / f"results-{args.grader}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: round(value, 4) if isinstance(value, float) else value for key, value in row.items()}
                         for row in rows)
    print(table, end="")
    print(f"Saved {args.suite / f'results-{args.grader}.md'} and .csv")


if __name__ == "__main__":
    main()
