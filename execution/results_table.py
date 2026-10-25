"""Helpers shared by the per-benchmark suite results tables."""

from __future__ import annotations

import csv
import json
from math import comb


# Display formats for the markdown table; the CSV keeps the raw numbers.
FORMATS = {"window": "{:,}", "prompt kept": "{:.0%}", "input tokens": "{:,.0f}", "documents covered": "{:.0%}",
           "accuracy": "{:.1%}", "exact match": "{:.1%}", "MRCR score": "{:.3f}", "prefix compliance": "{:.0%}",
           "p vs middle": "{:.3f}", "output tokens": "{:,.0f}", "generation s/question": "{:.1f}",
           "retrieval + packing s/question": "{:.1f}"}
PREFIX_FORMATS = {"agreement with ": "{:.0%}", "exact match at ": "{:.0%}"}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0


def policy(execution, chunk_tokens=7500):
    """Label a run's overflow policy; `chunk_tokens` is the suite's usual chunk size, left unlabelled."""
    if execution["mode"] == "direct":
        overflow = execution["direct_overflow"]
        return "full prompt" if overflow == "unsupported" else f"truncation: {overflow}"
    parts = [execution.get("evidence_selection", "dense")]
    if parts[0] == "random":
        parts[0] += f" seed {execution['selection_seed']}"
    if execution["child_tokens"] != chunk_tokens:
        parts.append(f"{execution['child_tokens']}-token chunks")
    if execution.get("neighbor_children"):
        count = execution["neighbor_children"]
        parts.append(f"{count} neighbour{'s' if count > 1 else ''} each side")
    if execution.get("packing", "stop") != "stop":
        parts.append(f"{execution['packing']} packing")
    if execution.get("document_floor"):
        parts.append("document floor")
    return ", ".join(parts)


def gained_lost(current, reference):
    """Paired counts against a reference run, over the questions both answered, with an exact McNemar test.

    The test treats questions as independent; questions share documents or conversations, so it can
    overstate significance. Paper intervals should resample whole document sets or conversations.
    """
    shared = current.keys() & reference.keys()
    gained = sum(current[q] and not reference[q] for q in shared)
    lost = sum(reference[q] and not current[q] for q in shared)
    tail = sum(comb(gained + lost, k) for k in range(min(gained, lost) + 1)) / 2 ** (gained + lost)
    return {"gained vs middle": gained, "lost vs middle": lost, "p vs middle": min(1.0, 2 * tail)}


def markdown(columns, rows):
    def cell(column, value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            prefixed = next((form for prefix, form in PREFIX_FORMATS.items() if column.startswith(prefix)), "{}")
            return FORMATS.get(column, prefixed).format(value)
        return str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(column, row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def write_table(directory, stem, title, columns, rows):
    table = markdown(columns, rows)
    (directory / f"{stem}.md").write_text(f"# {title}\n\n{table}")
    with (directory / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: round(value, 4) if isinstance(value, float) else value for key, value in row.items()}
                         for row in rows)
    print(table, end="")
    print(f"Saved {directory / stem}.md and .csv")
