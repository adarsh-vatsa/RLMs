"""Paired differences between the conditions of an AA-LCR suite, with 95% intervals over document sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aa_lcr.results_table import verdicts
from execution.intervals import PERMUTATION_SEED, PERMUTATIONS, REPLICATES, SEED, compare
from execution.results_table import is_middle_reference, read_rows, window, write_table


def conditions(suite, grader):
    runs = [path.parent for mode in ("direct", "hybrid") for path in (suite / mode).glob("*/manifest.json")]
    manifests = {run: json.loads((run / "manifest.json").read_text()) for run in runs}
    runs.sort(key=lambda run: manifests[run]["started_at"])
    found, clusters = {}, {}
    for run in runs:
        manifest, execution = manifests[run], manifests[run]["execution"]
        outcomes = verdicts(suite, grader, run)
        if outcomes is None:
            raise ValueError(f"{run.name} has no {grader} grades")
        questions = {}
        for answer in read_rows(run / "bridge_rows.jsonl"):
            clusters[answer["question_id"]] = answer["document_set_id"]
            questions[answer["case_id"]] = answer["question_id"]
        found[manifest["run_id"].removeprefix(f"{suite.name}_")] = {
            "window": window(execution, manifest["context_window_tokens"]), "outcomes": outcomes,
            "middle": is_middle_reference(execution),
            "capped": {questions[record["case_id"]] for record in read_rows(run / "execution.jsonl")
                       if record["finish_reason"] == "length"},
        }
    return found, clusters


def uncapped(found, clusters):
    """Each run against middle truncation at its window, on the questions where neither answer reached the cap."""
    rows = []
    for name, condition in found.items():
        middle = next((other for other, candidate in found.items()
                       if candidate["middle"] and candidate["window"] == condition["window"]), None)
        if condition["middle"] or middle is None:
            continue
        kept = {question: cluster for question, cluster in clusters.items()
                if question not in condition["capped"] | found[middle]["capped"]}
        _, pair = compare({name: condition, middle: found[middle]}, kept, metric="accuracy")
        rows += [{**row, "questions": len(kept)} for row in pair if row["baseline"] == middle]
    columns = ["condition", "window", "baseline", "baseline window", "questions", "accuracy", "baseline accuracy",
               "difference", "95% low", "95% high", "gained", "lost", "p"]
    return columns, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="benchmark_artifacts/aa_lcr/suites/<suite-id>")
    parser.add_argument("--grader", default="gpt-5.6-luna", help="Folder name under the suite's regrades/")
    args = parser.parse_args(argv)
    found, clusters = conditions(args.suite, args.grader)
    references = dict.fromkeys({condition["window"] for condition in found.values()}, "full")
    columns, rows = compare(found, clusters, metric="accuracy", references=references)
    write_table(args.suite, f"intervals-{args.grader}",
                f"AA-LCR suite {args.suite.name}, graded by {args.grader}: paired differences, 95% intervals from "
                f"{REPLICATES:,} resamples of the {len(set(clusters.values()))} document sets (seed {SEED}). "
                f"p: uncorrected two-sided randomization test, {PERMUTATIONS:,} swaps of the two runs' outcomes "
                f"by document set (seed {PERMUTATION_SEED}). "
                f"Gap recovered: (run − middle) / (full − middle) at the run's window",
                columns, rows)
    write_table(args.suite, f"intervals-{args.grader}-uncapped",
                f"AA-LCR suite {args.suite.name}, graded by {args.grader}: each run against middle truncation at "
                f"its window, leaving out the questions on which either answer reached the output cap; intervals "
                f"and p as in intervals-{args.grader}", *uncapped(found, clusters))


if __name__ == "__main__":
    main()
