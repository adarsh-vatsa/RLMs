"""Paired differences between the conditions of an AA-LCR suite, with 95% intervals over document sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aa_lcr.results_table import verdicts
from execution.intervals import PERMUTATION_SEED, PERMUTATIONS, REPLICATES, SEED, compare
from execution.results_table import read_rows, write_table


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
        for answer in read_rows(run / "bridge_rows.jsonl"):
            clusters[answer["question_id"]] = answer["document_set_id"]
        found[manifest["run_id"].removeprefix(f"{suite.name}_")] = {
            "window": manifest["context_window_tokens"], "outcomes": outcomes,
            "middle": execution["mode"] == "direct" and execution["direct_overflow"] == "middle",
        }
    return found, clusters


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


if __name__ == "__main__":
    main()
