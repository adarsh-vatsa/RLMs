"""Paired differences in exact match between the conditions of an MRCR v2 suite, with 95% intervals.

Four conversations are too few to resample, so each replicate resamples questions within each
conversation; the intervals hold for these conversations, not for conversations in general.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from execution.intervals import REPLICATES, SEED, compare
from execution.results_table import read_rows, write_table


def conditions(suite):
    runs = [path.parent for mode in ("direct", "hybrid") for path in (suite / mode).glob("*/manifest.json")]
    manifests = {run: json.loads((run / "manifest.json").read_text()) for run in runs}
    runs.sort(key=lambda run: manifests[run]["created_at"])
    found, strata = {}, {}
    for run in runs:
        manifest, execution = manifests[run], manifests[run]["execution"]
        if manifest["status"] != "complete":
            raise ValueError(f"{run.name} is {manifest['status']}")
        records = read_rows(run / "predictions.jsonl")
        strata.update((record["case_id"], record["source_id"]) for record in records)
        found[manifest["run_id"].removeprefix(f"{suite.name}_")] = {
            "window": manifest["context_window_tokens"],
            "outcomes": {record["case_id"]: bool(record["exact_match"]) for record in records},
            "middle": execution["mode"] == "direct" and execution["direct_overflow"] == "middle",
        }
    return found, strata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="benchmark_artifacts/mrcr_v2/suites/<suite-id>")
    args = parser.parse_args(argv)
    found, strata = conditions(args.suite)
    columns, rows = compare(found, {question: question for question in strata}, strata, metric="exact match")
    write_table(args.suite, "intervals",
                f"MRCR v2 suite {args.suite.name}: paired differences in exact match, 95% intervals from "
                f"{REPLICATES:,} resamples of the questions within each of the {len(set(strata.values()))} "
                f"conversations (seed {SEED})", columns, rows)


if __name__ == "__main__":
    main()
