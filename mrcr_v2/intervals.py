"""Paired differences in exact match between the conditions of an MRCR v2 suite, with 95% intervals.

Four conversations are too few to resample, so each replicate resamples questions within each
conversation; the intervals hold for these conversations, not for conversations in general. The
oracle at each window bounds what a run can recover. Full context exists only for the conversation
that middle truncation at the largest window sends whole, so a second table covers its questions alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from execution.intervals import PERMUTATION_SEED, PERMUTATIONS, REPLICATES, SEED, compare
from execution.results_table import read_rows, write_table
from mrcr_v2.results_table import size_label


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
            "window": manifest["context_window_tokens"], "oracle": bool(manifest.get("oracle")),
            "outcomes": {record["case_id"]: bool(record["exact_match"]) for record in records},
            "middle": execution["mode"] == "direct" and execution["direct_overflow"] == "middle",
            "sent whole": {record["case_id"]: record["full_rendered_input_tokens"]
                           for record in records if record["route"] == "direct_fit"},
        }
    return found, strata


def table(suite, stem, title, found, strata, references, reference):
    columns, rows = compare(found, {question: question for question in strata}, strata, metric="exact match",
                            references=references)
    write_table(suite, stem, f"{title}: paired differences in exact match, 95% intervals from {REPLICATES:,} "
                f"resamples of the questions within each conversation (seed {SEED}). p: uncorrected two-sided "
                f"randomization test, {PERMUTATIONS:,} swaps of the two runs' outcomes by question "
                f"(seed {PERMUTATION_SEED}). Gap recovered: "
                f"(run − middle) / ({reference} − middle) at the run's window", columns, rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="benchmark_artifacts/mrcr_v2/suites/<suite-id>")
    args = parser.parse_args(argv)
    found, strata = conditions(args.suite)
    oracles = {condition["window"]: name for name, condition in found.items() if condition["oracle"]}
    table(args.suite, "intervals", f"MRCR v2 suite {args.suite.name}, {len(set(strata.values()))} conversations",
          found, strata, oracles, "oracle")
    middles = [name for name, condition in found.items() if condition["middle"]]
    full = max(middles, key=lambda name: found[name]["window"])
    whole = found[full]["sent whole"]
    sizes = ", ".join(sorted({size_label(tokens) for tokens in whole.values()}))
    table(args.suite, "intervals-full-context",
          f"MRCR v2 suite {args.suite.name}, the {len(whole)} questions on the {sizes} conversation that `{full}` "
          f"sends whole", found, {question: strata[question] for question in whole},
          {found[name]["window"]: full for name in middles if name != full}, "full context")


if __name__ == "__main__":
    main()
