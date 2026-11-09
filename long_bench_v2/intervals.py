"""Paired differences between the conditions of a LongBench-v2 suite, with 95% intervals over documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from execution.intervals import REPLICATES, SEED, compare
from execution.results_table import read_rows, write_table


def conditions(suite):
    runs = [path.parent for folder in ("longbench_v2_api", "longbench_v2")
            for path in (suite / folder).glob("*/manifest.json")]
    manifests = {run: json.loads((run / "manifest.json").read_text()) for run in runs}
    configs = {run: json.loads((run / "execution_manifest.json").read_text()) for run in runs}
    runs.sort(key=lambda run: manifests[run]["started_at"])
    found, clusters = {}, {}
    for run in runs:
        config = configs[run]
        # Questions about the same document share its content hash.
        clusters.update((record["case_id"], record["source_content_sha256"])
                        for record in read_rows(run / "execution.jsonl"))
        found[manifests[run]["run_id"].removeprefix(f"{suite.name}_")] = {
            "window": config["context_window_tokens"],
            "outcomes": {row["case_id"]: bool(row["scores"].get("accuracy"))
                         for row in read_rows(run / "evaluation.jsonl")},
            "middle": config["mode"] == "direct" and config["direct_overflow"] == "middle",
        }
    return found, clusters


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="benchmark_artifacts/longbench_v2/suites/<suite-id>")
    args = parser.parse_args(argv)
    found, clusters = conditions(args.suite)
    columns, rows = compare(found, clusters, metric="accuracy")
    write_table(args.suite, "intervals",
                f"LongBench-v2 suite {args.suite.name}: paired differences, 95% intervals from {REPLICATES:,} "
                f"resamples of the {len(set(clusters.values()))} documents (seed {SEED})", columns, rows)


if __name__ == "__main__":
    main()
