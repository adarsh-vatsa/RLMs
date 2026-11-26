"""Visibility decomposition of an MRCR v2 suite: did each prompt contain the needle turns?

A turn is visible when all of it, request and response, is in the prompt. Exact match is
the share with the target visible times exact match given it is visible, plus any correct
answer without it. Where the target is visible, the oracle's exact match on the same
questions at the same window shows whether the run reads them as well as the oracle does.
A second table splits those questions by how many needle turns are visible, for the selection
runs, since a prompt that drops the other needle turns leaves the model fewer candidates.
Truncated prompts are rebuilt with the tokenizer the runs used, so run this where the suite ran;
with --selection-only it reads only the selection runs, needs no tokenizer and writes only the
second table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from execution.pipeline import Pipeline
from execution.results_table import mean, read_rows, window, write_table
from execution.tokens import truncate_prompt
from mrcr_v2.adapter import solver_task
from mrcr_v2.dataset import (
    DEFAULT_DATA_DIR, DEFAULT_REVISION, file_hash, load_prepared, load_tokenizer, read_source, tokenizer_metadata,
)
from mrcr_v2.oracle import needle_spans
from mrcr_v2.results_table import label, size_label


def visible(record, manifest, prefix, body, question, spans, tokenizer):
    """Whether each span is whole in the prompt the run built for this record."""
    if record["route"] == "direct_fit":
        return [True] * len(spans)
    if record["route"] == "dense_child_packed":
        ranges = [(item["char_start"], item["char_end"]) for item in record["selected_evidence_ranges"]]
        return [any(left <= start and end <= right for left, right in ranges) for start, end in spans]
    task = solver_task(record["case_id"], record["source_id"], prefix, body, question)
    request = task.render(None)
    overflow = manifest["execution"]["direct_overflow"]
    ending = Pipeline._ending(task, request) if overflow == "head" else ""
    content = truncate_prompt(tokenizer, request, manifest["max_input_tokens"], overflow, ending)[-1]["content"]
    return [body[start:end] in content for start, end in spans]


def given(answers, condition, outcome=lambda answer: answer["exact"]):
    matches = [outcome(answer) for answer in answers if condition(answer)]
    return mean(matches) if matches else ""


def summarize(suite, data_dir, *, tokenizer_factory=load_tokenizer, modes=("direct", "hybrid")):
    dataset, prepared = load_prepared(data_dir)
    dataset_sha256 = file_hash(data_dir / "dataset_manifest.json")
    questions = {row["case_id"]: row for row in prepared}
    runs = [path.parent for mode in modes for path in (suite / mode).glob("*/manifest.json")]
    manifests = {run: json.loads((run / "manifest.json").read_text()) for run in runs}
    runs = sorted((run for run in runs if manifests[run]["status"] == "complete"),
                  key=lambda run: manifests[run]["created_at"])
    sources, spans, sizes, results = {}, {}, {}, {}
    tokenizer = None
    for run in runs:
        manifest = manifests[run]
        if manifest["dataset_manifest_sha256"] != dataset_sha256:
            raise ValueError(f"{run.name} used a different prepared dataset than {data_dir}")
        print(f"[MRCR VISIBILITY] {run.name}")
        answers = []
        for record in read_rows(run / "predictions.jsonl"):
            row = questions[record["case_id"]]
            if record["source_id"] not in sources:
                sources[record["source_id"]] = read_source(data_dir, record["source_id"])
            prefix, body = sources[record["source_id"]]
            if record["case_id"] not in spans:
                spans[record["case_id"]] = needle_spans(body, row["question"], row["answer"], row["num_relevant"])
            needles, target = spans[record["case_id"]]
            if tokenizer is None and record["route"] not in {"direct_fit", "dense_child_packed"}:
                tokenizer = tokenizer_factory(manifest["executor_model"], DEFAULT_REVISION)
                if tokenizer_metadata(tokenizer, manifest["executor_model"]) != dataset["tokenizer"]:
                    raise ValueError("This tokenizer differs from the one the runs used; run where the suite ran")
            shown = visible(record, manifest, prefix, body, row["question"], needles, tokenizer)
            sizes[record["source_id"]] = min(sizes.get(record["source_id"], record["full_rendered_input_tokens"]),
                                             record["full_rendered_input_tokens"])
            answers.append({"case": record["case_id"], "source": record["source_id"],
                            "exact": bool(record["exact_match"]), "target": shown[target], "all": all(shown),
                            "needles": sum(shown)})
        results[run] = answers
    windows = {run: window(manifests[run]["execution"], manifests[run]["context_window_tokens"]) for run in results}
    oracles = {windows[run]: {answer["case"]: answer["exact"] for answer in answers}
               for run, answers in results.items() if manifests[run].get("oracle")}
    ordered = sorted(sizes, key=sizes.get)
    rows = []
    for run, answers in results.items():
        manifest = manifests[run]
        oracle = oracles.get(windows[run])
        rows.append({
            "condition": manifest["run_id"].removeprefix(f"{suite.name}_"),
            "policy": label(manifest),
            "window": windows[run],
            "questions": len(answers),
            "exact match": mean(answer["exact"] for answer in answers),
            "target visible": mean(answer["target"] for answer in answers),
            "all needles visible": mean(answer["all"] for answer in answers),
            "exact match given target visible": given(answers, lambda answer: answer["target"]),
            "oracle exact match where target visible":
                given(answers, lambda answer: answer["target"], lambda answer: oracle[answer["case"]]) if oracle else "",
            "exact match given target hidden": given(answers, lambda answer: not answer["target"]),
            "exact match given all needles visible": given(answers, lambda answer: answer["all"]),
            **{f"target visible at {size_label(sizes[source])}":
               mean(answer["target"] for answer in answers if answer["source"] == source) for source in ordered},
        })
    columns = ["condition", "policy", "window", "questions", "exact match", "target visible", "all needles visible",
               "exact match given target visible", "oracle exact match where target visible",
               "exact match given target hidden",
               "exact match given all needles visible",
               *[f"target visible at {size_label(sizes[source])}" for source in ordered]]
    needle_rows = []
    for run, answers in results.items():
        manifest = manifests[run]
        oracle = oracles.get(windows[run])
        if oracle is None or manifest.get("oracle") or manifest["mode"] != "hybrid":
            continue
        for count in sorted({answer["needles"] for answer in answers if answer["target"]}):
            group = [answer for answer in answers if answer["target"] and answer["needles"] == count]
            needle_rows.append({
                "condition": manifest["run_id"].removeprefix(f"{suite.name}_"),
                "policy": label(manifest),
                "window": windows[run],
                "needle turns visible": count,
                "questions": len(group),
                "exact matches": sum(answer["exact"] for answer in group),
                "oracle exact matches": sum(oracle[answer["case"]] for answer in group),
            })
    needle_columns = ["condition", "policy", "window", "needle turns visible", "questions", "exact matches",
                      "oracle exact matches"]
    return columns, rows, needle_columns, needle_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="benchmark_artifacts/mrcr_v2/suites/<suite-id>")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--selection-only", action="store_true",
                        help="Read only the selection runs and write only the needle table; they need no tokenizer")
    args = parser.parse_args(argv)
    columns, rows, needle_columns, needle_rows = summarize(
        args.suite, args.data_dir, modes=("hybrid",) if args.selection_only else ("direct", "hybrid"))
    if not args.selection_only:
        write_table(args.suite, "visibility", f"MRCR v2 suite {args.suite.name}: visibility", columns, rows)
    write_table(args.suite, "visibility-needles",
                f"MRCR v2 suite {args.suite.name}: exact matches where the target is visible, by needle turns visible",
                needle_columns, needle_rows)


if __name__ == "__main__":
    main()
