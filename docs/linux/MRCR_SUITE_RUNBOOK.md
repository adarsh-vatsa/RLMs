# MRCR v2 suite runbook

`linux/run_mrcr_suite.sh` runs every MRCR v2 condition of the
[experiment list](../workshop_paper_todo.md#mrcr-v2-jarvis) one after another,
instead of launching runs by hand from the
[benchmark runbook](BENCHMARK_RUNBOOK.md#overflow-policy-conditions). MRCR scores
each answer deterministically as it runs, so there is no grading step.

To run this suite on Jarvis as a Slurm job, see the
[Jarvis MRCR suite page](../jarvis_suites/MRCR_V2.md).

## Before you start

- Complete the [Linux setup](SETUP_RUNBOOK.md) and
  [prepare the MRCR data](BENCHMARK_RUNBOOK.md#mrcr-v2) for 4 needles, 100K to
  1.2M tokens, in `benchmark_data/mrcr_v2_4needle_100k_1200k` (set
  `MRCR_DATA_DIR` for another folder).
- Start the executor with its full 262,144-token context and
  [record the serving setup](SETUP_RUNBOOK.md#record-the-serving-setup) in
  `.cache/serving_metadata.json`. The script checks both before it starts.
- Run from the repository root, inside `tmux`. At the default 30 questions per
  conversation the first step takes about 14 to 15 hours: each run at the 262K
  budget takes about 55 minutes (about 31 seconds per question at full budget),
  and the budget sweep about 1.5 hours in all.

Hybrid runs embed on the GPU. vLLM's default memory setting leaves about 9 GB
for the embedding model; if embedding runs out of memory, lower
`SEMANTIC_CACHE_EMBEDDING_BATCH_SIZE` and rerun the same command.

## 1. Run the conditions

```bash
DRY_RUN=1 bash linux/run_mrcr_suite.sh 20261026   # print the commands only
bash linux/run_mrcr_suite.sh 20261026
```

This runs the 20 conditions on 30 questions per conversation, 120 in all, with
3,500-token chunks unless a condition sets its own. `MRCR_ROWS_PER_SOURCE=100`
takes 100 per conversation instead, and `0` takes all 953 questions; the
[run-time estimates](../run_time_estimates_20261025.md#mrcr-sample-size) give the
time for each. Use one value per suite ID, because a different value selects
different questions and runs could no longer be paired. The order puts the
references and the structure-or-selection controls first (`middle`, `dense`,
`position`, `random0`), then the other baselines, the retrieval variants, the
upgrade ladder and the budget sweep. The 133K conversation fits the 262K
budget, so its rows are the full-context reference.

The argument is the suite ID. Everything the suite writes stays in
`benchmark_artifacts/mrcr_v2/suites/<suite-id>/`, apart from earlier results:
runs in `direct/` and `hybrid/`, named `<suite-id>_<condition>`, and one log per
run in `logs/`. Rerunning the same command resumes: completed runs are skipped,
and a run that did not complete is renamed `*.incomplete-<time>` and retried.
A failed condition does not stop the others; the summary lists it and the
script exits non-zero. `ONLY=position,random0` runs a subset, and an unknown
name lists the valid ones.

## 2. Add the upgraded-recipe runs

Once the upgraded recipe is fixed from the MRCR and LongBench-v2 ladders, pass
its hybrid flags, including its chunk size, to add its runs at 262K, 32K, 64K
and 128K. The 20 finished runs are skipped. If the recipe is exactly one of the
ladder steps, its 262K run repeats that step; skip it with
`ONLY=upgraded_32k,upgraded_64k,upgraded_128k`.

```bash
MRCR_RECIPE="--evidence-selection fusion --child-tokens 1024 --child-overlap-tokens 128 --neighbor-children 1 --packing fill --document-floor" \
  bash linux/run_mrcr_suite.sh 20261026
```

## Results table

Each invocation ends by writing a results table, one row per run, to
`results.md` and `results.csv` in the suite folder. It includes exact match and
the MRCR score, questions gained and lost against middle truncation at the same
window with an exact McNemar p-value, and exact match for each conversation
length. A run still in progress shows its status. The CSV keeps plain numbers
for plotting. Regenerate it at any time:

```bash
.venv/bin/python -m mrcr_v2.results_table benchmark_artifacts/mrcr_v2/suites/20261026
```
