# LongBench-v2 suite runbook

`jarvis/run_longbench_suite.sh` runs every LongBench-v2 condition of the
[experiment list](../workshop_paper_todo.md#longbench-v2-jarvis) one after
another. Run from the login node, it submits one `client-gpu` job that works
through the conditions against your executor job; answers are scored by letter
as they run, so there is no grading step.

## Before you start

Follow the [LongBench-v2 runbook](HPC_RUNBOOK_EXPERIMENT.md) up to the run step:

1. [Prepare the data](HPC_RUNBOOK_EXPERIMENT.md#1-prepare-the-data):
   `benchmark_data/long_bench_v2/data.json` must be present.
2. [Start the executor](HPC_RUNBOOK_EXPERIMENT.md#2-start-the-executor) with the
   full 262,144-token window and `--max-num-seqs 1`, and export its endpoint as
   `OPENAI_COMPAT_EXECUTOR_BASE_URL`.
3. [Record the serving setup](HPC_RUNBOOK_EXPERIMENT.md#3-record-the-serving-setup)
   and export `LONGBENCH_SERVING_METADATA=.cache/serving_metadata.json`.

The script checks the endpoint and the metadata file before submitting.

## 1. Run the conditions

From the parent directory that contains the repository:

```bash
DRY_RUN=1 bash adarsh-rlms/jarvis/run_longbench_suite.sh 20261026   # print the commands only
bash adarsh-rlms/jarvis/run_longbench_suite.sh 20261026
```

This runs the 17 conditions on the same 221 questions: original rows whose full
prompt is over the 64K budget but fits the 262K window, so each has a
full-context reference. It takes about 6 hours, within one job's 24-hour limit.
The order puts the references and the structure-or-selection controls first
(`full`, `middle`, `dense`, `position`, `random0`), then the other baselines,
the upgrade ladder and the budget sweep. At 128K, 115 of the 221 prompts fit
whole, so those rows match the full-context run.

Everything the suite writes stays in
`benchmark_artifacts/longbench_v2/suites/<suite-id>/`: direct runs in
`longbench_v2_api/` and hybrid runs in `longbench_v2/`, named
`<suite-id>_<condition>`, and one log per run in `logs/`. The job's own log is
`$PROJECT_LOG_DIR/rlms-client-gpu-<job id>.out`.

Submitting the same command again resumes: finished runs are skipped, and a run
that did not finish is renamed `*.incomplete-<time>` and retried. Resubmit if
the job or the executor job reaches its time limit, after restarting the
executor. A failed condition does not stop the others; the job log's summary
lists it. `ONLY=position,random0` runs a subset, and an unknown name lists the
valid ones.

## 2. Add the upgraded-recipe runs

Once the upgraded recipe is fixed from the MRCR and LongBench-v2 ladders, pass
its hybrid flags to add its runs at 64K, 32K and 128K, about 1.7 hours. The
finished runs are skipped.

```bash
LONGBENCH_RECIPE="--evidence-selection fusion --child-tokens 1024 --child-overlap-tokens 128 --neighbor-children 1 --packing fill --document-floor" \
  bash adarsh-rlms/jarvis/run_longbench_suite.sh 20261026
```

## Results table

Each job ends by writing a results table, one row per run, to `results.md` and
`results.csv` in the suite folder: accuracy, questions gained and lost against
middle truncation at the same window with an exact McNemar p-value, answers
outside A–D, and token, time and embedding columns. The CSV keeps plain numbers
for plotting. Regenerate it at any time, from the repository directory:

```bash
.venv/bin/python -m long_bench_v2.results_table benchmark_artifacts/longbench_v2/suites/20261026
```
