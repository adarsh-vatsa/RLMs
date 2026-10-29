# Workshop suites on Jarvis

All three workshop benchmarks run on Jarvis, one at a time.
`jarvis/run_suite.sh <aa_lcr|mrcr|longbench> <suite-id>` submits one job that
runs every condition of that benchmark's
[experiment list](../workshop_paper_todo.md#experiment-list) in turn against an
executor job, and ends with a results table.

| Benchmark | Conditions now (+ recipe later) | Questions per run | Estimated time | Grading | Output folder |
|---|---:|---|---:|---|---|
| LongBench-v2 (`longbench`) | 17 (+3) | 221 | ~6 h | by answer letter, in the run | `benchmark_artifacts/longbench_v2/suites/<suite-id>/` |
| AA-LCR (`aa_lcr`) | 15 (+3) | 100 | ~4 h | executor after each run; GPT-5.6 Luna at the end | `benchmark_artifacts/aa_lcr/suites/<suite-id>/` |
| MRCR v2 (`mrcr`) | 20 (+4) | 120 (30 per conversation) | ~14 h | deterministic, in the run | `benchmark_artifacts/mrcr_v2/suites/<suite-id>/` |

The times are estimated from the Neselab and August Jarvis runs; the first job
of each benchmark measures them. Starting with LongBench-v2, which has run on
Jarvis before, shakes out the setup on the shortest suite.

## One-time setup

Run these from the parent directory that contains the repository.

1. Complete the [setup runbook](HPC_RUNBOOK_SETUP.md): the vLLM environment and
   the client environment in `adarsh-rlms/.venv`.
2. Pull the branch: `git -C adarsh-rlms pull`.
3. Prepare the data:
   - **LongBench-v2:** download `data.json` as in
     [step 1 of the LongBench-v2 runbook](HPC_RUNBOOK_EXPERIMENT.md#1-prepare-the-data).
   - **AA-LCR:** `(cd adarsh-rlms && .venv/bin/python -m aa_lcr.prepare_dataset --dataset-version 1.1)`.
     Prepare it on Jarvis rather than copying it: its manifest stores absolute
     paths.
   - **MRCR:** nothing to do. The prepared 4-needle dataset,
     `benchmark_data/mrcr_v2_4needle_100k_1200k`, is in Git.
4. For the final AA-LCR regrade, put `OPENROUTER_API_KEY` in `adarsh-rlms/.env`,
   which Git ignores.

## Run one benchmark

1. [Start the executor](HPC_RUNBOOK_EXPERIMENT.md#2-start-the-executor) with the
   full 262,144-token window and `--max-num-seqs 1`, and export its endpoint as
   `OPENAI_COMPAT_EXECUTOR_BASE_URL`. All three benchmarks use the same executor
   settings. Start a fresh executor for each benchmark, so its 24-hour limit
   covers the whole suite.
2. [Record the serving setup](HPC_RUNBOOK_EXPERIMENT.md#3-record-the-serving-setup)
   for that executor. Every suite reads `adarsh-rlms/.cache/serving_metadata.json`.
3. Submit the suite. A dry run prints its commands on the login node without
   submitting anything:

   ```bash
   DRY_RUN=1 bash adarsh-rlms/jarvis/run_suite.sh longbench 20261028
   bash adarsh-rlms/jarvis/run_suite.sh longbench 20261028
   ```

   The job waits up to 30 minutes for the executor to finish loading. For a
   first check of a new setup, submit a single condition with `ONLY=middle`;
   the full suite later skips it.
4. Follow the job log, `$PROJECT_LOG_DIR/rlms-client-gpu-<job id>.out`. Each run
   also writes its own log to `logs/` in the suite folder.
5. When the job ends, cancel the executor job (`scancel <executor job id>`)
   before starting the next benchmark.

Submitting the same command again resumes: finished runs are skipped, and an
unfinished run is moved aside and retried. If the executor job reaches its time
limit mid-suite, start a new executor, record its serving setup, export its
endpoint and resubmit. A failed condition does not stop the others; the job
log's summary lists it. `ONLY=position,random0` runs a subset, and an unknown
name lists the valid ones.

## Later steps

- **Upgraded recipe.** Once the recipe is fixed from the MRCR and LongBench-v2
  ladders, pass its hybrid flags with the same suite ID; only the new runs run:
  `AA_LCR_RECIPE="..."`, `MRCR_RECIPE="..."` or `LONGBENCH_RECIPE="..."`.
- **AA-LCR regrade with GPT-5.6 Luna**, after every AA-LCR run including the
  recipe runs: `AA_LCR_GRADER=luna bash adarsh-rlms/jarvis/run_suite.sh aa_lcr 20261028`.
  This submits a CPU job that calls OpenRouter and needs no executor. It checks
  one call for reasoning tokens before regrading every run.
- **Full MRCR.** A larger sample is a new suite with its own ID, because a
  different sample selects different questions:
  `MRCR_ROWS_PER_SOURCE=0 bash adarsh-rlms/jarvis/run_suite.sh mrcr <new-suite-id>`.
  All 953 questions take about 117 hours, so expect several executor jobs and
  resubmissions.

## Results tables

Each job ends by writing its benchmark's table to the suite folder:
`results-<grader>.md` and `.csv` for AA-LCR, `results.md` and `.csv` for MRCR and
LongBench-v2. The CSV keeps plain numbers for plotting. The
[AA-LCR](../linux/AA_LCR_SUITE_RUNBOOK.md) and
[MRCR](../linux/MRCR_SUITE_RUNBOOK.md) suite runbooks describe each table's
columns. Regenerate a table at any time, from the repository directory:

```bash
.venv/bin/python -m long_bench_v2.results_table benchmark_artifacts/longbench_v2/suites/20261028
.venv/bin/python -m aa_lcr.results_table benchmark_artifacts/aa_lcr/suites/20261028 --grader Qwen3.6-35B-A3B
.venv/bin/python -m mrcr_v2.results_table benchmark_artifacts/mrcr_v2/suites/20261028
```
