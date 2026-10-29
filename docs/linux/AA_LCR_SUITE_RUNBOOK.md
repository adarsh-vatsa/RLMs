# AA-LCR suite runbook

`linux/run_aa_lcr_suite.sh` runs every AA-LCR condition of the
[experiment list](../workshop_paper_todo.md#experiment-list) one after another
and grades each run as it finishes, instead of launching runs by hand from the
[benchmark runbook](BENCHMARK_RUNBOOK.md#overflow-policy-conditions). The same
command, with the same suite ID, is used for all three steps below.

On Jarvis, the [workshop suites runbook](../jarvis/WORKSHOP_SUITES_RUNBOOK.md)
submits this suite as a Slurm job against an executor job.

## Before you start

- Complete the [Linux setup](SETUP_RUNBOOK.md) and
  [prepare the AA-LCR data](BENCHMARK_RUNBOOK.md#aa-lcr).
- Start the executor and
  [record the serving setup](SETUP_RUNBOOK.md#record-the-serving-setup) in
  `.cache/serving_metadata.json`. The script checks both before it starts.
- Run from the repository root, inside `tmux`, since the first step takes about
  four hours.

Hybrid runs embed on the GPU. vLLM's default memory setting leaves about 9 GB
for the embedding model; if embedding runs out of memory, lower
`SEMANTIC_CACHE_EMBEDDING_BATCH_SIZE` and rerun the same command.

## 1. Run the conditions

```bash
DRY_RUN=1 bash linux/run_aa_lcr_suite.sh 20261023   # print the commands only
bash linux/run_aa_lcr_suite.sh 20261023
```

This runs the 15 conditions on all 100 questions and grades each run with the
executor (Qwen3.6) as soon as it finishes. The argument is the suite ID.
Everything the suite writes stays in
`benchmark_artifacts/aa_lcr/suites/<suite-id>/`, apart from earlier results:
runs in `direct/` and `hybrid/`, named `<suite-id>_<condition>`, grades in
`regrades/<grader>/` and one log per step in `logs/`.

Rerunning the same command resumes: finished runs and grades are skipped, and
an interrupted run's folder is renamed `*.incomplete-<time>` before the run is
retried. A failed condition does not stop the others; the summary lists it and
the script exits non-zero. `ONLY=position,random0` runs a subset, and an
unknown name lists the valid ones.

## 2. Add the upgraded-recipe runs

Once the upgraded recipe is fixed on MRCR and LongBench-v2, pass its hybrid
flags to add its three runs at 64K, 32K and 80K. The 15 finished runs are
skipped.

```bash
AA_LCR_RECIPE="--evidence-selection fusion --child-tokens 1024 --child-overlap-tokens 128 --neighbor-children 1 --packing fill --document-floor" \
  bash linux/run_aa_lcr_suite.sh 20261023
```

## 3. Regrade with GPT-5.6 Luna

After every AA-LCR run is done, regrade them all with GPT-5.6 Luna through
OpenRouter. The script reads `OPENROUTER_API_KEY` from `.env` when it is not
set, checks on one question that reasoning tokens come back, then regrades the
finished runs in parallel into the suite's `regrades/gpt-5.6-luna/`. It covers
the upgraded-recipe runs without `AA_LCR_RECIPE`. If credits run out partway,
top up and rerun: finished regrades are skipped.

```bash
AA_LCR_GRADER=luna bash linux/run_aa_lcr_suite.sh 20261023
```

## Results table

Each step ends by writing a results table, one row per run, to
`results-<grader>.md` and `.csv` in the suite folder. The CSV keeps plain
numbers for plotting. Regenerate it at any time, for example while the suite is
still running:

```bash
.venv/bin/python -m aa_lcr.results_table benchmark_artifacts/aa_lcr/suites/20261023
.venv/bin/python -m aa_lcr.results_table benchmark_artifacts/aa_lcr/suites/20261023 --grader gpt-5.6-luna
```
