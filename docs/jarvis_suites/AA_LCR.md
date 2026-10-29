# AA-LCR suite

Each AA-LCR question is answered from several documents, and the answer is free
text that a model grades. The main budget is 64K tokens: a 65,536-token window
with 61,440 input tokens, which leaves 4,096 tokens for the answer.

| | |
| --- | --- |
| Name in commands | `aa_lcr` |
| Questions per run | 100, all of AA-LCR v1.1 |
| Runs | 15 now, plus 3 upgraded-recipe runs later |
| Time | about 4 hours, plus about 1.25 hours for the recipe runs |
| Grading | The executor (Qwen3.6) grades each run as soon as it finishes. At the very end, GPT-5.6 Luna regrades every run through OpenRouter. |
| Output | `adarsh-rlms/benchmark_artifacts/aa_lcr/suites/<suite-id>/` |

## Conditions

*Direct* runs send the prompt itself, cut to the budget when it is too long.
*Hybrid* runs split the documents into chunks of 7,500 tokens, with 750
overlapping. They rank the chunks and fill the budget with the best-ranked
ones, shown in document order.

| Condition | Mode | Window | What the model sees |
| --- | --- | --- | --- |
| `full` | direct | 262K | The whole prompt. Upper-bound reference. |
| `middle` | direct | 64K | The start and end of the prompt; the middle is cut. |
| `middle_marked` | direct | 64K | The same cut, with an `[... omitted ...]` marker where text was removed. |
| `head` | direct | 64K | The start of the documents, then the question. |
| `dense` | hybrid | 64K | Chunks ranked by embedding similarity to the question. |
| `bm25` | hybrid | 64K | Chunks ranked by BM25 keyword score. |
| `fusion` | hybrid | 64K | Dense and BM25 rankings combined (reciprocal-rank fusion). |
| `position` | hybrid | 64K | Chunks nearest each document's start and end first; ignores the question. |
| `random0`, `random1`, `random2` | hybrid | 64K | Chunks in random order, with seeds 0, 1 and 2. |
| `middle_32k`, `middle_80k` | direct | 32K, 80K | `middle` at other budgets. |
| `position_32k`, `position_80k` | hybrid | 32K, 80K | `position` at other budgets. |
| `upgraded`, `upgraded_32k`, `upgraded_80k` | hybrid | 64K, 32K, 80K | The upgraded recipe (later; see below). |

## Before the first suite: prepare the data

This downloads AA-LCR v1.1 and extracts its documents into
`benchmark_data/aa_lcr/v1.1/`. Run it on Jarvis rather than copying the folder
from another server, because the folder's manifest stores absolute paths.

```bash
cd ~/adarsh-rlms
.venv/bin/python -m aa_lcr.prepare_dataset --dataset-version 1.1
ls benchmark_data/aa_lcr/v1.1   # AA-LCR_Dataset.csv, dataset_manifest.json, extracted_text, ...
cd ~
```

## Run the suite

1. Update the code: `git -C adarsh-rlms pull`.
2. Start the executor, export its address and record its setup. These are
   steps 1 to 4 of [EXECUTOR.md](EXECUTOR.md). The same executor also grades
   the runs.
3. Preview the suite:

   ```bash
   DRY_RUN=1 bash adarsh-rlms/jarvis/run_suite.sh aa_lcr 20261029
   ```

   The first line should read
   `[AA-LCR SUITE] Suite 20261029: 15 conditions, grader Qwen/Qwen3.6-35B-A3B`.
   Below it are a run command and a grading command for each condition.
4. Optional first check, for a new setup. This submits two runs, one direct and
   one hybrid, and grades them, which takes under an hour:

   ```bash
   ONLY=middle,dense bash adarsh-rlms/jarvis/run_suite.sh aa_lcr 20261029
   ```

   When the job ends, check that its log ends with
   `All 2 conditions complete`. The next step skips both runs.
5. Submit the suite:

   ```bash
   bash adarsh-rlms/jarvis/run_suite.sh aa_lcr 20261029
   ```

6. Follow it:

   ```bash
   squeue -u "$USER"
   tail -f "$PROJECT_LOG_DIR/rlms-client-gpu-<suite job id>.out"
   ```

   Each condition shows a `run:` line and then a `grade:` line.
7. When the log ends with `All 15 conditions complete`, stop the executor:
   `scancel "$EXECUTOR_JOB"`.

To resume after an interruption, submit the command from step 5 again.
Finished runs and grades are skipped.

## Results

The suite folder, `adarsh-rlms/benchmark_artifacts/aa_lcr/suites/20261029/`,
holds:

```text
direct/20261029_<condition>/                    direct runs
hybrid/20261029_<condition>/                    hybrid runs
regrades/Qwen3.6-35B-A3B/<mode>/20261029_<condition>/   grades by the executor
regrades/gpt-5.6-luna/<mode>/20261029_<condition>/      grades by GPT-5.6 Luna (at the end)
logs/<condition>.log, logs/<condition>.grade-<grader>.log
results-Qwen3.6-35B-A3B.md, .csv                results graded by the executor
results-gpt-5.6-luna.md, .csv                   results graded by GPT-5.6 Luna
```

Each table has one row per run. It gives the accuracy, and the questions gained
and lost against `middle` at the same window, with an exact McNemar p-value. It
also shows how much of the prompt was kept, document coverage for hybrid runs,
token counts and timings. The CSV keeps plain numbers for plotting. Regenerate
a table at any time:

```bash
cd ~/adarsh-rlms
.venv/bin/python -m aa_lcr.results_table benchmark_artifacts/aa_lcr/suites/20261029
.venv/bin/python -m aa_lcr.results_table benchmark_artifacts/aa_lcr/suites/20261029 --grader gpt-5.6-luna
cd ~
```

## Later: upgraded-recipe runs

Once the upgraded recipe has been chosen from the LongBench-v2 and MRCR ladder
results, add its three runs to the same suite. Start a new executor first, as
in steps 1 to 4 of [EXECUTOR.md](EXECUTOR.md). The 15 finished runs are
skipped.

```bash
AA_LCR_RECIPE="<the recipe's hybrid flags>" bash adarsh-rlms/jarvis/run_suite.sh aa_lcr 20261029
```

## At the end: regrade with GPT-5.6 Luna

Regrade once, after every AA-LCR run is finished, including the recipe runs.
This needs no executor. It submits a CPU job that calls GPT-5.6 Luna through
OpenRouter, using `OPENROUTER_API_KEY` from `adarsh-rlms/.env`. It first checks
on one question that the model returns reasoning tokens. Then it regrades all
18 runs in parallel, which takes under an hour.

```bash
AA_LCR_GRADER=luna DRY_RUN=1 bash adarsh-rlms/jarvis/run_suite.sh aa_lcr 20261029   # preview
AA_LCR_GRADER=luna bash adarsh-rlms/jarvis/run_suite.sh aa_lcr 20261029
tail -f "$PROJECT_LOG_DIR/rlms-client-<job id>.out"
```

The job ends by writing `results-gpt-5.6-luna.md` and `.csv`. If the OpenRouter
credits run out partway, top them up and submit the same command again:
finished regrades are skipped.
