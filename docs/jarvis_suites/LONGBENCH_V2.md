# LongBench-v2 suite

Each LongBench-v2 question is a multiple-choice question about one long
document. The main budget is 64K tokens: a 65,536-token window with 65,528
input tokens, which leaves 8 tokens for the answer letter.

| | |
| --- | --- |
| Name in commands | `longbench` |
| Questions per run | 221: every original question whose full prompt exceeds the 64K budget but fits the executor's 262K window, so each has a full-context reference |
| Runs | 17 now, plus 3 upgraded-recipe runs later |
| Time | about 6 hours, plus about 1.7 hours for the recipe runs |
| Scoring | the answer letter (A to D) is checked against the key during the run; no grader |
| Output | `adarsh-rlms/benchmark_artifacts/longbench_v2/suites/<suite-id>/` |

## Conditions

*Direct* runs send the prompt itself, cut to the budget when it is too long.
*Hybrid* runs split the document into chunks, 7,500 tokens with 750 overlapping
unless stated otherwise. They rank the chunks and fill the budget with the
best-ranked ones, shown in document order.

| Condition | Mode | Window | What the model sees |
| --- | --- | --- | --- |
| `full` | direct | 262K | The whole prompt. Upper-bound reference. |
| `middle` | direct | 64K | The start and end of the prompt; the middle is cut. |
| `middle_marked` | direct | 64K | The same cut, with an `[... omitted ...]` marker where text was removed. |
| `head` | direct | 64K | The start of the document, then the question. |
| `dense` | hybrid | 64K | Chunks ranked by embedding similarity to the question. |
| `bm25` | hybrid | 64K | Chunks ranked by BM25 keyword score. |
| `fusion` | hybrid | 64K | Dense and BM25 rankings combined (reciprocal-rank fusion). |
| `position` | hybrid | 64K | Chunks nearest the document's start and end first; ignores the question. |
| `random0` | hybrid | 64K | Chunks in random order (seed 0). |
| `ladder1` | hybrid | 64K | `fusion` with 1,024-token chunks (128 overlapping). |
| `ladder2` | hybrid | 64K | `ladder1`, with each chosen chunk widened by its neighbouring chunks. |
| `ladder3` | hybrid | 64K | `ladder2`, filling the leftover budget with lower-ranked chunks. The last one is trimmed to fit. |
| `ladder4` | hybrid | 64K | `ladder3`, giving each document its best chunk first. Each question here has one document, so this matches `ladder3`. |
| `middle_32k`, `middle_128k` | direct | 32K, 128K | `middle` at other budgets. |
| `position_32k`, `position_128k` | hybrid | 32K, 128K | `position` at other budgets. |
| `upgraded`, `upgraded_32k`, `upgraded_128k` | hybrid | 64K, 32K, 128K | The upgraded recipe (later; see below). |

## Before the first suite: get the data

The question list is part of the code. The documents are in `data.json`, about
465 MB, which you download once:

```bash
cd ~/adarsh-rlms
mkdir -p benchmark_data/long_bench_v2
curl -fL --retry 3 \
  'https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json' \
  -o benchmark_data/long_bench_v2/data.json.download &&
  mv benchmark_data/long_bench_v2/data.json.download benchmark_data/long_bench_v2/data.json
ls -lh benchmark_data/long_bench_v2/data.json
cd ~
```

## Run the suite

1. Update the code: `git -C adarsh-rlms pull`.
2. Start the executor, export its address and record its setup. These are
   steps 1 to 4 of [EXECUTOR.md](EXECUTOR.md).
3. Preview the suite:

   ```bash
   DRY_RUN=1 bash adarsh-rlms/jarvis/run_suite.sh longbench 20261029
   ```

   The first line should start with `[LONGBENCH SUITE] Suite 20261029: 17 conditions`.
   Below it is one command per condition.
4. Optional first check, for a new setup. This submits two runs, one direct and
   one hybrid, which take about 45 minutes:

   ```bash
   ONLY=middle,dense bash adarsh-rlms/jarvis/run_suite.sh longbench 20261029
   ```

   When the job ends, check that its log ends with
   `All 2 conditions complete`. The next step skips both runs.
5. Submit the suite:

   ```bash
   bash adarsh-rlms/jarvis/run_suite.sh longbench 20261029
   ```

6. Follow it:

   ```bash
   squeue -u "$USER"
   tail -f "$PROJECT_LOG_DIR/rlms-client-gpu-<suite job id>.out"
   ```

7. When the log ends with `All 17 conditions complete`, stop the executor:
   `scancel "$EXECUTOR_JOB"`.

To resume after an interruption, submit the command from step 5 again.

## Results

The suite folder, `adarsh-rlms/benchmark_artifacts/longbench_v2/suites/20261029/`,
holds:

```text
longbench_v2_api/20261029_<condition>/   direct runs
longbench_v2/20261029_<condition>/       hybrid runs
logs/<condition>.log                     one log per run
results.md, results.csv                  the results table
```

The table has one row per run. It gives the accuracy, and the questions gained
and lost against `middle` at the same window, with an exact McNemar p-value. It
also shows how much of the prompt was kept, token counts and timings. The CSV
keeps plain numbers for plotting. The job writes the table at the end;
regenerate it at any time, for example while the suite is still running:

```bash
cd ~/adarsh-rlms
.venv/bin/python -m long_bench_v2.results_table benchmark_artifacts/longbench_v2/suites/20261029
cd ~
```

## Later: upgraded-recipe runs

Once the upgraded recipe has been chosen from the ladder results, add its
three runs to the same suite. Start a new executor first, as in steps 1 to 4
of [EXECUTOR.md](EXECUTOR.md). The 17 finished runs are skipped.

```bash
LONGBENCH_RECIPE="<the recipe's hybrid flags>" bash adarsh-rlms/jarvis/run_suite.sh longbench 20261029
```

For example, the flags of `ladder3` are
`--evidence-selection fusion --child-tokens 1024 --child-overlap-tokens 128 --neighbor-children 1 --packing fill`.
