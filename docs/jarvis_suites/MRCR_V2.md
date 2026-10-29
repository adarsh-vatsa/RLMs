# MRCR v2 suite

Each MRCR v2 question comes at the end of a long conversation that contains
several similar requests, such as four poems about the same topic. The question
asks the model to repeat one of them, such as the second poem, behind a given
prefix. The answer is scored by its similarity to the expected text (the MRCR
score) and by exact match. The main budget is the executor's full window: a
262,144-token window with 258,048 input tokens, which leaves 4,096 tokens for
the answer.

| | |
| --- | --- |
| Name in commands | `mrcr` |
| Questions per run | 120: 30 from each of the four conversations, which are about 133K, 267K, 534K and 1.07M tokens long. The 133K conversation fits the 262K window, so it is sent whole under every 262K condition. |
| Runs | 20 now, plus 4 upgraded-recipe runs later |
| Time | about 14 hours, plus about 1.6 hours for the recipe runs |
| Scoring | during the run; no grader |
| Output | `adarsh-rlms/benchmark_artifacts/mrcr_v2/suites/<suite-id>/` |

The data, 953 questions in all, comes with the code in
`benchmark_data/mrcr_v2_4needle_100k_1200k`, so there is nothing to prepare.

## Conditions

*Direct* runs send the prompt itself, cut to the budget when it is too long.
*Hybrid* runs split the conversation into chunks, 3,500 tokens with 350
overlapping unless stated otherwise. They rank the chunks against the requested
item, such as "poem about space exploration", and fill the budget with the
best-ranked ones, shown in conversation order.

| Condition | Mode | Window | What the model sees |
| --- | --- | --- | --- |
| `middle` | direct | 262K | The start and end of the conversation; the middle is cut. |
| `middle_marked` | direct | 262K | The same cut, with an `[... omitted ...]` marker where text was removed. |
| `head` | direct | 262K | The start of the conversation, then the question. |
| `dense` | hybrid | 262K | Chunks ranked by embedding similarity. |
| `bm25` | hybrid | 262K | Chunks ranked by BM25 keyword score. |
| `fusion` | hybrid | 262K | Dense and BM25 rankings combined (reciprocal-rank fusion). |
| `position` | hybrid | 262K | Chunks nearest the conversation's start and end first; ignores the question. |
| `random0` | hybrid | 262K | Chunks in random order (seed 0). |
| `dense_7500` | hybrid | 262K | `dense` with 7,500-token chunks (750 overlapping). |
| `dense_question` | hybrid | 262K | `dense`, ranking against the whole final instruction instead of the requested item. |
| `ladder1` | hybrid | 262K | `fusion` with 1,024-token chunks (128 overlapping). |
| `ladder2` | hybrid | 262K | `ladder1`, with each chosen chunk widened by its neighbouring chunks. |
| `ladder3` | hybrid | 262K | `ladder2`, filling the leftover budget with lower-ranked chunks. The last one is trimmed to fit. |
| `ladder4` | hybrid | 262K | `ladder3`, giving each document its best chunk first. Each question here has one conversation, so this matches `ladder3`. |
| `middle_32k`, `middle_64k`, `middle_128k` | direct | 32K, 64K, 128K | `middle` at smaller budgets. |
| `position_32k`, `position_64k`, `position_128k` | hybrid | 32K, 64K, 128K | `position` at smaller budgets. |
| `upgraded`, `upgraded_32k`, `upgraded_64k`, `upgraded_128k` | hybrid | 262K, 32K, 64K, 128K | The upgraded recipe (later; see below). |

## Run the suite

1. Update the code: `git -C adarsh-rlms pull`.
2. Start the executor, export its address and record its setup. These are
   steps 1 to 4 of [EXECUTOR.md](EXECUTOR.md). At about 14 hours, the suite
   fits within one executor's 24 hours.
3. Preview the suite:

   ```bash
   DRY_RUN=1 bash adarsh-rlms/jarvis/run_suite.sh mrcr 20261029
   ```

   The first line should read
   `[MRCR SUITE] Suite 20261029: 20 conditions, data benchmark_data/mrcr_v2_4needle_100k_1200k, 30 questions per conversation (0 = all)`.
4. Optional first check, for a new setup. This submits two runs, one direct and
   one hybrid, which take about 2 hours:

   ```bash
   ONLY=middle,dense bash adarsh-rlms/jarvis/run_suite.sh mrcr 20261029
   ```

   When the job ends, check that its log ends with
   `All 2 conditions complete`. The next step skips both runs.
5. Submit the suite:

   ```bash
   bash adarsh-rlms/jarvis/run_suite.sh mrcr 20261029
   ```

6. Follow it:

   ```bash
   squeue -u "$USER"
   tail -f "$PROJECT_LOG_DIR/rlms-client-gpu-<suite job id>.out"
   ```

7. When the log ends with `All 20 conditions complete`, stop the executor:
   `scancel "$EXECUTOR_JOB"`.

To resume after an interruption, submit the command from step 5 again.

## Results

The suite folder, `adarsh-rlms/benchmark_artifacts/mrcr_v2/suites/20261029/`,
holds:

```text
direct/20261029_<condition>/   direct runs
hybrid/20261029_<condition>/   hybrid runs
logs/<condition>.log           one log per run
results.md, results.csv        the results table
```

The table has one row per run. It gives the exact-match rate, the MRCR score
and prefix compliance, and the questions gained and lost against `middle` at
the same window, with an exact McNemar p-value. It also shows how much of the
prompt was kept, token counts and timings. The CSV keeps plain numbers for
plotting. Regenerate it at any time:

```bash
cd ~/adarsh-rlms
.venv/bin/python -m mrcr_v2.results_table benchmark_artifacts/mrcr_v2/suites/20261029
cd ~
```

## Later: upgraded-recipe runs

Once the upgraded recipe has been chosen from the ladder results, add its four
runs to the same suite. Start a new executor first, as in steps 1 to 4 of
[EXECUTOR.md](EXECUTOR.md). The 20 finished runs are skipped.

```bash
MRCR_RECIPE="<the recipe's hybrid flags>" bash adarsh-rlms/jarvis/run_suite.sh mrcr 20261029
```

## Possible follow-up: the full benchmark

All 953 questions form a new suite with its own ID, because a different number
of questions selects different questions. `MRCR_ROWS_PER_SOURCE=0` selects all
of them:

```bash
MRCR_ROWS_PER_SOURCE=0 DRY_RUN=1 bash adarsh-rlms/jarvis/run_suite.sh mrcr 20261105-full   # preview
MRCR_ROWS_PER_SOURCE=0 bash adarsh-rlms/jarvis/run_suite.sh mrcr 20261105-full
```

This takes about 117 hours, so it spans several executor jobs. Each time the
executor or the suite job reaches its 24-hour limit, start a new executor,
following steps 1 to 4 of [EXECUTOR.md](EXECUTOR.md). Then resubmit the same
command, including `MRCR_ROWS_PER_SOURCE=0`. Without it, the remaining runs
would use 30 questions per conversation. The `Resume or repeat with:` line in
the job log includes it.
