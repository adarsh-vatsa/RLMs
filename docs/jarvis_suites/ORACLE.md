# Oracle and document-floor runs

These runs finish the workshop paper's Table 2 and the AA-LCR held-out check.
They add seven runs to the existing suites, `20261029`, and need one executor
job.

| | |
| --- | --- |
| AA-LCR | 3 runs of the recipe without `--document-floor`: `ladder3` (64K), `ladder3_32k` and `ladder3_80k`. The suite grades each with Qwen3.6, then Luna regrades them. |
| MRCR v2 | 4 oracle runs: `oracle` (262K), `oracle_32k`, `oracle_64k` and `oracle_128k`, then the visibility table. |
| Time | AA-LCR about 1 hour. MRCR about 2 hours, plus up to 30 minutes for the visibility table. The Luna regrade under 30 minutes, alongside MRCR. |
| Output | the existing suite folders, `adarsh-rlms/benchmark_artifacts/aa_lcr/suites/20261029/` and `adarsh-rlms/benchmark_artifacts/mrcr_v2/suites/20261029/` |

## What the runs are

**AA-LCR `ladder3`.** The recipe without `--document-floor`. The recipe keeps
every document for every question, but it also changes the chunk size, adds
neighbouring chunks and fills the budget. Comparing `ladder3` with `upgraded`
at the same budget shows what the document floor adds on its own.

**MRCR oracle.** The recipe with one change: the chunks that hold the needle
turns, the four turns that repeat the requested item, are offered to the packer
first, the target turn before the others. The needle turns are found from the
question's request and checked against the gold answer, so the oracle is a
ceiling, not a method. The prompt is built as in any hybrid run; only the order
in which chunks are offered uses gold data. Its rows in the results table read
`oracle: ...`.

**Visibility table.** At the end, the MRCR suite job writes `visibility.md` and
`visibility.csv`. For every MRCR run, they give how often the prompt held the
whole target turn and all four needle turns, and the exact-match rate given
each. Hybrid runs are checked against the ranges they kept. Truncated prompts
are rebuilt with the tokenizer the runs used, which is why the table is made on
Jarvis.

## Run them

1. Update the code: `git -C adarsh-rlms pull`. Check that no suite variable
   is left over from earlier commands:

   ```bash
   env | grep -E '^(ONLY|AA_LCR_|MRCR_|LONGBENCH_)'
   ```

   It should print nothing. A leftover `AA_LCR_RECIPE` or `MRCR_RECIPE` is
   harmless, but `unset ONLY AA_LCR_GRADER` if either appears: `ONLY` would
   skip the new runs, and `AA_LCR_GRADER=luna` would regrade instead of run.
2. Start the executor, export its address and record its setup: steps 1 to 4
   of [EXECUTOR.md](EXECUTOR.md).
3. Preview AA-LCR:

   ```bash
   DRY_RUN=1 bash adarsh-rlms/jarvis/run_suite.sh aa_lcr 20261029 | grep -c 'already finished'
   ```

   It should print `17`: only the three `ladder3` runs are new. If
   `AA_LCR_RECIPE` is still set from the recipe runs, it prints `20`; either is
   fine.
4. Submit AA-LCR and follow it:

   ```bash
   bash adarsh-rlms/jarvis/run_suite.sh aa_lcr 20261029
   tail -f "$PROJECT_LOG_DIR/rlms-client-gpu-<suite job id>.out"
   ```

   Wait until the log ends with `All 20 conditions complete` (23 if
   `AA_LCR_RECIPE` is set).
5. Preview MRCR:

   ```bash
   DRY_RUN=1 bash adarsh-rlms/jarvis/run_suite.sh mrcr 20261029 | grep -c 'already finished'
   ```

   It should print `23`: only the four oracle runs are new. If `MRCR_RECIPE` is
   still set, it prints `27`.
6. Submit MRCR:

   ```bash
   bash adarsh-rlms/jarvis/run_suite.sh mrcr 20261029
   ```

7. While MRCR runs, regrade the new AA-LCR runs with Luna. This is a CPU job
   that does not use the executor:

   ```bash
   AA_LCR_GRADER=luna DRY_RUN=1 bash adarsh-rlms/jarvis/run_suite.sh aa_lcr 20261029   # preview: 23 conditions
   AA_LCR_GRADER=luna bash adarsh-rlms/jarvis/run_suite.sh aa_lcr 20261029
   tail -f "$PROJECT_LOG_DIR/rlms-client-<job id>.out"
   ```

   The 20 runs Luna has already graded print `grade: already done`. The log
   ends with `All 23 conditions complete`.
8. Stop the executor, `scancel "$EXECUTOR_JOB"`, once the MRCR log reaches
   `Visibility table`: the table does not use the executor. The log ends with
   `All 27 conditions complete` (31 if `MRCR_RECIPE` is set).
9. Commit and push both suite folders, as for the earlier runs.

To resume after an interruption, submit the same command again. Finished runs
and grades are skipped. Do not `git pull` until every job above has ended: the
runs read the code as they start.

## Results

**AA-LCR.** `results-gpt-5.6-luna.md` gains rows for `ladder3`, `ladder3_32k`
and `ladder3_80k`. Compare each with the `upgraded` row at the same window;
both have the "documents covered" and "questions missing a document" columns.

**MRCR.** `results.md` gains the four oracle rows. `visibility.md` has one row
per run:

| Column | Meaning |
| --- | --- |
| `target visible` | Share of questions whose prompt held the whole target turn, request and response |
| `all needles visible` | Share whose prompt held all four needle turns |
| `exact match given target visible`, `exact match given target hidden` | Exact match among those questions; the second should be near 0 |
| `exact match given all needles visible` | The run's accuracy when nothing it needed was cut: an analytic oracle |
| `target visible at <length>` | `target visible` for each conversation |

The oracle rows should show 100% in both visibility columns. The recipe
already shows the target turn for every question at 262K, so `oracle` should
land close to `upgraded`; the oracle matters at 32K to 128K, where the recipe
shows the target for 48% to 89% of questions.

To make the visibility table again, run this from `~/adarsh-rlms`. It takes up
to half an hour:

```bash
.venv/bin/python -m mrcr_v2.visibility benchmark_artifacts/mrcr_v2/suites/20261029 \
  --data-dir benchmark_data/mrcr_v2_4needle_100k_1200k
```
