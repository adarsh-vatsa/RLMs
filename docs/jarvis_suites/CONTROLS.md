# Control runs

These runs separate the recipe's ranking from the rest of the recipe. They add
nine runs to the existing suites, `20261029`, and need one executor job, pinned
to one node.

| | |
| --- | --- |
| AA-LCR | 6 runs: `position_recipe` and `bm25_recipe` at 64K, and each with `_32k` and `_80k`. The suite grades each with Qwen3.6, then Luna regrades them. |
| MRCR v2 | 3 runs: `bm25_recipe_32k`, `bm25_recipe_64k` and `bm25_recipe_128k`, then the visibility table. |
| Node | g202, the node that served every recipe run (see [Why one node](#why-one-node)). |
| Time | AA-LCR about 2 hours with its grades. MRCR about 1 hour, plus up to 30 minutes for the visibility table. The Luna regrade under 30 minutes, alongside MRCR. |
| Output | the existing suite folders, `adarsh-rlms/benchmark_artifacts/aa_lcr/suites/20261029/` and `adarsh-rlms/benchmark_artifacts/mrcr_v2/suites/20261029/` |

## What the runs are

The recipe is fusion ranking plus four steps: 1,024-token chunks with 128 of
overlap, one neighbouring chunk on each side, `fill` packing and the document
floor. `position` and `bm25` ran without those steps, with 7,500-token chunks
(3,500 on MRCR) and `stop` packing. So the recipe's margin over them is partly
a difference in chunking, packing and, on AA-LCR, how many documents each
prompt covers.

Each control keeps the recipe's four steps and changes only the ranking:

- **`position_recipe`** ranks chunks by their distance from the nearer end of
  their document, ignoring the question. Compared with `upgraded`, it shows what
  ranking by the question adds to the same prompt structure. It runs on AA-LCR
  only. On one document, position keeps the head and tail, which is middle
  truncation with an omission marker. On MRCR, `position` is already within 2
  points of `middle` at every window, and `middle_marked` matches it exactly.
- **`bm25_recipe`** ranks chunks by BM25 alone. Compared with `upgraded`, it
  shows what fusion with dense ranking adds. Neither control embeds anything.

The interval pairs for both are in `execution/intervals.py`: each control
against its unmatched version, `bm25_recipe` and `upgraded` against
`position_recipe`, and `upgraded` against `bm25_recipe`, at each window.

## Why one node

Greedy decoding gives the same answer to the same prompt on g101 and g102, but
not on g202, which served the recipe runs and the BM25 sweep on 3 November. The
[experiment list](../workshop_paper_todo.md#experiment-list) has the details.
The controls are compared with the recipe runs, so they run on g202 too. If the
executor lands on another node, the runs are still valid, but every comparison
with the recipe also carries the difference between nodes.

## Run them

1. Update the code: `git -C adarsh-rlms pull`. Check that no suite variable
   is left over from earlier commands:

   ```bash
   env | grep -E '^(ONLY|AA_LCR_|MRCR_|LONGBENCH_|EXECUTOR_NODE)'
   ```

   It should print nothing. A leftover `AA_LCR_RECIPE` or `MRCR_RECIPE` is
   harmless, but `unset ONLY AA_LCR_GRADER` if either appears: `ONLY` would
   skip the new runs, and `AA_LCR_GRADER=luna` would regrade instead of run.
2. Check that g202 is in the partition and free:

   ```bash
   sinfo -N -p gpu-l40s -n g202
   ```

   `STATE` `idle` starts at once; `mixed` or `allocated` means the executor
   waits until the node frees up.
3. Start the executor on g202: step 1 of [EXECUTOR.md](EXECUTOR.md), with
   `EXECUTOR_NODE=g202` added to the command. Then steps 2 to 4 of that page:
   wait until it is ready, export its address and record its setup. The address
   should name g202:

   ```bash
   echo "$OPENAI_COMPAT_EXECUTOR_BASE_URL"   # http://g202...:8000/v1
   ```

4. Optional: print the node's GPU driver, to help find why g202 differs, and
   note it in the to-do's experiment list. The same command on an executor on
   g101 or g102 would show whether the drivers differ.

   ```bash
   srun --jobid "$EXECUTOR_JOB" --overlap nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
   ```

5. Preview AA-LCR:

   ```bash
   DRY_RUN=1 bash adarsh-rlms/jarvis/run_suite.sh aa_lcr 20261029 | grep -c 'already finished'
   ```

   It should print `20`: only the six control runs are new. If
   `AA_LCR_RECIPE` is still set, it prints `23`; either is fine. The first line
   of the preview reads `Suite 20261029: 26 conditions` (29 with the recipe).
6. Submit AA-LCR and follow it:

   ```bash
   bash adarsh-rlms/jarvis/run_suite.sh aa_lcr 20261029
   tail -f "$PROJECT_LOG_DIR/rlms-client-gpu-<suite job id>.out"
   ```

   Wait until the log ends with `All 26 conditions complete` (29 if
   `AA_LCR_RECIPE` is set).
7. Preview MRCR:

   ```bash
   DRY_RUN=1 bash adarsh-rlms/jarvis/run_suite.sh mrcr 20261029 | grep -c 'already finished'
   ```

   It should print `27`: only the three `bm25_recipe` runs are new. If
   `MRCR_RECIPE` is still set, it prints `31`.
8. Submit MRCR:

   ```bash
   bash adarsh-rlms/jarvis/run_suite.sh mrcr 20261029
   ```

9. While MRCR runs, regrade the new AA-LCR runs with Luna. This is a CPU job
   that does not use the executor:

   ```bash
   AA_LCR_GRADER=luna DRY_RUN=1 bash adarsh-rlms/jarvis/run_suite.sh aa_lcr 20261029   # preview: 29 conditions
   AA_LCR_GRADER=luna bash adarsh-rlms/jarvis/run_suite.sh aa_lcr 20261029
   tail -f "$PROJECT_LOG_DIR/rlms-client-<job id>.out"
   ```

   The 23 runs Luna has already graded print `grade: already done`. The log
   ends with `All 29 conditions complete`.
10. Stop the executor, `scancel "$EXECUTOR_JOB"`, once the MRCR log reaches
    `Visibility table`: the table does not use the executor. The log ends with
    `All 30 conditions complete` (34 if `MRCR_RECIPE` is set).
11. Rerun the intervals, from `~/adarsh-rlms`. Each takes a minute or two:

    ```bash
    .venv/bin/python -m aa_lcr.intervals benchmark_artifacts/aa_lcr/suites/20261029
    .venv/bin/python -m mrcr_v2.intervals benchmark_artifacts/mrcr_v2/suites/20261029
    ```

12. Commit and push both suite folders, as for the earlier runs.

To resume after an interruption, submit the same command again. Finished runs
and grades are skipped. Do not `git pull` until every job above has ended: the
runs read the code as they start.

## Results

`results-gpt-5.6-luna.md` (AA-LCR) and `results.md` (MRCR) gain one row per
control. Their policy column reads, for example, `position, 1024-token chunks,
1 neighbour each side, fill packing, document floor`. On AA-LCR, compare the
`documents covered` column with `upgraded`: with the floor, both should cover
nearly every document.

The comparisons that decide the paper's lead and headline are in
`intervals-gpt-5.6-luna.md` and `intervals.md`:

| Row | Question it answers |
| --- | --- |
| `upgraded` against `position_recipe` | Does ranking by the question help, with structure, chunking and coverage held equal? |
| `bm25_recipe` against `position_recipe` | The same for BM25 alone |
| `upgraded` against `bm25_recipe` | Does fusion with dense ranking add anything to BM25? |
| `position_recipe` against `position`, `bm25_recipe` against `bm25` | What the four steps add to each ranking |

Item 4 of the [workshop paper to-do](../workshop_paper_todo.md#remaining-work)
says how each outcome changes the paper.
