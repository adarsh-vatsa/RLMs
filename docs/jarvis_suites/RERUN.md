# Rerun the g202 runs on g101

g202 answers identical prompts differently from g101 and g102 (see
[Why one node](CONTROLS.md#why-one-node)). This page reruns every g202 run of
suite `20261029` on g101, together with the controls, with one command. Every
comparison in the paper then uses g101 or g102, which agree byte for byte.

| | |
| --- | --- |
| Reruns | 17 g202 runs. AA-LCR: `bm25_32k`, `bm25_80k`, `upgraded` at 64K, 32K and 80K. LongBench-v2: `bm25_32k`, `bm25_128k`, `upgraded` at 64K, 32K and 128K. MRCR: `bm25` at 32K, 64K and 128K, `upgraded` at 262K, 32K, 64K and 128K. |
| Controls | the 9 runs of [CONTROLS.md](CONTROLS.md), which are compared with the recipe runs and so go to the same node. |
| Jobs | one executor pinned to g101 and one suite job, submitted together, each with a 12-hour limit. `RERUN_TIME=HH:MM:SS` sets another; a limit that ends before a reserved job starts lets Slurm start both jobs in the gap. |
| Time | about 10 hours after the executor starts: AA-LCR 3.5, LongBench-v2 2, MRCR 4 (with the visibility table). |
| Output | the existing suite folders. Each g202 run moves, with its grades and logs, to a sibling folder `20261029_g202`. |

Decided on 2026-11-10, before the rerun: the g101 runs replace the g202 runs in
the paper, whatever their results. The g202 runs stay in `20261029_g202` for
the record but are not reported.

## What the command does

`jarvis/rerun_g202_runs.sh` submits the executor with the settings of
[EXECUTOR.md](EXECUTOR.md) and `EXECUTOR_NODE=g101`. It also submits a suite job
that starts once the executor does. That job:

1. waits until the executor is ready, checks that it is on g101 and records its
   setup in `.cache/serving_metadata-rerun.json`;
2. moves each run whose manifest names g202, its Qwen3.6 and Luna grades and its
   logs to `<suite>_g202`;
3. runs AA-LCR (5 reruns and 6 controls, each graded by Qwen3.6), then starts
   the Luna regrade of those 11 runs alongside the next suites;
4. runs LongBench-v2 (5 reruns), then MRCR (7 reruns and 3 controls, then the
   visibility table);
5. recomputes the intervals of all three suites and cancels the executor.

## Run it

1. Update the code: `git -C adarsh-rlms pull`. Check that `EXECUTOR_NODE` is not
   left over from [CONTROLS.md](CONTROLS.md):

   ```bash
   env | grep EXECUTOR_NODE
   ```

   It should print nothing; otherwise `unset EXECUTOR_NODE`. The command sets
   every suite variable itself, so other leftovers do not matter.
2. Check that g101 is in the partition and free:

   ```bash
   sinfo -N -p gpu-l40s -n g101
   ```

   `STATE` `idle` starts at once; `mixed` or `allocated` means the executor
   waits until the node frees up.
3. Preview it:

   ```bash
   DRY_RUN=1 bash adarsh-rlms/jarvis/rerun_g202_runs.sh > rerun-preview.txt
   grep -c 'RERUN\] moving' rerun-preview.txt          # 54
   grep -c 'run: already finished' rerun-preview.txt   # 17
   grep -c 'run: started' rerun-preview.txt            # 9
   ```

   The 54 moves are the 17 runs and their logs, plus the Qwen3.6 and Luna grades
   of the 5 AA-LCR runs. A preview moves nothing, so the 17 runs still show as
   finished; the job reruns them after moving them.
4. Submit:

   ```bash
   bash adarsh-rlms/jarvis/rerun_g202_runs.sh
   ```

   It prints two `Submitted batch job` lines: the executor, then the suite job.
5. Follow the suite job:

   ```bash
   tail -f "$PROJECT_LOG_DIR/rlms-client-gpu-<suite job id>.out"
   ```

   Once the executor has loaded the model, the log shows
   `[RERUN] Executor ready at http://g101...`. The log ends with
   `[RERUN] All runs, grades and intervals complete`, and the executor has
   been cancelled.
6. Commit and push the three suite folders and the three `20261029_g202`
   folders, as for the earlier runs.

The suite job waits for the executor for as long as the executor job is queued
or loading the model, and stops at once if the executor job ends first.
Whenever the suite job stops, it cancels the executor.

To resume after an interruption, submit the same command again. It starts a new
executor; finished runs and grades are skipped, and runs already moved are not
moved again. Do not `git pull` until both jobs have ended: the runs read the
code as they start.

## Check the rerun

At the main budget, `upgraded` builds the same prompts as `ladder4`. On g101 it
should give the same answers on MRCR and LongBench-v2, so the `upgraded` against
`ladder4` row of each `intervals.md` should show `gained` 0 and `lost` 0. The
g202 runs showed 2 and 2 on MRCR, and 4 and 2 on LongBench-v2.
