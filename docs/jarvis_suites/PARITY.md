# Make the conditions match across benchmarks

These 10 runs fill the gaps in the paper's conditions table, so that a policy
runs the same way on every benchmark where a run is possible and informative.
They are added to suite `20261029` on g101, like the [rerun](RERUN.md), with
one command.

| | |
| --- | --- |
| Runs | AA-LCR: `ladder1`, `ladder2`. LongBench-v2: `random1`, `random2`, `bm25_recipe`, `bm25_recipe_32k`, `bm25_recipe_128k`. MRCR: `random1`, `random2`, `bm25_recipe`. |
| Jobs | one executor pinned to g101 and one suite job, submitted together, each with an 8-hour limit. `PARITY_TIME=HH:MM:SS` sets another limit, and `EXECUTOR_NODE=g102` uses g102. |
| Time | about 5.5 hours after the executor starts: AA-LCR 0.5, LongBench-v2 2, MRCR 3 (with the visibility table). |
| Output | the existing suite folders, next to the runs they are compared with. |

## What the runs add

| Runs | Where | Result |
| --- | --- | --- |
| Random chunks, seeds 1 and 2 | MRCR 262K, LongBench-v2 64K | Seeds 0, 1 and 2 on every benchmark, as on AA-LCR. |
| Ladder rungs 1 and 2 | AA-LCR 64K | The full ladder on every benchmark. AA-LCR already has the other rungs: `fusion` below the ladder, `ladder3` (the recipe without the floor) and `upgraded` (the recipe). |
| BM25 recipe | LongBench-v2 32K, 64K and 128K; MRCR 262K | The BM25 recipe at every window of every benchmark. |

AA-LCR is held out: no setting of the recipe was chosen on its scores. Its
ladder rungs therefore run after the recipe was fixed, as an ablation, and the
recipe stays as it is whatever they show. AA-LCR is also the only benchmark on
which the recipe's steps clearly beat `fusion` at 64K (+11.0 points,
[+1.8, +21.1]), so its ladder shows which steps that gain comes from.

Not run, on purpose:

- `position_recipe` on MRCR and LongBench-v2. Each prompt there is one text, on
  which position keeps the head and the tail: in effect `middle_marked`.
- The recipe without the floor on MRCR and LongBench-v2. On one text the floor
  changes nothing, and `ladder3` already is that condition at the main window:
  it built the same prompts and gave the same answers as `ladder4`.
- `dense_7500` and `dense_question` on AA-LCR and LongBench-v2. They test the two
  MRCR settings that differ from the defaults, and the other two benchmarks use
  the defaults.

## What the command does

`jarvis/parity_runs.sh` submits the executor with the settings of
[EXECUTOR.md](EXECUTOR.md) and `EXECUTOR_NODE=g101`. It also submits a suite job
that starts once the executor does. That job:

1. waits until the executor is ready, checks that it is on g101 and records its
   setup in `.cache/serving_metadata-parity.json`;
2. runs AA-LCR (2 runs, each graded by Qwen3.6), then regrades them with Luna;
3. runs LongBench-v2 (5 runs), then MRCR (3 runs, then the results and
   visibility tables);
4. recomputes the intervals of all three suites and cancels the executor.

## Run it

1. Update the code: `git -C adarsh-rlms pull`. Check that `EXECUTOR_NODE` is not
   left over from an earlier page:

   ```bash
   env | grep EXECUTOR_NODE
   ```

   It should print nothing; otherwise `unset EXECUTOR_NODE`. The command refuses
   any node but g101 and g102, which served every reported run.
2. Check that g101 is in the partition and free:

   ```bash
   sinfo -N -p gpu-l40s -n g101
   ```

   `STATE` `idle` starts at once; `mixed` or `allocated` means the executor
   waits until the node frees up.
3. Preview it:

   ```bash
   DRY_RUN=1 bash adarsh-rlms/jarvis/parity_runs.sh > parity-preview.txt
   grep -c 'run: started' parity-preview.txt           # 10
   grep -c 'run: already finished' parity-preview.txt  # 0
   grep -c 'aa_lcr.regrade' parity-preview.txt         # 5
   ```

   The 5 regrades are Qwen3.6's grades of the two AA-LCR runs, one Luna check
   call and Luna's grades of the two runs.
4. Submit:

   ```bash
   bash adarsh-rlms/jarvis/parity_runs.sh
   ```

   It prints two `Submitted batch job` lines: the executor, then the suite job.
5. Follow the suite job:

   ```bash
   tail -f "$PROJECT_LOG_DIR/rlms-client-gpu-<suite job id>.out"
   ```

   Once the executor has loaded the model, the log shows
   `[PARITY] Executor ready at http://g101...`. The log ends with
   `[PARITY] All runs, grades and intervals complete`, and the executor has
   been cancelled.
6. Check the runs (below), then commit and push the three suite folders, as for
   the earlier runs.

The suite job waits for the executor for as long as the executor job is queued
or loading the model, and stops at once if the executor job ends first.
Whenever the suite job stops, it cancels the executor.

To resume after an interruption, submit the same command again. It starts a new
executor; finished runs and grades are skipped. Do not `git pull` until both
jobs have ended: the runs read the code as they start.

A resumed job moves the folder of each unfinished run or grade aside, to
`<folder>.incomplete-<time>`, before it starts that run or grade again. Those
folders should not be committed. In MRCR they also reach the tables: the
results table lists the unfinished run and the intervals stop on it, so a
resumed job that retried an MRCR run ends with `Failed: MRCR intervals`.
Submitting again does not change that. Once the job has ended, list the
folders:

```bash
cd adarsh-rlms
find benchmark_artifacts/*/suites/20261029 -name '*.incomplete-*' -prune -print
```

Delete them, and if one was under `mrcr_v2`, rebuild MRCR's two tables:

```bash
find benchmark_artifacts/*/suites/20261029 -name '*.incomplete-*' -prune -exec rm -rf {} +
.venv/bin/python -m mrcr_v2.results_table benchmark_artifacts/mrcr_v2/suites/20261029
.venv/bin/python -m mrcr_v2.intervals benchmark_artifacts/mrcr_v2/suites/20261029
```

## Check the runs

Every new run should name g101 in its manifest. This prints the manifests that
do not, so it should print nothing:

```bash
cd adarsh-rlms/benchmark_artifacts
grep -L '"executor_base_url": "http://g101' \
  aa_lcr/suites/20261029/hybrid/20261029_ladder{1,2}/manifest.json \
  longbench_v2/suites/20261029/longbench_v2/20261029_{random1,random2,bm25_recipe,bm25_recipe_32k,bm25_recipe_128k}/manifest.json \
  mrcr_v2/suites/20261029/hybrid/20261029_{random1,random2,bm25_recipe}/manifest.json
```

The new AA-LCR and LongBench-v2 runs should also have loaded the same chat
template as the rerun and the `ladder3` runs. Unlike MRCR, these two benchmarks
do not pin the tokenizer's revision, so a newer template on Hugging Face would
be loaded instead. This should print nothing too:

```bash
grep -L '"executor_chat_template_sha256": "1b836156' \
  aa_lcr/suites/20261029/hybrid/20261029_ladder{1,2}/manifest.json \
  longbench_v2/suites/20261029/longbench_v2/20261029_{random1,random2,bm25_recipe,bm25_recipe_32k,bm25_recipe_128k}/manifest.json
```

A run it lists loaded another template: find out what changed before comparing
that run with the others.

The intervals tables gain a row for each new run against `middle`, and these
comparisons:

- AA-LCR, in `intervals-gpt-5.6-luna.md`: `ladder1` against `fusion`, `ladder2`
  against `ladder1` and `ladder3` against `ladder2`.
- LongBench-v2 at each window and MRCR at 262K, in `intervals.md`:
  `bm25_recipe` against `bm25` and `upgraded` against `bm25_recipe`.

Item 7 of the [workshop paper to-do](../workshop_paper_todo.md#remaining-work)
lists what changes in the paper afterwards.
