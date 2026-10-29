# Workshop suites on Jarvis

The workshop experiments run as three suites, one per benchmark. A suite runs
every condition of its benchmark one after another as a single Slurm job and
ends by writing a results table.

## Pages

1. [SETUP.md](SETUP.md): once. Code, Python environments, keys and shell
   settings.
2. [EXECUTOR.md](EXECUTOR.md): before each suite. Start the model server and
   record its setup.
3. One page per suite, in the order to run them:
   1. [LONGBENCH_V2.md](LONGBENCH_V2.md): the shortest; run it first to check
      the setup.
   2. [AA_LCR.md](AA_LCR.md)
   3. [MRCR_V2.md](MRCR_V2.md)

| Suite        | Name in commands | Questions per run          | Runs now (+ later) | Time   | Scoring                                                  |
| ------------ | ---------------- | -------------------------- | ------------------ | ------ | -------------------------------------------------------- |
| LongBench-v2 | `longbench`      | 221                        | 17 (+3)            | ~6 h   | answer letter checked during the run                     |
| AA-LCR       | `aa_lcr`         | 100                        | 15 (+3)            | ~4 h   | graded by the executor after each run; by GPT-5.6 Luna at the end |
| MRCR v2      | `mrcr`           | 120 (30 per conversation)  | 20 (+4)            | ~14 h  | scored during the run                                    |

The times are estimates; the first suite of each benchmark measures them. The
runs added later use the upgraded recipe, which is chosen from the LongBench-v2
and MRCR results.

## Overall plan

1. Setup, once.
2. LongBench-v2 suite.
3. AA-LCR suite.
4. MRCR suite with 30 questions per conversation.
5. Choose the upgraded recipe, then add its runs to each suite.
6. Regrade every AA-LCR run with GPT-5.6 Luna.
7. If the MRCR suite went well: the full MRCR suite.

Run one suite at a time: they all use the same executor.

## How a suite runs

Each suite uses two Slurm jobs:

- **The executor job** serves the model, Qwen3.6-35B-A3B, with vLLM on 4 L40S
  GPUs. It stays up until you cancel it or it reaches its 24-hour limit.
- **The suite job** is submitted by `jarvis/run_suite.sh`. It gets one L40S GPU
  for the embedding model, 96 GB of memory and 24 hours. It waits up to 30
  minutes for the executor, then runs each condition in turn, sending its
  requests to the executor.

Both jobs keep running after you log out.

## Commands every suite uses

Run every command on the Jarvis login node, from the directory that contains
the `adarsh-rlms` folder. These pages assume that is your home directory. The
examples use the suite ID `20261029`. A suite ID can be any name; reusing it
continues the same suite.

| Purpose | Command |
| --- | --- |
| Preview: print each run's command, submit nothing | `DRY_RUN=1 bash adarsh-rlms/jarvis/run_suite.sh longbench 20261029` |
| Submit the suite | `bash adarsh-rlms/jarvis/run_suite.sh longbench 20261029` |
| Submit only some conditions | `ONLY=middle,dense bash adarsh-rlms/jarvis/run_suite.sh longbench 20261029` |
| List your jobs | `squeue -u "$USER"` |
| Follow the suite job | `tail -f "$PROJECT_LOG_DIR/rlms-client-gpu-<suite job id>.out"` |
| Cancel a job | `scancel <job id>` |

Submitting prints `Submitted batch job <id>`; that number is the job ID. The
suite job's log shows each step:

```text
[LONGBENCH SUITE] Suite 20261029: 17 conditions, executor http://<node>:8000/v1
[LONGBENCH SUITE] [1/17] full: direct --context-window-tokens 262144 --max-input-tokens 262136
[LONGBENCH SUITE]   run: started 14:02 (log: benchmark_artifacts/longbench_v2/suites/20261029/logs/full.log)
[LONGBENCH SUITE]   run: done in 0h31m
...
[LONGBENCH SUITE] All 17 conditions complete
```

Each run also writes its own log to the suite folder's `logs/`.

## Resuming and failures

To resume, submit the same command again, with the same variables. Finished
runs are skipped. An unfinished run's folder is renamed
`*.incomplete-<time>` and the run starts again. The job log prints the command
to use, after `Resume or repeat with:`.

A failed run does not stop the others. The end of the job log lists the failed
runs, and the run's own log in `logs/` shows the cause. Fix it and resubmit.

If the executor job stops partway (its 24-hour limit, or a crash), the
remaining runs fail. Start a new executor as described in
[EXECUTOR.md](EXECUTOR.md), then resubmit the suite.

Do not `git pull` while a suite job is running. The runs read the code as they
start, so a pull would change code partway through the suite.
