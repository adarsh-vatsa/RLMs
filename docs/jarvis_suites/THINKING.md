# Thinking runs (Qwen3.6)

Qwen3.6-35B-A3B with thinking on, on the conditions planned for every new model in
[extension_experiments.csv](../extension_experiments.csv), and the no-context baseline for
every model. The plan and its decisions are in [extension_todo.md](../extension_todo.md).
One command per node submits the executor and a suite job, as the
[rerun](RERUN.md) and [parity](PARITY.md) commands did.

## What changes with thinking

- **The same prompts.** Each condition keeps the input budget of its thinking-off run in
  suite `20261029`, so the prompts are the same and thinking on and off are compared on
  identical inputs. Every run is served the whole 262,144-token window, with an output
  allowance for the reasoning and the answer, set by `THINKING_ALLOWANCE`: 81,920 tokens
  in the pilot, then 32,768 or 81,920 as the pilot decides
  ([Thinking budget](../extension_todo.md#thinking-budget-decided-2026-11-25)).
  - MRCR runs at 32K, 64K and 128K only: 262K would leave 4,096 tokens to think and
    answer.
  - AA-LCR's full-context run uses an input budget of 262,144 minus the allowance. Its
    longest prompt is 122K, so every prompt is still sent whole.
  - On LongBench-v2, the longest full prompts leave less than the allowance: 13 of the 221
    at 32,768 and 46 at 81,920. Those requests' output is capped at what the window
    leaves, and `max_tokens` in `execution.jsonl` records each request's limit.
- **Decoding.** Qwen's thinking settings for precise tasks: temperature 0.6, top_p 0.95,
  top_k 20, min_p 0 and presence penalty 0. Each question's seed comes from its ID, so every
  policy draws the same seed for a question. The repeats `middle_repeat2/3` and
  `upgraded_repeat2/3` use `--sampling-seed 2` and `3` to measure how much a sampled
  run varies.
- **One request at a time, on g101 or g102.** A rerun repeats its answers only on the
  same node, vLLM version and batch size. The executor is the one in
  [EXECUTOR.md](EXECUTOR.md); thinking is set per request.
- **What each run adds.** `reasoning.jsonl` holds the reasoning text of each answer, and
  `execution.jsonl` adds `reasoning_tokens` and `max_tokens`. The results tables label
  each run with the window its input budget belongs to and add a reasoning-tokens
  column (and, on LongBench-v2, output limit hits).
- **No context.** `no_context` sends the prompt with no source text: the instructions
  and the question, with the four options on LongBench-v2. AA-LCR's document block is
  empty, and LongBench-v2 shows the omission marker where its text was, as a prompt
  that keeps nothing would. It measures what the model answers from what it already
  knows.

## Before you start

Pull the code (`git -C adarsh-rlms pull`); nothing new needs installing.
`adarsh-rlms/.env` needs `OPENROUTER_API_KEY` for the Luna regrade, as in
[SETUP.md](SETUP.md).

## 1. Pilot

A few questions per benchmark, at an 81,920-token allowance, to measure reasoning length
and time before the full runs. One command on g101; it takes about 1–3 hours once the
executor is up, depending on how long the model thinks:

```bash
EXECUTOR_NODE=g101 THINKING_ALLOWANCE=81920 MAX_ROWS=20 MRCR_ROWS_PER_SOURCE=3 \
AA_LCR_ONLY=full,middle,upgraded MRCR_ONLY=middle_128k,upgraded_128k LONGBENCH_ONLY=middle,upgraded \
  bash adarsh-rlms/jarvis/thinking_runs.sh 20261128_think_pilot aa_lcr,mrcr,longbench
```

Use a pilot suite ID that the full runs will not reuse: the pilot's runs hold 20
questions, and a suite skips finished runs. Each suite's results table
(`benchmark_artifacts/<benchmark>/suites/20261128_think_pilot/results*.md`) gives the
mean **reasoning tokens** per question, which sets the full runs' hours
([extension_todo.md](../extension_todo.md#experiments-proposal)), and the **generation
s/question**. On LongBench-v2, **invalid answers** should be 0 if the A–D constraint
applies after the reasoning.

The allowance turns on the answers that go past 32,768 tokens, which the smaller cap would
cut off, and those that reach 81,920. To count them per run:

```bash
adarsh-rlms/.venv/bin/python - <<'EOF'
import glob, json
for path in sorted(glob.glob("adarsh-rlms/benchmark_artifacts/*/suites/20261128_think_pilot/*/*/execution.jsonl")):
    rows = [json.loads(line) for line in open(path)]
    print(path.split("/suites/")[1].removesuffix("/execution.jsonl"), f"{len(rows)} answers,",
          f"mean reasoning {sum(row['reasoning_tokens'] for row in rows) / len(rows):,.0f} tokens,",
          f"past 32,768: {sum(row['output_tokens'] > 32768 for row in rows)},",
          f"at the cap: {sum(row['finish_reason'] == 'length' for row in rows)}")
EOF
```

To check that a seeded rerun repeats its answers, run `middle` again on the same node
under another suite ID, then compare:

```bash
EXECUTOR_NODE=g101 THINKING_ALLOWANCE=81920 MAX_ROWS=5 AA_LCR_ONLY=middle \
  bash adarsh-rlms/jarvis/thinking_runs.sh 20261128_think_repeat aa_lcr

adarsh-rlms/.venv/bin/python - <<'EOF'
import glob, json
def answers(suite):
    path, = glob.glob(f"adarsh-rlms/benchmark_artifacts/aa_lcr/suites/{suite}/direct/*_middle/execution.jsonl")
    return {row["case_id"]: row["prediction"] for row in map(json.loads, open(path))}
first, again = answers("20261128_think_pilot"), answers("20261128_think_repeat")
print(sum(first[case] == again[case] for case in again), "of", len(again), "answers identical")
EOF
```

## 2. Full runs

Two commands, one per node, with the same suite ID and the allowance the pilot chose.
Each benchmark's runs stay on one node, so the comparisons within a benchmark share a
node:

```bash
EXECUTOR_NODE=g101 THINKING_ALLOWANCE=<32768 or 81920> \
  bash adarsh-rlms/jarvis/thinking_runs.sh 20261130_think aa_lcr,longbench
EXECUTOR_NODE=g102 THINKING_ALLOWANCE=<32768 or 81920> \
  bash adarsh-rlms/jarvis/thinking_runs.sh 20261130_think mrcr
```

- Each suite runs its planned conditions with the headline runs first (P1, then P2, then
  P3 in the CSV). AA-LCR is graded by the executor after each run and by Luna at the end
  of the suite. The job then writes the results tables and intervals.
- Both jobs end after 24 hours. Submit the same command again to continue: finished runs
  and grades are skipped, and the run in progress starts again. The job log prints the
  command after `Resume or repeat with:`.
- Follow a job with `squeue -u "$USER"` and
  `tail -f "$PROJECT_LOG_DIR/rlms-client-gpu-<job id>.out"`; each run's own log is in
  the suite folder's `logs/`.
- To run only some conditions, set `AA_LCR_ONLY`, `MRCR_ONLY` or `LONGBENCH_ONLY` to
  condition names, in the order to run them.

## 3. No-context runs without thinking

Qwen3.6 with thinking off needs only the two no-context runs, added to suite `20261029`
on g101. Start an executor as in [EXECUTOR.md](EXECUTOR.md) with `EXECUTOR_NODE=g101`,
point the suites at it and record its setup (steps 2–4), then:

```bash
ONLY=no_context bash adarsh-rlms/jarvis/run_suite.sh aa_lcr 20261029
ONLY=no_context bash adarsh-rlms/jarvis/run_suite.sh longbench 20261029
```

When both jobs end, regrade the AA-LCR run with Luna and stop the executor:

```bash
AA_LCR_GRADER=luna ONLY=no_context bash adarsh-rlms/jarvis/run_suite.sh aa_lcr 20261029
scancel "$EXECUTOR_JOB"
```

Each takes a few minutes: without the documents, the prompts are short.
