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
s/question**. On LongBench-v2, **invalid answers** should equal the **output limit hit**
count: an answer cut off at the cap has no letter, so any other invalid answer means the
A–D constraint does not apply after the reasoning.

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

Thinking must leave the prompts as they were. To check that each pilot question was sent
the prompt of its thinking-off run in suite `20261029`, with the same length and the same
evidence:

```bash
adarsh-rlms/.venv/bin/python - <<'EOF'
import glob, json
NEW, OLD = "20261128_think_pilot", "20261029"
for path in sorted(glob.glob(f"adarsh-rlms/benchmark_artifacts/*/suites/{NEW}/*/*/execution.jsonl")):
    before = {row["case_id"]: row for row in map(json.loads, open(path.replace(NEW, OLD)))}
    rows = [json.loads(line) for line in open(path)]
    same = sum(row["final_rendered_input_tokens"] == before[row["case_id"]]["final_rendered_input_tokens"]
               and row["selected_evidence_ranges"] == before[row["case_id"]]["selected_evidence_ranges"]
               for row in rows)
    print(path.split("benchmark_artifacts/")[1].removesuffix("/execution.jsonl"), f"{same} of {len(rows)} prompts as in {OLD}")
EOF
```

Every line should read `20 of 20` (`12 of 12` on MRCR). Any other count means thinking
changed the prompts, and thinking on and off would no longer be compared on identical
inputs; find out why before the full runs.

To check that a seeded rerun repeats itself, run `middle` again on the same node under
another suite ID, then compare the answers and the reasoning. Identical reasoning is the
stronger test, since the same short answer can follow different reasoning:

```bash
EXECUTOR_NODE=g101 THINKING_ALLOWANCE=81920 MAX_ROWS=5 AA_LCR_ONLY=middle \
  bash adarsh-rlms/jarvis/thinking_runs.sh 20261128_think_repeat aa_lcr

adarsh-rlms/.venv/bin/python - <<'EOF'
import glob, json
def saved(suite, name, field):
    path, = glob.glob(f"adarsh-rlms/benchmark_artifacts/aa_lcr/suites/{suite}/direct/*_middle/{name}")
    return {row["case_id"]: row[field] for row in map(json.loads, open(path))}
for label, name, field in (("answers", "execution.jsonl", "prediction"), ("reasoning texts", "reasoning.jsonl", "reasoning")):
    first, again = saved("20261128_think_pilot", name, field), saved("20261128_think_repeat", name, field)
    print(sum(first.get(case) == text for case, text in again.items()), "of", len(again), label, "identical")
EOF
```

Both lines should read `5 of 5`.

## 2. No-context runs without thinking

Qwen3.6 with thinking off needs only the two no-context runs, added to suite `20261029`
on g101. Run them before the full runs, which hold g101 for days; an executor started
after those would wait behind them. They can also run on g102 during the pilot, since
g101 and g102 give the same greedy answers byte for byte. Start an executor as in
[EXECUTOR.md](EXECUTOR.md) with `EXECUTOR_NODE=g101` (or `g102`), point the suites at it and record
its setup (steps 2–4), then:

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

## 3. Full runs

Two commands, one per node, with the same suite ID and the allowance the pilot chose.
Each benchmark's runs stay on one node, so the comparisons within a benchmark share a
node. At the [plan's rates](../extension_todo.md#feasibility-numbers-2026-11-25) and 5K
reasoning tokens per question, LongBench-v2 takes about 21 hours, AA-LCR 15 and MRCR 12
(about twice that at 10K), so LongBench-v2 gets a node to itself. If the pilot's
generation s/question differs much between benchmarks, split them by its numbers
instead: any split that keeps each benchmark on one node works.

First check that no pilot setting is left in the shell. This should print nothing; a
leftover `MAX_ROWS` would make 20-question runs, which the suite would then skip as
finished:

```bash
env | grep -E '^(MAX_ROWS|MRCR_ROWS_PER_SOURCE|AA_LCR_ONLY|MRCR_ONLY|LONGBENCH_ONLY)='
```

Then:

```bash
EXECUTOR_NODE=g101 THINKING_ALLOWANCE=<32768 or 81920> \
  bash adarsh-rlms/jarvis/thinking_runs.sh 20261130_think longbench
EXECUTOR_NODE=g102 THINKING_ALLOWANCE=<32768 or 81920> \
  bash adarsh-rlms/jarvis/thinking_runs.sh 20261130_think aa_lcr,mrcr
```

- Each suite runs its planned conditions with the headline runs first (P1, then P2, then
  P3 in the CSV). AA-LCR is graded by the executor after each run and by Luna at the end
  of the suite. The job then writes the results tables and intervals.
- A node runs its benchmarks one after the other, so on g102 MRCR's headline runs start
  only after all of AA-LCR's, about 15 hours in at 5K reasoning tokens. To put both
  benchmarks' headline runs first, submit them alone, then the command above once that
  job ends:

  ```bash
  EXECUTOR_NODE=g102 THINKING_ALLOWANCE=<32768 or 81920> AA_LCR_ONLY=full,middle,upgraded \
  MRCR_ONLY=middle_128k,upgraded_128k,oracle_128k \
    bash adarsh-rlms/jarvis/thinking_runs.sh 20261130_think aa_lcr,mrcr
  ```

- Both jobs end after 24 hours. Submit the same command again to continue: finished runs
  and grades are skipped, and the run in progress starts again. The job log prints the
  command after `Resume or repeat with:`.
- Follow a job with `squeue -u "$USER"` and
  `tail -f "$PROJECT_LOG_DIR/rlms-client-gpu-<job id>.out"`; each run's own log is in
  the suite folder's `logs/`.
- To run only some conditions, set `AA_LCR_ONLY`, `MRCR_ONLY` or `LONGBENCH_ONLY` to
  condition names, in the order to run them.
