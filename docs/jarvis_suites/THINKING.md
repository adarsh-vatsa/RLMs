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
  allowance of 81,920 tokens for the reasoning and the answer, set by `THINKING_ALLOWANCE`
  and chosen from the pilot
  ([Thinking budget](../extension_todo.md#thinking-budget-decided-2026-11-25)).
  - MRCR runs at 32K, 64K and 128K only: 262K would leave 4,096 tokens to think and
    answer.
  - AA-LCR's full-context run uses an input budget of 262,144 minus the allowance. Its
    longest prompt is 122K, so every prompt is still sent whole.
  - On LongBench-v2, the longest full prompts leave less than the allowance: 46 of the
    221, 13 of them less than 32,768. Those requests' output is capped at what the window
    leaves, and `max_tokens` in `execution.jsonl` records each request's limit.
- **Decoding.** Qwen's thinking settings for precise tasks: temperature 0.6, top_p 0.95,
  top_k 20, min_p 0 and presence penalty 0. Each question's seed comes from its ID, so every
  policy draws the same seed for a question. The repeats `middle_repeat2/3` and
  `upgraded_repeat2/3` use `--sampling-seed 2` and `3` to measure how much a sampled
  run varies.
- **One request at a time, on g101 or g102.** A greedy rerun repeats its answers on the
  same vLLM version and batch size, on g101 and g102 alike; the checks after the runs show
  whether seeded sampling does too. The executor is the one in [EXECUTOR.md](EXECUTOR.md);
  thinking is set per request.
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

## 1. Pilot (done 2026-11-26)

Suite `20261128_think_pilot` ran `full`, `middle` and `upgraded` on 20 AA-LCR questions,
`middle_128k` and `upgraded_128k` on 12 MRCR questions and `middle` and `upgraded` on 20
LongBench-v2 questions, on g101, in 1h12m once the executor was up:

```bash
EXECUTOR_NODE=g101 THINKING_ALLOWANCE=81920 MAX_ROWS=20 MRCR_ROWS_PER_SOURCE=3 \
AA_LCR_ONLY=full,middle,upgraded MRCR_ONLY=middle_128k,upgraded_128k LONGBENCH_ONLY=middle,upgraded \
  bash adarsh-rlms/jarvis/thinking_runs.sh 20261128_think_pilot aa_lcr,mrcr,longbench
```

- Mean reasoning ran from 2,337 to 5,499 tokens per question (medians 1,432 to 3,125),
  and generation took 20–40 s per question. No answer reached the 81,920-token cap and no
  request was retried. LongBench-v2 had no invalid answers, so the A–D constraint applies
  after the reasoning.
- LongBench-v2's reasoning has a long tail: 5 of its 40 answers took 9,000 to 35,590
  tokens, much of it re-checking an answer the model had already chosen; each still ended
  with an answer.
- The allowance stays at 81,920. One answer went past 32,768 (35,590 tokens, on
  LongBench-v2 `upgraded`), which the smaller cap would have cut off and scored zero; the
  larger cap costs time only on the answers that need it.
- Every question's prompt matched its thinking-off run in suite `20261029`, in length and
  evidence.
- The full runs ask every pilot question again, with the same seed and the same allowance,
  so they also show whether a seeded run repeats itself (section 3).

## 2. Full runs

Two commands, one per node, with the same suite ID; submit both at once. Each benchmark's
runs stay on one node, so the comparisons within a benchmark share a node. At the pilot's
time per question and the conditions' prompt lengths, LongBench-v2 takes about 17 hours
on g101, and AA-LCR (about 9) and MRCR (about 10) take about 20 on g102 with grading, so
each fits in one 24-hour job if the other conditions reason about as long as the pilot's.

First check that no pilot setting is left in the shell. This should print nothing; a
leftover `MAX_ROWS` would make 20-question runs, which the suite would then skip as
finished:

```bash
env | grep -E '^(MAX_ROWS|MRCR_ROWS_PER_SOURCE|AA_LCR_ONLY|MRCR_ONLY|LONGBENCH_ONLY)='
```

Then:

```bash
EXECUTOR_NODE=g101 THINKING_ALLOWANCE=81920 \
  bash adarsh-rlms/jarvis/thinking_runs.sh 20261130_think longbench
EXECUTOR_NODE=g102 THINKING_ALLOWANCE=81920 \
  bash adarsh-rlms/jarvis/thinking_runs.sh 20261130_think aa_lcr,mrcr
```

- Each suite runs its planned conditions with the headline runs first (P1, then P2, then
  P3 in the CSV). AA-LCR is graded by the executor after each run and by Luna at the end
  of the suite. The job then writes the results tables and intervals.
- Both jobs end after 24 hours. If one ends before its suites finish, or a run fails,
  check with `squeue -u "$USER"` that both of that node's jobs have ended, then submit
  the same command again; the job log prints it after `Resume or repeat with:`. Finished
  runs and grades are skipped. A run counts as finished only once every question has an
  answer, since each run stops at its first failed request, so the run that was cut off
  starts again from its first question.
- Do not `git pull` while a suite job is running: each run reads the code as it starts.
  A pull while both jobs are pending is safe.
- A resumed job moves the folder of each unfinished run or grade aside, to
  `<folder>.incomplete-<time>`. Do not commit those folders. MRCR's also reach its results
  table and stop its intervals, so a resumed job that retried an MRCR run ends with
  `Failed: MRCR intervals`. Once the job has ended, list and delete them, and if one was
  under `mrcr_v2`, rebuild MRCR's two tables:

  ```bash
  cd adarsh-rlms
  find benchmark_artifacts/*/suites/20261130_think -name '*.incomplete-*' -prune -print
  find benchmark_artifacts/*/suites/20261130_think -name '*.incomplete-*' -prune -exec rm -rf {} +
  .venv/bin/python -m mrcr_v2.results_table benchmark_artifacts/mrcr_v2/suites/20261130_think
  .venv/bin/python -m mrcr_v2.intervals benchmark_artifacts/mrcr_v2/suites/20261130_think
  ```
- Follow a job with `squeue -u "$USER"` and
  `tail -f "$PROJECT_LOG_DIR/rlms-client-gpu-<job id>.out"`; each run's own log is in
  the suite folder's `logs/`.
- To run only some conditions, set `AA_LCR_ONLY`, `MRCR_ONLY` or `LONGBENCH_ONLY` to
  condition names, in the order to run them.

## 3. After the runs

One script on the login node reads every finished run of the suite. For each run it
prints the mean reasoning length and how many answers went past 32,768 tokens or reached
the cap. It then checks each question's prompt against the thinking-off run in suite
`20261029`, with the same length and the same evidence, and for the pilot's questions,
whether the answer and the reasoning came out as in the pilot:

```bash
adarsh-rlms/.venv/bin/python - <<'EOF'
import glob, json, os, re
NEW, OLD, PILOT = "20261130_think", "20261029", "20261128_think_pilot"

def load(run, name="execution.jsonl"):
    path = os.path.join(run, name)
    return {row["case_id"]: row for row in map(json.loads, open(path))} if os.path.exists(path) else {}

for run in sorted(os.path.dirname(path) for path in glob.glob(f"adarsh-rlms/benchmark_artifacts/*/suites/{NEW}/*/*/execution.jsonl")):
    rows = load(run)
    print(run.split("benchmark_artifacts/")[1].replace(f"/suites/{NEW}", ""), f"{len(rows)} answers,",
          f"mean reasoning {sum(row.get('reasoning_tokens', 0) for row in rows.values()) / len(rows):,.0f} tokens,",
          f"past 32,768: {sum(row['output_tokens'] > 32768 for row in rows.values())},",
          f"at the cap: {sum(row['finish_reason'] == 'length' for row in rows.values())}")
    off = load(re.sub(r"_repeat\d$", "", run.replace(NEW, OLD)))
    if off:
        same = sum(case in off and row["final_rendered_input_tokens"] == off[case]["final_rendered_input_tokens"]
                   and row["selected_evidence_ranges"] == off[case]["selected_evidence_ranges"]
                   for case, row in rows.items())
        print(f"  prompts as in {OLD}: {same} of {len(rows)}")
    pilot = load(run.replace(NEW, PILOT))
    if pilot:
        before, now = load(run.replace(NEW, PILOT), "reasoning.jsonl"), load(run, "reasoning.jsonl")
        answers = sum(row["prediction"] == rows[case]["prediction"] for case, row in pilot.items())
        reasoning = sum(row["reasoning"] == now.get(case, {}).get("reasoning") for case, row in before.items())
        print(f"  pilot questions asked again: {answers} of {len(pilot)} answers and {reasoning} of {len(before)} reasoning texts identical")
EOF
```

- **Prompts** should match on every question: `100 of 100` on AA-LCR, `120 of 120` on
  MRCR and `221 of 221` on LongBench-v2. Any other count means thinking changed the
  prompts, so thinking on and off were not compared on identical inputs; find out why
  before reporting the runs. A run whose thinking-off counterpart does not exist yet, such
  as `no_context` before section 4, gets no prompts line.
- **Pilot questions** should repeat in full: 20 of 20 on AA-LCR and LongBench-v2, 12 of
  12 on MRCR. LongBench-v2 ran on g101 as the pilot did, so it tests a seeded rerun on
  the same node; AA-LCR and MRCR ran on g102, so they also test a rerun on another node.
  Identical reasoning is the stronger test, since the same short answer can follow
  different reasoning. If the reruns differ, report the runs as single samples without a
  reproducibility claim; the comparisons do not depend on it, since each benchmark ran on
  one node.
- **Past 32,768 and at the cap** give the allowance hits by policy for the paper.

## 4. No-context runs without thinking

Qwen3.6 with thinking off needs only the two no-context runs, added to suite `20261029`.
Run them once g101 or g102 is free; the two give the same greedy answers byte for byte.
Start an executor as in [EXECUTOR.md](EXECUTOR.md) with `EXECUTOR_NODE=g101` (or `g102`),
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

Each takes a few minutes: without the documents, the prompts are short. Then run the
script in section 3 again to check the thinking `no_context` prompts against them.
