# Benchmarking Q&A Log

This file tracks newcomer-style benchmarking questions and answers.

## Q0
Setup after cloning:

Recommended setup after clone:
- 1) Create and activate a virtual environment:
  python3 -m venv .venv
  source .venv/bin/activate

- 2) Install this repository package dependencies:
  python -m pip install --upgrade pip
  python -m pip install -e .

- 3) Verify NoLiMa benchmark tools in this repository:
  python nolima/run_benchmark.py --help
  python nolima/score_nolima_predictions.py --help

- 4) Set API credentials for the synthesis/evaluator models used by this repository (example):
  export ANTHROPIC_API_KEY="..."

Optional: uv-based setup (use this instead of steps 1-3 above, not in addition):
- Create and activate environment:
  uv venv .venv
  source .venv/bin/activate
- Install dependencies:
  uv pip install -e .
- Verify tools:
  uv run python nolima/run_benchmark.py --help
  uv run python nolima/score_nolima_predictions.py --help

## Q1
Question: Which benchmark is used and what kind of tasks are executed to test this project?

Answer:
- We are using the high-parity NoLiMa benchmark path implemented in this repository.
- Runner entrypoint:
  - `nolima/run_benchmark.py`
- Scorer entrypoint:
  - `nolima/score_nolima_predictions.py`
- v1 scope currently targets the standard NoLiMa needle set (`needle_set.json`) and full context-length matrix:
  - 250, 500, 1K, 2K, 4K, 8K, 16K, 32K
- Default depth sweep:
  - 26 intervals
- Purpose of this scope: preserve NoLiMa-style placement stress while evaluating this repository's SemanticCache system.

## Q2
Question: Can you elaborate on the tasks? What are they trying to test?

Answer:
- NoLiMa focuses on long-context retrieval where literal overlap between question and inserted needle is minimized.
- In this implementation, `task` is derived from question type in the needle set (for example `direct`), and each test is expanded across:
  - multiple haystacks,
  - multiple context lengths,
  - multiple depth placements.
- Combined intent:
  - Stress retrieval and reasoning when lexical shortcuts are weak.
  - Measure how accuracy degrades as context length and placement difficulty increase.

## Q3
Question: Which datasets are used to test these tasks?

Answer:
- For NoLiMa runs, input assets are staged under:
  - `benchmark_data/nolima/needlesets/`
  - `benchmark_data/nolima/haystack/rand_shuffle/`
- For development-only smoke checks in this repository, use the synthetic fixture:
  - `benchmark_fixtures/nolima/needlesets/needle_set.json`
  - `benchmark_fixtures/nolima/haystack/rand_shuffle/*.txt`

Practical rule:
- Parity-oriented runs: use NoLiMa-style needle set + haystack assets under `benchmark_data/nolima/`.
- Local dry runs: use fixture assets only for wiring/sanity checks.

## Q4
Question: Where do we get the data from? I see that in the README there is parameter that takes the path. Do we have the dataset locally ready?

Answer:
- This repository expects NoLiMa assets to be present locally under `benchmark_data/nolima/`.
- Synthetic fixture data is already available under `benchmark_fixtures/nolima/`.
- Full parity runs require you to place/download larger NoLiMa needle-set and haystack assets into:
  - `benchmark_data/nolima/needlesets/`
  - `benchmark_data/nolima/haystack/rand_shuffle/`

## Q5
Question: How can I run the script? Could you summarize the commands for me?

Answer:
- 1) Activate environment and inspect CLI:
  source .venv/bin/activate
  python nolima/run_benchmark.py --help

- 2) Local smoke run (pipeline sanity check only, not full parity reporting):
  python nolima/run_benchmark.py \
    --needle-set-path benchmark_fixtures/nolima/needlesets/needle_set.json \
    --haystack-dir benchmark_fixtures/nolima/haystack/rand_shuffle \
    --lengths 250,1K \
    --depth-intervals 3 \
    --max-samples 20 \
    --mode cache \
    --output-dir benchmark_artifacts

- 3) Full NoLiMa-style run (with staged NoLiMa assets):
  python nolima/run_benchmark.py \
    --needle-set-path benchmark_data/nolima/needlesets/needle_set.json \
    --haystack-dir benchmark_data/nolima/haystack/rand_shuffle \
    --lengths 250,500,1K,2K,4K,8K,16K,32K \
    --depth-intervals 26 \
    --mode cache \
    --output-dir benchmark_artifacts

  Warm rerun behavior (same command, same selected dataset content):
  - Re-run the same command above to reuse persistent cache state by default.

  Cold-start behavior (explicit reset only):
  python nolima/run_benchmark.py \
    --needle-set-path benchmark_data/nolima/needlesets/needle_set.json \
    --haystack-dir benchmark_data/nolima/haystack/rand_shuffle \
    --lengths 250,500,1K,2K,4K,8K,16K,32K \
    --depth-intervals 26 \
    --mode cache \
    --cache-reset \
    --output-dir benchmark_artifacts

  Optional custom cache-state root:
  python nolima/run_benchmark.py \
    --needle-set-path benchmark_data/nolima/needlesets/needle_set.json \
    --haystack-dir benchmark_data/nolima/haystack/rand_shuffle \
    --lengths 250,500,1K,2K,4K,8K,16K,32K \
    --depth-intervals 26 \
    --mode cache \
    --cache-state-root benchmark_artifacts/official_nolima/cache_state \
    --output-dir benchmark_artifacts

- 4) Score generated predictions for an existing NoLiMa run:
  ./.venv/bin/python nolima/score_nolima_predictions.py \
    --run-dir benchmark_artifacts/official_nolima/<run_id> \
    --metric contains

Artifacts are written under `benchmark_artifacts/official_nolima/<run_id>/` with `predictions.jsonl`, `bridge_rows.jsonl`, and `manifest.json` after generation, then `official_nolima_eval_report.json` and `scored_predictions.jsonl` after scoring.

Quick rule:
- Use command 3 for generation.
- Use command 4 for scoring.
- In `--mode cache`, warm reuse is automatic unless `--cache-reset` is provided.

## Q6
Question: What is the bridge generation step? Also, what's the difference between command 2 and 4 you mentioned?

Answer:
- Bridge generation step means:
  - Expand NoLiMa needle-set tests into concrete samples across haystacks, lengths, and depth placements.
  - For each expanded sample, run this project's architecture (ingest context + answer question).
  - Write evaluator-ready predictions to `predictions.jsonl`.
  - Write run telemetry to `bridge_rows.jsonl`.
  - Save run metadata to `manifest.json`.

- Difference between command 2 and command 4 in this NoLiMa workflow:
  - Command 2 is a local fixture smoke run.
    - Goal: sanity-check wiring and execution quickly.
    - Not suitable for full parity reporting.
  - Command 4 is scoring over an existing generated run.
    - Goal: produce report metrics from `predictions.jsonl`.
    - Command 4 does not generate predictions by itself.

In short:
- Command 2 = pipeline test on synthetic fixture.
- Command 4 = scoring pass for an already generated run.

## Q7
Question: How much would it cost to run the benchmark?

Answer:
- Cost depends mainly on generation model calls during command 3.
- Scoring (command 4) is local and usually low-cost compared to generation.
- Practical estimate flow:
  - Run a small subset (for example with `--max-samples 20`).
  - Read average `delta_cost_usd` from `bridge_rows.jsonl`.
  - Extrapolate to full sample count.

Important caveats:
- Longer contexts and larger depth sweeps increase total token usage.
- Full NoLiMa-style matrices can be substantially larger than quick smoke runs.

Best practice:
- Run a small subset first and extrapolate before full matrix execution.

## Q8
Question: How many samples are there in the official benchmark dataset?

Answer:
- There is no single fixed sample count in this repository.
- Sample count is expanded dynamically from:
  - number of cases in needle set,
  - number of haystack files,
  - number of selected lengths,
  - depth intervals.

Approximate formula:
- total_samples ~= cases x haystacks x lengths x depth_intervals
  (then capped by `--max-samples` if set)

How to check what was actually executed:
- Open run `manifest.json` and read `samples_selected`.

## Q9
Question: In this repository, where are benchmark inputs and outputs saved? Are testing and full-run outputs stored in the same location?

Answer:
- Inputs:
  - Fixture inputs are under `benchmark_fixtures/nolima/`.
  - Full NoLiMa assets are expected under `benchmark_data/nolima/`.

- Outputs:
  - NoLiMa runner outputs are written under `benchmark_artifacts/official_nolima/<run_id>/`.
  - Each run writes separate artifacts in its own timestamped run folder (`predictions.jsonl`, `bridge_rows.jsonl`, `manifest.json`).

- Testing vs full-run location:
  - They use the same root output location (`benchmark_artifacts/official_nolima/`).
  - They are separated by run_id folders, so smoke/test and full runs do not overwrite each other.

- Cache state for warm reuse is stored separately under:
  - `benchmark_artifacts/official_nolima/cache_state/<namespace>/`

## Q10
Question: How are we evaluating "correct" answers?

Answer:
- In this NoLiMa implementation, scoring is done by `nolima/score_nolima_predictions.py`.
- Default metric is `contains` (NoLiMa-style default for v1 in this repo).
- Available metrics:
  - `EM`
  - `contains`
  - `lastline_EM`
  - `lastline_contains`

Practical implication:
- If scoring is omitted, you get generation artifacts only.
- Run `nolima/score_nolima_predictions.py` against the run folder to produce `official_nolima_eval_report.json`.

## Q11
Question: Is this NoLiMa flow invoking upstream NoLiMa run scripts directly?

Answer:
- No. This v1 path is repository-native.
- It preserves high-parity mechanics for sample expansion/placement/depth sweeps, but generation is always done by this repository's SemanticCache system.
- Upstream provider-specific connector scripts are not executed by this runner.

## Q12
Question: Do we still need NeMo-Skills for NoLiMa benchmark runs?

Answer:
- No. The NoLiMa runner/scorer implemented here do not depend on NeMo-Skills.
- You only need the normal dependencies required by this repository runtime.

## Q13
Question: Explain the --mode parameter.

Answer:
- `--mode` controls how this repository generates predictions.
- Allowed values are:
  - `baseline`
  - `cache`

What each mode does:
- `baseline`:
  - Calls search with cache reads disabled.
  - Intended to represent no-reuse behavior during generation.
- `cache`:
  - Calls search with cache reads enabled.
  - Uses the architecture's normal cache-enabled behavior during generation.

Important scope note:
- `--mode` affects generation behavior, not scoring math.
- You can compare baseline vs cache by running both modes and scoring both outputs.

Typical usage:
- Use `cache` as default for normal architecture benchmarking.
- Use `baseline` when you want an ablation-style comparison against cache-enabled behavior.

## Q14
Question: How does cache reuse behave for repeated NoLiMa runs?

Answer:
- In `--mode cache`, the runner reuses persistent cache state by default.
- Cache namespace is derived from content signature and key run parameters (needle set, cases, haystacks, lengths, depth intervals, seed, corpus id).
- Artifacts remain timestamped per run under `benchmark_artifacts/official_nolima/<run_id>/`.
- Reusable cache state is stored under:
  - default: `benchmark_artifacts/official_nolima/cache_state/<namespace>/`

Quick usage:
- Warm rerun: run the same command again with `--mode cache`.
- Cold restart: add `--cache-reset`.
- Optional custom state root: use `--cache-state-root`.

## Q15
Question: How do I verify warm-cache behavior?

Answer:

- Check run summary in stdout (`hits=...`, `entries_before=...`, `entries_after=...`).
- Check `manifest.json` under the run folder for `cache_reuse` fields.
- Check `bridge_rows.jsonl` for rows with `from_cache=true`.

## Q16
Question: Which CLI options are most useful for controlled dry runs?

Answer:
- `--max-cases`: limit number of expanded test cases.
- `--max-haystacks`: limit haystack files.
- `--max-samples`: hard cap on total generated samples.
- `--lengths`: reduce context matrix for faster smoke checks.
- `--depth-intervals`: lower sweep density for quick checks.

## Q17
Question: What are the current known limitations of this NoLiMa integration?

Answer:
- The runner is high-parity and repository-native; it does not execute upstream NoLiMa API connector scripts.
- v1 documents and defaults focus on standard `needle_set.json` first.
- Additional needle-set variants (`hard`, `MC`, `ONLYDirect`, `w_CoT`, `w_Distractor`) are roadmap items and can be added with the same runner once staged.

## Q18
Question: How do we fetch the NoLiMa dataset assets into this repository?

Answer:
- Use the helper script in this repository:
  - `nolima/scripts/fetch_dataset.sh`

```bash
# From repository root

# Download needlesets + rand_shuffle + rand_shuffle_long
bash nolima/scripts/fetch_dataset.sh

# Skip rand_shuffle_long if you only want standard files
bash nolima/scripts/fetch_dataset.sh --no-long

# Optional custom target root
bash nolima/scripts/fetch_dataset.sh --target-root benchmark_data/nolima
```

Quick verification:

```bash
ls -lh benchmark_data/nolima/needlesets
ls -lh benchmark_data/nolima/haystack/rand_shuffle
du -sh benchmark_data/nolima
```

## Q19
Question: Is there a dataset-size limit we can set to estimate spending first?

Answer:
- Yes. This runner supports multiple caps to control run size and estimate cost before full execution:
  - `--max-samples`: hard cap on total expanded samples (strongest budget control)
  - `--max-cases`: limit number of needle-set cases
  - `--max-haystacks`: limit number of haystack files
  - `--lengths`: reduce length matrix
  - `--depth-intervals`: reduce number of placements per length

Recommended budget-first workflow:
- 1) Run a small capped job.
- 2) Read average `delta_cost_usd` from `bridge_rows.jsonl`.
- 3) Extrapolate to planned full sample count.

Example small budgeted run:

```bash
python nolima/run_benchmark.py \
  --needle-set-path benchmark_data/nolima/needlesets/needle_set.json \
  --haystack-dir benchmark_data/nolima/haystack/rand_shuffle \
  --lengths 1K,4K \
  --depth-intervals 5 \
  --max-cases 20 \
  --max-haystacks 2 \
  --max-samples 100 \
  --mode baseline \
  --output-dir benchmark_artifacts
```

Then estimate:

```bash
# Uses helper script in this repository
python nolima/scripts/estimate_run_cost.py \
  --run-dir benchmark_artifacts/official_nolima/<run_id> \
  --project-samples 1000
```

