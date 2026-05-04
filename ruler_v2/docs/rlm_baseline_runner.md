# RULER RLM Baseline Runner

Use `ruler_v2/run_rlm_benchmark.py` to run an uncached RLM baseline on the
prepared RULER v2 datasets already used by `ruler_v2/run_benchmark.py`.

This runner is intentionally cache-free:

- no semantic cache
- no FAISS
- no reranker
- no cache state

Artifacts are written under:

- `benchmark_artifacts/official_ruler_v2_rlm/<run_id>/`

## Install RLM

The runner lazy-imports `rlm`, so the rest of the repository can work without
the RLM package installed. Install the upstream package or a local checkout of:

- <https://github.com/alexzhang13/rlm>

Example:

```bash
pip install rlms
```

If that package name is not available in your environment, install from the
GitHub checkout according to the upstream repo instructions.

## Environment

The default RLM backend is Anthropic with Haiku:

- backend: `anthropic`
- model: `claude-haiku-4-5-20251001`

Set your Anthropic key before running:

```bash
export ANTHROPIC_API_KEY=sk-...
```

The runner passes this value to RLM as `backend_kwargs["api_key"]` when present.

## Smoke Run

Start with one sample per task:

```bash
python ruler_v2/run_rlm_benchmark.py \
  --official-prepared-data benchmark_data/ruler2 \
  --official-tasks mk_niah_basic,mv_niah_basic,qa_basic \
  --official-lengths 8192,32768 \
  --official-max-samples-per-task 1 \
  --output-dir benchmark_artifacts
```

This writes:

- `predictions.jsonl`
- `bridge_rows.jsonl`
- `bridge_rows.csv`
- `manifest.json`

## Full Local RULER Run

Remove the sample cap to run all selected local prepared samples:

```bash
python ruler_v2/run_rlm_benchmark.py \
  --official-prepared-data benchmark_data/ruler2 \
  --official-tasks mk_niah_basic,mv_niah_basic,qa_basic \
  --official-lengths 8192,32768 \
  --output-dir benchmark_artifacts
```

## Useful RLM Options

```bash
python ruler_v2/run_rlm_benchmark.py \
  --official-prepared-data benchmark_data/ruler2 \
  --official-tasks mk_niah_basic \
  --official-lengths 8192 \
  --official-max-samples-per-task 1 \
  --rlm-backend anthropic \
  --rlm-model claude-haiku-4-5-20251001 \
  --rlm-environment local \
  --rlm-max-iterations 30 \
  --rlm-max-depth 1 \
  --rlm-log-trajectories \
  --output-dir benchmark_artifacts
```

Use `--rlm-log-trajectories` when you need execution traces. Trajectory logs are stored under the run directory when the upstream RLM logger supports it.

## Metrics

The manifest records:

- `official_target: "ruler_v2_rlm"`
- `baseline_type: "rlm_uncached"`
- total API calls
- input/output tokens
- estimated cost
- elapsed time
- selected tasks and lengths
- `cache_reuse.enabled: false`

Bridge rows include per-sample latency, parsed usage fields, raw
`rlm_usage_summary`, and RLM backend/model metadata.

If RLM usage parsing cannot find supported token fields, the run still
completes. Parsed token/cost fields are set to `0`, and `usage_parse_status`
explains what happened.

## Scoring

As with the standard RULER bridge, correctness is delegated to the RULER scoring logic. Most runs should omit inline scoring and score the completed run afterward:

```bash
python ruler_v2/score_ruler2_predictions.py \
  --run-dir benchmark_artifacts/official_ruler_v2_rlm/<run_id>
```

Example:

```bash
python ruler_v2/score_ruler2_predictions.py \
  --run-dir benchmark_artifacts/official_ruler_v2_rlm/20260504T150419Z
```

Inline scoring is optional. If you need it, pass the full evaluator shell command through `--official-eval-command`. The runner substitutes these placeholders:

- `{predictions}`
- `{bridge_rows}`
- `{prepared_data}`
- `{results_dir}`
- `{tasks_csv}`
- `{lengths_csv}`

## Comparison Role

This runner is an uncached baseline for speed, cost, API-call, and accuracy comparison against the semantic cache system. Do not reuse cache state or wrap RLM calls in this runner.
