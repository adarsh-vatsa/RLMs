# Cache-Mode Benchmark

This benchmark is for evaluating cache-route behavior directly:

- `exact`
- `semantic`
- `knowledge`
- `miss`

It is intentionally separate from RULER v2 and NoLiMa.

- RULER/NoLiMa: answer quality, replay economics, and warm-run behavior
- Cache-mode suite: whether the system chooses the expected cache route for a labeled follow-up query

## Why a synthetic suite exists

RULER and NoLiMa do not provide labels such as:

- "this follow-up should be a semantic hit"
- "this follow-up should be a knowledge hit"
- "this follow-up should miss because the logic is inverted"

So this benchmark uses a small hand-authored fixture with known expectations.

## Run command

```bash
python cache_bench/run_benchmark.py \
  --suite-path benchmark_fixtures/cache_bench/cache_mode_suite_v1.json \
  --mode cache \
  --output-dir benchmark_artifacts
```

Optional filters:

```bash
python cache_bench/run_benchmark.py \
  --suite-path benchmark_fixtures/cache_bench/cache_mode_suite_v1.json \
  --case-ids finance_semantic_arr,doctor_knowledge_director \
  --mode cache \
  --output-dir benchmark_artifacts
```

## Experiment modes

This runner now supports the three benchmark modes you asked for:

### 1. No cache

Disables cache reads and persistent cache reuse.

```bash
python cache_bench/run_benchmark.py \
  --suite-path benchmark_fixtures/cache_bench/cache_mode_suite_v1.json \
  --mode baseline \
  --output-dir benchmark_artifacts
```

### 2. Cold start

Runs with cache enabled, but resets the persistent namespace first.

```bash
python cache_bench/run_benchmark.py \
  --suite-path benchmark_fixtures/cache_bench/cache_mode_suite_v1.json \
  --mode cache \
  --cache-reset \
  --output-dir benchmark_artifacts
```

### 3. Warm start

Runs with cache enabled and reuses the same persistent namespace from a prior run.

```bash
python cache_bench/run_benchmark.py \
  --suite-path benchmark_fixtures/cache_bench/cache_mode_suite_v1.json \
  --mode cache \
  --output-dir benchmark_artifacts
```

Persistent cache state lives under:

- `benchmark_artifacts/cache_mode_suite/cache_state/<namespace>/`

You can override the location with `--cache-state-root`.

## Artifacts

Each run writes a timestamped directory under:

- `benchmark_artifacts/cache_mode_suite/<run_id>/`

Artifacts:

- `predictions.jsonl`
- `bridge_rows.jsonl`
- `bridge_rows.csv`
- `manifest.json`
- `official_cache_mode_eval_report.json`

When `--mode cache` is used, the manifest also includes a `cache_reuse`
section describing cold/warm state, hit rate, and persistent namespace paths.

## What to look at

Manifest highlights:

- `answer_accuracy`
- `cache_type_match_rate`
- `overall_pass_rate`
- `expected_route_counts`
- `actual_route_counts`
- `by_scenario`
- `by_expected_cache_type`
- `by_actual_cache_type`

Bridge-row highlights:

- `expected_cache_type`
- `actual_cache_type`
- `answer_correct`
- `cache_type_match`
- `overall_pass`
- `knowledge_verifier_called`
- `knowledge_verifier_allow`

Eval report highlights:

- top-level route and answer metrics in one JSON file
- grouped summaries by scenario and cache type
- `failing_cases` list for quick debugging

## Fixture design in v1

The initial fixture uses three small corpora:

- Finance metrics: semantic paraphrases and fact reuse over ARR, MRR, EBITDA, margin, churn, and COGS
- Doctor Strange facts: entity/fact QA and broad-summary-to-fact follow-ups
- Ops incident policy: semantic negatives and logical inversion checks

This keeps v1 small enough to debug while still covering the main cache routes.
