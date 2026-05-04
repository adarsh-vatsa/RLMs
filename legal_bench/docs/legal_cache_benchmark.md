# LegalBench/CUAD Cache-Route Benchmark

This benchmark extends the simple synthetic cache-mode suite with real contract review data. It is intentionally separate from `cache_bench/`.

- Simple fixture: `cache_bench/run_benchmark.py`
- Legal real-dataset track: `legal_bench/run_benchmark.py`

The legal track writes artifacts under:

- `benchmark_artifacts/legal_cache_suite/<run_id>/`
- `benchmark_artifacts/legal_cache_suite/cache_state/<namespace>/`

## CUAD Clause Labels

CUAD defines 41 contract-review categories. This benchmark filters out
`Document Name` because it is document metadata, not a clause-review issue.
The remaining categories include:

- `Parties`
- `Agreement Date`
- `Effective Date`
- `Expiration Date`
- `Renewal Term`
- `Notice Period to Terminate Renewal`
- `Governing Law`
- `Most Favored Nation`
- `Non-Compete`
- `Exclusivity`
- `No-Solicit of Customers`
- `Competitive Restriction Exception`
- `No-Solicit of Employees`
- `Non-Disparagement`
- `Termination for Convenience`
- `Rofr/Rofo/Rofn`
- `Change of Control`
- `Anti-Assignment`
- `Revenue/Profit Sharing`
- `Price Restrictions`
- `Minimum Commitment`
- `Volume Restriction`
- `IP Ownership Assignment`
- `Joint IP Ownership`
- `License Grant`
- `Non-Transferable License`
- `Affiliate License-Licensor`
- `Affiliate License-Licensee`
- `Unlimited/All-You-Can-Eat-License`
- `Irrevocable or Perpetual License`
- `Source Code Escrow`
- `Post-Termination Services`
- `Audit Rights`
- `Uncapped Liability`
- `Cap on Liability`
- `Liquidated Damages`
- `Warranty Duration`
- `Insurance`
- `Covenant Not to Sue`
- `Third Party Beneficiary`

## 1. Build a Suite

CUAD-QA is hosted on Hugging Face:

- Dataset page: <https://huggingface.co/datasets/theatticusproject/cuad-qa>

That Hub repo currently uses a Python loading script; the dataset viewer says it requires arbitrary Python execution. Current `datasets` releases no longer support dataset scripts, so this repo does not call `load_dataset()` for CUAD. Instead, it uses the same source URL and parsing logic shown in `legal_bench/cuad_qa.py`.

Generate a suite from the official CUAD source zip:

```bash
python legal_bench/build_suite.py \
  --from-cuad-source \
  --output-path benchmark_data/legal_bench/legal_cache_suite_cuad_v1.json \
  --selection-strategy balanced-corpora \
  --max-corpora 5 \
  --records-per-corpus 3
```

The generator creates `exact`, `semantic`, `knowledge`, and `miss` cases from
real contract questions and answer spans. CUAD's administrative `Document Name`
field is skipped because it is document metadata, not a clause-review issue.
Repeated rows for the same contract/clause label are collapsed so small suites
cover more review issues.

The recommended balanced command selects 5 contracts with 3 clause records each,
usually producing 60 total cases. This keeps cost similar to a 15-record
sequential run while adding contract variety. `miss` cases require at least two
questions from the same contract, so balanced selection skips contracts that do
not have enough distinct usable clause labels.

For quick debugging, you can still use sequential selection:

```bash
python legal_bench/build_suite.py \
  --from-cuad-source \
  --output-path benchmark_data/legal_bench/legal_cache_suite_cuad_v1.json \
  --max-records 15
```

## 2. Run the Legal Benchmark (see section 3 before running)

Baseline/no-cache run:

```bash
python legal_bench/run_benchmark.py \
  --suite-path benchmark_data/legal_bench/legal_cache_suite_cuad_v1.json \
  --mode baseline \
  --output-dir benchmark_artifacts
```

Cold route-accuracy run:

```bash
python legal_bench/run_benchmark.py \
  --suite-path benchmark_data/legal_bench/legal_cache_suite_cuad_v1.json \
  --mode cache \
  --cache-reset \
  --output-dir benchmark_artifacts
```

Warm rerun:

```bash
python legal_bench/run_benchmark.py \
  --suite-path benchmark_data/legal_bench/legal_cache_suite_cuad_v1.json \
  --mode cache \
  --output-dir benchmark_artifacts
```

The legal runner uses the Qwen reranker by default. Retrieval is still bounded: FAISS collects candidates, the reranker filters them, and synthesis receives at most the top three selected chunks.

If FAISS finds candidates but the reranker rejects all of them, the runner falls back to the top FAISS chunks instead of stopping with `No relevant documents found.` Bridge rows and manifests record this with `reranker_fallback_used` and related retrieval counts.

Use FAISS-only mode only as an ablation/debug setting:

```bash
python legal_bench/run_benchmark.py \
  --suite-path benchmark_data/legal_bench/legal_cache_suite_cuad_v1.json \
  --mode cache \
  --disable-reranker \
  --output-dir benchmark_artifacts
```

## 3. Case Cache Isolation

The legal runner uses a separate persistent cache namespace per case by default.
This is a diagnostic setting, not the realistic deployment setting.

Why it exists: route-labeled benchmarks are trying to answer narrow questions
such as "does this paraphrase take the semantic route?" or "does this summary
seed produce a knowledge hit?" If all cases share one warmed cache, an earlier
case can save the eval query for a later case. The later case may then exact-hit
before it exercises the intended route, which makes route diagnostics noisy.

Use the default isolated mode for route-mechanism debugging and ablations:

```bash
python legal_bench/run_benchmark.py \
  --suite-path benchmark_data/legal_bench/legal_cache_suite_cuad_v1.json \
  --mode cache \
  --cache-reset \
  --output-dir benchmark_artifacts
```

Use the global-cache mode for the real-life cache-efficiency experiment:

```bash
python legal_bench/run_benchmark.py \
  --suite-path benchmark_data/legal_bench/legal_cache_suite_cuad_v1.json \
  --mode cache \
  --no-case-cache-isolation \
  --output-dir benchmark_artifacts
```

In global-cache mode, route labels are still recorded, but some route mismatches
are expected and may be desirable: they show that the shared cache found an even
cheaper reuse path than the route the generated case was designed to test.
Report isolated runs for route correctness and global-cache runs for reuse
economics.

## Artifacts

Each run writes:

- `predictions.jsonl`
- `bridge_rows.jsonl`
- `bridge_rows.csv`
- `manifest.json`
- `official_legal_cache_eval_report.json`

Legal bridge rows include source metadata such as source record ID, contract
title, clause label, seed record ID, answer offsets, and per-case namespace.

The eval report also includes miss diagnostics:

- `expected_miss_rate`: fraction of cases designed to miss.
- `actual_miss_rate`: fraction of cases that actually used the miss/API path.
- `unexpected_miss_rate`: fraction of all cases where a positive cache-route case
  fell through to miss.
- `positive_case_miss_rate`: fraction of non-miss cases that fell through to miss.
