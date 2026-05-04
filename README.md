# Two-Stage Semantic Cache for Autonomous Agents

A production-grade semantic caching system that makes autonomous LLM agent execution economically viable. Achieves **96.7% cost reduction** on redundant workloads through a novel "Dragnet & Sniper" architecture.

## Key Features

| Feature | What it does |
|---------|-------------|
| **Two-Stage Dragnet & Sniper** | Fast vector search (FAISS) + LLM-verified semantic equivalence (Haiku) — eliminates catastrophic collisions that break pure-vector caches |
| **Context Collapse Guard** | Prevents oversized cache returns from degrading agent reasoning — ephemeral tagging + recursive parallel summarization |
| **Source Provenance** | Every cached entry is grounded against source text + independently verified by a second LLM (consensus) |
| **Knowledge Extraction** | Decomposes answers into `(subject, relation, object)` triples, FAISS-indexed for cross-query reuse |
| **Cache Pre-Warming** | Programmatic Day-1 sweep eliminates cold-start — cache is saturated before any human query |
| **Corpus Namespace Isolation** | Multi-domain deployment (legal, finance, medical) with isolated FAISS indices, caches, and knowledge graphs |
| **Heterogeneous Routing** | Dispatches simple tasks to Haiku ($0.25/MTok), complex to Sonnet ($3/MTok) |

## Architecture

```
Query → Cache Check (free/cheap)
  ├─ HIT  → Serve cached answer (with provenance metadata)
  └─ MISS → FAISS Dragnet → Qwen3-Reranker → Sonnet Synthesis
              → Grounding Check → Consensus Verify → Cache Store
              → Knowledge Extraction → Fact Index
```

## Stack

- **Embeddings**: Qwen3-Embedding-0.6B (596M params, 1024-dim, local CPU)
- **Reranker**: Qwen3-Reranker-0.6B (cross-encoder, local CPU)
- **Vector Index**: FAISS IndexFlatIP (exact cosine similarity)
- **LLM API**: Claude Sonnet 4.5 (synthesis) + Claude Haiku 4.5 (evaluation/sniper/consensus)
- **Python**: 3.9+, single-file library (~1,900 lines)

## Quick Start

```bash
# Install dependencies
pip install transformers torch faiss-cpu anthropic python-dotenv numpy

# Set your API key
echo "ANTHROPIC_API_KEY=sk-..." > .env

# Run the semantic cache system demo
python semantic_cache_system.py
```

## Official Benchmarking

Use `ruler_v2/run_benchmark.py` only for the official RULER v2 benchmark path.
The script expects prepared official samples, generates architecture predictions,
and can execute the official evaluator command without metric overrides.

### Cost Accounting Assumptions

Benchmark `delta_cost_usd` values are estimated from token usage using the
centralized pricing map in `semantic_cache_system.py`:

- `MODEL_FAMILY_PRICING_USD_PER_1K["sonnet"]`: input `$0.003` / 1K, output `$0.015` / 1K
- `MODEL_FAMILY_PRICING_USD_PER_1K["haiku"]`: input `$0.001` / 1K, output `$0.005` / 1K

These are explicit Anthropic-style reference rates (not live billing API values).
Run-level totals in each benchmark `manifest.json` use the same estimated token pricing.

### Official Run Command

```bash
python ruler_v2/run_benchmark.py \
  --official-prepared-data /path/to/ruler2_prepared_data \
  --official-tasks mk_niah_basic,mv_niah_basic,qa_basic \
  --official-lengths 8192,32768 \
  --mode cache \
  --output-dir benchmark_artifacts
```

### RLM Baseline Command

Use `ruler_v2/run_rlm_benchmark.py` for the uncached RLM baseline. It writes the
same core artifacts under `benchmark_artifacts/official_ruler_v2_rlm/<run_id>/`
and intentionally does not load cache, FAISS, or reranker components.

```bash
python ruler_v2/run_rlm_benchmark.py \
  --official-prepared-data benchmark_data/ruler2 \
  --official-tasks mk_niah_basic,mv_niah_basic,qa_basic \
  --official-lengths 8192,32768 \
  --official-max-samples-per-task 1 \
  --output-dir benchmark_artifacts
```

### Cache Reuse Across Runs

In `--mode cache`, official benchmark runs now reuse persistent cache state by default.
The cache namespace is derived from dataset content signature + selected tasks +
selected lengths + corpus id, so timestamped dataset folder names do not break
warm-cache behavior when selected sample content is unchanged.

- Default state root:
  - `benchmark_artifacts/official_ruler_v2/cache_state/<namespace>/`
- Reset only when explicitly requested:

```bash
python ruler_v2/run_benchmark.py \
  --official-prepared-data /path/to/ruler2_prepared_data \
  --official-tasks mk_niah_basic,mv_niah_basic,qa_basic \
  --official-lengths 8192,32768 \
  --mode cache \
  --official-cache-reset \
  --output-dir benchmark_artifacts
```

- Optional custom state root:

```bash
python ruler_v2/run_benchmark.py \
  --official-prepared-data /path/to/ruler2_prepared_data \
  --official-tasks mk_niah_basic,mv_niah_basic,qa_basic \
  --official-lengths 8192,32768 \
  --mode cache \
  --official-cache-state-root /path/to/cache_state_root \
  --output-dir benchmark_artifacts
```

Warm-run signals are available in stdout and in each run's `manifest.json`
under the `cache_reuse` section.

### Data Preparation (NeMo-Skills)

Install benchmark tooling in your active virtual environment first:

```bash
# pip workflow
python -m pip install "git+https://github.com/NVIDIA/NeMo-Skills.git"

# or uv workflow
uv pip install "git+https://github.com/NVIDIA/NeMo-Skills.git"

# verify
ns --help
python -m nemo_skills.dataset.ruler2.prepare --help
```

If `nemo_skills` was previously installed from a temporary editable path
(for example `/tmp/NeMo-Skills`), reinstall with one of the commands above
to avoid breakage when tmp directories are cleaned.

Prepare official RULER2 data first (example with `dataset_size=100`):

```bash
ns prepare_data ruler2 --skip_data_dir_check \
  --setup adarsh_8192 \
  --max_seq_length 8192 \
  --tokenizer_type hf \
  --tokenizer_path TOKENIZER_PATH \
  --tasks mk_niah_basic mv_niah_basic qa_basic \
  --dataset_size 100

ns prepare_data ruler2 --skip_data_dir_check \
  --setup adarsh_32768 \
  --max_seq_length 32768 \
  --tokenizer_type hf \
  --tokenizer_path TOKENIZER_PATH \
  --tasks mk_niah_basic mv_niah_basic qa_basic \
  --dataset_size 100
```

### Optional: Inline Official Eval Command in Runner

Supported placeholders:

- `{tasks_csv}`
- `{lengths_csv}`
- `{prepared_data}`
- `{results_dir}`
- `{predictions}`
- `{bridge_rows}`

```bash
python ruler_v2/run_benchmark.py \
  --official-prepared-data /path/to/ruler2_prepared_data \
  --official-tasks mk_niah_basic,mv_niah_basic,qa_basic \
  --official-lengths 8192,32768 \
  --mode cache \
  --official-eval-command "ns eval --output_dir {results_dir} --benchmarks ruler2 --server_type openai" \
  --output-dir benchmark_artifacts
```

Note: in this environment, NeMo-Skills is invoked through the `ns` CLI.
The old `python -m nemo_skills.evaluation.evaluate` module path is not available in the installed version.

### Official Artifacts

Each run writes a timestamped directory under `benchmark_artifacts/official_ruler_v2/`:

- `predictions.jsonl`
- `bridge_rows.jsonl`
- `bridge_rows.csv`
- `manifest.json`

The manifest includes implemented task progress against the 13-task baseline,
plus run timing and aggregate estimated token/cost totals.

### Implemented Scope vs 13-Task Baseline

- Implemented: 3/13 (`mk_niah_basic`, `mv_niah_basic`, `qa_basic`)
- Pending: 10/13

## NoLiMa Benchmarking (High-Parity)

Use `nolima/run_benchmark.py` for the NoLiMa benchmark path. This flow keeps
NoLiMa placement semantics (needle-set expansion and depth sweeps) while
evaluating this repository's semantic cache system as the system under test.

### Folder Layout

- NoLiMa inputs:
  - `benchmark_data/nolima/needlesets/`
  - `benchmark_data/nolima/haystack/rand_shuffle/`
- NoLiMa artifacts:
  - `benchmark_artifacts/official_nolima/<run_id>/`
  - `benchmark_artifacts/official_nolima/cache_state/<namespace>/`

### Smoke Run (Synthetic Fixture)

```bash
uv run python nolima/run_benchmark.py \
  --needle-set-path benchmark_fixtures/nolima/needlesets/needle_set.json \
  --haystack-dir benchmark_fixtures/nolima/haystack/rand_shuffle \
  --lengths 1K,4K \
  --depth-intervals 5 \
  --mode cache \
  --output-dir benchmark_artifacts
```

### Full NoLiMa-Style Run

```bash
uv run python nolima/run_benchmark.py \
  --needle-set-path benchmark_data/nolima/needlesets/needle_set.json \
  --haystack-dir benchmark_data/nolima/haystack/rand_shuffle \
  --lengths 250,500,1K,2K,4K,8K,16K,32K \
  --depth-intervals 26 \
  --mode cache \
  --output-dir benchmark_artifacts
```

### Score an Existing NoLiMa Run

```bash
uv run python nolima/score_nolima_predictions.py \
  --run-dir benchmark_artifacts/official_nolima/<run_id> \
  --metric contains
```

NoLiMa artifacts include:

- `predictions.jsonl`
- `bridge_rows.jsonl`
- `bridge_rows.csv`
- `manifest.json`
- `official_nolima_eval_report.json` (after scoring)

## Cache-Mode Benchmarking

Use `cache_bench/run_benchmark.py` when you want to evaluate cache-route
behavior directly instead of official benchmark accuracy.

This suite measures whether a follow-up query behaves as an:

- `exact` hit
- `semantic` hit
- `knowledge` hit
- `miss`

Run the labeled synthetic suite:

```bash
python cache_bench/run_benchmark.py \
  --suite-path benchmark_fixtures/cache_bench/cache_mode_suite_v1.json \
  --mode cache \
  --output-dir benchmark_artifacts
```

Supported cache-mode experiments:

- No cache:
  - `python cache_bench/run_benchmark.py --suite-path benchmark_fixtures/cache_bench/cache_mode_suite_v1.json --mode baseline --output-dir benchmark_artifacts`
- Cold start:
  - `python cache_bench/run_benchmark.py --suite-path benchmark_fixtures/cache_bench/cache_mode_suite_v1.json --mode cache --cache-reset --output-dir benchmark_artifacts`
- Warm start:
  - `python cache_bench/run_benchmark.py --suite-path benchmark_fixtures/cache_bench/cache_mode_suite_v1.json --mode cache --output-dir benchmark_artifacts`

Artifacts are written under `benchmark_artifacts/cache_mode_suite/<run_id>/`:

- `predictions.jsonl`
- `bridge_rows.jsonl`
- `bridge_rows.csv`
- `manifest.json`
- `official_cache_mode_eval_report.json`

Persistent cache state for cache-enabled runs is written under:

- `benchmark_artifacts/cache_mode_suite/cache_state/<namespace>/`

See [cache_mode_benchmark.md](/Users/engindenizdogu/Desktop/local_repos/adarsh-rlms/cache_bench/docs/cache_mode_benchmark.md) for fixture
shape, metrics, and usage notes.

## LegalBench/CUAD Cache-Route Benchmarking

Use `legal_bench/` for the real legal-dataset cache-efficiency track. This is
separate from the simple synthetic suite above.

Build a route-labeled CUAD-QA suite from the official CUAD source zip:

```bash
python legal_bench/build_suite.py \
  --from-cuad-source \
  --output-path benchmark_data/legal_bench/legal_cache_suite_cuad_v1.json \
  --max-records 25
```

Run a cold route-accuracy pass with per-case cache isolation:

```bash
python legal_bench/run_benchmark.py \
  --suite-path benchmark_data/legal_bench/legal_cache_suite_cuad_v1.json \
  --mode cache \
  --cache-reset \
  --output-dir benchmark_artifacts
```

Artifacts are written under `benchmark_artifacts/legal_cache_suite/<run_id>/`:

- `predictions.jsonl`
- `bridge_rows.jsonl`
- `bridge_rows.csv`
- `manifest.json`
- `official_legal_cache_eval_report.json`

Persistent legal cache state is written under:

- `benchmark_artifacts/legal_cache_suite/cache_state/<namespace>/`

The legal runner uses the Qwen reranker by default. If the reranker returns zero
chunks after FAISS found candidates, retrieval falls back to bounded FAISS
candidates and records the fallback in bridge rows and the manifest.

The legal runner isolates cache state per case by default so unrelated contract
questions do not contaminate route-label evaluation. See
`legal_bench/docs/legal_cache_benchmark.md` for the full workflow.

### Epstein Court Document Search (Domain Client Example)

```bash
# Ingest 1,000 court documents → FAISS index
python epstein_search.py --ingest

# Search with full cache pipeline
python epstein_search.py --search "What charges did Ghislaine Maxwell face?"

# Interactive mode
python epstein_search.py --interactive
```

## Files

| File | Description |
|------|-------------|
| `semantic_cache_system.py` | Core library — embeddings, FAISS, reranker, cache controller, pre-warmer, router, agent |
| `epstein_search.py` | Domain client — Epstein court document search using the core library |
| `system_architecture.md` | Full architecture documentation with scenarios and use cases |
| `paper_draft.tex` | LaTeX research paper draft |

## Results

| Metric | Baseline RLM | Optimized | Improvement |
|--------|-------------|-----------|-------------|
| API Cost | $0.00182 | $0.00006 | **96.7% ↓** |
| Execution Time | 11.20s | 5.87s | **1.9× faster** |
| Model Calls | 5 | 2 (+3 cached) | **60% fewer** |

## License

MIT
