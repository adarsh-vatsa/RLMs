# Memoized RLM Controller

A memo-first controller for recursive language-model work. The core abstraction
is a persistent memo graph:

```text
task + context scope -> result + evidence + dependencies
```

The previous semantic-cache, knowledge, FAISS, and reranker components still
exist, but they now sit in cleaner roles: legacy cache and knowledge rows are
compatibility adapters, while FAISS/DuckDB/vector/lexical lookup are candidate
generators. The memo graph is the single source of truth for reuse decisions.

## Key Features

| Feature | What it does |
|---------|-------------|
| **Language Memoization** | Stores scoped subproblem results as reusable memo entries |
| **Memo Graph** | Tracks evidence and dependency edges between full answers and partial work |
| **DuckDB Backend** | Persists memo rows, raw context chunks, and dependency edges locally |
| **Q3 Local Testing** | Uses the local MLX Q3 model as solver, aggregator, and verifier in live tests |
| **Structured Candidate Generators** | Exposes memo candidates, DuckDB context chunks, and FAISS document retrieval as bounded packets |
| **Scope Invalidation** | Marks stale memo fragments as rejected while keeping them auditable |
| **Legacy Compatibility Adapters** | Mirrors old final-answer cache rows and knowledge triples into memo entries |
| **Context Collapse Guard** | Prevents oversized cache returns from degrading agent reasoning — ephemeral tagging + recursive parallel summarization |
| **Source Provenance** | Every cached entry is grounded against source text + independently verified by a second LLM (consensus) |
| **Cache Pre-Warming** | Programmatic Day-1 sweep eliminates cold-start — cache is saturated before any human query |
| **Corpus Namespace Isolation** | Multi-domain deployment with isolated corpus IDs, scopes, memo rows, and indexes |
| **Heterogeneous Routing** | Dispatches simple tasks to Haiku ($0.25/MTok), complex to Sonnet ($3/MTok) |

## Architecture

```
Query + scope
  → mirror legacy cache/knowledge into memo entries
  → exact memo replay
  → candidate generation when needed
      ├─ memo text candidates
      ├─ DuckDB context chunks
      └─ FAISS document chunks + reranker
  → solver / aggregator / verifier
  → store memo entry + evidence + dependency edges
```

## Stack

- **Embeddings**: Qwen3-Embedding-0.6B (596M params, 1024-dim, local CPU)
- **Reranker**: Qwen3-Reranker-0.6B (cross-encoder, local CPU)
- **Memo Store**: in-memory `MemoStore` or persistent `DuckDBMemoStore`
- **Vector Index**: FAISS IndexFlatIP for document and legacy candidate generation
- **LLM API**: Claude Sonnet 4.5 (synthesis) + Claude Haiku 4.5 (evaluation/sniper/consensus)
- **Local LLM**: MLX Q3 KXL for live DP memo tests
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

### Local MLX Smoke Test

This repo is wired to use the local MLX adapter in `local_llm.py`. By default it
uses `Brooooooklyn/Qwen3.6-27B-UD-Q3_K_XL-mlx`, which is the Q3 27B model tested
on Apple Silicon with an observed peak memory around 18 GB for a short smoke
generation.

```bash
# Install local MLX extras
python -m pip install ".[local-mlx]"

# Run the repo-level local model smoke test
python scripts/local_llm_smoke.py
```

For smaller smoke tests or machines with less unified memory, override the model:

```bash
RLMS_MLX_MODEL=mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  python scripts/local_llm_smoke.py
```

To test the local model through the DP memoization path, run:

```bash
python scripts/live_dp_memo_smoke.py
```

The live scripts do not impose small repository-level generation caps by
default. `--max-tokens` and `--aggregate-max-tokens` are available for explicit
stress tests, but the normal path leaves generation length to the MLX backend
or `RLMS_MLX_MAX_TOKENS` if you set it.

That smoke test uses `SemanticCacheController.solve_with_memo(...)` to split
the context into chunk windows, solve missing windows with the local model,
aggregate the partial answers with the local model, store the full-scope answer,
replay it on the second call, then test verifier-approved semantic reuse for a
different prompt over the same scope. Exact and semantic replay should both
report `model_calls: 0`, `aggregate_calls: 0`, and the script should print
`MVP CHECK: PASS`.

Run the same MVP path against DuckDB-backed memo storage:

```bash
python scripts/live_dp_memo_smoke.py \
  --reset-db \
  --backend duckdb \
  --duckdb-path benchmark_artifacts/live_dp_memo.duckdb
```

### DP Memo NoLiMa Fixture Benchmark

Run the DP memoization stack against the local NoLiMa fixture with the local MLX
model and DuckDB:

```bash
python scripts/run_dp_memo_nolima.py \
  --reset-db \
  --max-cases 2 \
  --max-haystacks 1 \
  --max-samples 4 \
  --lengths 250 \
  --depth-intervals 2 \
  --chunk-words 60 \
  --chunk-size 2 \
  --solver-mode evidence
```

This writes predictions, bridge rows, a manifest, and a DuckDB memo store under
`benchmark_artifacts/dp_memo_nolima/`. The smoke fixture covers two needles at
two depths and verifies exact replay for every sample.
Use `--max-cases` and `--max-haystacks` to define the source slice, then
`--max-samples` to stop generation early inside that slice.
Use `--sample-offset N` to skip directly to a later generated sample when
debugging a failure without rerunning earlier samples.

The official NoLiMa data can be fetched with:

```bash
bash nolima/scripts/fetch_dataset.sh \
  --target-root benchmark_data/nolima \
  --with-long
```

Before spending local Q3 runtime on a large benchmark slice, inspect the
workload shape:

```bash
python scripts/inspect_long_context_workload.py \
  --needle-set-path benchmark_data/nolima/needlesets/needle_set.json \
  --haystack-dir benchmark_data/nolima/haystack/rand_shuffle_long \
  --lengths 32K,64K,128K,256K \
  --depth-intervals 4 \
  --max-cases 4 \
  --max-haystacks 1 \
  --chunk-words 1000 \
  --chunk-size 2 \
  --output benchmark_artifacts/long_context_workload_plan.json
```

This does not call the model. It estimates samples, scoped solver windows, and
cold model calls so the benchmark slice is large enough to stress recursive
memoization before you run it.

Summarize cold/warm benchmark manifests with:

```bash
python scripts/summarize_dp_memo_runs.py \
  benchmark_artifacts/dp_memo_nolima/<cold-run>/manifest.json \
  benchmark_artifacts/dp_memo_nolima/<warm-run>/manifest.json
```

Add `--compact` when preparing human-facing summaries; it omits null fields
that are irrelevant for a given benchmark family.

### DP Memo Shared-Context Benchmark

Run the benchmark that better exposes the DP layer's advantage: one shared
context, several related questions, and a memoized fact-extraction pass reused
across all questions.

```bash
python scripts/run_dp_memo_shared_context.py \
  --reset-db \
  --sentences-per-chunk 2 \
  --chunk-size 1
```

This writes artifacts under `benchmark_artifacts/dp_memo_shared_context/`. The
fixture extracts reusable facts once, answers five questions from those facts,
asks the local model to choose from compact memo candidate packets, and verifies
that the fact layer replays from DuckDB with zero model and aggregation calls.

### DP Memo Mutable-Workspace Benchmark

Run a deterministic workload trace that exercises document updates:

```bash
python scripts/run_dp_memo_mutable_workload.py --reset-db
```

The trace solves a four-chunk runbook, updates one chunk, invalidates that
scope, reuses the three unaffected window fragments, solves only the changed
window, then reopens DuckDB and warm-replays the updated answer. Expected shape:

```text
v1_model_calls: 4
invalidated_entries: 2
v2_model_calls: 1
v2_initial_coverage_ratio: 0.75
warm_model_calls: 0
```

For NoLiMa-style inference tasks, prefer `--solver-mode evidence`. It stores
scoped evidence fragments first, then lets aggregation perform the final
reasoning. `--solver-mode answer` is still available for simpler direct-answer
tasks.
Evidence mode is intentionally recall-oriented: scoped calls should emit named
people, places, landmarks, titles, codes, and relationships instead of returning
`NOT_FOUND` just because a chunk does not directly answer the final question.
That was necessary for two-hop NoLiMa cases where the bridge evidence is a
landmark or place relationship rather than the final region name.

For overlap-reuse experiments across length sweeps, opt into stable identities:

```bash
python scripts/run_dp_memo_nolima.py \
  --lengths 16K \
  --solver-mode evidence \
  --corpus-id-mode stable \
  --document-id-mode stable \
  --content-hash-mode source
```

Those flags deliberately remove length from the reuse namespace so a later 32K
run over the same source can reuse solved 16K prefix work. Keep the defaults for
strict isolated benchmark runs.

Observed official-data stable-overlap smoke pattern with local Q3:

| Run | Initial coverage | Model calls | Aggregate calls | Correct |
|-----|------------------|-------------|-----------------|---------|
| 16K cold | 0.000 | 41 | 1 | yes |
| 32K partial | 0.503 | 41 | 1 | yes |
| 64K partial | 0.502 | 81 | 1 | yes |
| 128K partial | 0.501 | 161 | 1 | yes |
| 128K warm | 1.000 | 0 | 0 | yes |
| 256K partial | 0.579 | 234 | 1 | yes |
| 256K warm | 1.000 | 0 | 0 | yes |

This is a workload-style check, not a claim that a single QA benchmark fully
measures real-world agent value. The useful signal is that overlapping,
domain-scoped work fills the memo graph and later calls avoid reprocessing
already solved scopes.

Benchmark rows now include memo reuse telemetry: initial and final coverage
ratios, covered intervals, missing scopes, reusable entry IDs, negative entry
IDs, hint entry IDs, fragment-kind counts, evidence/dependency counts, and
per-window telemetry. Candidate packets also expose a planner-facing
`fragment_kind` such as `exact_answer`, `supporting_fact`,
`aggregation_component`, `search_hint`, or `ruled_out_region`.
Memo and context packets include `*_chars` and `*_truncated` fields whenever
text may be bounded for planner prompts. Pass `None` for the relevant max-char
argument to inspect the full text instead of accepting silent truncation.

The DuckDB backend also stores raw chunked context when `solve_with_memo(...)`
runs against a `DuckDBMemoStore`. This is exposed as structured retrieval APIs,
not unrestricted model-authored SQL:

```python
store.fetch_context_range(corpus_id="...", document_id="...", start=0, end=4)
store.search_context_chunks("launch codeword", corpus_id="...", limit=10)
controller.context_candidate_packets("launch codeword", scope=scope)
```

When a document range changes, invalidate the affected memo scope instead of
deleting old rows:

```python
store.invalidate_scope(scope, reason="document updated")
```

Invalidated entries remain in DuckDB for audit/history, but exact replay,
partial composition, and planner candidate generation ignore them. Invalidation
also propagates through dependency parents by default, so a derived answer is
rejected when one of its source fragments is invalidated.
For mutable workspaces, use stable or empty content hashes with explicit
invalidation so unchanged windows can still be reused after a localized edit.
Use strict content hashes when any source change should force recomputation.

The same backend materializes the persistent memo graph. Dependencies are
mirrored into `memo_edges`, and graph inspection is available through:

```python
store.graph_edges()
store.children(entry_id)
store.parents(entry_id)
store.lineage(entry_id)
```

## Official Benchmarking

Use `ruler_v2/run_benchmark.py` only for the official RULER v2 benchmark path.
The script expects prepared official samples, generates architecture predictions,
and can execute the official evaluator command without metric overrides.

### Cost Accounting Assumptions

Benchmark `delta_cost_usd` values are estimated from token usage using the
centralized pricing map in `semantic_cache_system.py`:

- `MODEL_FAMILY_PRICING_USD_PER_1K["sonnet"]`: input `$0.003` / 1K, output `$0.015` / 1K
- `MODEL_FAMILY_PRICING_USD_PER_1K["haiku"]`: input `$0.00025` / 1K, output `$0.00125` / 1K

These are explicit Anthropic-style reference rates (not live billing API values).

### Official Run Command

```bash
python ruler_v2/run_benchmark.py \
  --official-prepared-data /path/to/ruler2_prepared_data \
  --official-tasks mk_niah_basic,mv_niah_basic,qa_basic \
  --official-lengths 8192,32768 \
  --mode cache \
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
- `manifest.json`

The manifest includes implemented task progress against the 13-task baseline.

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
- `manifest.json`
- `official_nolima_eval_report.json` (after scoring)

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
