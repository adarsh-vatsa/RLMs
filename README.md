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

Use `run_benchmark.py` only for the official RULER v2 benchmark path.
The script expects prepared official samples, generates architecture predictions,
and can execute the official evaluator command without metric overrides.

### Official Run Command

```bash
python run_benchmark.py \
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
python run_benchmark.py \
  --official-prepared-data /path/to/ruler2_prepared_data \
  --official-tasks mk_niah_basic,mv_niah_basic,qa_basic \
  --official-lengths 8192,32768 \
  --mode cache \
  --official-cache-reset \
  --output-dir benchmark_artifacts
```

- Optional custom state root:

```bash
python run_benchmark.py \
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
python run_benchmark.py \
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
