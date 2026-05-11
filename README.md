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

## Benchmarking

This repository keeps benchmark instructions in dedicated documentation pages.
The README only summarizes the purpose of each benchmark track.

| Benchmark | Role |
|-----------|------|
| CUAD cache suite | cache-route debugging and legal contract fixture |
| RULER | long-context retrieval stress test, mainly engineering validation |
| LongBench-v2 | next candidate for general long-context reasoning with deterministic multiple-choice scoring |
| CorpusQA | candidate for very high-context corpus-level reasoning if the dataset and scoring harness are reproducible |
| OOLONG | candidate for aggregation-heavy long-context reasoning that may require map-reduce style execution |
| LongMemEval | candidate for persistent memory and knowledge reuse across sessions |
| ContractNLI | optional focused legal reasoning sanity check |
| LegalBench-RAG | optional retrieval/evidence-grounding benchmark |

Current implemented benchmark tracks:

| Track | Purpose | Documentation |
|-------|---------|---------------|
| RULER v2 cache runner | Official RULER v2 runs against this cache system | `ruler_v2/run_benchmark.py` |
| RULER v2 RLM baseline | Uncached RLM baseline over prepared RULER v2 data | `ruler_v2/docs/rlm_baseline_runner.md` |
| NoLiMa | Long-context needle placement benchmark | `nolima/` |
| Synthetic cache-mode suite | Small hand-authored exact/semantic/knowledge/miss fixture | `cache_bench/docs/cache_mode_benchmark.md` |
| LegalBench/CUAD cache suite | Real contract-data cache-route benchmark | `legal_bench/docs/legal_cache_benchmark.md` |

The current benchmark recommendation is documented in `docs/benchmarks.md`. The next likely implementation target is `(Modified) LongBench-v2`: preserve original multiple-choice accuracy labels, then add cache-route wrappers for exact duplicates, semantic paraphrases, misses, and knowledge hits only when repeated contexts support them naturally.

Benchmark artifacts are written under `benchmark_artifacts/`. Historical
artifacts should be treated as read-only unless a run is intentionally being
regenerated.

### Cost Accounting

Benchmark `delta_cost_usd` values are estimated from token usage using the
centralized pricing map in `semantic_cache_system.py`:

- `MODEL_FAMILY_PRICING_USD_PER_1K["sonnet"]`: input `$0.003` / 1K, output `$0.015` / 1K
- `MODEL_FAMILY_PRICING_USD_PER_1K["haiku"]`: input `$0.001` / 1K, output `$0.005` / 1K

These are explicit Anthropic-style reference rates, not live billing API values.
Run-level totals in each benchmark `manifest.json` use the same estimated token
pricing.

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
