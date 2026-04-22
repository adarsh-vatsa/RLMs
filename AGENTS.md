# Agent Project Index

Use this file as a compact index. Read the linked source files only when the task needs implementation detail.

## Project Purpose

- `adarsh-rlms` researches and benchmarks a two-stage semantic cache for autonomous LLM workflows.
- The system combines local Qwen3 embedding/reranking roles, FAISS vector search, Anthropic executor/evaluator roles, source provenance, knowledge extraction, corpus namespace isolation, and benchmark evaluation.
- Avoid hardcoding unstable model IDs from docs. Confirm exact model constants in `semantic_cache_system.py` before editing model-related code.

## Source Of Truth Index

- `docs/system_architecture.md`: full architecture, component behavior, data flow, scenarios, and return shapes.
- `semantic_cache_system.py`: core implementation and constants, including model roles, cache controller, persistence, routing, and pricing.
- `README.md`: setup, run commands, benchmark examples, artifact layout, and current results summary.
- `ruler_v2/run_benchmark.py`: official RULER v2 benchmark orchestration.
- `nolima/run_benchmark.py`: NoLiMa benchmark orchestration.

## Core Architecture Map

- `EmbeddingEngine`: local query/document embedding role.
- `FAISSIndex`: vector index wrapper with metadata and persistence.
- `Reranker`: local cross-encoder relevance gate for document retrieval.
- `ExecutionMetrics`: cache/API/cost/provenance counters.
- `SemanticCacheController`: cache lookup, document retrieval, synthesis, grounding, consensus, fact extraction, save/load, and corpus validation.
- `CachePreWarmer`: programmatic cache saturation over query templates and corpus chunks.
- `Router`: dispatches simple extraction-style tasks to evaluator-class models and complex synthesis to executor-class models.
- `AutonomousAgent`: framework-agnostic cached query facade.

Cache/search flow:
- Check exact cache matches inside the source/corpus bucket.
- Use embedding + FAISS dragnet for near candidates.
- Use evaluator/sniper logic for semantic-equivalence cache hits.
- Query the extracted knowledge/fact index for cross-query reuse.
- On miss, retrieve documents with FAISS + reranker and synthesize a grounded answer.
- Verify provenance/consensus, then store the answer, embedding, and extracted facts.
- Apply context-collapse protections for oversized cached results.

## Benchmark Guidance

- Official RULER v2 work is in `ruler_v2/run_benchmark.py` and currently targets `mk_niah_basic`, `mv_niah_basic`, and `qa_basic`.
- NoLiMa work lives under `nolima/`; use `nolima/run_benchmark.py` for runs and `nolima/score_nolima_predictions.py` for scoring existing runs.
- Benchmark data and fixtures live under `benchmark_data/` and `benchmark_fixtures/`.
- Benchmark outputs and reusable cache state live under `benchmark_artifacts/`; treat historical artifacts as read-only unless the user explicitly requests edits or regeneration.
- Preserve persistent cache reuse, namespace derivation, manifest fields, and `delta_cost_usd` reproducibility when changing benchmark code.

## Coding Guidelines

- Keep changes scoped and reproducible; prefer existing patterns over new abstractions.
- Preserve corpus namespace isolation and load-time mismatch refusal.
- Keep pricing and benchmark cost assumptions centralized in `MODEL_FAMILY_PRICING_USD_PER_1K` in `semantic_cache_system.py`.
- Prefer `uv` workflows for Python commands in this repo when practical.
- Update relevant docs when architecture, benchmark flow, public return shapes, or evaluation assumptions change.

## Agent Behavior

- Treat this file as a high-level index, not complete architecture documentation.
- Read `docs/system_architecture.md` before substantial architecture changes.
- Read the relevant benchmark runner before changing benchmark behavior.
- Avoid broad rewrites, artifact churn, and unrelated formatting.
- Protect benchmark reproducibility, persistent cache behavior, and source provenance semantics.
