---
description: General project context and coding guidelines for the `adarsh-rlms` repository, which focuses on a two-stage semantic cache system for autonomous LLM workflows. This file provides an overview of the project structure, benchmark details, and coding best practices to ensure consistency and reproducibility across contributions.
applyTo: '*'
---

Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.

## Project Summary

- `adarsh-rlms` is a Python research and benchmarking repository for a two-stage semantic cache system designed for autonomous LLM workflows.
- The core implementation lives in `semantic_cache_system.py`, with benchmark orchestration in `ruler_v2/run_benchmark.py`.
- Supporting assets include prepared benchmark data (`benchmark_data/`), fixtures (`benchmark_fixtures/`), and technical documentation (`docs/`, `ruler_v2/docs/`).

## Benchmarks

- RULER v2 benchmark orchestration is handled in `ruler_v2/run_benchmark.py`, currently focused on `mk_niah_basic`, `mv_niah_basic`, and `qa_basic` tasks across long-context settings.
- NoLiMa benchmark work is included under `nolima/` to expand evaluation coverage.
- RULER v2 benchmark outputs and reusable cache state are stored under `benchmark_artifacts/official_ruler_v2/`.

## Coding Guidelines

- Keep changes scoped and reproducible; avoid editing historical benchmark artifacts unless explicitly requested.
- Preserve benchmark compatibility and cache-reuse behavior when modifying RULER v2 or NoLiMa benchmark logic.
- Keep benchmark cost assumptions explicit and centralized in `semantic_cache_system.py` (`MODEL_FAMILY_PRICING_USD_PER_1K`) so `delta_cost_usd` remains reproducible.
- Use `uv`-based Python workflows where possible in this repository.
- Update relevant docs when architecture, benchmark flow, or evaluation assumptions change.

## Instructions
- Update this file with any new project context, coding guidelines, or instructions for AI to follow when generating code, answering questions, or reviewing changes.