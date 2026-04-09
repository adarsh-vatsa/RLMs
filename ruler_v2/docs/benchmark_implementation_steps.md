# Benchmark Implementation Steps and Status

This file is the living implementation log for benchmark integration work.
It records completed work, current state, and next steps so progress is reviewable across sessions.

## Scope

- Build official RULER v2 benchmarking path
- Generate architecture predictions for official prepared samples
- Invoke official evaluator command without custom score overrides
- Emit reproducible official artifacts and manifest reporting

## Current State (as of 2026-03-29)

### Completed

1. Added architecture benchmark pipeline and telemetry capture.
2. Added run modes: baseline, cache, both.
3. Added query suite loaders: jsonl, json, txt.
4. Added output artifacts: per_query.jsonl, per_query.csv, summary.json.
5. Added cold/warm pass support and summary aggregation.
6. Added cache-read bypass support in retrieval search path.
7. Fixed cache-hit metrics accounting for retrieval search path.
8. Fixed fresh-run cache indexing by lazy-initializing cache and knowledge indices.
9. Added starter fixtures in benchmark_fixtures for smoke tests.
10. Refactored runner to benchmark-only flow (removed query/interactive paths).
11. Renamed runner from run_with_docs.py to run_benchmark.py (now located at `ruler_v2/run_benchmark.py`).
12. Updated README command references to use `ruler_v2/run_benchmark.py`.
13. Polished benchmark CLI semantics (required query suite, explicit pass choices).
14. Replaced ambiguous state flags with explicit --reuse-state-path.
15. Added expected-answer scoring gate with configurable mode/threshold and optional fail-on-gate.
16. Added function-level docstrings and benchmark-flow comments in `ruler_v2/run_benchmark.py`.
17. Added RULER adapter lane with auto/on/off parsing and context-to-doc materialization.
18. Added a starter RULER-style pilot fixture and README usage example.
19. Added embedding-based semantic scoring modes (`semantic`, `hybrid-semantic`).
20. Added official RULER v2 bridge flow in `ruler_v2/run_benchmark.py` with:
   - official prepared-data ingestion (`--official-prepared-data`)
   - optional official prep command execution (`--official-prep-command`)
   - per-sample architecture prediction generation for official records
   - optional official evaluator command execution (`--official-eval-command`)
   - official run artifacts: `predictions.jsonl`, `bridge_rows.jsonl`, `manifest.json`
21. Added baseline progress accounting in official manifests against 13-task baseline.
22. Removed local adapter/scoring benchmark pathways and kept `ruler_v2/run_benchmark.py` official-only.

### In Progress

1. Wire exact NeMo-Skills RULER v2 production commands for prep/eval in target environment.
2. Calibrate semantic scoring thresholds and compare against lexical-only baselines.
3. Execute milestone matrix once API credits are available.

### Implemented Task Count vs 13-Task Baseline

1. Implemented tasks (3/13): `mk_niah_basic`, `mv_niah_basic`, `qa_basic`.
2. Pending tasks (10/13): not yet bridged in milestone run scope.

### Known Risk to Address

1. Knowledge-hit path can return incorrect cached answers on some query types.
2. Lexical scoring can miss semantic equivalence in paraphrased correct answers.

## Step-by-Step Plan

1. Keep benchmark control plane in a dedicated script.
2. Ensure only benchmark-related CLI arguments and execution paths remain.
3. Preserve fair comparison protocol:
   - baseline: cache reads bypassed
   - cache: full architecture behavior
4. Keep artifact schema stable for downstream analysis.
5. Add quality scoring layer in the next increment.
6. Add semantic scorer enhancement after lexical quality gate baseline is stable.

## Change Log

### 2026-03-26

- Initialized living implementation log.
- Recorded benchmark integration status and pending refactor tasks.
- Converted benchmark runner to benchmark-only script.
- Renamed runner to `ruler_v2/run_benchmark.py` and updated README usage examples.
- Removed legacy run_with_docs.py path (no compatibility wrapper).
- Tightened benchmark CLI constraints for safer benchmark execution.
- Replaced --state and --reuse-state with explicit --reuse-state-path.
- Added scoring gate controls: --score-mode, --score-threshold, --fail-on-score-gate.
- Added concise function docstrings and parameter explanations in `ruler_v2/run_benchmark.py`.
- Added RULER adapter controls: --ruler-adapter off/auto/on.
- Added RULER adapter pilot fixture and docs.
- Added semantic scoring modes: --score-mode semantic and hybrid-semantic.

### 2026-03-29

- Added official RULER v2 bridge execution mode to `ruler_v2/run_benchmark.py`.

### 2026-04-09

- Moved benchmark docs under `ruler_v2/docs/`.
- Standardized benchmark command references to use `ruler_v2/run_benchmark.py` and `ruler_v2/score_ruler2_predictions.py`.
- Added official flow CLI flags for tasks/lengths, prepared data, prep command, and evaluator command.
- Added per-run manifest progress accounting against 13-task baseline.
- Validated official bridge smoke run on local fixture shape and generated official-flow artifacts.
- Simplified runner to official-only benchmark mode and removed unrelated local benchmark lanes.
