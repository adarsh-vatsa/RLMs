# RULER2 Run Findings (2026-04-05)

## Execution Summary
- `20260331T162513Z`: Early official bridge/scoring artifacts test. Added `--official-cache-reset` modifier as every run was using cold start. (580 seconds)
- `20260405T194658Z`: First deep-dive run; 5/6 cache hits with repeated answer pattern. First-token pooling was used for query/document embeddings which caused queries with shared instruction prefixes to collapse under identical vectors (defined in `encode()` method of `EmbeddingEngine` class, `semantic_cache_system.py`). This made fact/query similarity appear perfect and amplified false knowledge hits (increased false-positives in cache hits).
- `20260406T001339Z`: Cold-start baseline after reset; mixed cache usage (4/6 hits). Accuracy: 0.88 (327 seconds)
- `20260406T002621Z`: Warm run with same dataset; 6/6 hits and different overall score vs cold start.Accuracy: 0.95 (211 seconds)
- `20260406T134708Z`: Bug fix: Multi-value task was not calculating accuracy using the whole set of expected answers leading to wrong results.
- `20260406T140046Z`: Clean run after bug fix (cache reset, 328 seconds)
- `20260406T142949Z`: Execution with cache (207 seconds)

## Scope
- Goal: understand odd scoring values and repeated answers with high cache-hit rate.

## What We Observed
- Report showed fractional "accuracy" values (for example 0.5, 0.6) even when samples=1 per task-length bucket.
- Predictions across all 6 samples were effectively the same answer text.
- Manifest reported cache_hits=5/6.
- Bridge rows showed those 5 hits were all cache_type="knowledge".

## Why Fractional Scores Happened
- Scoring uses NeMo evaluator via score_ruler2_predictions.py.
- In this run shape, expected_answer is passed as a plain string.
- NeMo RULER2 "all" matching path can iterate over that string in a way that yields partial/continuous values.
- For sample_000005 (expected "9653250072"), score became 0.6 because partial character overlap in generated text contributed to the score.
- Conclusion: current report values are not strict binary correctness for these rows.

## Why We Got 5 Cache Hits
- First sample was a cold miss and created one cache entry + one knowledge fact.
- Subsequent samples loaded the same persisted cache namespace/state.
- Knowledge retrieval branch matched query embedding to stored fact embedding (threshold > 0.75).
- On knowledge hit, system returns the linked source cache answer.
- Since cache state had only one source entry, later hits reused that same answer text.

## Current Hit Definition (Important)
- A benchmark "cache hit" is counted when output.from_cache == true.
- This includes exact, semantic, and knowledge routes.
- It does not imply answer correctness.

## Risk for Benchmark Interpretation
- High cache-hit rate can coexist with wrong repeated answers.
- Fractional evaluator scores can look better than strict correctness when references are handled as plain strings.

## Bug Found During Investigation (Fixed)
- Root cause identified in embedding pipeline: first-token pooling was used for query/document embeddings.
- With a shared instruction prefix, different queries collapsed to identical vectors (observed cosine=1.0 and zero vector diff across distinct queries).
- This made fact/query similarity appear perfect and amplified false knowledge hits.
- Patch applied: switched to attention-masked mean pooling in the embedding encoder.
- Post-fix validation: distinct queries now produce distinct vectors (cosine values vary and are no longer all 1.0).

## Important Rerun Note
- Existing cache state and old artifacts were produced with the buggy embeddings.
- For valid post-fix evaluation, run with cache reset (or delete the old cache namespace) before generating new benchmark artifacts.

## New Finding From Warm Run Comparison
- Compared consecutive runs: `20260406T001339Z` (cold/reset) vs `20260406T002621Z` (warm/no reset).
- Dataset signature and selected samples stayed the same.
- Overall accuracy changed (0.8833 -> 0.95) because cache state differed at run start.
- In warm run, `sample_000002` (mv_niah_basic|32768) returned a QA-style cached answer instead of the prior NIAH-style cached answer.
- Root cause in current logic: knowledge-hit retrieval selected facts linked to `source_cache_idx=1` (QA-derived entry), then returned that source answer for a non-QA query.
- This confirms a cross-task cache-routing issue in knowledge-hit policy, even after fixing embedding collapse. NOTE: This might not be an issue depending how we want to use it.

## Future TODO Items
- [DONE] Add guardrails for knowledge hits (for example, validate candidate answer/query compatibility before reuse). Competed on: 2026-04-05
- [TODO] Report hit types separately in run summaries and dashboards: exact, semantic, knowledge.
- [TODO] Add strict/binary scoring mode for numeric-answer tasks (alongside NeMo-compatible score mode).
- [TODO] Add a reproducible evaluation mode that freezes initial cache state (or disables writes) for deterministic A/B comparisons.