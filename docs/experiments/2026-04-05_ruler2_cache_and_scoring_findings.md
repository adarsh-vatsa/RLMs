# RULER2 Run Findings (2026-04-05)

## Scope
- Run analyzed: benchmark_artifacts/official_ruler_v2/20260405T194658Z
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

## Suggested Immediate Follow-Ups
- Add guardrails for knowledge hits (for example, second-stage validation before returning cached answer).
- Separate metrics in reports: exact hit, semantic hit, knowledge hit.
- Add strict/binary scoring mode for numeric tasks to complement NeMo-compatible reporting.

## Bug Found During Investigation (Fixed)
- Root cause identified in embedding pipeline: first-token pooling was used for query/document embeddings.
- With a shared instruction prefix, different queries collapsed to identical vectors (observed cosine=1.0 and zero vector diff across distinct queries).
- This made fact/query similarity appear perfect and amplified false knowledge hits.
- Patch applied: switched to attention-masked mean pooling in the embedding encoder.
- Post-fix validation: distinct queries now produce distinct vectors (cosine values vary and are no longer all 1.0).

## Important Rerun Note
- Existing cache state and old artifacts were produced with the buggy embeddings.
- For valid post-fix evaluation, run with cache reset (or delete the old cache namespace) before generating new benchmark artifacts.
