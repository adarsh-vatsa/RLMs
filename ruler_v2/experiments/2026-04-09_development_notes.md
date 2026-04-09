# Development Notes (2026-04-09)

## Scope of This Update
- Goal: improve safety of knowledge-hit cache reuse without introducing task-family compatibility checks.
- Constraint: keep cache behavior benchmark-agnostic so it can scale as a shared knowledgebase.

## Code Changes Implemented
File:
- `semantic_cache_system.py`

Updates in `SemanticCacheController`:
- Added knowledge-hit confidence gate constants:
  - `KNOWLEDGE_MIN_SCORE = 0.78`
  - `KNOWLEDGE_MIN_MARGIN = 0.03`
- Added benchmark-agnostic knowledge verifier cascade controls:
  - `KNOWLEDGE_VERIFIER_ENABLED = True`
  - `KNOWLEDGE_VERIFIER_ALWAYS = False`
  - `KNOWLEDGE_VERIFIER_SCORE_TRIGGER = 0.82`
  - `KNOWLEDGE_VERIFIER_MARGIN_TRIGGER = 0.06`
  - `KNOWLEDGE_VERIFIER_MIN_LEXICAL_SUPPORT = 0.20`
  - `KNOWLEDGE_VERIFIER_MAX_FACTS = 5`
- Knowledge-hit path now uses a cascade:
  - Stage A (cheap): score gate + initial fact candidate build.
  - Stage B (conditional): run knowledge Sniper verifier only on ambiguous/risky matches.
  - If verifier rejects, skip knowledge hit and fall back to retrieval/synthesis path.
- Added telemetry fields to knowledge-hit return payload:
  - `knowledge_top_score`
  - `knowledge_margin`
  - `knowledge_verifier_called`
  - `knowledge_verifier_allow`
  - `knowledge_verifier_reason`
  - `knowledge_verifier_confidence`
  - `knowledge_verifier_trigger_reasons`
  - `knowledge_lexical_support`
- Added execution summary metrics:
  - `knowledge_verifier_calls`
  - `knowledge_verifier_allowed`
  - `knowledge_verifier_rejected`

## Why This Change
- Previous behavior could reuse cached answers from semantically broad fact matches.
- This produced incorrect answer reuse in warm runs under high cache-hit rates.
- Confidence/margin-only gates were not enough for all collisions.
- The cascade adds an LLM semantic verifier only when risk signals are present, improving correctness while controlling latency/cost.

## Concrete Examples

Example 1: warm run answer drift on the same dataset
- Run A (cold/reset): `benchmark_artifacts/official_ruler_v2/20260406T001339Z`
- Run B (warm/no reset): `benchmark_artifacts/official_ruler_v2/20260406T002621Z`
- Both runs used the same dataset signature, but Run B reused richer cache state.
- Observed impact:
  - `sample_000002` changed output between runs.
  - Run A output was the NIAH-style cached answer.
  - Run B output was a QA-style "Document 3813903" answer.
- Interpretation: a knowledge-hit routed to a different cached source entry in the warm run.

Example 2: why a confidence gate is needed
- In the warm run investigation, top matched facts for `sample_000002` were tied to the QA-derived source entry (`source_cache_idx=1`) with close scores.
- Even when these facts looked semantically related in embedding space, the returned answer did not satisfy the query expectation.
- New behavior now applies:
  - minimum top score (`KNOWLEDGE_MIN_SCORE`)
  - top1-top2 margin (`KNOWLEDGE_MIN_MARGIN`) plus conditional verifier escalation
- Expected result: weak matches are rejected directly; ambiguous/risky matches are verified and may be rejected before fallback.

Example 3: expected skip log patterns after this patch
- Low-confidence case:
  - `[KNOWLEDGE] ✗ Skip: top score 0.742 below threshold 0.78`
- Ambiguous/risky case now escalates:
  - `[KNOWLEDGE-SNIPER] Escalating verifier (reasons=[...], lexical_support=...)`
  - If rejected: `[KNOWLEDGE-SNIPER] ✗ Reject ...` and fallback to retrieval/synthesis.
- Accepted case:
  - `[KNOWLEDGE] ✓ Fact Hit! ... (top1=0.861, margin=0.067)`

Example 4: what remains benchmark-agnostic
- No task labels are used in gating decisions.
- A knowledge hit can still be reused across benchmarks when cascade checks pass.
- This keeps cache architecture aligned with "shared large knowledgebase" design.

## Threshold Explanation

Two baseline confidence gates are still used in the cascade:

1. `KNOWLEDGE_MIN_SCORE = 0.78`
- Meaning: top-1 retrieved fact must have sufficiently strong similarity score.
- Rule: if `top1 < 0.78`, knowledge hit is rejected.

2. `KNOWLEDGE_MIN_MARGIN = 0.03`
- Meaning: top-1 must be clearly better than top-2 (avoid ambiguous retrieval neighborhoods).
- Rule: this is now a risk signal for verifier escalation when verifier is enabled. If verifier is disabled, low-margin candidates are rejected directly.

Why both are still needed:
- Score threshold prevents weak matches.
- Margin threshold prevents near-tie ambiguous matches.
- Combined with conditional verifier, this reduces false positives without forcing an LLM call on every knowledge candidate.

Quick decision examples:
- Accept: `top1=0.861`, `top2=0.794`, `margin=0.067`
- Reject (low score): `top1=0.742`
- Reject (ambiguous): `top1=0.812`, `top2=0.798`, `margin=0.014`

## Feigned-Stone Example Through Thresholds

Observed behavior before this patch:
- A `feigned-stone` answer was repeatedly reused for different prompts (including unrelated names/targets).
- In warm state, knowledge-hit routing could select a cached source entry that did not satisfy the current query objective.

How thresholds help in this case:
- If the selected fact neighborhood is weak (`top1 < 0.78`), reuse is blocked.
- If the neighborhood is ambiguous/risky, verifier is invoked and can block reuse semantically.
- Result: the system falls back to fresh retrieval/synthesis instead of reusing the wrong cached answer.

Important nuance:
- If a wrong cached candidate passes raw similarity gates, verifier still has a chance to reject it on semantic mismatch.
- This is still a probabilistic guard, so telemetry and threshold tuning remain important.

## What Was Deliberately Not Added
- No task-family or benchmark-specific compatibility checks.
- No hard partitioning by benchmark/task labels.

## Validation Status
- Static diagnostics on `semantic_cache_system.py` reported no errors after patch.

## Recommended Next Validation
1. Run one cold-start benchmark with cache reset and capture artifacts.
2. Run one warm benchmark on the same dataset and compare:
   - cache hit types (`exact`, `semantic`, `knowledge`)
  - `knowledge_verifier_calls`, `knowledge_verifier_allowed`, `knowledge_verifier_rejected`
  - count of verifier-triggered knowledge fallbacks from logs
   - prediction drift between cold and warm runs
3. Tune `KNOWLEDGE_VERIFIER_SCORE_TRIGGER`, `KNOWLEDGE_VERIFIER_MARGIN_TRIGGER`, and lexical support trigger for cost/quality tradeoff.

## Open Questions
- Should confidence thresholds be global or adaptive by cache density?
- Should verifier escalation policy be trigger-based only, or partially always-on for certain domains?
- Should runner manifests include counters for:
  - `knowledge_hits_skipped_low_score`
  - `knowledge_verifier_rejected`

## Future TODO Items
- [TODO] If we have an exact query match but the data has changed, cache still returns the previous answer. How to act in that case?
- [TODO] Review total token usage and cost