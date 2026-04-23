# 2026-04-22 Development Log

## Summary

- Added agent instruction indexes for GitHub/Codex workflows:
  - `.github/instructions/project-instructions.instructions.md`
  - `AGENTS.md`
- Fixed a RULER `qa_basic` cache correctness issue where identical question text could reuse an answer from a different document context.
- Added data-scoped cache eligibility for retrieval-based `search()` hits.
- Added focused regression coverage for scoped exact hits, scoped knowledge hits, scoped persistence, and legacy unscoped cache entries.
- Ran a focused RULER QA validation run:
  - Run: `benchmark_artifacts/official_ruler_v2/20260422T001840Z`
  - Scored samples: 3
  - Overall accuracy: 1.0
  - Cache hit rate: 0.333333

## Cache Correctness Fix

Problem:
- `search()` previously allowed a global exact query match across cached retrieval answers.
- In RULER `qa_basic`, two samples can contain the same question text but different document contexts and different expected document IDs.
- The exact query comparison was correct at the string level, but unsafe because the underlying data scope differed.

Resolution:
- Added `data_scope_hash` as the active document-set identity for retrieval search.
- `ingest()` computes this scope from sorted `.txt` filenames, normalized file contents, chunk size, and overlap.
- `store()` records the active scope on cache entries and extracted facts.
- `search()` now filters exact, semantic, and knowledge cache hits by active scope.
- Legacy persisted entries without `data_scope_hash` are skipped for scoped `search()` hits.

Validation:
- Added `test/test_data_scope_cache.py`.
- Verified with:
  - `python -m unittest discover -s test -p 'test_*.py'`
  - `python -m py_compile semantic_cache_system.py test/test_data_scope_cache.py`

### Hashing Flow Clarification

The system already had hashing before `data_scope_hash`. The new field does not replace the old hash; it covers a different level of cache identity.

`source_chunk_hash`:
- Identity of the exact source context/chunk that produced an answer.
- Used by direct `cached_query(query, context)` calls.
- Flow:
  - compute `source_chunk_hash = hash(context)`
  - look only inside `cache[source_chunk_hash]`
  - same query over another document gets another hash bucket
  - miss, call model, and store under the new source chunk hash

`data_scope_hash`:
- Identity of the active ingested document set for retrieval `search(query)`.
- Computed by `ingest()` from sorted filenames, normalized file contents, chunk size, and overlap.
- Used as an eligibility gate for global retrieval cache hits.
- Flow:
  - `ingest(docs_dir)` computes active `data_scope_hash`
  - `search(query)` scans cache candidates
  - exact/semantic/knowledge hits are allowed only when `entry.data_scope_hash == current data_scope_hash`
  - same query over a different ingested document set misses and regenerates

Example:
- Sample 1 asks the Scott Derrickson question over a context where the correct document is `1544120`.
- The system stores the answer with `data_scope_hash = A` and `source_chunk_hash = hash(retrieved_source_text)`.
- Sample 2 asks the same question text over a context where the correct document is `9214801`.
- `ingest()` computes `data_scope_hash = B`.
- The old answer has matching query text but `A != B`, so `search()` rejects the cache hit and regenerates from the new active document set.

Short rule:
- `source_chunk_hash` answers: which exact source context produced this cached answer?
- `data_scope_hash` answers: which active document set was being searched when this answer was produced?

## Benchmark Notes

- The focused run confirmed the same-question/different-context failure mode is addressed for the small `qa_basic` scenario.
- The manifest for `20260422T001840Z` reports only `qa_basic` as implemented in that run, even though the command requested the milestone task list.
- Next benchmark work should restore broader milestone coverage and add more diagnostic telemetry for retrieval, cache eligibility, and answer provenance.

## Follow-Up Design Decisions

### 5. Knowledge Cache Reliability

Current state:
- Cache entries and extracted facts now include `data_scope_hash`, so facts cannot be reused across different active document sets.
- Extracted facts also include `source_cache_idx` and `source_chunk_hash`, so each fact can be traced back to the cached answer and source context that produced it.

Remaining gap:
- `source_chunk_hash` tells us which cached source context produced the fact.
- It does not tell us where the fact appears inside that source context, whether the exact fact text was found in the source, or whether the extracted relation is supported by a specific span.
- For example, `Scott Derrickson | directed | Doctor Strange` has scope and chunk identity, but no `(char_start, char_end)` span proving where that relation appears.

Design decision:
- Keep `data_scope_hash`, `source_cache_idx`, and `source_chunk_hash`; add source-span support on top.
- Target fact shape should include:
  - `source_doc_id`
  - `source_chunk_hash`
  - `data_scope_hash`
  - `char_start`
  - `char_end`
  - `support_text`
  - `verifier_status`
  - `extraction_confidence`

Should related chunks be fed into an LLM again?
- Not for exact or semantic-equivalent cache hits; those can replay the cached answer.
- For knowledge hits, yes, but selectively:
  - If the query asks for a fact that can be answered directly from verified fact fields, answer from the structured fact without another full synthesis call.
  - If the query needs explanation, aggregation, or natural-language synthesis, feed only the supporting fact spans/chunks into a cheap verifier/synthesizer.
  - Do not replay the whole old cached answer for a related query unless the verifier confirms it directly answers the new query.

Practical next implementation:
- Add a deterministic source-span matcher after `_extract_facts()`.
- For each extracted fact, search the source context for exact or fuzzy support of the subject/object and relation context.
- Store matched spans on the fact.
- Require span support, or an explicit verifier approval, before serving a knowledge hit.

### 6. Provenance And Grounding

Current state:
- `_grounding_check()` verifies quantitative strings such as dollar amounts, percentages, and comma-separated numbers.
- This is useful for finance-style hallucination protection.
- It is not enough for RULER QA, where answers often depend on document IDs, entity names, titles, or exact text snippets.

Where to add this:
- In `semantic_cache_system.py`:
  - Extend `_grounding_check()` or add a second method such as `_grounding_check_spans()`.
  - Run it in `store()` alongside the existing grounding check.
  - Include grounding/provenance fields in `search()` return values for both cache hits and misses.
- In `ruler_v2/run_benchmark.py`:
  - Add benchmark telemetry fields to `bridge_rows.jsonl`.
  - Useful fields: `data_scope_hash`, `cache_rejection_reason`, `faiss_top_ids`, `rerank_top_ids`, `final_source_ids`, `expected_answer_in_retrieval`, `expected_answer_in_rerank`, `expected_answer_in_prediction`, `grounding_status`.

Design decision:
- Keep the current numeric grounding as a cheap first pass.
- Add span-based grounding for nonnumeric claims:
  - document IDs
  - dates
  - entity names
  - titles
  - quoted text
  - expected benchmark answers
- Treat grounding as evidence metadata, not just a label. Store the supporting source span whenever possible.

Practical next implementation:
- Add extraction helpers for document IDs and expected-answer-like strings.
- During benchmark runs, compare expected answers against:
  - retrieved FAISS candidates
  - reranked candidates
  - synthesis source context
  - final prediction
- This gives a failure taxonomy: retrieval miss, reranker miss, synthesis miss, cache-scope miss, or answer-format miss.

### 8. Router Improvements

Current state:
- `Router.select_model(query)` is a deterministic keyword heuristic.
- It does not make an LLM call.
- It returns only a model name, so benchmark telemetry cannot explain why a route was chosen.

Should the router be an LLM call?
- Default answer: no.
- A router runs on every miss, so making it an LLM call can erase part of the cost savings.
- Start with deterministic or lightweight routing and reserve LLM routing for ambiguous/high-risk cases.

How to decide task types:
- First use metadata when available:
  - RULER task name: `mk_niah_basic`, `mv_niah_basic`, `qa_basic`
  - domain: `legal`, `finance`, `benchmark`, etc.
  - mode: `baseline` or `cache`
- Then use query-shape rules:
  - `extract`, `find`, `list`, `count`, `classify` → extraction/lookup
  - `summarize`, `explain`, `compare`, `analyze`, `why` → synthesis/reasoning
  - `Most relevant document index:` → benchmark document-ID lookup
- Then use runtime signals:
  - cache hit type
  - retrieval confidence
  - reranker margin
  - context length
  - budget remaining

Design decision:
- Replace `select_model(query) -> str` with a richer route decision object over time:
  - `model`
  - `task_type`
  - `confidence`
  - `estimated_cost_usd`
  - `reasons`
  - `fallback_policy`
- Keep backward compatibility by allowing callers to use the selected model string.

Practical next implementation:
- Add a deterministic task classifier for benchmark/domain clients.
- Add router telemetry to `bridge_rows.jsonl`.
- Add ablation modes:
  - current keyword router
  - always executor model
  - always evaluator model for simple extraction
  - task-aware deterministic router
  - optional LLM router only for ambiguous cases

## Recursive Language Model Framing

This project should be framed as memory and memoization infrastructure for Recursive Language Models.

An RLM decomposes a hard task into many smaller language-model calls. That recursive pattern creates repeated subquestions, repeated document chunks, repeated extraction templates, and repeated intermediate summaries. This semantic cache is useful because it turns those repeated recursive subcalls into deterministic, scoped lookups.

Recommended positioning:

> Recursive Language Models decompose reasoning into repeated language-model subcalls; this system makes those recursive subcalls reusable, grounded, and economically viable through data-scoped semantic caching.

How the system fits:
- `AutonomousAgent.cached_query(query, context)` can replace raw LLM calls inside an RLM loop.
- Exact and semantic cache hits reduce redundant recursive subcalls.
- `data_scope_hash` and `source_chunk_hash` prevent identical query templates from crossing document or document-set boundaries.
- Knowledge extraction lets one recursive branch reuse verified facts discovered by another branch.
- Context Collapse Guard prevents parent recursive nodes from receiving oversized child outputs.
- Provenance, grounding, and consensus make recursive intermediate results auditable instead of ephemeral.

Minimal integration shape:

```python
def rlm_call(query: str, context: str) -> dict:
    return agent.cached_query(query, context)

def map_over_chunks(task: str, chunks: list[str]) -> list[dict]:
    return [rlm_call(task, chunk) for chunk in chunks]

def reduce_results(question: str, child_results: list[dict]) -> dict:
    context = "\n\n".join(item["result"] for item in child_results)
    return rlm_call(question, context)
```

Why this is useful:
- It attacks recursive cost explosion by memoizing repeated subcalls.
- It attacks recursive context explosion by summarizing or marking large cached results as ephemeral.
- It attacks recursive nondeterminism by reusing verified answers for equivalent scoped subproblems.
- It creates an audit trail from root answers back to cached leaf computations.

RLM-specific future work:
- Add a formal `RecursiveExecutor` or `RLMRuntime` wrapper around `cached_query()`.
- Track recursion metadata such as `parent_call_id`, `depth`, `branch_id`, `task_template`, and `source_chunk_hash`.
- Cache leaf extraction calls and reduction/synthesis calls separately.
- Add DAG reuse so two branches asking the same scoped subproblem run once and share the result.
- Add recursion-aware telemetry showing cache hits, misses, cost, and grounding by depth.
