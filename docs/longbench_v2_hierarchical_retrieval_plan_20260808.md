# LongBench-v2 Fast Hybrid Cache Plan

**Status:** Base hybrid v1 implemented locally; Jarvis validation pending  
**Created:** August 8, 2026  
**Revised:** August 9, 2026  
**Scope:** The local Qwen LongBench-v2 semantic-cache path. The direct API
baseline and hosted-provider paths remain separate.

## Decision

Optimize for the fastest hybrid cache system, not for forcing document
retrieval on every cache miss.

The hybrid path will:

1. Establish the active source scope.
2. Check exact and semantic cache entries.
3. On a cache miss, send the complete request directly when it fits.
4. Use token-budgeted child retrieval only when the complete request is
   overlength, adding parent expansion only if child-only packed retrieval loses
   accuracy.
5. Store the answer for exact and semantic reuse.

This replaces the completed run's expensive iterative reader. The iterative
implementation remains a historical comparison until the hybrid path is
validated; it is not the intended fallback in the finished design.

## Evidence From The Completed Run

The cold run `20260716T204202Z` processed 1,509 rows in 34,425.537 seconds, or
9.56 hours.

| Stage | Hours | Share of elapsed time |
| --- | ---: | ---: |
| Search and synthesis | 8.69 | 90.9% |
| Document ingestion | 0.86 | 9.0% |
| Saving, context writes, and other overhead | 0.01 | 0.1% |

The 509 cache misses caused most of the work:

- 6,941 API calls, or 13.64 calls per miss.
- 175.4M input tokens and 3.59M output tokens.
- 12,209 of 17,112 available chunks visited, or 71.3%.
- 313 misses exhausted their configured scan budget.
- Only 23 misses stopped early.
- The 503 originals accounted for 9.41 hours of row time.
- The run produced 503 exact hits, 497 semantic hits, and zero knowledge hits.

The bottleneck belongs to the historical iterative configuration. Current code
also contains a packed path that retrieves and optionally reranks chunks, packs
them by an estimated token budget, and performs one synthesis call. That path
must be measured as a control before adding more retrieval machinery.

## Target Request Flow

```text
Activate source scope
    |
    +-- Exact cache hit --------------------------> return stored answer
    |
    +-- Semantic cache hit
    |       |
    |       +-- cached-query FAISS + verifier ---> return stored answer
    |
    +-- Cache miss
            |
            +-- Exact rendered request fits
            |       |
            |       +-- one full-context executor call
            |
            +-- Exact rendered request is overlength
                    |
                    +-- chunk and embed source children
                    +-- retrieve child anchors with document FAISS
                    +-- if accuracy requires it, expand and merge parents
                    +-- pack selected children or parents by exact token budget
                    +-- one executor call
```

There are two independent FAISS uses:

- The cached-query index remains active for semantic cache lookup on every
  source-scoped request.
- The document-chunk index is built and searched only for an overlength cache
  miss.

Direct-fit rows therefore skip document chunking, document embeddings,
document-index construction, document top-k retrieval, reranking, and iterative
inspection.

## Source Scope Before Cache Lookup

The current benchmark establishes `data_scope_hash` as a side effect of
document ingestion. Direct-fit routing must not depend on ingestion, because a
fit row should avoid that work entirely.

The runner must activate a deterministic source identity before calling the
cache lookup. The identity should be derived from the normalized source ID and
source content. Exact, semantic, direct-fit, and hierarchical routes for the
same source must use the same value.

This is required to prevent a query from reusing an answer stored for another
document. Loading persistent state must not leave the last saved source as the
active scope for the next row; every row or source group must activate its own
scope explicitly.

Document chunks created for an overlength source must carry this same source
identity. Chunk size and retrieval policy belong in the cache namespace, not in
the definition of the source itself.

## Exact Direct-Fit Routing

The direct route must reuse the strict non-thinking Qwen system/user message
shape from the direct API baseline:

```text
system: strict LongBench-v2 MCQ instruction
user:   complete context + complete question + choices
```

Use the executor tokenizer and chat template to count the complete rendered
request, including the generation prompt. Do not route using the CSV estimate,
word counts, character counts, or the current heuristic LLM token estimator.

Initial 256K service profile:

```text
Served context window: 262,144 tokens
Maximum rendered input: 240,000 tokens
Maximum output:         8 tokens
Safety margin:          22,136 tokens
Thinking:               disabled
Temperature:            0
```

The 240K route is enabled only after the vLLM startup log reports maximum
concurrency of at least `1.00x` for a 262,144-token request and the smoke test
shows no context-length failure.

If the complete rendered request is at most 240,000 tokens, send it unchanged in
one executor request. The hybrid route does not middle-truncate an overlength
request; overlength requests use retrieval. Middle truncation remains a direct
API baseline policy.

The completed run's chunk counts suggest that many originals will fit, but the
route count must be reported from exact measurements rather than inferred from
historical chunks.

## Post-Answer Work For Strict MCQ Rows

The current packed and iterative paths run consensus verification and fact
extraction after producing an answer. Those calls should not be part of the new
strict MCQ route:

- Consensus verification reads the source again but does not alter the answer.
- Fact extraction receives only the one-letter answer and cannot produce useful
  source facts.
- The completed run recorded zero knowledge-cache hits.

For this benchmark path, validate the answer locally as one of `A`, `B`, `C`, or
`D`, then store the query embedding, answer, source identity, route, and compact
provenance needed for exact and semantic reuse. Do not store the complete direct
context inside every cache entry. Its source hash and source ID provide the
scope; hierarchical rows additionally store selected source ranges.

This preserves the cache behavior the suite exercises while avoiding two
unproductive evaluator calls and repeated persistence of very large contexts.
The general non-MCQ controller behavior is outside this plan.

## Existing Packed Path As The First Control

Before implementing parent expansion, compare the existing packed path against
the historical iterative path on matched overlength originals.

The existing path already provides:

```text
FAISS retrieval
    -> optional reranking
    -> source packing
    -> one executor synthesis
    -> cache storage
```

Its current limitations are:

- `SYNTHESIS_MAX_CHUNKS` can stop packing before the token budget is used.
- Packing uses a heuristic token estimate rather than the rendered Qwen chat
  count.
- Retrieved 10K chunks exceed the embedding model's 8,192-token input limit, so
  some chunk content is invisible to the embedding.
- The current wrapper uses masked mean pooling, while the official Qwen3
  embedding contract uses left padding and last-token pooling.
- Retrieved chunks are sent as isolated evidence and are not expanded to regain
  surrounding context.

The control determines whether parent expansion is necessary. If existing
packed retrieval already meets the accuracy gate, stop before adding a planner,
sparse ledger, or multi-round reader.

## Overlength Retrieval Path

If the complete request exceeds 240,000 tokens, chunk only the source context.
The complete question and all choices remain in the answering request.

### Retrieval Children

Children are search anchors, not executor batches. The initial experiment should
use an embedding-tokenizer size safely below the 8,192-token embedding limit,
with overlap, and verify the final encoded length in tests. A starting profile is
approximately 7.5K tokens with 750 tokens of overlap.

Each child retains:

```python
{
    "source_id": "...",
    "source_scope_hash": "...",
    "child_index": 12,
    "token_start": 60000,
    "token_end": 67500,
    "char_start": 240000,
    "char_end": 270000,
    "text": "...",
}
```

The current tokenizer chunker already records token and character offsets.
Natural-language, code, table, and dialogue structural splitters are not part of
the first implementation.

### Dense Embedding Contract

Before judging dense retrieval quality, make the local embedding wrapper match
the official `Qwen/Qwen3-Embedding-0.6B` usage:

- Left-pad batched inputs.
- Select the final non-padding token representation rather than masked mean
  pooling.
- Apply the task-specific English instruction to queries only.
- Do not add that instruction to document children.
- L2-normalize query and document vectors before inner-product search.

Establish the dense FAISS control only after this correction. Record the pooling
method, padding side, query instruction, and embedding input limit so the new
vectors cannot share an incompatible persisted index or cache namespace.

### Child Retrieval

Embed the complete question and choices as the query and rank every child in the
document FAISS index. At the current LongBench-v2 scale, exact exhaustive
`IndexFlatIP` search remains cheap; the exact 240K packing budget, rather than a
separate top-k parameter, determines how many regions the executor reads.

Do not scan a fixed percentage of all source chunks. Candidate-count and
reranker changes should be evaluated only if the initial child retrieval misses
known evidence.

### Optional Lexical Hybrid

If corrected dense retrieval misses exact identifiers, names, numbers, symbolic
codes, or answer-choice wording, add BM25 over the same children as a
complementary candidate source. Keep FAISS for dense semantic retrieval.

Retrieve dense and BM25 candidate lists locally, combine them with reciprocal
rank fusion or a deduplicated union, and then apply the existing reranker only if
the paired experiment shows that it helps. BM25 must not introduce another LLM
call, and it does not determine how many regions the executor reads; exact token
packing still controls that.

Do not add BM25 by default merely because it is available. Enable it only when
matched retrieval or answer failures show that corrected dense retrieval lacks
the lexical signal.

### Parent Expansion If Needed

First evaluate exact-budget packing of the retrieved children themselves. If
that meets the accuracy gate, omit parent expansion.

If isolated children lose necessary surrounding context, expand each selected
child into a larger contiguous region of the original source, normally placing
the child near the center. Begin by testing a parent target near 32K executor
tokens.

Merge overlapping or adjacent parents and retain the child indices that caused
each region to be selected. A merged parent remains bounded; dense nearby hits
must not reconstruct an arbitrarily large document section.

No parent embeddings or separate parent index are required.

### Exact Token-Budget Packing

Rank the selected evidence regions—children or merged parents—by their child
scores and pack them until the next region would make the complete rendered
request exceed 240,000 tokens. Measure the actual chat-template token count
after every addition.

The token budget, not a legacy maximum-chunk setting, controls the executor
input. Remove the fixed synthesis chunk cap from the hybrid path rather than
retaining it as a compatibility override.

Send the selected regions in one strict executor request. Do not introduce the
iterative evidence ledger, sparse evidence JSON, or a second retrieval round in
the first version.

## Accuracy And Runtime Gates

“Accuracy tolerance” means the permitted benchmark accuracy difference between
matched approaches; it is not model confidence.

Use staged validation:

1. A three-row technical smoke containing a direct-fit row, a near-cap row, and
   an overlength row. This validates routing and context safety, not accuracy.
2. A paired 12-18-source diagnostic sample spanning fit and overlength rows,
   multiple domains, and previously fast and slow cases. Inspect every answer
   disagreement.
3. A paired 30-50-source gate before a full run.
4. The 503-original full evaluation only after the paired gate passes.

Working acceptance criteria:

- No cross-source cache reuse.
- Exact dependent rows reuse their original answer.
- Semantic hit behavior is preserved or improved on matched source groups.
- All executor answers are valid choice letters.
- No more than one additional incorrect original on a 50-source paired gate.
- On the full 503 originals, overall accuracy is within one percentage point of
  the historical iterative result, with no obvious domain-specific collapse.
- Original miss-path wall time is at least twice as fast on the paired sample.
- No context-length errors.

Report direct-fit and overlength accuracy separately so a strong direct-fit
majority cannot hide weak retrieval behavior.

## Required Artifact Data

The manifest and bridge rows must make the run attributable without copying
historical iterative fields into the new design.

Record at least:

- Hybrid route and route version.
- Served context window, maximum rendered input, output cap, and safety margin.
- Executor tokenizer and chat-template identity.
- Rendered request tokens used for each routing decision.
- Source scope hash and cache namespace.
- Exact, semantic, direct-fit, and overlength-packed route counts.
- Child tokenizer, size, overlap, and encoded-length maximum.
- Embedding pooling method, padding side, query instruction, and normalization.
- Document embedding count and time for overlength rows.
- FAISS candidate count and selected child indices.
- Whether BM25 was enabled, its candidate count, fusion method, and fused child
  indices.
- Selected evidence ranges, whether parent expansion was used, supporting child
  indices, merged-parent count, and packed tokens.
- Executor and semantic-verifier calls separately.
- Input/output tokens, ingestion time, search time, and row wall time.
- Cache-state size and save time.
- Valid-choice and answer correctness.
- Context-length and API errors.

The cache namespace must include every answer-affecting setting: models, strict
prompt version, direct-input budget, tokenizer/chat-template identity, child
profile, embedding contract, optional lexical-fusion policy, parent expansion
policy, retrieval settings, and hybrid route version. Historical artifacts and
the July 16 run remain unchanged.

## Implementation Sequence

### Phase 0: Verify Capacity And Establish Controls

1. Verify the 256K vLLM startup capacity and run the three-row context smoke.
2. Count exact rendered tokens for all 503 originals without calling the model.
3. Report exact direct-fit and overlength counts.
4. Run matched iterative and existing-packed controls on the first diagnostic
   sample.

### Phase 1: Cache-Safe Direct-Fit Route

1. Activate source scope independently of ingestion.
2. Perform exact and semantic cache lookup before routing.
3. Reuse the strict direct-baseline message builder and exact tokenizer handling.
4. Send fitting misses directly and skip document ingestion.
5. Store compact exact/semantic cache entries without MCQ consensus or fact
   extraction calls.
6. Add route, token, call-purpose, scope, and persistence telemetry.
7. Retain iterative routing temporarily only for overlength rows so Phase 1 can
   be validated independently.

### Phase 2: Minimal Overlength Packed Route

1. Correct the Qwen embedding wrapper to use left padding and last-token pooling.
2. Replace 10K retrieval chunks with embedding-safe children.
3. Establish the corrected dense FAISS control.
4. Replace heuristic/fixed-count packing with exact token-budget packing.
5. Produce one strict answer call and store compact provenance.
6. Evaluate child-only packed retrieval against the paired gate.

Only if child-only packed retrieval misses necessary surrounding context, expand
selected children into bounded parents, merge overlaps, and repeat the paired
gate.

### Phase 3: Evidence-Driven Corrections

Add only behavior justified by failures observed in the paired evaluation. For
example, add BM25 rank fusion when dense retrieval misses lexical evidence,
restore document order for ordering questions, or diversify candidates across
distant source regions when comparison questions miss evidence.

Do not introduce a general selection-planner abstraction. Call existing task
detectors directly when a demonstrated failure requires a narrow rule.

### Phase 4: Full Validation And Documentation

1. Run the paired 30-50-source gate.
2. Freeze the accepted policy and cache namespace.
3. Update the LongBench and Jarvis commands for that policy.
4. Run all 503 originals plus their exact and semantic dependents.
5. Validate artifact totals and produce a new date-stamped comparison note.

## Deferred Work

The following are intentionally outside the initial implementation:

- Structural parsing for code, dialogue, tables, or multi-document inputs.
- A multi-policy selection planner or MMR framework.
- Iterative or sparse evidence ledgers.
- Multi-round retrieval.
- Concurrent source-group execution.
- Cross-run child-index persistence.
- General semantic-cache controller refactoring outside LongBench strict MCQ.

Revisit concurrency only if the hybrid route passes the accuracy gate and the
remaining measured runtime is still too high.

## Remaining Experimental Choices

These values remain deferred until the base hybrid measurements justify them:

- Whether overlength retrieval benefits from the existing reranker.
- Whether corrected dense retrieval requires optional BM25 fusion and, if so,
  which simple fusion method passes the paired gate.
- Parent target and maximum size if parent expansion is required.
- Narrow task-specific corrections justified by paired failures.

Choose them from matched measurements rather than dataset medians alone.
