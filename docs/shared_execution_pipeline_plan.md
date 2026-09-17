# Shared execution pipeline: LongBench-v2, AA-LCR, and MRCR v2

Status: direct/hybrid implementation completed with mocked validation; real
benchmark runs and held-out transfer validation remain separate. Existing entry
points preserve legacy defaults; select `--execution-profile common` explicitly.
See the [runbook](jarvis/SHARED_EXECUTION_RUNBOOK.md) for commands, artifacts, and
remaining persistent-cache/provider limitations. The sections below retain the
implementation contract and migration sequence.

See the [architecture diagrams](shared_execution_architecture.md) for component
boundaries, request routing, and the separation of scoring from cache verification.

## Objective and scope

Make all three benchmarks use one direct/hybrid execution pipeline. An adapter
should define the task's inputs, prompt format, output contract, and scoring;
it should not implement retrieval, routing, cache policy, or evidence selection.

The first migration covers direct inference and the existing dense hybrid
architecture. Packed synthesis, iterative reading, and RLM experiments remain
explicitly separate architectures until migrated and validated against the same
contracts. Do not describe those paths as unified merely because they share
embedding code. Do not add a plugin framework or rewrite the embedding engine.

Separate two changes: first consolidate existing behavior under explicit legacy
configurations; then introduce and evaluate a common configuration for paper
results. Preserve historical artifacts and give changed behavior new versions.

## What currently differs

| Area | LongBench-v2 | AA-LCR | MRCR v2 |
|---|---|---|---|
| Sources | One context per source ID | Ordered set of named documents | Conversation body; few-shot prefix kept outside retrieval |
| Prompt/output | MCQ system/user messages; constrained A-D decoding | Official document/question wrapper; free-text answer | Original text prompt in one user message; copied answer with marker |
| Hybrid cache | Reads/writes; persistent state supported | Reads/writes within run; saves state | Answer cache disabled |
| Packing | Score order; overlapping text retained | Score order; each selected child rendered as a document | Merge ranges; restore source order; mark omitted regions |
| Ingestion | Core `ingest()`; estimated-token fallback possible | Core `ingest()`; estimated-token fallback possible | Exact offsets required; adapter builds index |
| Oversized direct | OpenAI-compatible baseline removes middle | 64K removes middle; 262K preflight rejects overflow | Unsupported; no inference |
| Grading | Choice matching | Separate LLM grader | Official similarity plus strict metrics |

Execution evidence: [LongBench runner](../long_bench_v2/run_benchmark.py),
[LongBench direct runner](../long_bench_v2/run_api_benchmark.py),
[AA-LCR runner](../aa_lcr/run_benchmark.py),
[AA-LCR prompt/packing helpers](../aa_lcr/prompting.py),
[MRCR runner](../mrcr_v2/run_benchmark.py), and
[shared core](../semantic_cache_system.py).

AA-LCR also hardcodes experiment budgets and hybrid chunk sizes, and requires a
local vLLM evaluator for hybrid cache verification. Its grading and cache
verification currently share service configuration even though they serve
different purposes. These constraints become explicit configuration or separate
service roles, not benchmark-specific execution branches.

## Adapter boundary

Use small typed records and ordinary functions. Keep existing dataset preparation
formats; normalize examples when loading them instead of converting every dataset.

| Contract | Required information |
|---|---|
| Source group | Ordered documents, stable document IDs, content hashes, exact text loaded one group at a time |
| Solver input | Example ID, source group, question/retrieval query, fixed instructions/few-shot text, response contract |
| Evidence range | Document ID, half-open character offsets, retrieval score, child ID |
| Execution configuration | Models/revisions, budgets, routing, indexing, packing, cache and generation settings |
| Execution result | Prediction, route/status, evidence ranges, counts, timings, usage, finish reason and failures |
| Evaluation record | Reference answer, scoring metadata, and optional gold positions; never passed to the solver |

Adapters load/select examples, produce the solver input, and render either the
full source or an engine-selected evidence view. Rendering may add official
wrappers and labels, but cannot retrieve, select, reorder, summarize, or discard
evidence. The engine owns candidate selection and calls the renderer to measure
the actual request. Retrieval-query construction is explicit and versioned; it
uses only task input (including answer options where applicable), never gold data.

Scoring consumes the immutable execution result and a separate evaluation record.
Do not pass the original dataset row, with references attached, through the solver.

## Shared execution flow

1. Resolve an immutable configuration and validate budgets and service capabilities.
2. Load one source group; render/count each complete request and apply any requested
   source-length bounds. Keep MRCR's preparation/tokenizer compatibility checks.
3. Optionally look up an answer using the configured cache policy and source scope.
4. On a miss, send the complete request if it fits. Otherwise apply the explicit
   direct overflow policy, or enter hybrid retrieval.
5. For hybrid overflow, create/reuse the group's document index, rank candidates,
   optionally rerank, and pack evidence under the complete rendered input budget.
6. Generate with the shared client, retries, output contract, and usage accounting.
7. Optionally cache the generated answer, then persist the execution result.
8. Score independently and persist evaluation results and reports.

Preflight uses the same renderer, tokenizer, bounds, and routing functions. It
does not instantiate embedding models, contact a grader/verifier, or run inference.
With caching enabled, preflight reports routes assuming a cache miss; it cannot
predict semantic hits. Fitting prompts do not require document embeddings.

### Indexing and packing

- Provide a public in-memory indexing API; adapters must not assign controller
  private fields. Reuse the existing tokenizer chunker, embedding engine, FAISS,
  and dense retrieval implementation through this API.
- Chunk each document independently using exact tokenizer offsets in the common
  profile. Never silently estimate tokens or truncate embedding inputs.
- Key index reuse by ordered source content plus tokenizer, chunking, and embedding
  configuration. Group questions deterministically and hold one source group at
  a time. Reuse is in-process initially; persistent index storage is outside scope.
- Rank all children in the common dense profile. Add candidates in score order,
  merge overlapping/adjoining ranges within the same document, restore document
  order then character order, render, and count the complete request. Stop before
  the next candidate would overflow. Record the stop reason and selected ranges.
- Never merge across documents or deduplicate equal text at distinct source
  positions: repeated occurrences can be meaningful.
- Preserve exact source slices and signal omitted regions. Keep fixed instructions,
  few-shot examples, questions, and output instructions outside retrieval.
- For AA-LCR, retain original document identities/numbers when rendering fragments;
  two retrieved chunks from document 3 must not become documents 1 and 2. Version
  this rendering change separately from the legacy child-as-document behavior.

### Explicit system settings

Expose the same relevant options through all three thin runner CLIs. Resolve
existing experiment names and environment variables into configuration once;
the pipeline must not read mutable global settings to decide per-run behavior.

| Setting | Common quality profile | Other supported use |
|---|---|---|
| Execution mode | `direct` or `hybrid` | Same engine and prompt contract |
| Direct overflow | `unsupported` | Explicit middle truncation or preflight rejection for legacy reproduction |
| Chunk size/overlap | 7,500 / 750 embedding tokens | Configurable, validated against embedding limit |
| Candidate ranking | All children by dense score | Optional existing reranker with explicit limits |
| Evidence order | Source order | Score order for legacy runs/ablation |
| Merge overlaps | Enabled | Disabled for legacy runs/ablation |
| Answer-cache reads/writes | Both disabled | Independent switches for cache experiments |
| Cache matching | Off | Exact, or existing semantic matching plus verifier |
| Document-index reuse | Enabled within run | Independent of answer caching |
| Temperature/thinking | 0 / disabled | Record effective settings |
| Source bounds | Explicit inclusive interval when requested | Independent of executor input allowance |
| Input/output/context budgets | Explicit, with input + output <= context | Shared input budget for matched comparisons |

Keep executor model/revision, embeddings, chunking, packing, and input budget fixed
across benchmarks for the primary comparison. Output contracts and output
allowances can differ by task: an A-D answer and a long copied response have
different needs. Declare these differences before evaluation and retain the
same settings for each benchmark's direct/hybrid pair. Do not inherit conflicting
runner defaults silently or claim that all three tasks have identical prompts.

### Separate cache verification from grading

Cache verification is an optional solver operation. It may inspect the current
query, previous query/answer, source identity, and response contract, but never
reference answers, grades, or gold positions. Cache writes must not depend on
whether the answer later receives a correct grade.

Namespace answer-cache state by source scope, execution/prompt/response-contract
versions, model, and behavior-affecting configuration. Exact hits must respect
the full query and formatting requirements. Semantic verification must preserve
requirements such as MRCR's occurrence number and marker; do not rewrite an old
answer programmatically to make it match a new question. Version namespaces so
new runs cannot silently reuse incompatible historical state.

AA-LCR's grader receives the reference only after execution. Give grader and cache
verifier separate endpoint/model/credential configuration and usage accounting;
they may intentionally point to the same service. Hosted grading must not be
rejected merely because hybrid retrieval is enabled. Execution-only runs can
save predictions and use the existing regrading workflow later.

LongBench and MRCR require no answer-grading service. A verifier is required only
when semantic answer-cache matching is enabled. Service startup and readiness
checks must follow these effective requirements.

## Implementation sequence

1. **Capture current contracts.** Add small characterization fixtures for each
   runner: exact messages, route, selected ranges, cache calls, usage and failure
   handling. Inventory current CLI/experiment presets and artifact consumers.
   Use mocks only; do not regenerate historical benchmark runs.
2. **Create the shared execution package.** Add `execution/` with contracts,
   client/token-count helpers, retrieval/indexing, packing, cache integration,
   and a single `execute()` orchestration function. Move genuinely shared helpers
   out of `aa_lcr`/`long_bench_v2`; preserve narrow import shims where existing
   tools need them. Reuse the current low-level implementation rather than build
   a second embedding or cache engine.
3. **Migrate MRCR.** Use it to establish exact source preservation, no-cache
   operation, grouped index reuse, source bounds, and incremental result writing.
   Replace its manual private-field indexing and local routing/packing loop.
4. **Migrate AA-LCR.** Add multi-document rendering, separate grading from cache
   verification, and map the four existing experiment names to explicit legacy
   configurations. Generalize chunk/budget settings for new runs. Preserve pinned
   dataset releases, grader versions, regrading, and comparison checks.
5. **Migrate LongBench.** Route the OpenAI-compatible direct runner and hybrid
   branch through the engine. Allow hybrid with answer caching off. Preserve MCQ
   decoding and explicit original/exact/semantic row selection. Keep other
   provider and iterative/RLM paths clearly labeled until separately migrated.
6. **Switch to the common profile.** Select source-order/merged evidence, strict
   offsets, unsupported direct overflow, and cache off for all three. Give this
   profile a new version. Legacy rank-order/no-merge and truncation policies remain
   explicit reproduction choices implemented by the shared functions, not copied
   benchmark loops. Document intended changes from the characterization fixtures.
7. **Unify artifacts and launch behavior.** Keep existing entry scripts as thin
   argument/adapter wrappers. Update Jarvis forwarding, dry runs, and conditional
   service checks. Keep operational commands under `docs/jarvis/`. Remove migrated
   duplicate execution code once parity and intended-change tests pass.

The completion criterion is one implementation of each execution stage used by
all three adapters, not merely a new wrapper that dispatches to three old runners.

## Results and reproducibility

Write an immutable manifest and incremental execution JSONL before grading, plus
evaluation JSONL, bridge CSV, and report JSON. Record code/pipeline version,
resolved configuration/hash, dataset/source hashes, tokenizer/template revisions,
prompt and scoring versions, row order, and serving metadata. Do not serialize
credentials. Keep adapter-specific report fields as extensions and preserve the
existing AA-LCR regrade/comparison interfaces through explicit readers/exporters.

Separate ingestion, retrieval/packing, cache verification, generation, and grading
time/tokens. Report end-to-end totals as well. Charge index construction once per
source group, show amortized cost, and identify cold versus reused state. Record
actual API attempts and usage when supplied; do not fabricate failed-call usage.

Use distinct statuses for unsupported input, execution failure, successful
prediction, and grading failure/pending. Execution failures count as zero in
operational quality over supported selected examples; unsupported examples are
excluded and coverage is explicit. AA-LCR grading failures remain unresolved,
not silently wrong or silently dropped: mark the headline report incomplete,
report grading coverage, and allow regrading without new inference. Keep official
task scores separate from operational summaries and label diagnostic denominators.

For direct/hybrid comparisons, report quality on their common supported examples
and separately report oversized hybrid capability. Do not pool MRCR similarity,
LongBench accuracy, and AA-LCR judged accuracy into one undifferentiated score.
Keep answer-cache experiments and repeated/paraphrased rows separate from the
cache-disabled original-question quality evaluation.

## Validation and acceptance

- Run existing MRCR, AA-LCR, LongBench direct/hybrid, and touched core tests.
- Use one parametrized engine test suite with injected tokenizer, embedder,
  retriever, executor, cache verifier, and scorer; no paid APIs or model downloads.
- With identical normalized inputs and configuration, benchmark labels must not
  change routing, chunks, selected ranges, cache behavior, or solver requests.
- Test multi-document offsets, preserved document IDs, chronological ordering,
  overlapping and adjoining ranges, equal text at different positions, omitted
  regions, exact budget boundaries, and instructions too large to fit.
- Test independent cache reads/writes, cold/warm state, scope/config invalidation,
  format-sensitive queries, and index reuse with answer caching disabled.
- Verify gold metadata never reaches retrieval, generation, cache verification,
  or cache-write decisions. Grader failure must not discard saved predictions.
- Verify preflight makes no inference/embedding/grader calls and matches runtime
  routing on cache misses. Preserve MRCR tokenizer mismatch and bounds rejection.
- Test legacy behavior parity and common-profile changes separately, including
  truncation/unsupported denominators and AA-LCR document-fragment rendering.
- Mock end-to-end direct and hybrid runs for all three adapters, including failure
  paths, incremental artifacts, and AA-LCR saved-answer regrading.
- Verify Jarvis dry-run argument forwarding and required services for generation,
  grading, and semantic cache verification independently.

After implementation, perform separately authorized real smoke runs, freeze the
primary profile, and evaluate all three benchmarks. Report packing/cache/reranker
ablations explicitly. A later held-out benchmark should need only loading,
rendering, and scoring code; no engine edits. That transfer test is stronger
evidence of generalization than shared code or passing unit tests alone.
