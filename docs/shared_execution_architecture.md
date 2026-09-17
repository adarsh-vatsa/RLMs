# Shared execution architecture

**Implemented for direct and dense hybrid execution** for LongBench-v2's
OpenAI-compatible path, AA-LCR, and MRCR v2. Select `--execution-profile common`
explicitly; existing entry points retain legacy defaults. See the
[shared execution runbook](jarvis/SHARED_EXECUTION_RUNBOOK.md) for commands and
remaining limitations.
See the [implementation plan](shared_execution_pipeline_plan.md) for migration
steps, legacy configurations, and acceptance tests.

## Component boundaries

Every adapter supplies the same solver contract. The engine does not branch on
benchmark names. Scoring is separate and optional for tasks without references.

```mermaid
flowchart TB
    subgraph adapters["Task adapters"]
        LB["LongBench-v2<br/>Context, question, choices, MCQ format"]
        AA["AA-LCR<br/>Ordered documents, question, official format"]
        MR["MRCR v2<br/>Conversation, few-shot text, final instruction"]
    end

    INPUT["Shared solver contract<br/>Sources, query, fixed instructions,<br/>response contract and pure prompt renderer"]
    CONFIG["Explicit execution configuration<br/>Budgets, models, packing and component switches"]

    subgraph execution["One shared execution pipeline"]
        ENGINE["Route, index, retrieve, pack and generate"]
        INDEX[("Document index<br/>Reuse within a source group")]
        CACHE[("Optional answer cache")]
        VERIFY["Optional semantic cache verifier<br/>No reference-answer access"]
        ENGINE <--> INDEX
        ENGINE <--> CACHE
        ENGINE <--> VERIFY
    end

    RESULT["Saved execution result<br/>Prediction, route, evidence, usage and timings"]
    REFERENCES["Separate evaluation records<br/>Reference answers and scoring metadata"]
    SCORER["Selected scorer<br/>LongBench: choice matching<br/>MRCR: text similarity and strict metrics<br/>AA-LCR: LLM grading"]
    REPORT["Report<br/>Quality, coverage, failures and costs"]

    LB --> INPUT
    AA --> INPUT
    MR --> INPUT
    INPUT --> ENGINE
    CONFIG --> ENGINE
    ENGINE --> RESULT
    RESULT --> SCORER
    REFERENCES --> SCORER
    RESULT --> REPORT
    SCORER --> REPORT
```

The adapter's renderer formats engine-selected evidence; it cannot select or
reorder evidence. References and gold positions stay outside the solver, cache,
and verifier. Scoring never controls cache writes or feeds back into generation.
The AA-LCR grader and cache verifier are separate roles even if they use the same
model service.

## Request flow

This diagram shows the proposed common profile: exact offsets, source-order
packing with overlap merging, and `unsupported` for direct overflow. Answer-cache
reads and writes are independent switches, both off for primary quality runs.

```mermaid
flowchart TD
    START["Load task input and resolve configuration"]
    COUNT["Render full prompt, count tokens,<br/>validate budgets and source bounds"]
    PREFLIGHT{"Preflight only?"}
    AUDIT["Report expected route assuming cache miss<br/>No embedding or model calls"]
    READ{"Answer-cache reads enabled?"}
    LOOKUP["Look up source-scoped answer<br/>Verify semantic candidate only if configured"]
    HIT{"Accepted hit?"}
    FIT{"Full rendered prompt fits?"}
    MODE{"Execution mode?"}
    UNSUPPORTED["Record unsupported_context<br/>No executor call"]
    INDEX["Build or reuse document index<br/>Exact chunks, embeddings and FAISS"]
    RANK["Rank all children by dense relevance<br/>Optional configured reranker"]
    PACK["Add candidates in score order<br/>Merge ranges within each document<br/>Render exact slices in source order<br/>Stop before the next candidate exceeds budget"]
    GENERATE["Shared executor client<br/>Full prompt or packed evidence<br/>Output contract, retries and usage"]
    WRITE{"Answer-cache writes enabled<br/>and prediction eligible?"}
    STORE["Store generated answer and provenance<br/>Independent of its eventual grade"]
    SAVE["Persist execution result incrementally"]
    SCORE["Optional scoring after prediction is saved"]

    START --> COUNT --> PREFLIGHT
    PREFLIGHT -->|Yes| AUDIT
    PREFLIGHT -->|No| READ
    READ -->|Yes| LOOKUP --> HIT
    READ -->|No| FIT
    HIT -->|Yes: cached prediction| SAVE
    HIT -->|No| FIT
    FIT -->|Yes: direct_fit| GENERATE
    FIT -->|No| MODE
    MODE -->|Direct| UNSUPPORTED --> SAVE
    MODE -->|Hybrid| INDEX --> RANK --> PACK --> GENERATE
    GENERATE --> WRITE
    WRITE -->|Yes| STORE --> SAVE
    WRITE -->|No| SAVE
    SAVE -->|Prediction available| SCORE
```

Runtime failures also produce saved error results; configuration and preparation
errors fail before inference. Unsupported examples have no prediction to score.
Grading failures leave saved predictions available for regrading. Legacy
truncation/rejection policies are explicit reproduction settings, omitted here
to keep the common flow readable.

## What changes between benchmarks

| Changes in the adapter/evaluation layer | Remains the same shared implementation |
|---|---|
| Dataset parsing and example selection | Routing and exact request-budget enforcement |
| Document boundaries and fixed instructions | Chunking, indexing and document-index reuse |
| Prompt wrappers and required output format | Retrieval, optional reranking and evidence selection |
| Scoring rule and reference records | Packing policies, executor calls and cache controls |

For example, AA-LCR can supply several documents and MRCR one conversation to
the same indexer and packer. Ranges retain document IDs and original positions;
overlaps merge only within one document. Fixed instructions and few-shot text
are always outside the retrieval corpus and inside the measured request.

Document-index reuse saves embedding work without reusing answers. Answer caching
can skip generation and is evaluated separately. Neither requires a different
benchmark-specific execution pipeline.
