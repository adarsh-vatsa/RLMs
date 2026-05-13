# Semester Progress Report: Recursive Language Models and Semantic Caching

Course: CS 800 - Special Problems in CS  
Lab: Neurosymbolic Software Engineering Lab  
Collaborator: Adarsh Vatsa, PhD student in Prof. William Eiers's lab  
[OPEN: Add student name, semester label, and final submission date.]

## Introduction

Large language models are increasingly used as autonomous problem solvers: they search through long documents, decompose tasks into smaller steps, call other models or tools, and combine intermediate results into final answers. Recursive Language Models (RLMs) are one example of this direction. Instead of forcing the entire task into a single prompt, an RLM can break a large input into smaller pieces and ask additional language-model calls to process those pieces. This makes RLMs promising for long-context work, but it also creates a practical problem: recursive workflows often repeat the same or similar subquestions many times. Each repeated call costs money, takes time, and can introduce another opportunity for an unsupported or inconsistent answer.

Existing approaches do not fully solve this problem. A standard language-model call has no memory of equivalent work that was already completed. A simple string cache can reuse exact duplicate questions, but it misses paraphrases and related questions. A pure vector-similarity cache can find similar questions, but it can also return dangerous false matches, especially when two queries are worded similarly but ask for different information. Long-context retrieval systems can reduce the amount of text sent to a model, but they do not automatically solve repeated recursive calls, source contamination, or the need to prove that a cached answer came from the correct document.

This project proposes a scoped semantic caching system for RLM-style and autonomous LLM workflows. The system stores answers from previous model calls and reuses them only when the new request is sufficiently related and tied to the correct source context. It first uses fast local retrieval to find candidate cached answers, then applies a stricter semantic check before serving a cached result. It also tracks source identity, document-set identity, estimated cost, benchmark metadata, and extracted knowledge so that cache reuse can be measured and audited rather than treated as an opaque shortcut.

The main finding so far is that semantic caching can substantially reduce repeated API usage and estimated cost while preserving the measured accuracy of the original cold-cache run in selected benchmarks. In the RULER v2 benchmark runs, a warm-cache rerun eliminated API calls and estimated API cost for the repeated workload while preserving the same measured accuracy as the cold-cache run. In the LegalBench/CUAD runs, the warm-cache setting reduced API calls, token usage, estimated cost, and runtime by large margins while maintaining the same answer accuracy observed in the cache-enabled cold run. These results suggest that caching is not just an implementation convenience: it is a necessary infrastructure layer for making future recursive and agentic language-model systems more efficient, reproducible, and practical on large workloads.

At a high level, the motivation of this work is to make recursive and long-context LLM systems economically usable. The problem is that repeated model calls are expensive and difficult to control, while unsafe reuse can return answers from the wrong source. The solution developed in this project is a semantic cache that combines reuse with source isolation and benchmark tracking. The finding is that this approach can turn repeated work into inexpensive lookups without changing the answer quality measured by the current benchmark runs, although more evaluation is still needed on broader reasoning benchmarks such as LongBench-v2.

## Related Work

Several recent papers study the same broad problem as this project: how to make LLMs reliable and useful when the relevant input is much longer than a single convenient prompt. [Recursive Language Models](https://huggingface.co/papers/2512.24601) (Zhang, Kraska, and Khattab, 2025) propose an inference strategy where a model treats the prompt as part of an external environment and recursively decomposes long inputs into smaller model calls. [MemGPT](https://huggingface.co/papers/2310.08560) (Packer et al., 2023) approaches the same context-window limitation through operating-system-inspired memory management, moving information between active context and external storage. Long-context benchmarks show why these systems are needed: [LongBench](https://aclanthology.org/2024.acl-long.172/) (Bai et al., ACL 2024) introduced a bilingual multitask benchmark for long-context understanding; [RULER](https://openreview.net/forum?id=kIoBbc76Sy) (Hsieh et al., COLM 2024) showed that simple needle-in-a-haystack retrieval is too shallow and that models degrade as context length and task complexity increase; [NoLiMa](https://research.adobe.com/publication/nolima-long-context-evaluation-beyond-literal-matching/) (Modarressi et al., ICML 2025) further showed that long-context retrieval becomes much harder when questions and evidence do not share obvious lexical overlap; and [LongBench-v2](https://huggingface.co/papers/2412.15204) (Bai et al., 2025) moves toward more realistic long-context reasoning with multiple-choice questions over contexts ranging from thousands to millions of words. This project is closest to RLMs because it studies repeated recursive model calls, but it focuses on an infrastructure layer that those systems need: safe reuse of prior subcall results. Unlike the benchmark papers, this work does not only evaluate long-context failure modes; it builds a cache mechanism intended to reduce the cost and runtime of systems that operate in those settings.

Other papers use related caching and memory ideas, but often for different deployment problems. [GPTCache](https://www.researchgate.net/publication/376404523_GPTCache_An_Open-Source_Semantic_Cache_for_LLM_Applications_Enabling_Faster_Answers_and_Cost_Savings) (Bang, NLP-OSS 2023) introduced an open-source semantic cache for LLM applications, reusing previous answers for semantically similar prompts to reduce latency and cost. [MeanCache](https://huggingface.co/papers/2403.02694) (Gill et al., 2024) studies semantic caching for LLM web services with attention to user-centric behavior and privacy. [VectorQ](https://huggingface.co/papers/2502.03771) (Schroeder et al., 2025) improves semantic prompt caching by adapting similarity thresholds instead of relying on one fixed cutoff. [RAGCache](https://huggingface.co/papers/2404.12457) (Jin et al., ACM Transactions on Computer Systems 2024) targets retrieval-augmented generation systems by caching intermediate knowledge states to improve serving latency and throughput. Apple’s [Krites work on asynchronous verified semantic caching](https://machinelearning.apple.com/research/semantic-caching) (Singh et al., 2026) is especially relevant because it uses an LLM judge to verify borderline semantic-cache candidates before promoting them for future reuse. This project also uses semantic reuse and an evaluator-style verification step, but the target problem is different: it applies caching to RLM-style and benchmarked autonomous workflows, where source-context isolation, cache-route diagnostics, persistent benchmark state, and answer provenance are central requirements rather than optional service-level features.

## Method

The proposed method is a scoped semantic cache for repeated LLM calls. Each cache entry stores the original query, generated answer, query embedding, source context, model metadata, provenance metadata, and optional extracted facts. More formally, a cache entry can be written as:

```text
e_i = (q_i, a_i, z_i, h_i, s_i, m_i, p_i)
```

where `q_i` is the stored query, `a_i` is the answer, `z_i = embed(q_i)` is the embedding of the query, `h_i` is the hash of the exact source chunk, `s_i` is the hash of the active document set, `m_i` records model/cost metadata, and `p_i` records provenance or verification metadata. For direct RLM-style calls over a given context `c`, the system computes `h = H(normalize(c))` and only compares entries in the same source bucket. For retrieval-style calls over an ingested document collection `D`, the system computes a data-scope hash:

```text
s = H(sorted filenames, normalized file contents, chunk size, overlap)
```

This hash is used as an eligibility gate. A cached answer is not allowed to serve a retrieval query unless it was produced from the same active document set. This design is stricter than a global query cache, but it is necessary because two benchmark samples can contain the same question text while requiring different answers from different document contexts.

The cache lookup proceeds in stages. First, the system checks for exact matches inside the eligible source or data scope:

```text
hit_exact(q, e_i) = 1[lower(q) = lower(q_i)] and 1[scope(q) = scope(e_i)].
```

If no exact match exists, the system performs a semantic search over cached query embeddings. Given a new query `q`, it computes `z = embed(q)` and scores each eligible candidate with cosine similarity:

```text
sim(q, q_i) = dot(z, z_i) / (norm(z) * norm(z_i)).
```

FAISS is used to retrieve the top candidates efficiently. A candidate enters the semantic verification stage only if its similarity is above a configured threshold. The second stage is an evaluator-model check, called the "Sniper" in the project notes. Instead of trusting vector similarity alone, the evaluator decides whether the new query and cached query ask for the same information. This two-stage design was chosen because exact caching is too conservative, but vector-only semantic caching is too risky: questions such as "include timeout errors" and "exclude timeout errors" can be close in embedding space while requiring opposite answers.

The system also maintains a knowledge cache. When an answer is generated, the cache extracts structured facts that can be represented as triples:

```text
f_j = (subject_j, relation_j, object_j, support_j, s_j, h_j).
```

These facts are embedded and indexed separately. A later query may match a fact even when it is not equivalent to the original cached query. The intended use is cross-query reuse: a broad setup question can populate reusable facts, and later narrower questions can retrieve those facts. The current implementation records these facts and indexes them, but the benchmark results show that knowledge-hit behavior still needs stronger span support and route diagnostics before it can be treated as a final contribution.

On a cache miss, the system runs a retrieval and synthesis pipeline. The active corpus is chunked and embedded. FAISS retrieves a broad candidate set, a local reranker filters and ranks the candidates, and the executor model synthesizes an answer from the selected evidence. The generated answer is then checked for grounding, optionally verified through a second model, stored in the cache, and used to extract new facts. The high-level miss path is:

```text
query -> FAISS retrieval -> reranking -> LLM synthesis -> grounding/consensus -> cache write -> fact extraction
```

There are several reasonable alternatives to this design. One alternative is a simple string cache; this is safe but misses paraphrases and therefore gives low hit rates in realistic recursive workflows. Another alternative is a pure embedding cache; this improves hit rate but can return false positives when two queries are lexically similar but semantically different. A third alternative is to use only retrieval-augmented generation without answer caching; this avoids cache collisions but still pays for repeated synthesis. The chosen design combines exact matching, vector retrieval, evaluator verification, source scoping, and persistent benchmark state because the project goal is not only to reduce cost, but to reduce cost without allowing cached answers to cross into the wrong document or benchmark sample.

## Data

The project uses several datasets and benchmark fixtures because each one tests a different part of the system. RULER v2 is used as a long-context retrieval stress test. The prepared local RULER data contains 102 samples: 17 samples for each combination of three tasks (`mk_niah_basic`, `mv_niah_basic`, and `qa_basic`) and two context lengths (8,192 and 32,768 tokens). In a `qa_basic` example, the model receives many numbered documents and a target text, then must return the most relevant document index. This tests whether the system can retrieve and preserve the correct evidence over long contexts. The RULER data was not generated by an LLM for this project; it was prepared in RULER-style JSONL files and then passed through the repository's benchmark runner.

The LegalBench/CUAD cache suite is used to test route behavior on real contract text. The local suite contains 400 route-labeled cases from 10 contract corpora: 100 exact cases, 100 semantic cases, 100 knowledge cases, and 100 miss cases. The reported May 2026 runs selected 60 of these cases. A typical CUAD-style query is: `Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer.` The expected answer is a span or phrase from the contract, such as `Distributor`. This dataset is useful because the task is realistic and source-grounded: a correct answer must come from the contract, not from world knowledge. The route-labeled cases are generated programmatically from CUAD-style contract QA records; the semantic route uses deterministic paraphrases such as asking which contract language counsel should review for a given clause issue.

The synthetic cache-mode suite is a small debugging fixture designed specifically for cache routes. It contains 23 cases over three small corpora: a finance report, Doctor Strange film notes, and an operations incident policy. The expected route distribution is 3 exact, 7 semantic, 6 knowledge, and 7 miss cases. For example, the finance corpus states that ACME's ARR was `$12.4M`; an exact case repeats `What was ACME's ARR in Q1 2023?`, while a semantic case asks `How much annual recurring revenue did ACME report for Q1 2023?`. This dataset is not intended to prove real-world benchmark performance. Its purpose is to make cache behavior observable in a small controlled setting.

NoLiMa is included as an integration track for long-context retrieval beyond literal matching. The current repository fixture is small: it has two needle templates and one haystack file. One example needle is `The traveler finally settled in Lisbon after years of moving`, and the corresponding question is `Which city did the traveler finally settle in?` This fixture verifies that the NoLiMa runner and scorer work, but it should be treated as a smoke test unless a larger official NoLiMa run is added.

LongBench-v2 is the next broader reasoning dataset being prepared. The local LongBench-v2 export contains 503 original multiple-choice questions and 503 exact duplicate rows, covering 462 unique contexts. A pilot cache-suite file currently contains 1,032 rows: 503 original rows, 503 exact rows, 10 semantic rewrite rows, 5 setup rows, and 11 knowledge rows. LongBench-v2 examples include tasks such as translating from a grammar book, answering questions from reports, comparing government documents, and reasoning over long structured or multi-document contexts. This dataset is attractive because the answer format is multiple choice, making scoring more deterministic than free-form span matching.

Some LongBench-v2 cache rows were generated with an LLM. The semantic rewrite rows were generated with the Codex SDK using `gpt-5.5` with `reasoning_effort=high`. The rewrite prompt instructed the model to rewrite only the question text so that it asked for the same answer with different wording, while preserving names, dates, numbers, quoted strings, code identifiers, and answer choices without revealing the answer. The knowledge setup rows were also generated with `gpt-5.5` at high reasoning effort. That prompt asked for exactly one open-ended setup question useful for extracting reusable facts from the same hidden long context, while avoiding answer choices, benchmark metadata, yes/no questions, or overly broad requests for every detail. Example generated setup question: `Summarize the Kalamang vocabulary, word order, grammatical markers, pronoun/possession patterns, tense/aspect cues, and short example translations that would help translate later Kalamang sentences into English.`

## Project Summary

This semester project investigates how to make Recursive Language Models (RLMs) more efficient and more practical for long-context autonomous workflows. The project started as an exploration of the RLM repository and its execution model, then developed into a semantic caching system for repeated LLM subcalls. The current repository, `adarsh-rlms`, implements and evaluates a two-stage semantic cache designed to reduce repeated API calls, preserve source provenance, and support benchmark-driven evaluation.

The motivating observation was that RLMs decompose large tasks into many smaller language-model calls. This decomposition can let an LLM work over inputs larger than a single model context window, but it also creates repeated subquestions, repeated document chunks, repeated extraction templates, and repeated intermediate summaries. Without caching, each recursive call is blocking, incurs API cost, and may repeat work that has already been performed. The main research direction became whether recursive LLM workflows can be made cheaper, faster, and more reliable by memoizing repeated subcalls in a scoped semantic cache.

By the current point in the semester, the project has produced:

- An initial understanding and diagram of RLM execution flow, including root/sub-RLM calls, local REPL environments, persistence, depth, iterations, and stopping behavior.
- Early local and API-based RLM experiments using GPT-style and local Qwen/Ollama models.
- A proposed caching direction for RLMs, motivated by limitations in cost control, runtime control, blocking recursive calls, and lack of prefix/cache reuse.
- A working two-stage semantic cache implementation with local embeddings, FAISS vector search, reranking, LLM-based semantic verification, provenance tracking, knowledge extraction, persistent cache state, and corpus/data-scope isolation.
- Benchmark infrastructure for RULER v2, NoLiMa, a synthetic cache-mode suite, LegalBench/CUAD, and a modified LongBench-v2 cache dataset.
- Recorded evaluation artifacts showing strong cost and API-call reductions on warm-cache runs while preserving measured benchmark accuracy in selected settings.

## Collaboration Context

This work is being conducted with Adarsh Vatsa, a PhD student in Prof. William Eiers's Neurosymbolic Software Engineering Lab. The work has combined research reading, system design, implementation, benchmark construction, experiment execution, and evaluation analysis.

The semester work can be divided into two phases. The first phase focused on understanding Recursive Language Models: how the provided RLM codebase works, what limitations it has, and how it behaves on long-context tasks. The second phase focused on building and evaluating infrastructure around RLM-style workflows, especially semantic caching and benchmark runners that make the cost, runtime, and correctness tradeoffs measurable.

## Initial RLM Exploration

At the start of the semester, the work focused on understanding the RLM codebase and reproducing simple examples. This included setting up VPN and JARVIS/HPC access, reading the repository structure, and constructing an execution-flow diagram to make the system easier to reason about. The diagram in `report/Notes on RLMs.png` summarizes the root RLM process, how prompts are loaded into a local environment, how sub-RLM calls are spawned, and how final answers are returned.

![RLM execution flow diagram](Notes%20on%20RLMs.png)

The initial architecture notes captured several important details:

- An RLM instance has parameters such as backend, environment, depth, `max_depth`, `max_iterations`, system prompt, persistence, and logging.
- The root RLM runs with `depth=0`; sub-RLMs run at deeper levels.
- The local REPL environment allows the model to manipulate the prompt as a variable, split context, call sub-RLMs, and combine intermediate outputs.
- The default behavior falls back to a regular LM completion when maximum depth is reached.
- Persistence changes whether the environment is reused across calls.
- The current RLM implementation appeared to support only `max_depth=1` in practice, even when higher values were configured.

This early analysis led to several research questions that shaped the rest of the semester:

- What is the practical difference between increasing `max_depth` and increasing `max_iterations`?
- Should deeper recursive calls run sequentially, in parallel, or only when earlier calls cannot find an answer?
- Can an RLM signal that it should stop once it has found the answer?
- Are repeated recursive calls wasting cost and runtime because they recompute equivalent work?
- Can a memory or caching layer make repeated subcalls reusable without returning answers from the wrong source context?

## Early Experiments

An Ollama client was implemented and tested, and remote/API model behavior was compared with local-model behavior on the quickstart RLM workflow. The weekly logs record the following early measurements:

| Setup | Input/Output Budget | Approximate Runtime | Notes |
| --- | ---: | ---: | --- |
| `gpt-5-nano-2025-08-07` | 400K input / 128K output tokens | about 120 seconds | Tested on `quickstart.py` |
| `qwen2.5:7b` via Ollama, Q4 quantization | 128K input / 8K output tokens | about 43 seconds | Worked locally; HPC was not yet tested |

Small synthetic RLM-style tasks were also run to understand answer quality under repeated long-context search. The CSV log `report/Experiments_03032026.csv` records tasks such as finding one or two hidden numbers and answering a question about the location of Stevens Institute of Technology using only provided context, including contexts with intentionally false information.

These experiments showed that simple extraction tasks could work, but reliability varied by prompt, model, context size, and whether the model defaulted to outside knowledge. For example:

- Number-finding tasks generally performed well across `gpt-5-mini` and `gpt-5-nano`.
- Location questions were more brittle, especially for smaller models, because the model sometimes defaulted to the real-world answer instead of obeying the supplied false context.
- Longer contexts, including approximately 65K-token contexts, made prompt design and grounding more important.

The experiments helped motivate later project requirements: benchmark tasks need clear scoring, answers need to be tied to the supplied source context, and cache reuse must not cross from one source context into another.

## Research Reading and Benchmark Survey

A major part of the semester was surveying long-context and RLM-relevant benchmarks. Candidate benchmarks were collected in `report/Benchmarks.csv` and later organized into implemented and candidate benchmark tracks in `docs/benchmarks.md`.

The benchmark survey included:

- RULER and Needle-in-a-Haystack style retrieval benchmarks.
- NoLiMa, which reduces literal overlap between query and needle.
- LongBench and LongBench-v2 for broader long-context QA/reasoning.
- OOLONG and CorpusQA for aggregation-heavy or corpus-level reasoning.
- Legal and contract benchmarks such as CUAD, ContractNLI, and LegalBench-RAG.
- Memory-oriented benchmarks such as LongMemEval and MemoryAgentBench.

This research changed the evaluation plan. RULER and NoLiMa are useful engineering stress tests, but they are mostly retrieval-focused. LongBench-v2 became the next primary candidate because it offers broader long-context reasoning and deterministic multiple-choice scoring. LegalBench/CUAD became useful for real-domain cache-route debugging because it contains real contracts, clause labels, and answer spans.

## Architecture Direction

The central architecture direction became a two-stage semantic cache for autonomous agents and RLM-style workflows. The system is designed around the idea that recursive LLM systems repeatedly ask similar questions over the same or related source material. A simple string cache only catches exact duplicates, while a pure vector cache can produce dangerous false hits. The project therefore uses a two-stage "Dragnet and Sniper" design:

1. The Dragnet stage uses local embeddings and FAISS search to cheaply collect likely similar cached entries.
2. The Sniper stage uses a smaller/evaluator LLM to decide whether the new query is truly semantically equivalent to a cached query.

The implemented architecture in `semantic_cache_system.py` includes:

- `EmbeddingEngine`: local Qwen3 embedding model for query and document embeddings.
- `FAISSIndex`: vector index wrapper with metadata and persistence.
- `Reranker`: local Qwen3 cross-encoder reranker for retrieved documents.
- `ExecutionMetrics`: counters for cache hits, misses, API calls, token usage, cost, and provenance.
- `SemanticCacheController`: exact, semantic, and knowledge cache lookup; document retrieval; synthesis; grounding; consensus; fact extraction; persistence; and data-scope validation.
- `CachePreWarmer`: programmatic cache saturation over templates and chunks.
- `Router`: model routing for simple extraction versus more complex synthesis tasks.
- `AutonomousAgent`: framework-agnostic cached-query facade for agent/RLM-style workflows.

The system uses local Qwen3 models for embedding/reranking and Anthropic Claude models for synthesis/evaluation roles. [OPEN: Confirm whether the final report should describe current model names exactly or keep them generic to avoid stale model-ID issues.]

## Cache Correctness and Source Isolation

A recurring challenge was that cache reuse can improve efficiency while also introducing correctness risks. The project therefore added multiple isolation mechanisms.

The first layer is `source_chunk_hash`, which identifies the exact source chunk or context that produced a cached answer. This protects direct `cached_query(query, context)` calls by preventing a query over one document from reusing an answer generated from another document.

The second layer is `data_scope_hash`, which identifies the active ingested document set for retrieval-based `search(query)` calls. This was added after discovering a RULER `qa_basic` failure mode: two samples could contain identical question text but different document contexts and different correct document IDs. Query-only exact matching was not safe. The fix made exact, semantic, and knowledge cache hits eligible only when the cached entry's `data_scope_hash` matches the currently ingested document set.

This distinction became an important implementation and reportable research contribution:

- `source_chunk_hash` answers: which exact source context produced this cached answer?
- `data_scope_hash` answers: which active document set was being searched when this answer was produced?

Regression tests were added for scoped exact hits, scoped knowledge hits, scoped persistence, and legacy unscoped cache entries.

## Benchmark Infrastructure Completed

Several benchmark tracks were implemented or prepared during the semester.

### RULER v2

The RULER v2 cache runner in `ruler_v2/run_benchmark.py` evaluates the semantic cache system on selected RULER tasks and context lengths. The implemented task set currently includes `mk_niah_basic`, `mv_niah_basic`, and `qa_basic` at 8,192 and 32,768 token contexts. The runner writes reproducible artifacts including predictions, bridge rows, manifests, sample documents, and official-style evaluation reports.

The RULER v2 RLM baseline in `ruler_v2/run_rlm_benchmark.py` provides an uncached RLM comparison path over prepared RULER v2 data. This is important because it lets the project compare the semantic cache system with a recursive baseline rather than only direct LLM calls.

### NoLiMa

The NoLiMa integration in `nolima/` includes a benchmark runner, scorer, and parity bridge. It expands needle sets into placement sweeps and writes repository-native artifacts. NoLiMa is useful because it tests long-context retrieval when the query and needle have reduced literal overlap.

The current NoLiMa fixture is still small and should be treated as an integration/smoke-test track rather than a final headline result.

### Synthetic Cache-Mode Suite

The synthetic cache-mode suite in `cache_bench/` is a small hand-authored fixture designed to test cache route behavior directly. It includes exact, semantic, knowledge, and miss cases. This suite is not meant to be a final accuracy benchmark; it is a debugging harness for route behavior.

The latest recorded run, `benchmark_artifacts/cache_mode_suite/20260428T153235Z`, selected 23 cases and reached 100% answer accuracy, but route-type matching was only 43.48%. This showed that the system often answered correctly while taking a different cache path than the case label expected. In particular, many expected knowledge or miss cases became exact hits because the warmed cache already contained a directly reusable answer.

### LegalBench/CUAD Cache Suite

The legal benchmark track in `legal_bench/` builds route-labeled cases from CUAD-style contract QA data. It generates exact, semantic, knowledge, and miss cases from real contract clauses and answer spans.

This benchmark is useful because it moves beyond hand-authored toy documents. It tests cache behavior on real legal text and asks contract-review questions such as highlighting language related to parties, agreement dates, effective dates, and other CUAD clause categories.

The implementation also added case-level cache isolation. This is important for route diagnostics because otherwise one case can warm the cache in a way that prevents a later case from exercising the intended route.

### Modified LongBench-v2

The current branch includes a modified LongBench-v2 dataset preparation workflow. The utilities under `long_bench_v2/` export the original dataset to CSV, generate semantic rewrites, generate setup/knowledge rows, and combine rows into a cache-suite CSV.

Current local files show:

| File | CSV Records | Purpose |
| --- | ---: | --- |
| `benchmark_data/long_bench_v2/data.csv` | 1,006 | 503 original rows and 503 exact rows |
| `benchmark_data/long_bench_v2/data_semantic_codex.csv` | 10 | Generated semantic rewrite pilot |
| `benchmark_data/long_bench_v2/data_knowledge_codex.csv` | 16 | 5 setup rows and 11 knowledge rows |
| `benchmark_data/long_bench_v2/data_cache_suite.csv` | 1,032 | Combined cache suite |

The modified LongBench-v2 approach is the current next direction for the project. The goal is to preserve LongBench-v2's original multiple-choice answer labels while adding cache-route structure around the benchmark. Original rows establish cold-cache behavior; exact rows duplicate the same question/context pair to test string-level reuse; semantic rows rewrite the question while preserving the answer to test paraphrase reuse; setup rows ask broad unscored questions intended to populate reusable facts; and knowledge rows test whether later scored questions can reuse those facts. The route labels are diagnostic metadata only: the benchmark runner should still pass the model only the context, question, choices, and answer format, so the cache system cannot use the label to choose an easier route.

## Results

The project is still in progress, so the results below should be interpreted as current evidence rather than final claims. The strongest completed evaluations so far measure whether persistent semantic-cache reuse reduces API calls, token usage, estimated cost, and runtime while preserving the measured accuracy of the original cache-producing run.

### Quantitative Results

The strongest RULER v2 comparison currently visible is the cold-cache run on April 23, 2026 and the warm-cache run on April 27, 2026. Both runs use the same 102 selected samples across `mk_niah_basic`, `mv_niah_basic`, and `qa_basic` at 8,192 and 32,768 token contexts.

| Run | Mode | Samples | API Calls | Tokens | Estimated Cost | Elapsed Time | Cache Hits | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260423T221015Z` | Cache, cold start | 102 | 306 | 513,515 | $1.212457 | 8,017.457 sec | 0 / 102 | 0.8113 |
| `20260427T151403Z` | Cache, warm start | 102 | 0 | 0 | $0.000000 | 3,764.086 sec | 102 / 102 | 0.8113 |

The warm RULER run preserved the measured accuracy of the cold run while eliminating API calls and estimated API cost for the repeated workload. Runtime decreased by about 53.1%, although it did not go to zero because the benchmark still performs local orchestration, loading, scoring, and artifact writing.

The by-task results were:

| Task and Context | Samples | Accuracy |
| --- | ---: | ---: |
| `mk_niah_basic`, 32,768 | 17 | 0.6471 |
| `mk_niah_basic`, 8,192 | 17 | 0.9412 |
| `mv_niah_basic`, 32,768 | 17 | 0.5882 |
| `mv_niah_basic`, 8,192 | 17 | 0.7500 |
| `qa_basic`, 32,768 | 17 | 0.9412 |
| `qa_basic`, 8,192 | 17 | 1.0000 |

This result supports the claim that persistent cache reuse can eliminate repeated API cost when the same benchmark workload is rerun, but it also shows that accuracy depends on the underlying retrieval/synthesis quality from the cold run.

The LegalBench/CUAD runs provide a clearer cost comparison across baseline, cold cache, and warm cache modes on 60 cases.

| Run | Mode | Cases | API Calls | Tokens | Estimated Cost | Elapsed Time | Answer Accuracy | Cache Hit Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260504T020700Z` | Baseline | 60 | 360 | 611,410 | $1.962066 | 4,848.257 sec | 0.5833 | N/A |
| `20260504T033206Z` | Cache, cold start | 60 | 285 | 406,647 | $1.278803 | 3,402.981 sec | 0.6667 | 0.75 |
| `20260504T042946Z` | Cache, warm start | 60 | 30 | 8,008 | $0.009688 | 659.463 sec | 0.6667 | 1.00 |

Compared with the baseline, the warm-cache LegalBench run reduced estimated cost by approximately 99.5%, API calls by approximately 91.7%, and runtime by approximately 86.4%, while preserving the cache-mode answer accuracy measured in the cold-cache run.

The synthetic cache-mode suite provides route-level diagnostics rather than a headline benchmark score. The latest recorded run selected 23 cases. It achieved 100% answer accuracy and a 95.65% cache hit rate, but only 43.48% cache-route match rate. This means the system often returned the right answer from cache, but through a different route than the case was designed to test.

| Benchmark | Cases/Samples | Answer Accuracy | Cache Hit Rate | Route Match Rate | Estimated Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic cache-mode suite `20260428T153235Z` | 23 | 1.0000 | 0.9565 | 0.4348 | $0.005834 |
| LegalBench/CUAD cold cache `20260504T033206Z` | 60 | 0.6667 | 0.7500 | 0.7500 | $1.278803 |
| LegalBench/CUAD warm cache `20260504T042946Z` | 60 | 0.6667 | 1.0000 | 0.5000 | $0.009688 |

The RULER RLM baseline run `benchmark_artifacts/official_ruler_v2_rlm/20260504T150419Z` evaluated 3 samples, one each from `mk_niah_basic`, `mv_niah_basic`, and `qa_basic` at 32,768 tokens. It reached 1.0 accuracy on those 3 samples with 702,406 total tokens, $0.75049 estimated cost, and 733.477 seconds elapsed. This is a small baseline, so it should be presented as an initial cost-profile comparison rather than a final RLM-vs-cache conclusion.

### Qualitative Results

A successful semantic-cache example appears in the synthetic cache-mode suite. The seed query asks: `What was ACME's ARR in Q1 2023?` The evaluation query asks the same question with different wording: `How much annual recurring revenue did ACME report for Q1 2023?` The expected answer is `$12.4M`. The system correctly served the result as a semantic cache hit, with the answer: `ACME's ARR in Q1 2023 was $12.4M.` This example shows the intended behavior of the two-stage design: exact string matching would miss the paraphrase, but semantic lookup plus verification can reuse the earlier answer.

A failure mode also appears in the synthetic cache-mode suite. In the case `ops_negative_timeout_inversion`, the test expected a miss because the query involved an exclusion/inclusion distinction in the incident policy. The system still served the answer from cache through an exact route because a previous warmed entry already matched the final query text. The returned answer was factually correct: `sandbox timeout warnings should be excluded from the incident review.` However, the route behavior failed the diagnostic test because the case was meant to verify that the system would avoid unsafe semantic reuse. This illustrates an important distinction for future evaluation: answer correctness, cache hit rate, and intended cache route are different metrics and must be reported separately.

The LegalBench/CUAD results show a similar qualitative pattern. Exact and semantic reuse often worked, but expected knowledge cases were frequently served through exact or semantic routes instead of the knowledge branch. From a product perspective, this can still be useful because the system reduces cost and returns the right answer. From a research perspective, it means the current experiments do not yet prove that the extracted-knowledge route is reliable. Future work should add span-supported fact extraction and stricter route isolation for knowledge-hit evaluation.

### Ablation-Style Comparisons

The main ablation-style evidence so far compares three operating modes: no cache, cold cache, and warm cache. In LegalBench/CUAD, the no-cache baseline required 360 API calls and cost an estimated $1.962066. The cold-cache run reduced this to 285 API calls and $1.278803 while improving measured answer accuracy from 0.5833 to 0.6667. The warm-cache run reduced the repeated workload to 30 API calls and $0.009688 while preserving 0.6667 accuracy. This comparison suggests that cache persistence, rather than only retrieval quality, is responsible for the largest cost reduction.

The RULER cold/warm comparison isolates persistent cache reuse even more directly because the same 102-sample workload is run twice. The cold run writes 102 entries and makes 306 API calls; the warm run loads those 102 entries, obtains 102 cache hits, and makes zero API calls while preserving the same 0.8113 accuracy. This supports the core design goal: once a repeated RLM-style workload has been computed and safely scoped, future equivalent work can be served as lookup rather than regenerated.

There was also an implementation-level ablation during development. An earlier embedding design used first-token pooling for query/document embeddings. Because many queries shared the same instruction prefix, distinct queries collapsed toward nearly identical vectors, which amplified false knowledge hits. The corrected implementation uses attention-masked mean pooling, producing distinct vectors for distinct queries. This was not a full benchmark ablation, but it materially changed cache reliability and should be included as an engineering lesson: semantic caching is only as safe as the embedding and verification policy behind it.

The project still needs stronger final ablations. The next report should compare exact-only caching, vector-only semantic caching, vector-plus-verifier caching, knowledge caching with and without source-span verification, and direct RAG without answer caching. These ablations are future work because the current project stage has focused on building the system, generating benchmark tracks, and validating the main cold/warm cache behavior.

## Key Technical Problems Solved

The project solved or made progress on several technical problems.

First, it moved from simple RLM experimentation to a reusable semantic cache architecture. Instead of treating every recursive subcall as a fresh LLM call, the system can reuse exact matches, semantic paraphrases, and extracted knowledge.

Second, it added scoped cache correctness. The system now distinguishes between exact source chunks and active document sets, reducing the risk that a cached answer from one document or benchmark sample will contaminate another.

Third, it added benchmark reproducibility infrastructure. Runs now write manifests, predictions, bridge rows, evaluation reports, cache state, and dataset signatures. This makes it possible to compare cold and warm runs and preserve historical artifacts.

Fourth, it added cost accounting. Benchmark `delta_cost_usd` and run-level totals are estimated from centralized pricing assumptions in `semantic_cache_system.py`, which keeps cost comparisons consistent.

Fifth, it identified and fixed important evaluation/caching bugs. One major bug involved embedding pooling: first-token pooling caused different queries with shared instruction prefixes to collapse toward identical vectors, amplifying false knowledge hits. The fix switched to attention-masked mean pooling, making distinct query embeddings meaningfully different. Another issue involved multi-value RULER scoring, where accuracy was not calculated using the full set of expected answers.

Sixth, it expanded the benchmark plan from one benchmark to a portfolio. RULER, NoLiMa, LegalBench/CUAD, and LongBench-v2 now each serve different purposes: retrieval stress testing, latent retrieval, real-domain route debugging, and broader long-context reasoning.

## Discussion

The designed solution is a good fit for the chosen problem because RLM-style systems naturally create repeated subproblems. When a model decomposes a large task into many smaller calls, some calls are exact duplicates, some are paraphrases, and some ask for facts that were already extracted by a previous branch of the computation. A cache is therefore not an add-on optimization; it matches the structure of the workload. The important design constraint is that reuse must be scoped. A global cache can reduce cost, but it can also return an answer generated from the wrong source document. The source-chunk and data-scope hash mechanisms are therefore central to the method because they let the system reuse work while still respecting the boundary between documents, corpora, and benchmark samples.

The solution is not a perfect fit for every long-context task. It works best when workloads contain repeated questions, paraphrases, repeated contexts, recurring extraction templates, or reusable facts. It is less directly useful for one-off questions where there is no future reuse. It also does not by itself solve global aggregation tasks, where the answer depends on evidence distributed across many chunks rather than a small retrieved subset. For those tasks, the cache likely needs to be combined with a map-reduce or structured aggregation path: extract local facts from many chunks, cache those intermediate results, and then reduce them into a final answer.

This project uses several capabilities of frontier models. First, it uses frontier models as synthesizers: they can read retrieved evidence and generate a natural-language answer. Second, it uses smaller or cheaper frontier models as evaluators: they can judge whether two differently worded questions ask for the same information, which is difficult to do safely with embedding similarity alone. Third, the project uses models as data-generation assistants for LongBench-v2, where semantic paraphrases and setup questions are generated under strict prompts and saved for review. This use of LLMs is different from simply asking a model to answer the benchmark. The models help construct, verify, and route the workflow around the main task.

The main recommendation for future work is to separate three metrics that are easy to confuse: answer correctness, cache hit rate, and cache-route correctness. The current experiments show that the system can return correct answers and reduce cost even when the route does not match the intended diagnostic label. That is useful for deployment, but it is not enough for a research claim about the knowledge route or semantic route specifically. Future evaluations should report exact, semantic, knowledge, and miss behavior separately, and should include cases where the cheapest cache route is not necessarily the route under test.

A second recommendation is to strengthen provenance. The current system has useful source scoping and numeric grounding, but future versions should store source spans for extracted facts. Each extracted fact should include the source document, character offsets, support text, and verifier status. This would make knowledge hits safer because the system could answer from a verified span instead of relying only on a related cached answer or fact embedding.

A third recommendation is to run stronger ablations. The final evaluation should compare exact-only caching, vector-only semantic caching, vector-plus-verifier semantic caching, knowledge caching with and without span verification, and direct RAG without answer caching. These comparisons would clarify which part of the system is responsible for each gain. The current cold/warm-cache results show that persistence can produce large cost reductions, but the next stage should isolate the value of the semantic verifier, the reranker, the knowledge index, and the source-scope gates.

## Current Limitations and Open Issues

Several limitations remain.

The RULER RLM baseline is currently small. The recorded run has only 3 samples, so it cannot support a broad conclusion about RLM versus semantic-cache performance yet.

The knowledge-hit route needs more work. Synthetic and legal route diagnostics show that expected knowledge cases are often answered by exact or semantic routes instead of the knowledge branch. This may be acceptable for cost reduction, but it weakens claims specifically about extracted-knowledge reuse unless the knowledge route is improved and measured separately.

The current grounding checks are strongest for numeric facts. More robust span-based grounding is needed for document IDs, entity names, dates, titles, and exact benchmark answers. The development notes propose adding `source_doc_id`, character spans, support text, verifier status, and extraction confidence to extracted facts.

LongBench-v2 is prepared but not fully evaluated in the visible artifacts. The dataset preparation scripts and pilot generated rows are in place, but a complete benchmark run and scoring report still appear to be future work.

Some benchmark scores need careful interpretation. RULER and NoLiMa are useful retrieval stress tests, but they do not fully measure complex reasoning. CUAD has real legal text, but answer-span scoring can be brittle when compared with free-form LLM outputs. LongBench-v2 should help address this because multiple-choice labels make scoring cleaner.

## Work Remaining

The next steps are:

- Finish the modified LongBench-v2 benchmark runner and run the combined cache suite.
- Compare three modes on LongBench-v2: direct LLM baseline, uncached RLM baseline, and semantic cache system.
- Decide whether knowledge hits should be a required LongBench-v2 route or only reported when repeated contexts naturally support them.
- Add stronger span-based provenance for nonnumeric facts.
- Improve route telemetry so reports separate exact, semantic, knowledge, and miss behavior from answer correctness.
- Expand the RULER RLM baseline beyond the 3-sample smoke run.
- Frame the final project claim around RLM motivation while explaining that the cache infrastructure can also apply to broader autonomous-agent workflows.

## Conclusion

This report described the semester's progress from understanding Recursive Language Models to building a semantic caching and evaluation system for recursive and autonomous LLM workflows. The central problem is that recursive decomposition can help process long inputs, but it also creates many repeated model calls. Those calls are expensive, slow, and difficult to control. The proposed solution is a scoped semantic cache that combines exact matching, embedding-based retrieval, evaluator-model verification, source/data-scope isolation, persistent cache state, and benchmark artifact logging.

The experiments completed so far show that this approach can substantially reduce repeated API usage. On RULER v2, the warm-cache run preserved the cold-cache accuracy of 0.8113 while reducing API calls from 306 to 0 and estimated API cost from $1.212457 to $0. On LegalBench/CUAD, the warm-cache run reduced API calls from 360 in the baseline to 30, reduced estimated cost from $1.962066 to $0.009688, and reduced runtime from 4,848.257 seconds to 659.463 seconds, while preserving the cache-enabled answer accuracy of 0.6667. The synthetic cache-mode suite further showed 100% answer accuracy and a 95.65% cache hit rate, but also exposed route-diagnostic issues because answer correctness and route correctness were not always aligned.

The main conclusion is that semantic caching is a promising infrastructure layer for RLM-style workloads, but the project is not finished. The current results support the cost-reduction claim for repeated workloads, while the route-level diagnostics show where the system still needs improvement. The next stage should complete the LongBench-v2 evaluation, expand the RLM baseline, add span-supported provenance for knowledge hits, and run cleaner ablations. These steps will test whether the same efficiency gains hold for broader long-context reasoning tasks, not only repeated benchmark reruns and cache-route fixtures.

## Remaining Open Items

1. Add student name, semester label, and final submission date.
2. Review attribution at the end and decide which parts should be described as individual work versus joint work with Adarsh.
3. Decide whether benchmark costs should be described only as estimated API costs or whether local compute time/cost should also be discussed.
4. Confirm whether final model names should be listed exactly or described generically to avoid stale model identifiers.
