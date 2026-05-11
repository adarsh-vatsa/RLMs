# Benchmarks

This document separates the benchmark tracks that are already implemented from the candidate benchmarks discussed for the next evaluation phase.

The goal is to evaluate whether the semantic cache system preserves reasoning quality while reducing latency, API calls, and cost compared with uncached baselines.

## Implemented Benchmarks

### Benchmarks Table

| Track | Status | Purpose | Main Artifacts |
|-------|--------|---------|----------------|
| RULER v2 cache runner | Implemented | Long-context retrieval stress test for the semantic cache system | `benchmark_artifacts/official_ruler_v2/<run_id>/` |
| RULER v2 RLM baseline | Implemented | Uncached RLM baseline over the same prepared RULER v2 samples | `benchmark_artifacts/official_ruler_v2_rlm/<run_id>/` |
| NoLiMa | Implemented | Needle-placement long-context benchmark with repository-native runner and scorer | `benchmark_artifacts/official_nolima/<run_id>/` |
| Synthetic cache-mode suite | Implemented | Small hand-authored route test for exact, semantic, knowledge, and miss behavior | `benchmark_artifacts/cache_mode_suite/<run_id>/` |
| LegalBench/CUAD cache suite | Implemented | Real contract-data cache-route benchmark for debugging cache behavior on legal text | `benchmark_artifacts/legal_cache_suite/<run_id>/` |

### RULER v2

The RULER v2 runner lives in `ruler_v2/run_benchmark.py`. It uses prepared RULER-style data and evaluates the semantic cache system on selected official RULER tasks and context lengths.

RULER is useful for engineering validation because it stresses long-context retrieval and source scoping. It is less ideal as the main reasoning benchmark because many tasks are closer to finding needles or retrieving facts than evaluating complex reasoning.

The uncached RLM baseline lives in `ruler_v2/run_rlm_benchmark.py`. It writes the same core artifact files under the `official_ruler_v2_rlm` namespace and does not use cache, FAISS, reranking, or the semantic cache controller. Usage instructions are in `ruler_v2/docs/rlm_baseline_runner.md`.

### NoLiMa

The NoLiMa integration lives under `nolima/`. It expands NoLiMa needle sets into placement sweeps and evaluates the semantic cache system with repository-native artifacts.

NoLiMa is valuable as a long-context retrieval benchmark where literal overlap between the question and the inserted needle is minimized. Like RULER, it is mostly a retrieval stress test rather than the strongest reasoning benchmark.

### Synthetic Cache-Mode Suite

The synthetic cache-mode suite lives under `cache_bench/` and uses the fixture `benchmark_fixtures/cache_bench/cache_mode_suite_v1.json`.

This suite exists to directly test cache routing: `exact`, `semantic`, `knowledge`, and `miss`.

It is intentionally small and hand-authored. It should stay separate from real dataset benchmarks because it is a debugging fixture, not a headline accuracy benchmark. Usage notes are in `cache_bench/docs/cache_mode_benchmark.md`.

### LegalBench/CUAD Cache Suite

The legal cache suite lives under `legal_bench/`. It builds route-labeled cases from CUAD-style contract QA rows and writes artifacts under `benchmark_artifacts/legal_cache_suite/<run_id>/`.

CUAD is useful because it gives real contract text, real clause labels, and real answer spans. It is currently best viewed as a cache-route and legal-domain debugging benchmark. It is weaker as a final accuracy benchmark because CUAD answers are span annotations, and some spans are too terse for robust free-form answer scoring.

The legal runner uses the reranker by default. If FAISS finds candidates but the reranker filters all of them out, the runner falls back to bounded FAISS candidates and records that fallback in the bridge rows. Usage notes are in `legal_bench/docs/legal_cache_benchmark.md`.

## Candidate Benchmarks

### Benchmarks Table

| Benchmark | Approximate Context Length | Best Use | Main Strength | Main Risk |
|-----------|----------------------------|----------|---------------|-----------|
| ContractNLI | Mean contract length is a little over 2,250 tokens | Legal reasoning sanity check | Clean three-way labels and deterministic accuracy | Contracts are not very long on average |
| (Modified) LongBench-v2 | 8K to 2M words, majority under 128K words | General long-context reasoning | Standard benchmark, multiple-choice scoring, broader than RULER | Cache route cases must be constructed around the dataset |
| MemoryAgentBench | Incremental multi-turn histories; exact per-sample token length should be profiled from the parquet splits | Agent memory, knowledge reuse, and conflict-resolution behavior | Directly evaluates memory agents across retrieval, learning, long-range understanding, and conflict resolution | Mixed task formats and scoring may be less clean than LongBench-v2 multiple-choice accuracy |
| OOLONG | Public Oolong-synth subsets include 1K, 2K, 4K, 8K, 16K, 32K, 64K, and 128K tokens | Aggregation-heavy long-context tasks | Tests behavior beyond simple retrieval | May require map-reduce or aggregation logic beyond top-k retrieval |
| LongMemEval / LME-MC10 | LongMemEval-S is about 115K tokens per history; LongMemEval-M uses 500 sessions and about 1.5M tokens | Long-term memory and multi-session reuse | Naturally aligned with persistent cache and knowledge reuse | Free-form scoring can require judge-style evaluation |
| CorpusQA | Up to 10M tokens in the paper framing; 1M-token variant is reported in model evaluations | Corpus-level reasoning over very high context inputs | Directly targets repository-scale analysis where evidence is dispersed | Dataset/code availability and evaluation harness need verification before implementation |
| LegalBench-RAG | Corpus is over 79M characters across legal documents | Retrieval and evidence grounding | Gold evidence spans support retrieval evaluation | Final answer accuracy can still be less clean |

### ContractNLI

- Deterministic labels: YES, one of `Entailment`, `Contradiction`, or `NotMentioned`.
- Input context length: mean contract length is a little over 2,250 tokens.
- Downside: good reasoning labels, but not a strong long-context benchmark because the contracts are relatively short.

ContractNLI remains a good reasoning benchmark, especially if we want a legal sanity check with deterministic labels. Each example asks whether a contract entails, contradicts, or does not mention a hypothesis: `Entailment`, `Contradiction`, or `NotMentioned`.

This is cleaner than CUAD for accuracy because the model can be asked to return one of three labels, and evaluation is exact label matching. It also tests negation, absence of evidence, and contract-specific reasoning.

**The limitation is context length. ContractNLI contracts are shorter than many long-context benchmarks, so it should not be used alone to support a strong long-context claim.** A possible extension is a controlled ContractNLI-long variant where distractor contracts are added around the target contract while preserving the original label.

### (Modified) LongBench-v2

- Deterministic labels: YES, multiple-choice `A`/`B`/`C`/`D`.
- Input context length: 8K to 2M words, with the majority under 128K words.
- Downside: cache route labels are not native to the dataset, so exact/semantic/miss cases must be constructed.

LongBench-v2 is the strongest candidate for the next primary benchmark if the goal is broader long-context reasoning. It contains multiple-choice questions over long contexts, which makes scoring deterministic: the system returns a choice such as `A`, `B`, `C`, or `D`, and the evaluator checks exact match.

For this project, LongBench-v2 would support three comparable modes: semantic cache system, uncached RLM baseline, and direct document-to-LLM baseline.

Cache behavior is not built into the raw dataset, so route cases should be constructed explicitly: **exact hits by duplicating selected question/context pairs** and inserting them back into the run stream at random positions; **semantic hits by using conservative deterministic rewrites or offline LLM-generated paraphrases** saved into the suite; misses by pairing the same context with different questions or using similar questions from different contexts.

Knowledge hits may be possible if the dataset contains repeated `context` fields with multiple questions. Before making knowledge hits a required metric, inspect the local dataset for repeated or near-repeated contexts. If repeated contexts exist, seed queries can populate reusable facts and later questions over the same context can test whether the knowledge layer helps. If repeated contexts are sparse, LongBench-v2 should be treated mainly as an exact/semantic/miss cache benchmark plus direct reasoning accuracy benchmark.

### MemoryAgentBench

- Deterministic labels: MIXED; some QA/classification-style tasks can likely be scored directly, while summarization or long-range understanding tasks may need task-specific scoring or a judge model.
- Input context length: incremental multi-turn histories; the Hugging Face dataset is small (`<1K` rows, currently 146 rows), but exact token lengths should be profiled from the released parquet splits before implementation.
- Downside: strong conceptual fit for cache/memory, but less clean as the next primary benchmark because task formats are broader and scoring may not be uniformly deterministic.

MemoryAgentBench is directly relevant to the memory side of this project. It evaluates memory agents through incremental multi-turn interactions rather than a single static long document. The benchmark focuses on four competencies: accurate retrieval, test-time learning, long-range understanding, and conflict resolution.

This is a closer match to the semantic cache's knowledge and persistence mechanisms than many static long-context QA datasets. Prior turns can populate cache entries and extracted facts, while later turns can test whether the system retrieves, updates, or rejects stale/conflicting memory correctly.

The reason not to choose it as the immediate next benchmark is scoring and comparability. LongBench-v2 gives cleaner multiple-choice accuracy for the cache system, RLM baseline, and direct LLM baseline. MemoryAgentBench is better as a follow-up once we are specifically evaluating persistent memory behavior, especially false reuse, stale facts, and conflict resolution across an interaction stream.

### OOLONG

- Deterministic labels: mostly YES, with exact-match style scoring for categorical, numeric, date, comparison, and user answers.
- Input context length: public Oolong-synth subsets include 1K, 2K, 4K, 8K, 16K, 32K, 64K, and 128K tokens.
- Downside: many tasks require global aggregation over many records, so bounded top-k retrieval may be the wrong execution strategy without map-reduce or similar aggregation.

OOLONG is attractive because it targets tasks that are not just passage lookup. The benchmark emphasizes aggregation over many context items, such as counting, grouping, comparing distributions, or combining information across many chunks.

This is useful for challenging the system beyond RULER-style retrieval. The tradeoff is that the current semantic cache system retrieves a bounded number of chunks and synthesizes from those chunks. Some OOLONG tasks ask for global properties such as the most frequent label, a count across many records, a temporal distribution, or a user-specific pattern. A top-k retriever can miss most of the evidence needed for those answers because the answer is distributed across many individually relevant records rather than concentrated in one or two passages.

For that reason, OOLONG may require a map-reduce style path or a similar aggregation algorithm: map over many chunks to extract local labels/counts/facts, reduce those intermediate outputs into a global answer, and cache the intermediate computations so repeated questions over the same context window become cheaper. Without that kind of aggregation path, a poor score on OOLONG may mostly show that bounded top-k retrieval is the wrong execution strategy for corpus-wide distributional questions.

### LongMemEval / LME-MC10

- Deterministic labels: NO (for the official LongMemEval version); LME-MC10 version is multiple-choice but less established.
- Input context length: LongMemEval-S is about 115K tokens per history; LongMemEval-M uses 500 sessions and about 1.5M tokens.
- Downside: strong fit for memory/cache behavior, but official scoring is less clean and may require a judge model or constrained adaptation.

LongMemEval is relevant to persistent cache and memory behavior. It evaluates whether an assistant can remember and reason over information from previous interactions, including multi-session reasoning, temporal reasoning, knowledge updates, and abstention.

This aligns with the knowledge-cache story better than standard long-context QA. Previous interactions can seed cache entries or extracted facts, and later queries can test whether the system reuses them correctly. **The main downside is evaluation: the official benchmark is more free-form than LongBench-v2, so clean scoring may require a judge model or a carefully constrained adaptation.**

**There are multiple-choice derivatives such as LME-MC10, but they appear less established than the official LongMemEval benchmark.** They may be useful for internal deterministic scoring, but they should not be the first headline benchmark without noting that they are derivative adaptations.

### CorpusQA

- Deterministic labels: LIKELY YES for benchmark accuracy, BUT the public scoring format needs verification before implementation.
- Input context length: up to 10M tokens in the paper framing; 1M-token variant is reported in model evaluations.
- Downside: dataset/code availability, license, and scoring harness need verification before making it a primary target.

CorpusQA is a strong candidate if we want a benchmark aimed directly at higher context inputs and corpus-level analysis. The paper frames CorpusQA as scaling up to 10M tokens, and public model reports include a CorpusQA 1M variant for question answering over approximately million-token contexts.

The appeal is that CorpusQA targets the limitation we are discussing: many benchmarks assume sparse retrieval, where a few chunks are enough, but corpus-level reasoning can require dispersed evidence across many documents. This makes it conceptually close to OOLONG and potentially a better fit than RULER for evaluating whether a system can reason over a large repository rather than simply retrieve a needle.

The main caution is maturity and availability. Before making CorpusQA a primary implementation target, verify the dataset release, license, scoring harness, and whether the 1M and 10M variants are accessible in a reproducible form. If those pieces are available, CorpusQA could become a strong high-context benchmark for the cache system, direct LLM baseline, and RLM baseline.

### LegalBench-RAG

- Deterministic labels: YES for retrieval/evidence spans; WEAKER for final free-form answer accuracy.
- Input context length: corpus is over 79M characters across legal documents.
- Downside: primarily evaluates retrieval and grounding, not end-to-end reasoning quality.

LegalBench-RAG is better suited for retrieval and evidence-grounding evaluation than for pure closed-form answer accuracy. Its gold evidence spans make it possible to measure whether retrieved chunks overlap the correct legal text.

Useful metrics would include evidence recall at k, evidence precision at k, first relevant rank, answer accuracy when deterministic labels are available, and cache hit distribution/cost savings for the semantic cache system.

**We are not choosing LegalBench-RAG as the next primary benchmark because it mainly evaluates the retrieval component of a RAG pipeline. That is useful, but it does not directly answer the current question: whether the overall system can preserve reasoning quality while reducing cost and latency.** Its strongest labels are evidence spans, so the cleanest evaluation is retrieval overlap, not final reasoning accuracy. It is also still legal-domain-specific, while the current direction is broader long-context reasoning rather than legal benchmarks specifically.

It is a good candidate after the main reasoning benchmark is in place, especially if we want to measure evidence grounding and retrieval quality separately from answer accuracy.

## Reference Links

- ContractNLI summary: https://aclanthology.org/2021.findings-emnlp.164/
- LongBench-v2: https://longbench2.github.io/
- MemoryAgentBench: https://huggingface.co/datasets/ai-hyz/MemoryAgentBench
- OOLONG paper: https://openreview.net/forum?id=lrDr6dmXOX
- OOLONG public dataset: https://huggingface.co/datasets/oolongbench/oolong-synth
- CorpusQA paper: https://arxiv.org/abs/2601.14952
- LongMemEval: https://github.com/xiaowu0162/LongMemEval
- LongMemEval (LME-MC10): https://huggingface.co/datasets/Percena/lme-mc10/viewer/default/train?row=1
- LegalBench-RAG: https://github.com/zeroentropy-cc/legalbenchrag
