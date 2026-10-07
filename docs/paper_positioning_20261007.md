# Paper positioning: 7 October 2026

A record of the literature review and positioning discussion for a workshop
paper based on the current experiments. It covers what prior work already
claims, what we can still claim, the recommended pitch, the weaknesses a
reviewer will find and the experiments that would close them. Results and
system details are in the [project status](project_status_20261001.md) and the
two September reports; this document does not repeat them in full.

## Summary

- **The method is not new.** Retrieving chunks of an over-long input and
  presenting them in source order is published (OP-RAG, DOS RAG) and was run as
  an ablation in the LongBench v2 paper. We should not present the pipeline as a
  novel architecture.
- **The paper should be an evaluation paper.** Its subject is the *overflow
  policy*: what a system does when the input is longer than the model's window.
  The default in benchmark harnesses is to cut out the middle. We show that
  retrieval is a much better policy and explain why.
- **Lead with the comparison across task types.** Retrieval beats truncation at
  the same budget on multi-document reasoning (AA-LCR), exact reproduction
  (MRCR v2) and mixed multiple-choice tasks (LongBench-v2). This is the one
  claim all three benchmarks support.
- **The explanation is shown on MRCR only.** Beyond the window, accuracy is
  roughly the chance that the needed text is in the model's input, times the
  model's accuracy when it is. Retrieval raises the first factor and leaves the
  second unchanged. MRCR is a needle-style task, and neither AA-LCR nor
  LongBench-v2 releases evidence locations, so the paper should present this as
  the explanation, not as its central claim.
- **A second measurement should come from AA-LCR,** not from another synthetic
  benchmark. Reviewers already expect retrieval to win on needle tasks; what
  they doubt is multi-document reasoning.
- **How spread out the evidence is may be the more useful axis.** It is sparse
  in MRCR, spread over several documents in AA-LCR and the whole context in
  some LongBench-v2 tasks. Retrieval should help less as evidence gets more
  diffuse, which would also show where the method stops working.
- **The semantic cache, grounding check and consensus verification stay out of
  the claims.** No current benchmark run exercises them, and each has close
  prior work.
- **Six weeks of experiments would make the paper defensible, mostly without
  code changes.** A second model, a budget sweep, an independent grader and
  LongBench-v2 on the shared pipeline run with existing flags. The extra
  baselines need small additions to the shared pipeline.

## The results we are positioning

All runs use Qwen3.6-35B-A3B served by vLLM, one run per setting.

| Benchmark | Setting | Truncation | Retrieval | Paired result |
|---|---|---:|---:|---|
| AA-LCR | 64K window, prompts of 88K–122K | 37% | 54% | +17 points, 95% interval +8.4 to +25.9 |
| MRCR v2, 4 needles | 262K window, 533K conversation | 27% | 60% | 12 gained, 2 lost, p = 0.013 |
| MRCR v2, 4 needles | 262K window, 1.06M conversation | 10% | 30% | 8 gained, 2 lost, p = 0.11 |
| MRCR v2, 4 needles | All 120 questions | 38% | 51% | 20 gained, 5 lost, p = 0.004 |
| LongBench-v2 | 240K budget, 107 over-window questions | 41.1% | 56.1% | 21 gained, 5 lost, p = 0.0025 |

Full context on AA-LCR, which fits the 262K window, scores 61%. The LongBench-v2
figures are from the August runner, before the shared pipeline.

Sources:
[AA-LCR report](reports/aa_lcr_results_20260930/aa_lcr_results_20260930.md),
[MRCR report](reports/mrcr_results_20260930/mrcr_results_20260930.md),
[project status](project_status_20261001.md).

## What prior work already claims

| Prior work | What it shows | What we can no longer claim as new |
|---|---|---|
| OP-RAG [1], DOS RAG [2] | Retrieve chunks and present them in source order. This simple recipe matches or beats multi-stage pipelines such as ReadAgent and RAPTOR. | The packing method. Our retrieval route is this method. |
| LongBench v2 [9] | Its own RAG ablation concatenates retrieved chunks in original order. Qwen2.5 with 32K of retrieved text beat its full 128K context by 4.1 points. | That a focused prompt can beat full context. |
| Self-Route [3], Pre-Route [4], LaRA [5] | Routing each query to RAG or to long context, by model self-reflection or document metadata. | Routing. Ours is a length check, which is simpler but not newer. |
| Distractor-Aware Truncation [11] | Naive middle truncation destroys answer-bearing content, including on MRCR v2. Removing only irrelevant content preserves or improves accuracy. | That truncation is a poor baseline. But this work uses oracle knowledge of what is relevant, closed models and no retriever. |
| Recall Is Not Enough [12] | "Answer-in-context", whether the gold answer survives packing, predicts budgeted RAG accuracy better than retrieval recall. | The diagnostic behind our visibility analysis. We should cite it and use its term. |
| BudgetBench [13] | Treats the per-call token budget as the independent variable for local models (including a Qwen3 30B) at 2K–32K. Results were inconclusive. | Budget as the experimental variable, at small budgets. |
| vCache [26], Krites [28], MeanCache [27], GPTCache, Cortex | Semantic caches with a verification step: error-rate guarantees, an LLM judge, or a learned similarity model. | A verified two-stage semantic cache. |
| Recursive Language Models [24], reproduction [25] | Recursive model calls over an external copy of the prompt handle inputs far beyond the window. The reproduction reports large increases in time and tokens. | Nothing directly, but it is the comparison reviewers will think of, and we have no current numbers for it. |

Broader context a related-work section should cover:

- **Long context versus RAG.** When the input fits, long context usually wins on
  average and RAG is cheaper [3, 6]. Adding more retrieved passages helps and
  then hurts [7]. A small model with retrieval can match a larger-window model
  [8].
- **Effective context is shorter than the advertised window.** Accuracy falls as
  input grows even on simple tasks [23]; open-weight models lag most on tasks
  that need the full context [20]; real-document benchmarks show the same
  [10, 15].
- **Long-context tasks are often solvable with short context** if the system
  chooses what to read [17]. KV-level retrieval is another route [18].
- **MRCR is designed to test more than retrieval** [21], which makes a retrieval
  gain on it more notable than on a needle-in-a-haystack task.

## The gap we can occupy

Most studies ask whether a small retrieved context can replace a full context
that fits, usually with closed models and retrieval budgets of a few thousand
tokens. The review found no paper that studies overflow directly with all of:

- one open-weight model, served locally, with inputs from 0.5× to 4× the served
  window;
- truncation and retrieval at the same token budget, paired by question;
- current benchmarks (AA-LCR, MRCR v2, LongBench v2);
- errors split into "the needed text was not in the input" and "it was, and the
  model still failed".

The Self-Route authors noted in passing that RAG wins when the input far exceeds
the window, without studying it [3]. The oracle truncation paper [11] is the
most useful point of contrast: it shows what is possible when the relevant
content is known, and we show how much of that a small off-the-shelf embedding
model recovers without labels.

## Recommended positioning

### One-sentence pitch

When an input exceeds the window, replacing middle truncation with the simplest
retrieval recovers most of the lost accuracy on three different kinds of task,
for one embedding pass, because it keeps the needed text in the model's input.

### Framing: query-aware truncation

The retrieval route does not build a separate "RAG prompt". It re-renders the
original prompt with parts of the documents removed: selected chunks are mapped
back to character ranges, put in source order, merged where they overlap and
shown with `[... omitted source ...]` markers and the original document numbers
([`execution/packing.py`](../execution/packing.py)). When the prompt fits, the
direct and retrieval requests are identical.

Truncation and retrieval can therefore be described as two *redaction policies*
over the same prompt, differing only in which spans they keep:

| Policy | Chooses spans by |
|---|---|
| Head and tail (the benchmark default) | Position, ignoring the question |
| Random chunks | Nothing |
| Dense retrieval (ours) | Similarity to the question |
| Distractor-aware truncation [11] | Oracle knowledge of the answer |

This changes what we are compared against, not what we built. The mechanism is
still OP-RAG's, and the paper should say so in its first paragraph. We have not
checked whether OP-RAG or DOS RAG also merge spans and mark omissions; that
needs checking before we say the rendering differs.

Two naming points for the paper:

- Call the method RAG plainly, then qualify it (for example "budget-filling,
  order-preserving RAG used as the overflow policy"). It differs from textbook
  RAG in that the corpus is the prompt itself, it fills the window instead of
  taking a top-k, and it runs only on overflow.
- Do not call it "hybrid". In the retrieval literature that means dense plus
  sparse retrieval. The shared pipeline is dense-only, with reranking off by
  default.

### The visibility decomposition

Accuracy can be split into two factors that multiply:

1. **Visibility:** how often the text needed to answer is in the model's input.
   This depends only on the overflow policy.
2. **Accuracy given visibility:** how often the model answers correctly when
   that text is present. This depends on the model.

From the MRCR report, using the fact that every correct answer had its target
visible:

| Conversation | Policy | Target visible | Correct, given visible | Accuracy |
|---|---|---:|---:|---:|
| 133K (fits) | either | 30/30 | 19/30 (63%) | 63% |
| 533K | truncation | 10/30 | 8/10 (80%) | 27% |
| 533K | retrieval | 24/30 | 18/24 (75%) | 60% |
| 1.06M | truncation | 5/30 | 3/5 (60%) | 10% |
| 1.06M | retrieval | 13/30 | 9/13 (69%) | 30% |

Retrieval more than doubles visibility. Accuracy given visibility stays near the
in-window level under both policies, which suggests that splitting the
conversation into chunks does not itself hurt the model. What remains is the
model's own limit, which on MRCR is counting occurrences.

Limits of this claim:

- The counts are small (52 visible-target questions beyond the window).
- It is measured only on MRCR, where needle positions are known. On AA-LCR the
  report infers it indirectly, from about 10 truncated answers that say the
  information is not in the text.
- Neither other dataset says where the evidence is. AA-LCR lists the source
  files of the whole document set, the same for every question in it.
  LongBench-v2 has only the question, choices, answer and context, and some of
  its tasks have no single evidence location.
- MRCR is a needle-style task: the answer sits in a few known places. Retrieval
  is expected to do well there, so the result does not show that the mechanism
  holds for reasoning over several documents.
- AA-LCR hints that the strict form may not hold there. Retrieval answered 10
  questions that full context missed, so removing text seems to change the
  model's accuracy as well as visibility.

### Claims and their strength

| Claim | Strength | Basis |
|---|---|---|
| Retrieval beats truncation at equal budget | Strong | Significant on three benchmark types; LongBench from the old runner |
| The gain is evidence visibility, not better reading | Promising, under-sampled | MRCR table above |
| A quarter-size window with retrieval stays close to full context | Weak | AA-LCR: 54% against 61%, not significant, interval −16.7 to +2.0. On the longest prompts retrieval scores 17 of 39 against 26 |
| A focused prompt can beat full context | Anecdotal | 10 AA-LCR questions |

The third claim is better framed as "where it breaks": no loss when the budget
covers more than about 55% of the prompt, a real loss near 50%. A budget sweep
would turn this into a curve.

### Contributions the code supports

1. **A controlled comparison.** Only the overflow policy varies between
   conditions. Tokens are counted with the executor's chat template, a request
   over budget raises an error, and gold answers never reach the solver
   ([`execution/pipeline.py`](../execution/pipeline.py)).
2. **Per-example provenance.** Each record stores the character ranges the model
   saw and the score of every chunk. This is what makes the visibility analysis
   possible.
3. **The decomposition** as a way to read any overflow result.

## What to leave out

- **The semantic answer cache.** Its results come from the August LongBench
  runner with constructed paraphrases; it is tested only with mocks on the
  shared pipeline; the area is crowded [26, 27, 28].
- **Grounding check and consensus verification.** The shared pipeline never
  calls them, and the August LongBench runner records consensus as disabled. The
  grounding check is a regex test for numbers that appear verbatim in the
  source: it passes a wrong figure taken from the documents and flags a correct
  computed answer. Consensus only annotates, treats verifier failures as
  agreement and would agree with itself if the verifier is the same model. Both
  ideas have established stronger forms [29–33]. `docs/system_architecture.md`
  labels them "Novel"; that label should not be carried into the paper.
- **Any comparison with Recursive Language Models.** We have no current numbers.
- **"Fewer tokens and less time" as stated.** The AA-LCR retrieval run took
  2 h 45 min against 17 min for full context, because embedding ran on CPU.

One cheap test could bring the grounding check back as a short paragraph: run it
offline over the 300 saved AA-LCR answers and see whether its label predicts
correctness. The expected signal is weak.

## Weaknesses in the current comparison

- **Truncation gets no omission marker.** `truncate_middle` joins head and tail
  silently, while retrieval marks every gap. This matches the LongBench
  protocol but is a second difference between the conditions.
- **The retrieval query is built differently per benchmark.** MRCR strips the
  random marker and the ordinal with a regex; AA-LCR uses the bare question;
  LongBench includes the four answer choices and the answer-format instruction.
  The paper must state these, and MRCR needs the full-question query reported
  too.
- **Packing is slower than necessary.** It re-renders and re-tokenizes the whole
  prompt for each candidate chunk, about 12 seconds per MRCR question.
- **AA-LCR at 64K is a simulated overflow.** The model can read the whole
  prompt. This is defensible as deliberate, because it is the only setting with
  a full-context ceiling, but a cost argument for small windows needs memory and
  latency numbers we do not have.
- **Self-grading on AA-LCR, one model, one run per setting, small samples.**
  Already listed as open work in the project status.

## Likely reviewer objections

| Objection | Answer |
|---|---|
| Truncation is a strawman. | It is the published default in benchmark harnesses. A random-chunk baseline at the same budget shows whether selection or mere coverage drives the gain. |
| This is OP-RAG. | Agreed, and stated up front. The contribution is the overflow regime and the decomposition. |
| One model. | Add a second open-weight family. This is the most likely reason for rejection. |
| The MRCR query is hand-built. | Report the full-question query alongside it. |
| Retrieval fails on aggregation tasks. | True. Our 8-needle MRCR runs with 7,500-token chunks tied truncation; report that as a limitation. |
| Why not an agentic or recursive method? | Out of scope: this is a single model call. Cite the cost findings in [25]. |
| The mechanism is only shown on a needle task. | True today. Label evidence on AA-LCR, and report results by how spread out the evidence is. |

## Experiments, in priority order

1. **Offline visibility analysis on saved MRCR runs.** Chunk scores are saved
   for every question and packing runs on CPU, so visibility at any budget can
   be recomputed without a GPU. This gives a visibility-against-budget curve and
   a predicted accuracy curve.
2. **More MRCR questions.** About 40 seconds each after embedding; needed to
   firm up the decomposition.
3. **Independent grader for AA-LCR.**
4. **LongBench-v2 on the shared pipeline**, for the over-window subset.
5. **Baselines at the same budget:** random chunks, BM25, score-ordered packing,
   head-only truncation and truncation with an omission marker. The flags
   `--evidence-order score`, `--no-merge-overlaps` and `--pipeline-rerank-top`
   exist but have no test coverage and have not been run recently.
6. **A budget sweep** (for example 32K, 64K, 128K) for both policies. Accuracy
   against the share of the input that fits, across three benchmarks, is the
   central figure. One or two of these runs also test the prediction from
   item 1.
7. **A second model family.**
8. **MRCR chunk size and query**, separated as already planned.
9. **Cost numbers:** GPU or amortised embedding time, and memory and latency at
   a 64K against a 262K served window.
10. **Evidence labels for AA-LCR.** Either run full context once more asking the
    model to quote the passages it relied on, keeping only quotes that match
    the source exactly, or annotate the 100 questions by hand. The quoted run
    changes the prompt, so it is a labelling run and not a replacement for the
    full-context condition. Its labels cover only questions the model answered
    correctly and may be incomplete.
11. **LongBench-v2 split by task type**, on the August artifacts, to see whether
    the gain shrinks as evidence becomes more diffuse.

Code changes needed: none to the datasets, prompts or scoring. Items 2–4, 6
and 8 run with existing flags, and 7 probably does once token counting is
checked against the new model's chat template. Random chunks, the two
truncation variants and the MRCR full-question query are a few lines each in
`execution/` and the MRCR adapter. BM25 and the quoted-evidence run are the only
larger pieces. Items 1 and 11 are analysis scripts.

## Decisions to confirm

- **The lead.** Recommended: the comparison across task types, with the
  decomposition as its explanation. Leading with the decomposition is an option
  only if it is also measured on AA-LCR. Other framings are a practitioner one
  (serving long inputs on a small window) or a benchmark-methodology one
  (truncation confounds leaderboards), which sits closest to [11].
- **Scope of the cache and memoization work** relative to this paper.
- **Venue and experiment budget.** Items 1–5 fit before the AAAI deadline;
  adding 6 and 7 is tight, and the EACL date gives room for both.

## Venues

| Venue | Paper deadline | Notes |
|---|---|---|
| AAAI-27 workshops [35] | 20 November 2026 | Montréal, 22–23 February 2027, in person. Accepted workshop list posted by 16 October 2026. |
| EACL 2027 workshops [37] | 15 December 2026 (direct submission) | Athens, March 2027. |
| ICLR 2027 workshops [36] | Suggested 1 February 2027 | San Francisco. Workshop list not yet announced. |
| NeurIPS 2026 workshops [38] | Suggested 29 August 2026 | Passed. |
| Context Beyond the Window, COLM 2026 [39] | 23 June 2026 | Passed. The closest topical fit; worth watching for a next edition. |

The venues in `docs/workshop_venues_2026.md` on `main` (ICML 2026 workshops)
have all passed.

## Caveats on this review

- Most 2026 papers were read at abstract or summary level, not in full. Check
  each before citing it.
- Dates and details for [14] disagree between its arXiv identifier and its
  listed submission date, and it was not compared closely.
- References [29–33] and Cortex are cited from memory or from older project
  notes and were not searched in this review.
- Novelty of the pipeline was judged from `execution/`, the three adapters and
  the relevant parts of `semantic_cache_system.py`, not from a full read of that
  file.
- The drafts on `main` (`docs/paper_scope.md`, `docs/paper_draft.tex`) pitch the
  two-stage cache with O(1) scaling as a main-track contribution. Current
  results do not support that framing.

## References

### Retrieval and long context

1. OP-RAG: *In Defense of RAG in the Era of Long-Context Language Models*, 2024.
   https://arxiv.org/abs/2409.01666
2. DOS RAG: Laitenberger, Manning and Liu, *Stronger Baselines for
   Retrieval-Augmented Generation with Long-Context Language Models*, EMNLP
   2025. https://arxiv.org/abs/2506.03989
3. Self-Route: Li, Li, Zhang, Mei and Bendersky, *Retrieval Augmented Generation
   or Long-Context LLMs? A Comprehensive Study and Hybrid Approach*, EMNLP 2024
   Industry Track. https://arxiv.org/abs/2407.16833
4. Pre-Route: Chen et al., *Route Before Retrieve: Activating Latent Routing
   Abilities of LLMs for RAG vs. Long-Context Selection*, 2026.
   https://arxiv.org/abs/2605.10235
5. *LaRA: Benchmarking Retrieval-Augmented Generation and Long-Context LLMs — No
   Silver Bullet for LC or RAG Routing*, 2025. https://arxiv.org/abs/2502.09977
6. Li et al., *Long Context vs. RAG for LLMs: An Evaluation and Revisits*, 2025.
   https://arxiv.org/abs/2501.01880
7. Jin, Yoon, Han and Arık, *Long-Context LLMs Meet RAG: Overcoming Challenges
   for Long Inputs in RAG*, ICLR 2025. https://arxiv.org/abs/2410.05983
8. Xu et al., *Retrieval meets Long Context Large Language Models*, ICLR 2024.
   https://arxiv.org/abs/2310.03025

### Benchmarks and evaluation

9. Bai et al., *LongBench v2: Towards Deeper Understanding and Reasoning on
   Realistic Long-context Multitasks*, 2024. https://arxiv.org/abs/2412.15204
10. He et al., *LooGLE v2: Are LLMs Ready for Real World Long Dependency
    Challenges?*, NeurIPS 2025 Datasets and Benchmarks.
    https://arxiv.org/abs/2510.22548
11. Arjmandi, *Distractor-Aware Truncation: Disentangling Context-Length Effects
    from Signal Loss in Long-Context LLM Benchmarks*, 2026.
    https://arxiv.org/abs/2608.03297
12. Bala, *Recall Is Not Enough: A Reader-Context Diagnostic for
    Budget-Constrained Retrieval-Augmented Generation*, 2026.
    https://arxiv.org/abs/2607.00725
13. Rao and Jaggi, *BudgetBench: A Budget-Tiered Protocol and Pilot Harness for
    Memory Strategy Evaluation in Local Large Language Model Agents*, 2026.
    https://arxiv.org/abs/2609.13149
14. Vabbilisetty et al., *Beyond Static RAG: An Adaptive, Tri-Metric Routing
    Framework for Efficient Long-Context Inference on Commodity GPUs*, 2026.
    https://arxiv.org/abs/2609.17564
15. Huang et al., *ATLAS: All-round Testing of Long-context Abilities across
    Scales*, 2026. https://arxiv.org/abs/2605.28079
16. Song, Zhu, Haque and Zhao, *MegaMem: A Retrieval Solution for Ultra-Large
    Context Windows*, 2026. https://arxiv.org/abs/2608.22137
17. LC-Boost: *Are Long-LLMs A Necessity For Long-Context Tasks?*, 2024.
    https://arxiv.org/abs/2405.15318
18. RetroLM: *Does RAG Really Perform Bad For Long-Context Processing?*, 2025.
    https://arxiv.org/abs/2502.11444
19. LOFT: *Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and
    More?*, 2024. https://arxiv.org/abs/2406.13121
20. *HELMET: How to Evaluate Long-Context Language Models Effectively and
    Thoroughly*, ICLR 2025. https://arxiv.org/abs/2410.02694
21. Vodrahalli et al., *Michelangelo: Long Context Evaluations Beyond Haystacks
    via Latent Structure Queries*, 2024 (source of MRCR).
    https://arxiv.org/abs/2409.12640
22. Artificial Analysis, *Announcing AA-LCR*.
    https://artificialanalysis.ai/articles/announcing-aa-lcr
23. Chroma, *Context Rot: How Increasing Input Tokens Impacts LLM Performance*,
    2025. https://research.trychroma.com/context-rot

### Recursive and agentic approaches

24. Zhang, Kraska and Khattab, *Recursive Language Models*, 2025.
    https://arxiv.org/abs/2512.24601
25. Wang, *Think, But Don't Overthink: Reproducing Recursive Language Models*,
    2026. https://arxiv.org/abs/2603.02615

### Semantic caching

26. Schroeder et al., *vCache: Verified Semantic Prompt Caching*, ICLR 2026.
    https://arxiv.org/abs/2502.03771
27. *MeanCache: User-Centric Semantic Caching for LLM Web Services*, 2024.
    https://arxiv.org/abs/2403.02694
28. Krites: *Asynchronous Verified Semantic Caching for Tiered LLM
    Architectures*, 2026. https://arxiv.org/abs/2602.13165

### Answer verification (from memory; verify before citing)

29. Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in
    Language Models*, ICLR 2023. arXiv:2203.11171
30. Manakul, Liusie and Gales, *SelfCheckGPT: Zero-Resource Black-Box
    Hallucination Detection for Generative Large Language Models*, EMNLP 2023.
    arXiv:2303.08896
31. Cohen et al., *LM vs LM: Detecting Factual Errors via Cross Examination*,
    EMNLP 2023. arXiv:2305.13281
32. Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation*,
    2023. arXiv:2309.15217
33. Min et al., *FActScore: Fine-grained Atomic Evaluation of Factual Precision
    in Long Form Text Generation*, EMNLP 2023. arXiv:2305.14251

### Context and venues

34. *Long Context Benchmarks: All Three Hit 1M — Now What?*, 2026.
    https://yage.ai/share/long-context-benchmark-en-20260315.html
35. AAAI-27 workshops. https://aaai.org/conference/aaai/aaai-27/workshops-call/
36. ICLR 2027 call for workshops.
    https://iclr.cc/Conferences/2027/CallForWorkshops
37. EACL 2027 workshops. https://2027.eacl.org/calls/workshops/
38. NeurIPS 2026 workshops.
    https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/
39. Context Beyond the Window, COLM 2026 workshop.
    https://context-beyond-window.github.io/
