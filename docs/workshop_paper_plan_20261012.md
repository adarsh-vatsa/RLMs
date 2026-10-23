# Workshop paper plan: 12 October 2026

What to run, in what order, to turn the overflow-policy results into a
defensible workshop paper. It builds on the
[paper positioning](paper_positioning_20261007.md) of 7 October and adopts its
framing: the subject is the **overflow policy**, what a system does when the
input is longer than the window, and retrieval is presented as
*query-aware truncation*, one of several redaction policies over the same
prompt. This plan adds the code changes made since then, the experiment
matrix they enable, and a schedule.

## Summary

- **The pipeline now covers every baseline the positioning asked for.** Branch
  `query-aware-truncation-upgrades` adds random-chunk, BM25, per-document and
  head-only truncation, marked truncation, dense + BM25 fusion, budget-filling
  packing, neighbour widening, a per-document floor, the MRCR full-question
  query and a reranker fix. Defaults reproduce the September runs.
- **The paper's claim becomes a three-way comparison.** Naive truncation,
  query-aware truncation and an oracle that keeps exactly the needed text
  (Distractor-Aware Truncation [11] in the positioning). The finding is how much
  of the oracle's recovery a label-free selector gets on real tasks, and beyond
  the 180K tokens that work stopped at.
- **One control decides how strong the paper is.** "Head and tail of every
  document" (`--evidence-selection position`) is rendered exactly like
  retrieval and differs only in which chunks it keeps. If it closes most of the
  17-point AA-LCR gap, structure was doing the work and the paper is about the
  benchmark protocol. If it stays near truncation, selection did the work and
  the paper confirms the mechanism with a fair baseline. Run it first.
- **Target the AAAI-27 workshops (20 November).** The code is ready, and every
  new condition except fusion and the upgraded recipe runs without embedding.
  EACL workshops (15 December) are the fallback and leave room for the second
  model.

## What changed in the code

All options are on the shared pipeline, so AA-LCR, MRCR and LongBench-v2 get
them through the same flags. Commands are in the
[benchmark runbook](linux/BENCHMARK_RUNBOOK.md#overflow-policy-conditions) and
details in the [options reference](linux/SHARED_EXECUTION_RUNBOOK.md#retrieval-and-packing-options).

| Positioning weakness or objection | Change | Flag |
|---|---|---|
| Truncation is a strawman | Head and tail of every document, rendered like retrieval; random chunks at the same budget | `--evidence-selection position`, `--evidence-selection random --selection-seed N` |
| Truncation gets no omission marker | Middle truncation with a marker at the cut | `--direct-overflow middle-marked` |
| Head-only truncation missing | Keep the start of the documents, then the question | `--direct-overflow head` |
| BM25 baseline missing | Okapi BM25 over the same chunks | `--evidence-selection bm25` |
| "Hybrid" should mean dense + sparse | Reciprocal-rank fusion of dense and BM25 | `--evidence-selection fusion` |
| The MRCR query is hand-built | Rank by the full final question instead | `--retrieval-query question` (MRCR) |
| Packing is slower than necessary | The prompt is re-counted only when token bounds cannot decide | automatic |
| Coarse chunks leave budget unused | Keep trying lower-ranked chunks, or trim the next one to fill | `--packing skip`, `--packing fill` |
| Small chunks lose context | Retrieve small chunks, widen each by its neighbours | `--child-tokens`, `--neighbor-children N` |
| Multi-document questions miss documents | Each document's best chunk first | `--document-floor` |
| Reranking was untested | `--pipeline-rerank-top N` truncated the candidate list to about N; it now reorders the top N and keeps the rest | bug fix |

Things to know when reading results:

- **Packing selections are unchanged.** With `--packing stop` (the default) the
  new packer selects exactly what re-counting after every chunk selects. This
  was checked on synthetic fixtures with three tokenizers, and on 45 random
  cases with the Qwen3.6 tokenizer and chat template (no differences, about six
  times fewer full-prompt counts).
- **Route names are unchanged.** Every hybrid run is still `dense_child_packed`;
  each record now states its `evidence_selection`. Truncated direct runs are
  `middle_truncated`, `middle_marked_truncated` or `head_truncated`.
- **Non-dense selections skip embedding.** `bm25`, `random` and `position` chunk
  with the same tokenizer and boundaries as `dense` but never embed, so on the
  single-GPU server they take minutes instead of hours.
- **Manifests gain five settings** (`evidence_selection`, `selection_seed`,
  `neighbor_children`, `document_floor`, `packing`). Runs from before this
  change lack them; their values were the defaults.

The test suite has 232 tests (17 new). Two fail locally for environment
reasons that predate this change: one needs `torch`, one needs the LongBench
`data.json`.

### Review notes to check during implementation

Minor points from the merge review on 21 October. None blocks a run.

- **Packing equivalence is tested, not proven.** The bounds come from one probe
  fragment plus 16 tokens of slack. A bound that is too loose upward is caught
  by the pipeline's final budget check, but one that is too tight downward would
  silently reject a chunk that fits, and nothing reports it. Check
  `packing_stop_reason` and `packing_renders` if a run's selections look short.
- **`fill` tries one chunk.** It trims the best-ranked chunk that adds new text,
  using only that chunk's first uncovered range. If the trim fails it does not
  try the next chunk.
- **Reranking now fails without a reranker.** `--pipeline-rerank-top N` calls
  `controller.reranker` directly; before, the controller skipped reranking
  silently. The LongBench-v2 API runner does not build a reranker, so the flag
  raises an `AttributeError` there. Wire one in before using reranking on
  LongBench-v2.
- **`--retrieval-query` is MRCR only.** AA-LCR and LongBench-v2 always rank by
  their task query, which matches the docs.

## The paper in one figure and three tables

| Item | Content | Answers |
|---|---|---|
| Figure 1 | Accuracy against the share of the input that fits the budget, one line per policy, one panel per benchmark | The central claim, and where each policy breaks |
| Table 1 | All policies at one budget per benchmark: truncation variants, position, random, BM25, dense, fusion, upgraded, full context or oracle | Selection versus structure versus coverage |
| Table 2 | The visibility decomposition (target visible × accuracy given visible) for truncation, dense, upgraded and oracle on MRCR | Why it works, and how far from the oracle |
| Table 3 | Cost per question: embedding (CPU and GPU), packing and generation time, input tokens | "For one embedding pass", stated honestly |

The oracle for Table 2 can start analytically. MRCR needle positions are known,
so an oracle's visibility is 100% and its predicted accuracy is the model's
accuracy given visibility. A real oracle run would confirm it. It needs a small,
clearly labelled MRCR script that orders chunks containing the needle messages
first; keep it out of the shared pipeline, whose solver input never sees gold
data.

## Experiment matrix

Run conditions within a benchmark with identical data, budgets and serving.
Times are from the 30 September runs on Neselab with CPU embedding.

### Stage 1: the deciding controls (this week)

| Run | Benchmark | Flags | Approximate time |
|---|---|---|---|
| Position | AA-LCR, 64K | `hybrid --evidence-selection position` | 15 min |
| Random, seeds 0–2 | AA-LCR, 64K | `hybrid --evidence-selection random --selection-seed N` | 45 min |
| Marked truncation | AA-LCR, 64K | `direct --direct-overflow middle-marked` | 15 min |
| Head only | AA-LCR, 64K | `direct --direct-overflow head` | 15 min |
| Position, random seed 0 | MRCR, 262K | same flags | 1 h each |

Read the result before continuing. Where `position` and `random` land between
truncation (37%) and dense retrieval (54%) on AA-LCR sets the paper's lead (see
Summary). The existing 30 September dense and middle-truncation runs remain the
reference.

### Stage 2: lexical retrieval and the upgraded recipe

| Run | Benchmark | Flags |
|---|---|---|
| BM25 | AA-LCR, MRCR | `--evidence-selection bm25` |
| Fusion | AA-LCR, MRCR | `--evidence-selection fusion` |
| MRCR chunk size | MRCR | 7,500-token chunks with the request query, as already planned |
| MRCR full-question query | MRCR | `--retrieval-query question` |
| Upgrade ladder | MRCR, then LongBench-v2 | add one change at a time to `fusion`: `--child-tokens 1024 --child-overlap-tokens 128`, then `--neighbor-children 1`, then `--packing fill`, then `--document-floor` |
| Upgraded recipe | AA-LCR | the final ladder configuration, once |

Choose the upgraded recipe on MRCR and LongBench-v2 only, and run it on AA-LCR
once as a held-out check. AA-LCR has 100 questions, so choosing settings on it
would overfit. Use one recipe for all three benchmarks in the paper; tuning per
benchmark is the first thing a reviewer will look for. Report the whole ladder,
including steps that did not help.

### Stage 3: what makes it defensible

These are items from the positioning list, unchanged in substance:

1. **Independent AA-LCR grader.** Regrade every AA-LCR run with a model other
   than the executor, preferably from another family, using `aa_lcr.regrade`.
   Hand-label about 50 grades to report the grader's accuracy.
2. **LongBench-v2 on the shared pipeline**, at a 64K window on questions whose
   full prompt fits 262K, so a full-context reference exists. Multiple choice,
   so no grader is needed. Run the Table 1 policies here too.
3. **Budget sweep for Figure 1:** 32K, 64K and 128K windows for truncation,
   position and the upgraded recipe on all three benchmarks.
4. **GPU embedding timing** for Table 3; see the runbook for the memory setting
   on the single-GPU server.
5. **A second model family.** The positioning ranks this as the most likely
   reason for rejection. If it does not fit before AAAI, it is the main
   addition for EACL.

## Schedule to the AAAI-27 workshops (20 November)

| Week | Work |
|---|---|
| 12–18 Oct | Merge the branch after review. Stage 1 on AA-LCR and MRCR. Independent regrade of the 30 September and Stage 1 AA-LCR runs. Decide the lead |
| 19–25 Oct | Stage 2 on MRCR; LongBench-v2 shared-pipeline runs start on Jarvis |
| 26 Oct–1 Nov | Ladder on LongBench-v2; upgraded recipe on AA-LCR; GPU embedding timing |
| 2–8 Nov | Budget sweep; MRCR oracle (analytic, then a run if time allows); figures |
| 9–20 Nov | Write. Related work from the positioning references, verified in full text before citing |

If Stage 1 shows that structure explains most of the gap, move the budget sweep
earlier. A protocol paper needs the curve more than the ladder.

## Claims

| Make | Avoid |
|---|---|
| Query-aware truncation recovers X% of what middle truncation loses, at the same budget, on three task types | That the method is new; it is OP-RAG and DOS RAG, said in the first paragraph |
| It reaches Y% of an oracle's recovery on MRCR, and works at 533K–1.06M tokens | That focused prompts beat full context; 10 AA-LCR questions are anecdotal |
| The gain is visibility, shown on MRCR; position and random controls rule out structure and coverage (if Stage 1 says so) | "Fewer tokens and less time" without GPU embedding numbers |
| Where it breaks: budget share and evidence spread | Anything about the semantic cache, grounding check or consensus |

## Risks

- **Stage 1 may show structure matters most.** That is still a paper, about
  the benchmark protocol and close to [11], but the lead changes. Decide by
  26 October.
- **The upgrades may not help.** Then the ladder supports "the simplest
  selector is enough", which suits the pitch. Report it as such.
- **Single model.** Mitigate with the second family; otherwise state it first
  in the limitations.
- **Concurrent work.** Distractor-Aware Truncation (August) shows how close the
  space is. Check arXiv for overflow-policy papers once more before submitting.
