# Overflow-policy conditions

Every benchmark run answers one question: the full prompt exceeds the executor's
input budget, so what does the model see? The conditions below are the answers
we compare. They apply to any benchmark that runs through
`execution.pipeline.Pipeline` (AA-LCR, MRCR, LongBench's OpenAI-compatible
paths); only the dataset parsing, prompt format and scoring differ.

Budgets are set per run with `--context-window-tokens` and
`--max-input-tokens` (the window minus room for output and the chat template).
A prompt that already fits is sent unchanged under every condition, so the
conditions only differ on prompts longer than the budget.

## Direct: send the raw prompt, cut it if too long

Selected with `direct --direct-overflow <policy>`. Truncation is applied to the
final user message by token count; the question stays intact
(`execution/tokens.py`).

| Condition     | Flag                                                    | What the model sees                                                                              |
| ------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Full context  | `--context-window-tokens` large enough for every prompt | The whole prompt. Upper-bound reference; no overflow.                                            |
| Middle        | `middle`                                                | Equal head and tail, middle dropped silently (the LongBench protocol).                           |
| Marked middle | `middle-marked`                                         | Same cut, with an `[... omitted ...]` marker at the cut. Tests whether signalling the gap helps. |
| Head          | `head`                                                  | Only the start of the context, plus the question. Tests whether the end matters.                 |
| Error         | `error`                                                 | Raises on overflow instead of answering. For checking nothing overflows unnoticed.               |

## Hybrid: chunk, rank, pack

Selected with `hybrid --evidence-selection <ranking>`. The documents are split
into overlapping chunks, the chunks are ordered by the ranking, and the highest
ranked ones are packed into the budget (`execution/retrieval.py`). All
rankings share the same chunking and packing, so a difference between them is
the effect of the ranking alone.

| Condition | Flag                        | Ranking                                                                                                                    |
| --------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Dense     | `dense` (default)           | Embedding similarity to the question.                                                                                      |
| BM25      | `bm25`                      | Lexical keyword score.                                                                                                     |
| Fusion    | `fusion`                    | Reciprocal-rank fusion (k = 60) of the dense and BM25 rankings.                                                            |
| Random    | `random --selection-seed N` | Seeded shuffle per case. No relevance signal; run several seeds to get variance.                                           |
| Position  | `position`                  | Query-independent: chunks nearest a document's start or end first, so every document keeps its edges and loses its middle. |

## What the comparisons isolate

- **Question-aware vs. question-blind selection:** dense, BM25 and fusion
  against random and position. If random or position matches dense, the benchmark
  is not rewarding retrieval.
- **Selection vs. plain truncation:** hybrid conditions against the direct
  truncation policies at the same budget.
- **Which part of the context carries the answer:** middle vs. head vs. position
  show whether evidence sits at the start, the end or the edges of documents.
- **Effect of signalling the cut:** middle vs. marked middle.
- **Budget sensitivity:** repeat one direct and one hybrid condition at several
  budgets. Pick budgets so some prompts still overflow; a budget that fits
  every prompt makes all conditions identical.

## Running a condition

Keep each condition attributable: give it its own `--run-id`, keep all other
flags (model, chunk size and overlap, output limit, temperature) identical
across conditions, and record the budget. The experiment list and run status
are in [workshop_paper_todo.md](workshop_paper_todo.md); per-benchmark suite
runners are described in the [Linux runbooks](linux/), and the
[Jarvis suite pages](jarvis_suites/README.md) cover running them on Jarvis.
