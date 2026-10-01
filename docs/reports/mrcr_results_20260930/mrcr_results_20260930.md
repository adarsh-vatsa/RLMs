# MRCR v2 results, 29–30 September 2026

Hybrid and direct runs on the 4-needle MRCR v2 release, on the shared execution
pipeline, on the Neselab server. MRCR is scored deterministically, so these
scores involve no model grader.

## Summary

- **Retrieval keeps working where the model's window runs out.** On 120 questions,
  hybrid answered **51%** exactly and direct **38%**, gaining 20 questions and
  losing 5 (paired test p = 0.004).
- **The gain appears only once the conversation exceeds the window, and grows
  with length.** Up to 267K tokens the two modes score the same. At 533K, direct
  keeps half of the conversation and scores 27%; hybrid scores **60%**. At 1.06M,
  direct keeps a quarter and scores 10%; hybrid scores **30%**.
- **Retrieval finds the needles about twice as often as chance.** It packed 76% of
  each question's needle group while reading 48% of the 533K conversation, and 50%
  while reading 24% of the 1.06M one.
- **Counting occurrences remains the limit, as expected.** Even with the whole
  133K conversation in view, the model answers 63% correctly; nearly all its
  misses copy a real needle, but the wrong occurrence.
- **Runs are reproducible.** The 40 questions shared by the two run pairs got
  identical answers in both, in both modes.

## The task

MRCR (multi-round coreference resolution) hides several responses to the *same*
request in a long synthetic conversation, for example four poems written after
the identical line "Write a poem about flamingoes in a humorous style." The final
question asks the model to reproduce, say, the third one, prefixed with a random
12-character marker. Answering requires finding the matching responses, counting
them in order and copying the right one exactly.

Each released file holds a single conversation just under a power-of-two length,
with many questions about it, so each length below is one conversation:

| Conversation | Prompt tokens | Questions available | Needle groups |
|---|---:|---:|---:|
| 133K | 133,073–133,080 | 133 | 77 |
| 267K | 266,624–266,632 | 218 | 162 |
| 533K | 532,962–532,971 | 301 | 245 |
| 1.06M | 1,064,310–1,064,321 | 301 | 245 |

These runs use the 4-needle release (953 questions in total), so each question
asks for the first to fourth of four identical requests.

## What was run

| Pair | Date | Questions per conversation | Total | Hybrid run | Direct run |
|---|---|---:|---:|---|---|
| A | 29 Sep | 10 | 40 | `20260929T185154649177Z` | `20260929T202814048970Z` |
| B | 30 Sep | 30 | 120 | `20260930T172140824150Z` | `20260930T192314239638Z` |

Both modes use the executor's full served window: 262,144 tokens, of which 4,096
are reserved for output, leaving a 258,048-token input budget. The executor is
Qwen/Qwen3.6-35B-A3B on vLLM, thinking disabled, temperature 0.

- **Direct** sends the whole prompt when it fits. Otherwise it keeps an equal head
  and tail (`--direct-overflow middle`), so it keeps all of the 133K conversation,
  97% of the 267K one, 48% of the 533K one and 24% of the 1.06M one.
- **Hybrid** splits each conversation into 3,500-token chunks overlapping by 350
  tokens. It ranks them against the request the question describes (for example
  `poem about flamingoes in a humorous style`, without the marker or the
  ordinal), fills the budget with the best-ranked chunks and presents them in
  conversation order. On the 133K conversation, which fits, it sends the same
  request as direct.

Questions are taken evenly across each conversation's needle groups
(`--max-rows-per-source`), so the 10 per conversation of pair A are a subset of the
30 of pair B. On those 40 shared questions, both modes gave identical answers in
both pairs, so pair B is the main result below and pair A confirms it repeats.
Pair A scored higher (hybrid 63%, direct 40%) only because the 80 questions
added in pair B happen to be harder.

## Results

Exact match, pair B:

| Conversation | Direct | Hybrid | Gained / lost | Paired test |
|---:|---:|---:|---:|---:|
| 133K | 19/30 (63%) | 19/30 (63%) | 0 / 0 | identical requests |
| 267K | 16/30 (53%) | 15/30 (50%) | 0 / 1 | — |
| 533K | 8/30 (27%) | **18/30 (60%)** | 12 / 2 | p = 0.013 |
| 1.06M | 3/30 (10%) | **9/30 (30%)** | 8 / 2 | p = 0.11 |
| **All** | **46/120 (38%)** | **61/120 (51%)** | **20 / 5** | **p = 0.004** |

The paired test is an exact binomial test on the questions where the modes
disagree. The official MRCR similarity score shows the same pattern (0.40 direct,
0.53 hybrid overall). No answer reached the output limit.

At 533K, hybrid roughly matches what the model achieves with a whole 133K
conversation in view, although these are different conversations. Direct falls
steadily with length: 63%, 53%, 27%, 10%.

### Why hybrid wins: it sees the needle

A question can only be answered if the requested response is in what the model
receives. Every correct answer in either mode had its target visible.

| Conversation | Target visible: direct | Target visible: hybrid | Hybrid: share of needle group packed | Share of conversation packed |
|---:|---:|---:|---:|---:|
| 533K | 10/30 | 24/30 | 76% | 48% |
| 1.06M | 5/30 | 13/30 | 50% | 24% |

Head-and-tail truncation sees a needle only if it happens to sit near the start or
end. Retrieval finds needles at roughly twice the rate that reading the same
amount of text at random would. At 1.06M it still misses half of each group,
which is where the remaining gap lies.

### What still goes wrong: counting

Most wrong answers copy one of the group's real responses, but not the one asked
for. Among answers whose target was visible yet wrong, 31 of 36 for hybrid and 27
of 29 for direct had every earlier occurrence visible too. The model saw all it
needed and still counted wrong. On the 133K conversation, where both modes see
everything, 11 of 30 answers in each mode were the wrong occurrence.

| Requested occurrence | Questions | Direct | Hybrid |
|---|---:|---:|---:|
| First | 26 | 16 | 19 |
| Second | 38 | 14 | 17 |
| Third | 28 | 7 | 12 |
| Fourth | 28 | 9 | 13 |

Later occurrences are harder in both modes. This was expected: the limit belongs
to the model, not to retrieval, and it caps both modes on every conversation.

## Note: 4 needles versus 8

### What the needle count changes

In each needle group, the same request appears N times in the conversation, each
time followed by a different response, and the question asks for the k-th
response, with k up to N.

| | 8 needles | 4 needles |
|---|---|---|
| Identical requests per group | 8 | 4 |
| Questions ask for | first to eighth | first to fourth |
| To answer the k-th, the model must see and count | up to 8 occurrences in order | up to 4 |
| Needle groups on the 133K conversation (in its questions) | 38 | 77 |

The conversations are about the same length in both releases, so fewer needles
per group means more, smaller groups and less counting. For hybrid it also means
fewer earlier occurrences to capture before the target can be counted.

### What changed what

The 29 September 8-needle pair and pair A both used 40 questions, 10 per
conversation. Between them, three things changed: the needle count (8 to 4), the
retrieval query (from the full question, marker included, to the described
request only) and the chunk size (7,500 to 3,500 tokens). Exact matches:

| Run | 133K | 267K | 533K | 1.06M | Total /40 |
|---|---:|---:|---:|---:|---:|
| 8 needles, direct | 3 | 4 | 1 | 2 | 10 |
| 8 needles, hybrid (7,500-token chunks, full question as query) | 3 | 4 | 2 | 1 | 10 |
| 4 needles, direct | 6 | 7 | 3 | 0 | 16 |
| 4 needles, hybrid (3,500-token chunks, request-only query) | 6 | 7 | **9** | **3** | **25** |

- **Fewer needles raised both modes.** Direct uses no chunks or retrieval query,
  yet rose from 10 to 16. On the 133K conversation, where both modes see
  everything and no retrieval happens, accuracy doubled from 3 to 6: counting
  became easier.
- **The retrieval changes created hybrid's advantage.** With 8 needles, hybrid and
  direct tied on the two conversations larger than the window (3 each). With 4
  needles, hybrid leads there 12 to 3. Retrieval also improved directly: at 1.06M
  it packed 30% of each needle group while reading 23% of the conversation with 8
  needles, against 50% while reading 24% with 4.
- **Chunk size and query cannot be separated here.** Both changed at once. The
  smaller chunks are the more likely cause, since a needle's only link to its
  question is a request line of about 15 tokens, which carries more weight in a
  smaller chunk. One hybrid run on the same 120 questions with 7,500-token chunks
  would settle it (see [Next steps](#next-steps)).

The questions differ between the two releases, so these comparisons are
indicative rather than exact.

## Run time

| Run | Questions | Wall time | Generation | Embedding | Packing |
|---|---:|---:|---:|---:|---:|
| Hybrid, pair B | 120 | 117 min | 52 min | 44 min | 18 min |
| Direct, pair B | 120 | 57 min | 54 min | — | — |
| Hybrid, pair A | 40 | 67 min | | | |
| Direct, pair A | 40 | 19 min | | | |

Hybrid embeds each conversation once per run on CPU: 6.3, 12.4 and 25.4 minutes for
the 267K, 533K and 1.06M conversations with 3,500-token chunks (83, 166 and 330
chunks). Packing re-tokenizes the prompt for each candidate chunk, about 12
seconds per question.

## Limitations

- **One conversation per length.** Each length is a single synthetic
  conversation, so results per length describe that conversation. The length
  trend rests on four data points.
- **30 questions per conversation.** Questions on one conversation share its
  text and are not fully independent; the 1.06M difference is not significant on
  its own (p = 0.11).
- **Combined changes.** The improvement over the 8-needle runs mixes needle
  count, query and chunk size; see the [note on needle count](#note-4-needles-versus-8).
- **No serving record.** These runs predate the `--serving-metadata` step.

## Next steps

1. Run more questions per conversation, or all 953, to tighten the per-length
   results; after embedding, extra hybrid questions cost about 40 seconds each.
2. Add the 1M–2M file (one conversation of about 2.1M tokens) to extend the
   length trend.
3. Rerun hybrid on the same 120 questions with 7,500-token chunks
   (`MRCR_CHILD_TOKENS=7500 MRCR_CHILD_OVERLAP_TOKENS=750`), keeping the
   request-only query. If accuracy at 533K and 1.06M falls back toward direct,
   chunk size drove the gain; if it stays near 60% and 30%, the query change did.
   Direct does not need rerunning.
4. Try smaller chunks at 1.06M, where half of each needle group is still missed.

## Artifacts

Runs are in `benchmark_artifacts/mrcr_v2/{hybrid,direct}/<run id>`, with the run
IDs listed above; the earlier 8-needle runs are in
`benchmark_artifacts/mrcr_v2/{hybrid,direct}/archive/`. The dataset is
`benchmark_data/mrcr_v2_4needle_100k_1200k`. Commands are in the
[Linux benchmark runbook](../../linux/BENCHMARK_RUNBOOK.md); the task and scoring
are described in the [MRCR overview](../../mrcr_v2.md).
