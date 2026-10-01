# AA-LCR results, 30 September 2026

Three AA-LCR runs on the shared execution pipeline, on the Neselab server. Each
answers all 100 questions of the official v1.1 dataset. Scores are preliminary:
the answering model also graded them (see [Grading](#grading)).

## Summary

- **Retrieval recovers most of what truncation loses.** With a 64K model window,
  keeping the start and end of each prompt scored **37%**; retrieving and packing
  the relevant passages scored **54%**, a gain of **+17 points** (95% interval
  +8.4 to +25.9).
- **Retrieval stays close to reading everything.** Reading each whole prompt in
  the full 262K window scored **61%**. The 64K hybrid run kept **89%** of that
  accuracy while reading about half of each prompt; its −7 point gap is not
  statistically significant (95% interval −16.7 to +2.0). Truncation kept 61% of
  it, a significant −24 points.
- **The gap between hybrid and full context opens only on the longest prompts**
  (115K–123K tokens), where the 64K budget holds the smallest share of each prompt.
- **The grader is the main caveat.** The executor grades its own answers, and a
  hand check of about 55 grades found five clear errors and one inconsistency, in
  both directions. Correcting them leaves every conclusion unchanged, but absolute
  scores need an independent grader before they are reported.

## What was run

AA-LCR asks free-form reasoning questions over sets of long documents: 100
questions on 30 document sets, drawn mostly from company documents (63
questions). Full prompts are 87,847–122,112 tokens (median 107,136).

| Run | Model window / input budget | How oversized prompts are handled | Route |
|---|---|---|---|
| Full context | 262,144 / 258,048 | Not needed: every prompt fits | `direct_fit` for all 100 |
| Truncated (64K) | 65,536 / 61,440 | Keep an equal head and tail, cut the middle | `middle_truncated` for all 100 |
| Hybrid (64K) | 65,536 / 61,440 | Retrieve 7,500-token chunks, pack them in source order | `dense_child_packed` for all 100 |

The input budget is the model window minus a 4,096-token output allowance. All
three runs use the same executor (Qwen/Qwen3.6-35B-A3B served by vLLM, thinking
disabled, temperature 0), the same prompt, dataset release 1.1, and the `common`
execution profile with answer caching off. Within each pair compared below, the
comparison tool found no settings difference other than the mode and budget.
Hybrid used CPU embeddings (Qwen3-Embedding-0.6B) with 750-token chunk overlap.

## Results

| Run | Accuracy | Input sent (median) | Share of prompt | Answers hitting the output limit |
|---|---:|---:|---:|---:|
| Full context | **61%** | 107,136 tokens | 100% | 1 |
| Hybrid (64K) | **54%** | 58,296 tokens | 54% | 4 |
| Truncated (64K) | **37%** | 61,439 tokens | 57% | 5 |

Paired comparisons use the repository's comparison tool
(`aa_lcr.compare_conditions`). Its 95% intervals resample whole document sets,
because questions on the same documents tend to succeed or fail together.

| Comparison | Accuracy | Difference | 95% interval | Questions gained / lost |
|---|---|---:|---|---:|
| Truncated → hybrid | 37% → 54% | **+17** | +8.4 to +25.9 | 23 / 6 |
| Full context → hybrid | 61% → 54% | −7 | −16.7 to +2.0 | 10 / 17 |
| Full context → truncated | 61% → 37% | **−24** | −35.4 to −13.8 | 4 / 28 |

Hybrid sent slightly *less* text than truncation, so its gain comes from which
passages it chose, not from reading more. It recovered 17 of the 24 points that
truncation lost.

### By prompt length

| Prompt length | Questions | Full context | Truncated | Hybrid |
|---|---:|---:|---:|---:|
| 88K–100K | 37 | 22 | 18 | 22 |
| 100K–115K | 24 | 13 | 6 | 15 |
| 115K–123K | 39 | 26 | 13 | 17 |

Hybrid matches or beats full context below 115K tokens and falls behind on the
longest prompts, where the 61,440-token budget covers roughly half of each prompt.
Truncation loses most at 100K–115K.

### By document category

| Category | Questions | Full context | Truncated | Hybrid |
|---|---:|---:|---:|---:|
| Company documents | 63 | 37 | 24 | 34 |
| Government consultations | 11 | 6 | 5 | 4 |
| Industry reports | 8 | 5 | 2 | 6 |
| Legal | 6 | 4 | 1 | 3 |
| Marketing | 6 | 6 | 2 | 3 |
| Academia | 5 | 3 | 3 | 3 |
| Survey reports | 1 | 0 | 0 | 1 |

Category counts are small; only company documents have enough questions to read
on their own. Across the 30 document sets, hybrid matched full context on 16,
did better on 5 and worse on 9.

### What the errors look like

- **Truncation removes the evidence.** About 10 of the 23 questions that hybrid
  gained over truncation have truncated answers stating that the information is
  not in the provided text, which is what the cut middle held.
- **Hybrid's losses are mostly reasoning or selection errors,** not missing
  documents: wrong figures (for example 18% used instead of 37.6%), misreading
  "acquired" as acquisitions, or a date taken from the wrong baseline.
- **Hybrid sometimes beats full context.** It answered 10 questions that the
  full-context run missed, such as counts and calculations where the full prompt
  offered more distracting material. This fits the view that a focused prompt can
  help reasoning, though 10 questions is a small sample.

## Grading

All three runs were graded by the executor model itself, Qwen3.6, with the
`aa_lcr_equality_v1.1` prompt, which includes the official AA-LCR equivalence
rules. Every grade was valid. Grading with the same model keeps the comparison
between modes even, since all answers come from that model, but it limits trust in
the absolute scores.

A hand check covered the questions where runs disagreed: truncated versus hybrid,
full context versus hybrid, and one cut-off answer, about 55 grades in all. It
found five clear errors and one inconsistency:

| Question | Run | Grade | Problem |
|---|---|---|---|
| 83 | Hybrid | Correct | The answer was cut off mid-reasoning and never gives a value |
| 23 | Hybrid | Wrong | "$901,170 (in thousands)" is the reference "$901 million" |
| 75 | Hybrid | Wrong | Same ranking as the reference, with country names spelled out |
| 93 | Full context | Wrong | The answer states "Baby Boomer", the reference |
| 20 | Full context vs hybrid | Correct vs wrong | Both gave "11%, provision for impairment"; one of the two grades is wrong |

Four of these break rules written into the grading prompt. Correcting the clear
errors gives roughly 61–62% for full context, 55% for hybrid and 37% for
truncation: the ordering and the size of the gaps hold. The 70-odd questions
where the runs agreed were not checked, so these corrected figures are not final
scores.

## Output limit

With thinking disabled, the model still reasons inside its answer. Ten of the 300
answers reached the 4,096-token output limit before giving a final value (1 full
context, 4 hybrid, 5 truncated), and only the one grading error above was scored
correct. The cut-offs are spread across runs, so they do not explain the
differences, but they lower all three scores. Raising the output limit within a
64K window would require a correspondingly smaller input budget.

## Run time

| Run | Wall time | Generation | Indexing |
|---|---:|---:|---:|
| Full context | 17 min | 17 min | — |
| Truncated (64K) | 14 min | 13 min | — |
| Hybrid (64K) | 2 h 45 min | 13 min | 151 min |

Hybrid embedded each of the 30 document sets once, on CPU, and reused the index
for that set's questions. Embedding accounts for over 90% of its run time;
retrieval and packing took about a minute in total.

## Limitations

- **Self-grading.** See [Grading](#grading).
- **Small, correlated sample.** 100 questions on 30 document sets, 63 of them on
  company documents. The document-set intervals account for this, but category
  results are anecdotal.
- **One run per condition.** Serving was deterministic in earlier checks on this
  server, but repeats would show run-to-run variation.
- **Coarse chunks.** 7,500-token chunks left a median of about 3,000 tokens of
  the budget unused (58,296 of 61,440). Smaller chunks might fit more evidence.
- **No serving record.** These runs predate the `--serving-metadata` step, so
  their manifests do not record the GPU or vLLM version.

## Earlier results

The 30 August Jarvis runs reported 44% at full context, 41% for hybrid and 21% for
truncation at 64K. They are not directly comparable: they used an earlier
pipeline, dataset release 1.0 answer keys, the older grading prompt, a 512-token
output limit and a 60,000-token budget. The pattern is the same: hybrid recovers
most of what truncation loses.

## Next steps

1. Regrade all three runs with an independent, stronger grader, and compare again
   before reporting absolute scores.
2. Try smaller chunks (for example 3,500 tokens) to use more of the 64K budget.
3. Consider a second model window (for example 32K) to show how retrieval and
   truncation degrade as the window shrinks.
4. Record serving metadata for future runs.

## Artifacts

| Run | Answers | Grades |
|---|---|---|
| Full context | `benchmark_artifacts/aa_lcr/direct/20260930T134555Z_full` | `benchmark_artifacts/aa_lcr/regrades/Qwen3.6-35B-A3B/direct/20260930T134555Z_full` |
| Truncated (64K) | `benchmark_artifacts/aa_lcr/direct/20260930T134555Z` | `benchmark_artifacts/aa_lcr/regrades/Qwen3.6-35B-A3B/direct/20260930T134555Z` |
| Hybrid (64K) | `benchmark_artifacts/aa_lcr/hybrid/20260930T025706Z` | `benchmark_artifacts/aa_lcr/regrades/Qwen3.6-35B-A3B/hybrid/20260930T025706Z` |

Commands are in the [Linux benchmark runbook](../../linux/BENCHMARK_RUNBOOK.md).
