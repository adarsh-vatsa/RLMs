# MRCR v2: Benchmark Overview

**MRCR (multi-round coreference resolution) tests whether a system can find and
reproduce the correct earlier response in a long conversation.** We use it to
measure retrieval, occurrence ordering, and faithful copying as source length
increases.

## Example task

This shortened, illustrative two-needle example contains two responses to the
same request, separated by a distractor:

```text
User: Write a poem about the moon in a cheerful style.
Assistant: The moon shines bright tonight!

User: Write a riddle about the ocean in a formal style.
Assistant: What moves without feet and spans the world?

User: Write a poem about the moon in a cheerful style.
Assistant: Moonbeams dance across the sky!

User: Prepend AbCd1234EfGh to the second poem about the moon in a
cheerful style. Do not include any other text in your response.
```

Expected answer:

```text
AbCd1234EfGhMoonbeams dance across the sky!
```

Finding any moon poem is insufficient: the system must select the **second**
matching response and copy it, with the required 12-character marker.

## How length and difficulty vary

Released datasets have **2, 4, or 8 matching responses**, called needles. Longer
synthetic conversations contain more surrounding exchanges and distractors;
the release extends to approximately **8M tokens**. Filler can repeat, so this
does not imply 8M tokens of unique information. The preparation script defaults
to 8 needles; the current results use 4.

Source-length filtering and executor limits are independent:

- `--min-source-tokens 100000 --max-source-tokens 1200000` selects complete
  examples whose rendered executor prompts fall inside that inclusive interval.
- `--max-input-tokens 258048` limits each executor request after retrieval.
- Published file bands use a Gemini tokenizer; our source bounds use the
  executor tokenizer. Selection searches only the supplied files and never
  truncates examples to meet the bounds.

### Released files from 100K to 1.2M tokens

Each released 8-needle file (the `_fast` variant we download) holds **one**
conversation, sitting just under its band's upper limit, plus many questions
about it. Each question targets one group of eight matching responses. A file
therefore contributes a single source length, and results for a file describe
one conversation. Source bounds only filter these files; they cannot produce
other lengths.

| Band file | Conversation (executor tokens) | Questions | Download | Report band |
|---|---:|---:|---:|---|
| 64K–128K | 132,994–133,001 | 103 | 62.5 MB | [131,072, 262,144) |
| 128K–256K | 266,651–266,659 | 141 | 173.7 MB | [262,144, 524,288) |
| 256K–512K | 534,018–534,025 | 236 | 572.4 MB | [524,288, 1,048,576) |
| 512K–1M | 1,066,311–1,066,319 | 310 | 1,524.2 MB | [1,048,576, 2,097,152) |
| **Total** | | **790** | **2.33 GB** | |

Lengths and counts come from the prepared `mrcr_v2_100k_1200k` dataset (29
September 2026); lengths vary by a few tokens because the final questions
differ. Download sizes were read on 28 September 2026. Each conversation sits
just under its band's upper limit, so the next file, 1M–2M (3.06 GB), should
hold one conversation of about 2.1M tokens.

The 2- and 4-needle files for the same bands share this structure: our
preparation code parses their first rows, which show one conversation just
under the 128K limit and only the ordinals the needle count allows. Their
question counts are not yet measured. Download sizes (MB, read 29 September
2026):

| Needles | 64K–128K | 128K–256K | 256K–512K | 512K–1M | Total |
|---:|---:|---:|---:|---:|---:|
| 2 | 119.9 | 346.5 | 693.9 | 1,390.8 | 2,551.1 |
| 4 | 81.4 | 266.4 | 738.8 | 1,485.9 | 2,572.5 |
| 8 | 62.5 | 173.7 | 572.4 | 1,524.2 | 2,332.8 |

Questions within a file are ordered by needle group, so the first ten cover only
one or two groups. `--max-rows-per-source` instead takes evenly spaced questions
from each conversation; ten of them cover ten groups in the measured files.

## How our system runs it

Hybrid routing matches LongBench. Runs use the executor's full served window:
a 262,144-token context with 4,096 tokens reserved for output, leaving a
258,048-token input budget.

| Full prompt length | Direct mode | Hybrid mode |
|---|---|---|
| 133K tokens | Send the full prompt | Send the full prompt |
| 534K tokens | Keep a head and a tail slice with `--direct-overflow middle`; otherwise record unsupported with no call | Index the source and retrieve evidence |

For oversized sources, hybrid selects chunks by relevance, merges overlapping
ranges, and presents them in original conversation order. Few-shot examples
are preserved. The retrieval query is the request the question describes, such
as `poem about stars in a formal style`: the matching responses all follow that
exact request, while the marker is random and the ordinal never appears in the
conversation. The model still receives the full question. Missing earlier matches can still cause an incorrect occurrence
count. Answer caching is disabled, and gold answers/positions never guide
retrieval. An 8M-token source does not require an 8M-token executor window.

## How results are scored

No evaluator model is needed. We report:

| Metric | Meaning |
|---|---|
| Official MRCR score | Text similarity from 0 to 1 after the last required marker; missing marker scores 0 |
| Exact-match accuracy | Fraction matching the complete reference, ignoring only outer whitespace |
| Prefix compliance | Fraction beginning with the required marker after outer whitespace is stripped |

For the example above, the exact expected answer passes all three metrics.
Adding an explanation before it still yields official similarity **1**, but
fails exact match and prefix compliance. Similarity **0.8** therefore does not
mean 80% of answers were completely correct.

Execution failures count as zero; unsupported direct examples are excluded from
quality scores and reported separately as coverage. Reports include source
length, route, token usage, and timings. Retrieval-assisted results measure our
whole system, rather than the model's native context capacity.

## Further reading

- [Linux runbook: preparation, preflight, and run commands](linux/BENCHMARK_RUNBOOK.md)
- [Linux setup runbook: environment and executor startup](linux/SETUP_RUNBOOK.md)
- [Upstream MRCR v2 benchmark](https://github.com/google-deepmind/eval_hub/tree/master/eval_hub/mrcr_v2)
