# LongBench-v2 Baseline vs. Full Semantic-Cache Run

**Meeting note — August 3, 2026**

Compared runs: direct API baseline `20260802T215043Z` and full semantic-cache run `20260716T204202Z`.

## Executive summary

On the same 503 LongBench-v2 original questions, the full semantic-cache run scored **46.9% (236/503)** versus the ==direct full-context baseline's reported **13.3%== (67/503)**, a **+33.6 percentage-point** end-to-end gain. The full run also served repeated traffic effectively: **100% exact-hit rate** and **98.8% semantic-hit rate** (**99.4% across all 1,006 repeat/paraphrase rows**), while preserving the original answers and therefore the same 46.9% accuracy across original, exact, and semantic rows.

==The main caveat is that the direct baseline was operationally incomplete. It returned only **182/503 successful API responses (36.2%)**; all **321 failures** were HTTP 400 context-length errors. Among the 142 rows where the baseline returned a valid `A`/`B`/`C`/`D` choice, its accuracy was **47.2%**==, close to the full run's **45.1% on those same rows**. The current evidence therefore supports a strong **reliability and cache-reuse** result, but not yet a claim that the cache path improves the underlying model's reasoning accuracy.

## Key results

| Metric                                       |        Direct API baseline |                               Full semantic-cache run | Interpretation                                                      |
| -------------------------------------------- | -------------------------: | ----------------------------------------------------: | ------------------------------------------------------------------- |
| Original questions                           |                        503 |                                                   503 | Same source and suite hashes                                        |
| Successful end-to-end responses              |                182 (36.2%) |                                            503 (100%) | Baseline had 321 context-limit failures                             |
| Valid answer choices                         |                142 (28.2%) |                                            503 (100%) | 40 successful baseline calls did not yield a parseable choice       |
| Official original accuracy                   |             67/503 (13.3%) |                                       236/503 (46.9%) | +33.6 percentage points, driven substantially by baseline failures  |
| Accuracy on baseline's 142 valid-choice rows |             67/142 (47.2%) |                                        64/142 (45.1%) | Reasoning quality is broadly comparable on completed, valid outputs |
| Exact cache-hit rate                         |                        N/A |                                        503/503 (100%) | Exact repeats required no model calls                               |
| Semantic cache-hit rate                      |                        N/A |                                       497/503 (98.8%) | Only 6 paraphrases fell back to a miss                              |
| Accuracy on exact and semantic rows          |                        N/A |                                            46.9% each | Cached responses preserved original-answer quality                  |
| Median query latency                         | 1.68 s on successful calls | 29.29 s original miss; 0.63 ms exact; 164 ms semantic | Large repeat-query speedup, with a slower iterative miss path       |

## Caveat and next step

The baseline artifacts recorded every prompt as two tokens, marked zero rows as truncated, and then received the same context-window HTTP 400 on 321 rows. Both runs also report `$0` estimated cost because the local OpenAI-compatible Qwen models do not have priced usage in the benchmark. The full-run manifest separately reports **17.8% input-token savings** against its 213.8M-token full-context counterfactual, but the failed direct baseline does not independently validate that estimate. Fix the baseline prompt-token accounting/truncation path, add a small context-window safety margin, and rerun all 503 originals before publishing cost or model-quality comparisons.

Sources: [baseline manifest](../benchmark_artifacts/longbench_v2_api/20260802T215043Z/manifest.json), [baseline evaluation](../benchmark_artifacts/longbench_v2_api/20260802T215043Z/official_longbench_v2_api_eval_report.json), [full-run manifest](../benchmark_artifacts/longbench_v2/20260716T204202Z/manifest.json), and [full-run evaluation](../benchmark_artifacts/longbench_v2/20260716T204202Z/official_longbench_v2_eval_report.json).
