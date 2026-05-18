# DP Memo Benchmark Report

Generated during the 2026-05-18 long-horizon substrate session.

## What We Tested

The current memoized RLM substrate was tested on two workload shapes:

1. Official NoLiMa long-context slices with local Q3 as solver/aggregator.
2. A deterministic mutable-workspace trace with scoped invalidation.

The NoLiMa path tests long-context indirect evidence. The mutable-workspace
path tests the real agent pattern of repeated work over a changing corpus.

## Official NoLiMa Stable-Overlap Results

All runs use:

```text
model: Brooooooklyn/Qwen3.6-27B-UD-Q3_K_XL-mlx
solver_mode: evidence
backend: DuckDBMemoStore
corpus mode: stable
document mode: stable
content hash mode: source
chunk_words: 200
chunk_size: 2
```

| Run | Initial Coverage | Model Calls | Aggregate Calls | Correct |
| --- | ---: | ---: | ---: | --- |
| 16K cold | 0.000 | 41 | 1 | yes |
| 32K partial | 0.503 | 41 | 1 | yes |
| 64K partial | 0.502 | 81 | 1 | yes |
| 128K partial | 0.501 | 161 | 1 | yes |
| 128K warm | 1.000 | 0 | 0 | yes |
| 256K partial | 0.579 | 234 | 1 | yes |
| 256K warm | 1.000 | 0 | 0 | yes |

Interpretation: as the requested context grew, the memo graph reused prior
prefix work and solved only the missing suffix. Warm replay reopened DuckDB in
a fresh process and answered with zero model calls.

Key artifacts:

```text
benchmark_artifacts/dp_memo_nolima_overlap_reuse_stable_256k_partial/20260518T191447Z/manifest.json
benchmark_artifacts/dp_memo_nolima_overlap_reuse_stable_256k_warm/20260518T193120Z/manifest.json
```

## Official NoLiMa Two-Sample Smoke

To check that the evidence-mode path was not only working for one sample, we
ran a fresh 8K official slice with two samples:

```text
cold: samples=2, correct=2, model_calls=42, aggregate_calls=2
warm: samples=2, correct=2, model_calls=0, aggregate_calls=0
```

Artifacts:

```text
benchmark_artifacts/dp_memo_nolima_official_8k_two_sample/20260518T194155Z/manifest.json
benchmark_artifacts/dp_memo_nolima_official_8k_two_sample_warm/20260518T194514Z/manifest.json
```

This run also exposed and validated a robustness fix: Q3 can return confidence
as labels such as `high`; the controller now coerces label-style confidence
values instead of crashing on `float("high")`.

## Two-Hop Failure And Fix

A four-sample 8K slice found a real failure:

```text
samples: 4
correct: 3
failed sample: 0401_T17_C02_twohop__rand_book_1__L8000__D00
question: Which character has been to Uusimaa?
wrong answer: NoLiMa
gold: Katie
```

Diagnosis: the evidence extractor over-filtered. It returned `NOT_FOUND` even
for the first chunk containing `Katie lives next to the Kiasma museum`, so the
aggregator never saw the bridge evidence needed for the two-hop location
question.

Fix: evidence-mode subcalls now explicitly avoid returning `NOT_FOUND` when a
slice contains any named person, place, landmark, title, code, or relationship.
The downstream question is marked as non-filtering context.

Targeted rerun:

```text
sample: 0401_T17_C02_twohop__rand_book_1__L8000__D00
correct: 1
answer: Katie
model_calls: 21
aggregate_calls: 1
supporting_fact fragments: 19
ruled_out_region fragments: 2
```

Tradeoff: this improves recall for two-hop questions but creates more
supporting fragments and is slower than aggressive `NOT_FOUND` filtering.

## Mutable-Workspace Result

The mutable workload solves a four-chunk runbook, updates one chunk, invalidates
that scope, reuses unaffected fragments, solves only the changed window, and
then warm-replays from DuckDB.

```text
v1_model_calls: 4
invalidated_entries: 2
v2_model_calls: 1
v2_initial_coverage_ratio: 0.75
v2_reused_windows: 3
v2_missing_windows: 1
warm_model_calls: 0
```

Artifact:

```text
benchmark_artifacts/dp_memo_mutable_workload_report_v2/20260518T193524Z/manifest.json
```

Interpretation: persistent memoization is useful only if the graph can survive
localized changes without trusting stale fragments. Scope invalidation plus
dependency propagation gives us that behavior.

## Benchmark Limitation

A 512K probe over the same single NoLiMa long-haystack slice exact-replayed
immediately because the selected source was exhausted at the same 1108 chunk
scope as the 256K run.

That means larger nominal length flags are not automatically harder workloads.
For stronger experiments, use larger sources, more haystacks, more cases, or a
workload trace with evolving documents.

## Practical Claim

The useful claim is not just that this is a cache. The useful claim is:

```text
Repeated domain-bounded agent work creates reusable scoped subproblems.
A memo graph can store those fragments, compose them over larger scopes,
invalidate stale dependencies, and reduce future model calls while preserving
evidence and lineage.
```
