# Long-Horizon Memo Substrate Worklog

Started: 2026-05-18 13:23 EDT  
Target stop: 2026-05-18 16:23 EDT

## Goal

Use the memoization substrate itself while improving the project, and let real
long-context benchmark work expose implementation choices. The point is not to
make another small demo. The point is to harden the system toward the workload
we care about: repeated, domain-scoped, long-horizon agent work over contexts
larger than one model window.

## Benchmark Direction

The strongest immediate target is NoLiMa because this repository already has a
NoLiMa bridge and the benchmark is designed to defeat literal-match shortcuts.
The official NoLiMa repository describes it as an ICML 2025 benchmark for
"Long-Context Evaluation Beyond Literal Matching" and downloads needle sets and
haystacks from Hugging Face:

- https://github.com/adobe-research/NoLiMa
- https://arxiv.org/abs/2502.05167

The repo also has RULER v2 integration, and InfiniteBench / LongBench v2 remain
important comparison points:

- InfiniteBench: https://github.com/OpenBMB/InfiniteBench
- LongBench v2: https://longbench2.github.io/

Practical choice for this work block: make the current DP memo NoLiMa path more
observable first, then scale it to full NoLiMa assets. This avoids building a
new toy benchmark when the repo already has a higher-value benchmark adapter.

## 2026-05-18 13:30 EDT Update

Implemented planner telemetry in the memo substrate.

Before this change, the planner returned reusable entries and missing scopes,
but benchmark rows mostly saw only final accuracy, model calls, and replay
counts. That made it hard to answer the important question: did memoization
actually cover useful work, or did the model just get the answer anyway?

New telemetry on `MemoReusePlan`:

- `coverage_ratio`
- `covered_length`
- `missing_length`
- `covered_intervals`
- `missing_scopes`
- exact / reusable / negative / hint entry IDs
- counts for each entry class

This makes the benchmark capable of reporting, for example:

```text
initial coverage: 0.00
after first solve: 1.00
replay: exact full-scope hit
```

Also added `fragment_kind` to `MemoEntry`. Planners now see whether a candidate
is an `exact_answer`, `supporting_fact`, `aggregation_component`,
`search_hint`, or `ruled_out_region`. This is a small but important fragment
design improvement: the model should not have to infer the role of a memo entry
from a loose list of labels.

Files changed so far:

- `language_memoization.py`
- `semantic_cache_system.py`
- `scripts/run_dp_memo_nolima.py`
- `scripts/run_dp_memo_shared_context.py`
- `test/test_language_memoization.py`
- `test/test_data_scope_cache.py`

Verification:

```text
python -m pytest test/test_language_memoization.py test/test_data_scope_cache.py test/test_duckdb_memoization.py test/test_dp_memo_shared_context.py
28 passed
```

## 2026-05-18 13:34 EDT Update

Fetched the official NoLiMa data into `benchmark_data/nolima/`, including the
`rand_shuffle_long` haystacks. Local size is about 8 MB. The long haystacks are
large enough to generate samples in the 64K--256K token range with the existing
NoLiMa bridge.

Added `scripts/inspect_long_context_workload.py`. This is a model-free
inspection tool for estimating the size of a NoLiMa workload before spending
local model calls. Example run:

```text
python scripts/inspect_long_context_workload.py \
  --needle-set-path benchmark_data/nolima/needlesets/needle_set.json \
  --haystack-dir benchmark_data/nolima/haystack/rand_shuffle_long \
  --lengths 32K,64K,128K,256K \
  --depth-intervals 4 \
  --max-cases 4 \
  --max-haystacks 1 \
  --chunk-words 1000 \
  --chunk-size 2 \
  --agent-window-tokens 32768 \
  --output benchmark_artifacts/long_context_workload_plan.json
```

The selected slice produces 64 samples and an estimated 3,904 cold model calls
under that chunking plan. The 64K, 128K, and 256K rows exceed the comparison
window, with 256K at roughly 7.8x the comparison window. This is the kind of
workload that forces recursive decomposition instead of direct prompting.

Also fixed a real composition bug in `solve_with_memo()`: previously it skipped
a chunk window only if that exact window had an exact replay entry. If a larger
memo fragment already covered the window, the system still called the model.
That was wrong for partial composition. The loop now skips any window whose
reuse plan is complete, whether the coverage comes from an exact window, a
larger solved region, supporting facts, or a ruled-out region.

Added a regression test:

```text
test_solve_with_memo_skips_windows_covered_by_larger_fragment
```

Verification:

```text
python -m pytest test/test_data_scope_cache.py test/test_language_memoization.py
21 passed
```

Session substrate check:

```text
benchmark_artifacts/long_horizon_substrate_session/session_memo.duckdb
entries: 3
```

This store records the benchmark decision, the first implementation
improvement, and a parent work-session summary whose dependencies point at
those two child memo entries. That is intentionally small, but it uses the same
memo graph abstraction for the work session itself.

## 2026-05-18 13:35 EDT Update

Added `MemoStore.stats()`, inherited by `DuckDBMemoStore`, so long runs can
report what the memo substrate is actually accumulating. Per-solve telemetry is
useful, but for a long-horizon agent we also need store-level state:

- entry count
- counts by `fragment_kind`
- counts by `result_type`
- counts by verifier status
- counts by reuse mode
- dependency edge count
- evidence span count
- average confidence

The NoLiMa and shared-context benchmark manifests now include `memo_stats`.
This should make it easier to compare runs where accuracy is similar but one
run is filling the memo graph with better reusable fragments.

Verification:

```text
python -m pytest test/test_language_memoization.py test/test_duckdb_memoization.py
16 passed
```

## 2026-05-18 13:36 EDT Update

Started a real official-data NoLiMa DP memo smoke:

```text
python scripts/run_dp_memo_nolima.py \
  --reset-db \
  --needle-set-path benchmark_data/nolima/needlesets/needle_set.json \
  --haystack-dir benchmark_data/nolima/haystack/rand_shuffle_long \
  --lengths 64K \
  --depth-intervals 1 \
  --max-samples 1 \
  --chunk-words 600 \
  --chunk-size 2 \
  --duckdb-path benchmark_artifacts/dp_memo_nolima_official_smoke/memo.duckdb \
  --output-dir benchmark_artifacts/dp_memo_nolima_official_smoke
```

While that was running, added progress and context-size fields to
`scripts/run_dp_memo_nolima.py`:

- `--progress-every`
- `context_chars`
- `context_words`
- `chunk_words`

This is a practical long-run improvement: a real 64K or 128K local-model run
should show scoped solver progress and should record enough context-size
metadata to explain runtime and call counts afterward.

Verification:

```text
python -m pytest test/test_long_context_workload_inspector.py test/test_dp_memo_shared_context.py
3 passed
```

## 2026-05-18 13:37 EDT Update

Added reused-window accounting to `solve_with_memo()` and benchmark rows:

- `window_count`
- `reused_window_count`
- `missing_window_count`
- `reused_window_ratio` in NoLiMa manifest totals
- `fact_reused_window_ratio` in shared-context manifest totals

This makes the benchmark closer to the actual claim. If the memo graph is doing
dynamic-programming-style work avoidance, we should be able to point to windows
that were skipped because they were already covered by fragments, not merely
show final answer accuracy.

Verification:

```text
python -m pytest test/test_data_scope_cache.py test/test_language_memoization.py
22 passed
```

## 2026-05-18 13:39 EDT Update

Updated `scripts/live_dp_memo_smoke.py` to print memo coverage and window
reuse telemetry during the live Q3 MVP check. The smoke already proves exact
and verifier-approved replay, but the output now also shows the same fields the
benchmark rows rely on.

Updated the LaTeX draft with the concrete implementation details from this
session: telemetry, fragment kinds, store stats, the NoLiMa workload inspector,
and the composition fix. Rebuilt
`docs/context_management_and_language_memoization.pdf` with `pdflatex`.

Created `docs/workshop_venues_2026.md` with current workshop-positioning notes.
Best fit is SCALE @ ICML 2026, with CATS @ ICML 2026 and FAGEN @ ICML 2026 as
secondary targets depending on whether we emphasize efficient agent memory,
continual adaptation through reusable fragments, or failure modes from repeated
subproblem work.

Full deterministic verification after the current changes:

```text
python -m pytest
36 passed
```

## 2026-05-18 13:44 EDT Update

While the official 64K NoLiMa smoke was running, I tried to inspect the DuckDB
memo table from a second process. DuckDB correctly held a write lock for the
running benchmark, so external read-only inspection failed. That is a practical
long-run issue: we need progress visibility without touching the locked memo
database.

Added a sidecar progress file to `scripts/run_dp_memo_nolima.py`:

```text
progress.jsonl
```

It records:

- `sample_start`
- `solver_call`
- `sample_done`

The manifest now records the progress path. This gives us a tail-able progress
stream for long Q3 runs while DuckDB remains the durable memo backend.

Verification:

```text
python -m pytest test/test_long_context_workload_inspector.py test/test_data_scope_cache.py
15 passed
```

## 2026-05-18 13:46 EDT Update

Interrupted the first official 64K NoLiMa smoke after about 11 minutes because
it was still inside the first scoped Q3 generation and had no row-level output.
This was useful: it showed that the benchmark target is directionally right but
the local Q3 path needs smaller scoped windows or a faster local serving path
before 64K/128K sweeps are practical.

Started a smaller official-data smoke instead:

```text
python scripts/run_dp_memo_nolima.py \
  --reset-db \
  --needle-set-path benchmark_data/nolima/needlesets/needle_set.json \
  --haystack-dir benchmark_data/nolima/haystack/rand_shuffle_long \
  --lengths 8K \
  --depth-intervals 1 \
  --max-samples 1 \
  --chunk-words 200 \
  --chunk-size 2 \
  --max-tokens 64 \
  --aggregate-max-tokens 256 \
  --progress-every 5 \
  --duckdb-path benchmark_artifacts/dp_memo_nolima_official_smoke_8k/memo.duckdb \
  --output-dir benchmark_artifacts/dp_memo_nolima_official_smoke_8k
```

The output caps here are explicit benchmark-output caps, not hidden context
budgets: NoLiMa expects short answers or `NOT_FOUND`.

Result:

```text
samples: 1
correct: 0
model_calls: 21
aggregate_calls: 1
exact_replay_checks: 1
initial_coverage_ratio: 0.0
final_coverage_ratio: 1.0
replay_model_calls: 0
replay_aggregate_calls: 0
```

The memo substrate worked mechanically: it covered all 41 chunks, stored 21
negative scoped entries plus one exact full-scope entry, and replayed the
full-scope answer with zero model calls. The answer was wrong (`NoLiMa` instead
of `Katie`), so the next benchmark-quality work is prompt/aggregation quality,
not memo persistence. This is exactly the kind of implementation decision that
only surfaces when using the substrate on a real benchmark.

Also fixed the NoLiMa manifest label. It now reports:

```text
benchmark: dp_memo_nolima
data_kind: fixture | official_or_downloaded | custom
```

instead of calling every run `dp_memo_nolima_fixture`.

Verification:

```text
python -m pytest test/test_long_context_workload_inspector.py
3 passed
```

## 2026-05-18 13:55 EDT Update

The NoLiMa failure exposed the most important fragment-design lesson so far.
The direct-answer scoped solver was too brittle: it asked each chunk to decide
whether it alone answered the final question. On NoLiMa, the relevant chunk
said:

```text
Katie lives next to the Kiasma museum.
```

The question was:

```text
Which character has been to Helsinki?
```

The model needed to preserve the local evidence and let aggregation perform the
final inference. So I added:

```text
--solver-mode evidence
```

In evidence mode, scoped calls extract broad named-entity and relationship
facts instead of over-filtering by the final question. This stores better
fragments:

```text
chunk 0..2 -> aggregation_component:
Katie lives next to the Kiasma museum.
Felix grabbed Van by the arm.
...
```

Then the aggregator answers from those fragments.

Rerun on the same official 8K NoLiMa sample:

```text
correct: 1 / 1
answer: Katie
model_calls: 21
aggregate_calls: 1
exact_replay_checks: 1
initial_coverage_ratio: 0.0
final_coverage_ratio: 1.0
memo_entries: 22
aggregation_component entries: 1
ruled_out_region entries: 20
exact_answer entries: 1
```

Artifact:

```text
benchmark_artifacts/dp_memo_nolima_official_smoke_8k_broad_evidence/20260518T175247Z/manifest.json
```

This is a concrete example of why the project is not "just a cache." The useful
cache line was not the final answer; it was the local evidence fragment. The
memo graph became useful only after the fragment design matched the reasoning
structure of the task.

Updated README and the LaTeX draft to document `--solver-mode evidence`.
Rebuilt the PDF.

Full deterministic verification:

```text
python -m pytest
37 passed
```

## 2026-05-18 13:59 EDT Update

The controller now lets a solver label positive fragments with explicit
`reusable_as` modes. This matters because an evidence fragment should not be
stored only as a generic aggregation component. `solve_with_memo()` and
`memoized_subproblem()` now honor a solver-returned field like:

```python
{
    "result": "Katie lives next to the Kiasma museum.",
    "reusable_as": ("supporting_fact", "aggregation_component"),
}
```

Reran the official 8K NoLiMa evidence smoke with typed fragments:

```text
correct: 1 / 1
answer: Katie
memo_entries: 22
by_fragment_kind:
  supporting_fact: 1
  ruled_out_region: 20
  exact_answer: 1
by_reuse_mode:
  supporting_fact: 1
  aggregation_component: 1
  ruled_out: 20
  exact_answer: 1
```

Artifact:

```text
benchmark_artifacts/dp_memo_nolima_official_smoke_8k_typed_evidence/20260518T175708Z/manifest.json
```

Verification:

```text
python -m pytest
38 passed
```

## 2026-05-18 14:00 EDT Update

Ran a warm-start replay against the same official 8K typed-evidence DuckDB
store, without `--reset-db`.

Result:

```text
correct: 1 / 1
answer: Katie
model_calls: 0
aggregate_calls: 0
initial_coverage_ratio: 1.0
final_coverage_ratio: 1.0
exact_replay_checks: 1
```

Artifact:

```text
benchmark_artifacts/dp_memo_nolima_official_smoke_8k_typed_evidence_warm/20260518T175940Z/manifest.json
```

This is the smallest real benchmark demonstration of the intended lifecycle:

1. cold run builds scoped evidence fragments and a full exact memo;
2. warm run replays the answer from DuckDB with no scoped solver calls and no
   aggregation call.

Added `scripts/summarize_dp_memo_runs.py` to compare benchmark manifests. This
turns cold/warm comparisons into a reproducible command instead of manual JSON
inspection.

Example output for the official 8K typed-evidence cold/warm pair:

```text
cold: accuracy=1.0, model_calls=21, aggregate_calls=1, initial_coverage=0.0
warm: accuracy=1.0, model_calls=0,  aggregate_calls=0, initial_coverage=1.0
```

Verification:

```text
python -m pytest test/test_summarize_dp_memo_runs.py
2 passed
```

## 2026-05-18 20:30 EDT Update

Added the post-depth-sweep research state. The current thesis is no longer just
that memoization can replay answers. The stronger question is how to make
scoped language fragments reusable, trustworthy, and composable under noisy
evidence.

Open research questions after today's NoLiMa work:

1. Fragment design. What should the reusable unit be? Final answers are too
   coarse, while broad fact extraction can flood aggregation with distractors.
   The useful NoLiMa fragment was a compact bridge fact such as `Megan lives
   next to the Kiasma museum.`
2. Aggregation under noise. The 16K D03 failure showed that the correct
   fragment can exist in the memo graph while the aggregator still chooses a
   frequent distractor. Aggregation needs stronger ranking, evidence typing,
   and possibly verifier passes.
3. Domain-specific scope design. The generic memo graph can stay shared, but
   code, textbooks, logs, legal documents, financial tables, and spreadsheets
   need different chunking, stable IDs, and invalidation policies.
4. Stable addressing. The system must identify whether new context is the same
   chunk, appended context, edited context, moved context, or overlapping
   context with shifted boundaries. Content hashes are necessary but not
   sufficient; structural anchors and neighbor hashes are likely needed.
5. Beyond exact replay. The warm replay results are strong, but the larger
   research claim depends on broader partial composition and cross-question
   reuse: new tasks should be solved as old solved fragments plus a small
   amount of new work.
6. Semantic reuse verification. The system needs to decide when an old memo is
   an exact answer, supporting evidence, a search hint, irrelevant, or a
   dangerous distractor for a new but related question.
7. Retrieval over the memo graph. Current DuckDB lookup is deterministic and
   safe, mostly keyed by task and scope. The next layer needs richer candidate
   generation using SQL filters, lexical search, embeddings, graph traversal,
   and model-ranked bounded packets without flooding the aggregator.
8. Benchmark design. Single-shot QA undermeasures this substrate. Better
   benchmarks should report cold cost, warm replay cost, partial reuse,
   cross-question reuse, stale-memory handling, accuracy under noisy fragments,
   model calls saved, and wall-clock time saved.
9. Cache safety. Memoization amplifies both correct work and mistakes. We need
   confidence calibration, source grounding, verifier checks, staleness
   detection, human correction hooks, parent invalidation, and audit trails.
10. Scheduling policy. The agent needs policies for when to pre-warm, when to
    solve on demand, when to extract broad versus narrow facts, when to
    aggregate, when to verify, and when to discard noisy fragments.

The concise research framing:

```text
Dynamic programming over language tasks works only if fragment design,
scope identity, and composition under noise are solved together.
```

Today's useful failure was the 16K D03 two-hop sample. The memo graph contained
the right evidence, but aggregation selected `Ray` instead of `Megan` because
irrelevant Ray facts were frequent. A general aggregation prompt fix that
prefers explicit character-to-place bridge facts made the targeted rerun answer
`Megan` correctly. This supports the idea that the next layer is not more
storage alone; it is better fragment typing and composition policy.

## 2026-05-18 14:47 EDT Update

Extended the stable overlap experiment to 64K official NoLiMa context using the
same persistent DuckDB memo graph that already contained the 16K and 32K runs.

Partial 64K run:

```text
correct: 1 / 1
answer: Katie
model_calls: 81
aggregate_calls: 1
initial_covered_length: 161
initial_missing_length: 160
initial_coverage_ratio: 0.501558
reused_windows: 80
missing_windows: 81
reused_window_ratio: 0.496894
final_coverage_ratio: 1.0
avg_latency_ms: 455347.803
```

Warm 64K replay over the same DuckDB store:

```text
correct: 1 / 1
model_calls: 0
aggregate_calls: 0
initial_coverage_ratio: 1.0
avg_latency_ms: 140.782
```

Artifacts:

```text
benchmark_artifacts/dp_memo_nolima_overlap_reuse_stable_64k_partial/20260518T183707Z/manifest.json
benchmark_artifacts/dp_memo_nolima_overlap_reuse_stable_64k_warm/20260518T184453Z/manifest.json
```

The 64K run is now the cleanest live evidence for partial composition: the
system reused the solved 32K prefix, processed only the new 64K suffix, stored
the resulting fragments, then replayed the full 64K answer without loading Q3.

The session memo store was updated with this result:

```text
benchmark_artifacts/long_horizon_substrate_session/session_memo.duckdb
entries: 7
fragment mix: supporting_fact=6, exact_answer=1
dependency edges: 6
```

## 2026-05-18 14:55 EDT Update

Documented the benchmark-limitation lesson in the LaTeX draft. The NoLiMa path
is useful as a stress test for long-context indirect evidence, but it is not a
complete proxy for real-world agent performance. The better evaluation target
for this substrate is a workload trace: repeated related tasks over a shared or
slowly changing corpus, with cold-start cost, warm replay, partial-overlap
reuse, stale-memory safety, and fragment-kind reuse all reported.

Small runner cleanup:

- `scripts/run_dp_memo_nolima.py` now exposes `--max-cases`.
- `scripts/run_dp_memo_nolima.py` now exposes `--max-haystacks`.
- benchmark manifests record both settings.
- README and the LaTeX draft explain that these define the source slice, while
  `--max-samples` is only an early stop inside the generated slice.

Verification:

```text
python -m py_compile scripts/run_dp_memo_nolima.py
python -m pytest test/test_long_context_workload_inspector.py test/test_summarize_dp_memo_runs.py
7 passed
```

## 2026-05-18 15:03 EDT Update

Extended the stable-overlap experiment to 128K official NoLiMa context.

Partial 128K run:

```text
correct: 1 / 1
answer: Katie
model_calls: 161
aggregate_calls: 1
initial_covered_length: 321
initial_missing_length: 320
initial_coverage_ratio: 0.500780
reused_windows: 160
missing_windows: 161
reused_window_ratio: 0.498442
final_coverage_ratio: 1.0
avg_latency_ms: 746148.796
memo_entries: 328
```

Warm 128K replay in a fresh process:

```text
correct: 1 / 1
answer: Katie
model_calls: 0
aggregate_calls: 0
initial_coverage_ratio: 1.0
avg_latency_ms: 150.627
```

Artifacts:

```text
benchmark_artifacts/dp_memo_nolima_overlap_reuse_stable_128k_partial/20260518T184928Z/manifest.json
benchmark_artifacts/dp_memo_nolima_overlap_reuse_stable_128k_warm/20260518T190203Z/manifest.json
```

This gives the clean growth pattern:

```text
16K cold:    41 model calls
32K partial: 41 model calls, about half already covered
64K partial: 81 model calls, about half already covered
128K partial: 161 model calls, about half already covered
128K warm:   0 model calls
```

The 128K result is important because the requested context is now far beyond a
single practical local-agent prompt, while the memo graph still prevents
reprocessing the solved prefix.

Session memo store updated:

```text
benchmark_artifacts/long_horizon_substrate_session/session_memo.duckdb
entries: 8
fragment mix: supporting_fact=7, exact_answer=1
evidence spans: 8
```

Also fixed a DuckDB/in-memory parity issue: task-scoped DuckDB candidate lookup
now filters rejected entries before they can appear as planner hints. Rejected
entries already could not cover scope, but rejected hints should not be visible
to the planner at all.

Verification:

```text
python -m pytest test/test_duckdb_memoization.py test/test_language_memoization.py
17 passed

python -m pytest
44 passed
```

## 2026-05-18 15:06 EDT Update

Added explicit stale-memory handling to the memo substrate.

New API:

```python
store.invalidate_scope(scope, reason="document updated")
```

Behavior:

- finds non-rejected memo entries overlapping the supplied scope;
- marks them with `verifier_status = rejected`;
- records `rejection_reason` and `rejected_at` metadata;
- keeps the entries in the store for audit and lineage;
- removes them from exact replay, partial composition, and planner-visible
  candidate lookup.

This is important for real-world agent workloads because a persistent memo
graph must support document updates and user corrections. The system should not
delete history, but it also cannot keep trusting stale fragments.

README and the LaTeX draft now document scope invalidation.

Verification:

```text
python -m pytest test/test_language_memoization.py test/test_duckdb_memoization.py
19 passed

python -m pytest
46 passed
```

## 2026-05-18 15:07 EDT Update

Added a controller-level regression test for mutable-workspace reuse.

Scenario:

1. solve a four-chunk document with stable/empty content hash;
2. update only chunk 1;
3. call `invalidate_scope()` for chunk 1;
4. solve the updated document.

Expected behavior now covered by tests:

- old full-scope answer is rejected;
- old chunk-1 fragment is rejected;
- chunks 0, 2, and 3 remain reusable;
- the next solve calls the solver only for chunk 1;
- the new full answer contains the updated chunk-1 result.

This is the practical versioning distinction:

- strict content hashes force clean recomputation after any document change;
- stable/empty content hashes plus explicit scope invalidation allow localized
  updates while preserving unaffected memo work.

Verification:

```text
python -m pytest test/test_data_scope_cache.py::DataScopedSearchCacheTests::test_solve_with_memo_reuses_unaffected_windows_after_scope_invalidation
1 passed

python -m pytest
47 passed
```

## 2026-05-18 15:08 EDT Update

Added richer reuse-plan telemetry for fragment design analysis.

`MemoReusePlan.to_telemetry()` now includes:

- `fragment_kind_counts`;
- `covering_fragment_kind_counts`;
- `evidence_span_count`;
- `dependency_edge_count`.

This matters because coverage alone is too coarse. For the talk and later
ablations, we need to know whether the system avoided work because of exact
answer replay, supporting facts, aggregation components, or negative
memoization.

README and the LaTeX draft now describe these telemetry fields.

Verification:

```text
python -m pytest test/test_language_memoization.py
10 passed

python -m pytest
47 passed
```

Fresh warm replay after telemetry changes:

```text
benchmark_artifacts/dp_memo_nolima_overlap_reuse_stable_128k_warm_telemetry/20260518T190915Z/manifest.json
correct: 1 / 1
model_calls: 0
aggregate_calls: 0
initial_coverage_ratio: 1.0
avg_latency_ms: 75.720
```

## 2026-05-18 15:10 EDT Update

Made candidate packet truncation explicit and disableable.

Before this, memo result/evidence text and raw context packet text were bounded
for planner prompts, but callers could not tell from the packet whether useful
text had been cut. That is exactly the kind of hidden budget that can become a
failure mode.

Now candidate packets expose:

- `result_chars`;
- `result_truncated`;
- per-evidence `text_chars`;
- per-evidence `text_truncated`;
- `evidence_count`;
- raw context `text_chars`;
- raw context `text_truncated`.

The max-character arguments also accept `None`, which returns full text for
inspection/debugging instead of silently truncating.

Verification:

```text
python -m pytest test/test_data_scope_cache.py::DataScopedSearchCacheTests::test_candidate_packets_make_truncation_explicit_and_disableable test/test_data_scope_cache.py::DataScopedSearchCacheTests::test_solve_with_memo_persists_context_chunks_when_duckdb_backed
2 passed

python -m pytest
48 passed
```

## 2026-05-18 15:13 EDT Update

Added a deterministic mutable-workspace benchmark:

```text
scripts/run_dp_memo_mutable_workload.py
```

This is the workload trace we said the project needed:

1. solve a four-chunk runbook;
2. update one chunk;
3. invalidate the changed scope;
4. reuse the three unaffected fragments;
5. solve only the changed chunk;
6. reopen DuckDB and warm-replay the updated answer.

Live run:

```text
benchmark_artifacts/dp_memo_mutable_workload/20260518T191311Z/manifest.json
v1_model_calls: 4
invalidated_entries: 2
v2_model_calls: 1
v2_initial_coverage_ratio: 0.75
v2_reused_windows: 3
v2_missing_windows: 1
warm_model_calls: 0
```

Updated `scripts/summarize_dp_memo_runs.py` so mutable-workload manifests show
their v1/v2/warm replay fields.

Verification:

```text
python -m pytest test/test_dp_memo_mutable_workload.py test/test_summarize_dp_memo_runs.py
4 passed

python -m pytest
50 passed
```

## 2026-05-18 15:17 EDT Update

Strengthened invalidation semantics with dependency propagation.

Before this, `invalidate_scope()` rejected entries whose own scopes overlapped
the changed region. That was not enough for a graph: a derived memo entry can
depend on a changed child while living at another scope. Invalidation now walks
dependency parents by default, so changed source fragments reject downstream
answers too.

Added in-memory and DuckDB tests for:

- `parents(entry_id)`;
- `children(entry_id)`;
- scope invalidation rejecting direct overlaps;
- scope invalidation rejecting dependency parents.

Verification:

```text
python -m pytest test/test_language_memoization.py test/test_duckdb_memoization.py test/test_dp_memo_mutable_workload.py
22 passed

python -m pytest
52 passed
```

Session memo store updated:

```text
benchmark_artifacts/long_horizon_substrate_session/session_memo.duckdb
entries: 9
fragment mix: supporting_fact=8, exact_answer=1
```

## 2026-05-18 15:31 EDT Update

Extended the official NoLiMa stable-overlap run to 256K.

Partial 256K run:

```text
correct: 1 / 1
answer: Katie
model_calls: 234
aggregate_calls: 1
initial_covered_length: 641
initial_missing_length: 467
initial_coverage_ratio: 0.578520
reused_windows: 320
missing_windows: 234
reused_window_ratio: 0.577617
final_coverage_ratio: 1.0
avg_latency_ms: 984723.526
memo_entries: 563
```

Warm 256K replay in a fresh process:

```text
correct: 1 / 1
answer: Katie
model_calls: 0
aggregate_calls: 0
initial_coverage_ratio: 1.0
avg_latency_ms: 135.433
```

Artifacts:

```text
benchmark_artifacts/dp_memo_nolima_overlap_reuse_stable_256k_partial/20260518T191447Z/manifest.json
benchmark_artifacts/dp_memo_nolima_overlap_reuse_stable_256k_warm/20260518T193120Z/manifest.json
```

Session memo store updated:

```text
benchmark_artifacts/long_horizon_substrate_session/session_memo.duckdb
entries: 10
fragment mix: supporting_fact=9, exact_answer=1
evidence spans: 11
```

Verification:

```text
python -m pytest
52 passed
```

## 2026-05-18 15:33 EDT Update

Ran the live repo-level DP memo smoke test through DuckDB after the long-session
changes.

Command:

```text
python scripts/live_dp_memo_smoke.py --reset-db --backend duckdb --duckdb-path benchmark_artifacts/live_dp_memo_after_long_session.duckdb
```

Result:

```text
first solve: model_calls=2, aggregate_calls=1
exact replay: model_calls=0, aggregate_calls=0
semantic verifier replay: model_calls=0, aggregate_calls=0
MVP CHECK: PASS
DUCKDB PERSISTENCE CHECK: PASS
```

## 2026-05-18 15:34 EDT Update

Checked a 512K probe over the same single NoLiMa long-haystack slice.

Result:

```text
benchmark_artifacts/dp_memo_nolima_overlap_reuse_stable_512k_probe/20260518T193410Z/manifest.json
correct: 1 / 1
model_calls: 0
aggregate_calls: 0
initial_covered_length: 1108
initial_missing_length: 0
avg_latency_ms: 81.017
```

Interpretation: this is not a new harder workload. The selected long haystack
has about 221K words, so the 512K nominal length collapses to the same 1108
chunk scope as the prior 256K run and exact-replays immediately. For genuinely
larger experiments, we need larger sources or multiple haystacks/cases rather
than only increasing the nominal length flag.

Ran the workload inspector for a broader source slice:

```text
python scripts/inspect_long_context_workload.py \
  --lengths 256K,512K \
  --depth-intervals 1 \
  --max-cases 2 \
  --max-haystacks 5 \
  --chunk-words 200 \
  --chunk-size 2 \
  --output benchmark_artifacts/long_context_workload_plan_5_haystacks.json
```

Result:

```text
samples: 20
estimated cold model calls: 19,220
haystack source words total: 1,104,959
```

## 2026-05-18 15:35 EDT Update

Tightened mutable-workload reporting:

- mutable workload manifests now include `corpus_id`;
- mutable workload manifests now include top-level `memo_entries`;
- the summary utility falls back to `memo_stats.entry_count` when
  `memo_entries` is absent.

Fresh mutable-workload artifact:

```text
benchmark_artifacts/dp_memo_mutable_workload_report_v2/20260518T193524Z/manifest.json
v1_model_calls: 4
invalidated_entries: 2
v2_model_calls: 1
v2_initial_coverage_ratio: 0.75
warm_model_calls: 0
```

Verification:

```text
python -m pytest test/test_summarize_dp_memo_runs.py test/test_dp_memo_mutable_workload.py
4 passed

python -m pytest
52 passed
```

## 2026-05-18 15:36 EDT Update

Added compact summaries for human-facing benchmark output.

`scripts/summarize_dp_memo_runs.py` now accepts:

```text
--compact
```

This omits null-valued fields, which makes mixed benchmark-family output easier
to use in talks and docs without changing the full machine-readable summary.

Verification:

```text
python -m pytest test/test_summarize_dp_memo_runs.py
4 passed
```

## 2026-05-18 15:37 EDT Update

Added a compact human-facing benchmark report:

```text
docs/dp_memo_benchmark_report.md
```

It summarizes:

- official NoLiMa stable-overlap results through 256K;
- 128K/256K warm replay;
- mutable-workspace invalidation result;
- the 512K single-haystack limitation;
- the practical claim for the project.

Verification:

```text
python -m pytest
53 passed
```

## 2026-05-18 15:45 EDT Update

Ran a fresh official NoLiMa 8K two-sample Q3 slice to check whether evidence
mode was only fitting the original one-sample path.

The first attempt exposed a robustness bug: Q3 returned aggregate confidence as
the string `high`, and the controller tried to cast it directly with
`float(...)`.

Fix:

- added `coerce_confidence(...)`;
- accepts numeric strings and label-style values such as `high`, `medium`,
  `low`;
- used it in memo entries, solver outputs, aggregate outputs, planner outputs,
  and verifier outputs.

Rerun result:

```text
benchmark_artifacts/dp_memo_nolima_official_8k_two_sample/20260518T194155Z/manifest.json
samples: 2
correct: 2
model_calls: 42
aggregate_calls: 2
accuracy_contains: 1.0
```

Warm replay:

```text
benchmark_artifacts/dp_memo_nolima_official_8k_two_sample_warm/20260518T194514Z/manifest.json
samples: 2
correct: 2
model_calls: 0
aggregate_calls: 0
avg_latency_ms: 94.138
```

Verification:

```text
python -m pytest
54 passed
```

README note added for `--sample-offset`, which is now the practical way to
debug a later generated benchmark sample without rerunning earlier samples.

Small confidence parsing cleanup: numeric confidence values between 1 and 100
now parse as percentages, so `75` becomes `0.75` instead of being clamped to
`1.0`.

Verification:

```text
python -m pytest test/test_language_memoization.py
12 passed
```

Session memo store updated:

```text
benchmark_artifacts/long_horizon_substrate_session/session_memo.duckdb
entries: 11
fragment mix: supporting_fact=10, exact_answer=1
evidence spans: 13
```

Small robustness addition: `coerce_confidence(...)` now also accepts percentage
strings such as `92%`.

Verification:

```text
python -m pytest test/test_language_memoization.py
12 passed
```

## 2026-05-18 16:06 EDT Update

Ran a four-sample official 8K slice. This found a real two-hop failure.

Four-sample result before the evidence prompt fix:

```text
benchmark_artifacts/dp_memo_nolima_official_8k_four_sample/20260518T194852Z/manifest.json
samples: 4
correct: 3
accuracy_contains: 0.75
failed sample: 0401_T17_C02_twohop__rand_book_1__L8000__D00
question: Which character has been to Uusimaa?
wrong answer: NoLiMa
gold: Katie
```

Diagnosis:

- the relevant needle was `Actually, Katie lives next to the Kiasma museum.`;
- the evidence extractor returned `NOT_FOUND` for every scoped window;
- therefore the aggregator never saw the bridge evidence needed for the
  Uusimaa/Kiasma two-hop inference.

Change:

- evidence-mode extraction now says not to return `NOT_FOUND` when the slice
  contains any named person, place, landmark, title, code, or relationship;
- the downstream question is explicitly marked as non-filtering context for
  evidence extraction.

Targeted failed-sample rerun:

```text
benchmark_artifacts/dp_memo_nolima_official_8k_twohop_probe_v2/20260518T200038Z/manifest.json
correct: 1 / 1
answer: Katie
model_calls: 21
aggregate_calls: 1
supporting_fact fragments: 19
ruled_out_region fragments: 2
avg_latency_ms: 313566.005
```

Warm replay:

```text
benchmark_artifacts/dp_memo_nolima_official_8k_twohop_probe_v2_warm/20260518T200659Z/manifest.json
correct: 1 / 1
model_calls: 0
aggregate_calls: 0
avg_latency_ms: 149.957
```

Interpretation: this is the fragment-design tradeoff in concrete form. Aggressive
`NOT_FOUND` filtering is cheap but can destroy two-hop recall. Recall-oriented
evidence extraction is slower and stores more fragments, but it gives the
aggregator enough material to answer.

Session memo store updated:

```text
benchmark_artifacts/long_horizon_substrate_session/session_memo.duckdb
entries: 12
fragment mix: supporting_fact=11, exact_answer=1
evidence spans: 16
```

Verification:

```text
python -m pytest
54 passed
```

Full deterministic verification:

```text
python -m pytest
43 passed
```

## 2026-05-18 14:45 EDT Update

Extended the stable overlap experiment to 64K.

64K partial composition over the same stable DuckDB store:

```text
correct: 1 / 1
model_calls: 81
aggregate_calls: 1
initial_covered_length: 161
initial_missing_length: 160
initial_coverage_ratio: 0.501558
reused_windows: 80
missing_windows: 81
reused_window_ratio: 0.496894
final_coverage_ratio: 1.0
avg_latency_ms: 455347.803
```

64K exact warm replay:

```text
correct: 1 / 1
model_calls: 0
aggregate_calls: 0
initial_coverage_ratio: 1.0
avg_latency_ms: 140.782
```

Artifacts:

```text
benchmark_artifacts/dp_memo_nolima_overlap_reuse_stable_64k_partial/20260518T183707Z/manifest.json
benchmark_artifacts/dp_memo_nolima_overlap_reuse_stable_64k_warm/20260518T184453Z/manifest.json
```

This is now a proper larger-than-window live result: 64K official NoLiMa data,
typed evidence fragments, partial prefix reuse from prior 32K work, and exact
warm replay afterward.

## 2026-05-18 14:10 EDT Update

Optimized the exact-replay path in `solve_with_memo()`. Previously the
controller upserted raw context chunks into DuckDB before checking whether a
full-scope exact memo already existed. That made warm runs do unnecessary
database writes.

The controller now checks exact and semantic replay first, and only persists raw
context chunks before actual scoped solving.

16K warm replay after this change:

```text
correct: 1 / 1
model_calls: 0
aggregate_calls: 0
initial_coverage_ratio: 1.0
avg_latency_ms: 83.463
```

Earlier lazy warm run was about 270 ms, so this removed another source of warm
path overhead.

Verification:

```text
python -m pytest test/test_data_scope_cache.py test/test_duckdb_memoization.py
21 passed
```

Added regression test:

```text
test_solve_with_memo_exact_replay_skips_context_chunk_upsert
```

Verification:

```text
python -m pytest test/test_data_scope_cache.py
15 passed
```

Full deterministic verification:

```text
python -m pytest
41 passed
```

## 2026-05-18 14:18 EDT Update

Ran the typed-evidence path at 32K official NoLiMa.

Cold run:

```text
correct: 1 / 1
answer: Katie
model_calls: 81
aggregate_calls: 1
initial_coverage_ratio: 0.0
final_coverage_ratio: 1.0
avg_latency_ms: 383584.147
```

Warm run over the same DuckDB store:

```text
correct: 1 / 1
answer: Katie
model_calls: 0
aggregate_calls: 0
initial_coverage_ratio: 1.0
avg_latency_ms: 148.135
```

Artifacts:

```text
benchmark_artifacts/dp_memo_nolima_official_smoke_32k_typed_evidence/20260518T181118Z/manifest.json
benchmark_artifacts/dp_memo_nolima_official_smoke_32k_typed_evidence_warm_no_upsert/20260518T181750Z/manifest.json
```

This is the strongest live result so far. The cold run takes real work, but the
warm run avoids 81 scoped Q3 calls and the aggregate call, with the exact same
answer.

Session memo store updated:

```text
benchmark_artifacts/long_horizon_substrate_session/session_memo.duckdb
entries: 5
latest task_type: benchmark_result
```

Summary utility over 16K and 32K cold/warm manifests:

```text
16K cold: model_calls=41, aggregate_calls=1, avg_latency_ms=167616.257
16K warm: model_calls=0,  aggregate_calls=0, avg_latency_ms=83.463

32K cold: model_calls=81, aggregate_calls=1, avg_latency_ms=383584.147
32K warm: model_calls=0,  aggregate_calls=0, avg_latency_ms=148.135
```

The exact-replay speedup is enormous because the warm path avoids both local Q3
generation and unnecessary context persistence.

## 2026-05-18 14:32 EDT Update

Implemented opt-in overlap reuse controls for NoLiMa:

```text
--corpus-id-mode stable
--document-id-mode stable
--content-hash-mode source
```

Why: the default benchmark namespace intentionally isolates every generated
sample by full dataset signature, document ID, and full context hash. That is
safe, but it prevents a 16K solve from helping a later 32K solve over the same
source prefix. The stable modes are experimental and explicit; they remove
length from the corpus/document/scope identity so overlapping length sweeps can
reuse solved prefix work.

Overlap experiment:

1. Run 16K official NoLiMa with stable corpus/document/source hash.
2. Run 32K official NoLiMa over the same DuckDB store with the same stable
   identity settings.

16K cold:

```text
correct: 1 / 1
model_calls: 41
aggregate_calls: 1
final_coverage_ratio: 1.0
```

32K partial composition:

```text
correct: 1 / 1
model_calls: 41
aggregate_calls: 1
initial_covered_length: 81
initial_missing_length: 80
initial_coverage_ratio: 0.503106
reused_windows: 40
missing_windows: 41
reused_window_ratio: 0.493827
final_coverage_ratio: 1.0
```

Artifact:

```text
benchmark_artifacts/dp_memo_nolima_overlap_reuse_stable_32k_partial/20260518T182841Z/manifest.json
```

This is the first real partial-composition result. It is not exact replay. The
larger 32K task reused the solved 16K prefix and called Q3 only for the expanded
tail.

Updated README and the LaTeX draft to explain the stable overlap modes and the
partial-composition result. Rebuilt the PDF.

Full deterministic verification:

```text
python -m pytest
43 passed
```

Session memo store updated:

```text
benchmark_artifacts/long_horizon_substrate_session/session_memo.duckdb
entries: 6
latest task_type: benchmark_result
```

Added `corpus_id` to NoLiMa progress events, bridge rows, and manifests so
stable-vs-dataset namespace experiments are auditable.

Verification:

```text
python -m py_compile scripts/run_dp_memo_nolima.py
python -m pytest test/test_long_context_workload_inspector.py
5 passed
```

Updated `scripts/summarize_dp_memo_runs.py` to surface `corpus_id` as well.

Verification:

```text
python -m pytest test/test_summarize_dp_memo_runs.py
2 passed
```

Full deterministic verification after the summary utility:

```text
python -m pytest
40 passed
```

Session memo store updated:

```text
benchmark_artifacts/long_horizon_substrate_session/session_memo.duckdb
entries: 4
latest task_type: benchmark_result
```

## 2026-05-18 14:04 EDT Update

Added aggregate latency to benchmark manifests and summaries:

- NoLiMa manifest totals now include `latency_ms` and `avg_latency_ms`.
- Shared-context manifest totals now include `latency_ms` and `avg_latency_ms`.
- `scripts/summarize_dp_memo_runs.py` surfaces `avg_latency_ms`.

Verification:

```text
python -m pytest test/test_summarize_dp_memo_runs.py test/test_long_context_workload_inspector.py
5 passed
```

## 2026-05-18 14:05 EDT Update

Removed eager Q3 loading from the NoLiMa and shared-context benchmark runners.
The local model now loads lazily only if a solver, aggregator, or planner
actually calls it. This matters for warm replay: a fully memoized run should not
pay local model startup overhead.

Warm replay after lazy loading change:

```text
correct: 1 / 1
model_calls: 0
aggregate_calls: 0
initial_coverage_ratio: 1.0
avg_latency_ms: 161.001
```

Artifact:

```text
benchmark_artifacts/dp_memo_nolima_official_smoke_8k_typed_evidence_warm_lazy/20260518T180428Z/manifest.json
```

Verification:

```text
python -m pytest
40 passed
```

## 2026-05-18 14:08 EDT Update

Ran the same typed-evidence path at 16K official NoLiMa.

Cold run:

```text
correct: 1 / 1
answer: Katie
model_calls: 41
aggregate_calls: 1
initial_coverage_ratio: 0.0
final_coverage_ratio: 1.0
avg_latency_ms: 167616.257
```

Warm run over the same DuckDB store:

```text
correct: 1 / 1
answer: Katie
model_calls: 0
aggregate_calls: 0
initial_coverage_ratio: 1.0
avg_latency_ms: 270.547
```

Artifacts:

```text
benchmark_artifacts/dp_memo_nolima_official_smoke_16k_typed_evidence/20260518T180459Z/manifest.json
benchmark_artifacts/dp_memo_nolima_official_smoke_16k_typed_evidence_warm_lazy/20260518T180756Z/manifest.json
```

This is now a stronger live demonstration: same official benchmark path, longer
context, correct cold answer, persistent exact replay, and a warm run that avoids
both Q3 solving and aggregation.

Updated the summary utility to include `dependency_edge_count` and
`evidence_span_count`, so cold/warm reports include provenance shape as well as
accuracy and calls.

Verification:

```text
python -m pytest test/test_summarize_dp_memo_runs.py
2 passed
```
