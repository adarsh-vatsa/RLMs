# Project status: 21 September 2026

This is an orientation for someone joining the project. It covers what the
system does, which results hold up, what is unfinished or known to be wrong,
and where to start. Figures were checked against the committed artifacts on
2026-09-21. When a document disagrees with the code, trust the code. See
[AGENTS.md](../AGENTS.md) for more on this.

## The project in brief

We are building an execution layer for long-context reasoning with LLMs. Each
question comes with source text. If the source fits the model's input budget,
the system sends all of it. If it doesn't fit, the system retrieves the most
relevant parts of the source and packs them into the budget. Optionally, it
reuses verified answers from a semantic cache when a question is repeated or
paraphrased.

The research question: **compared with sending the full context to the same
model, does this keep or improve reasoning accuracy while using fewer tokens,
calls and less time?** We test this on three long-context benchmarks, using an
open-weight model served locally with vLLM.

The project started in March 2026 as a "Two-Stage Semantic Cache" prototype. It
paired a vector search (the "dragnet") with an LLM verifier (the "sniper") and
ran on Claude models. Since then the focus has moved from the cache to retrieval
under a context budget, and all current experiments use local Qwen models.

## Current state

- **Work on the `shared-execution-linux-runners` branch.** It is 129 commits
  ahead of `main`. `main` was last updated on 2026-05-18.
- **There are two unmerged lines of work.** This branch holds the benchmarks and
  the shared pipeline; nearly all commits since late March are Engin Deniz
  Dogu's. `main` has three commits from 2026-05-18 by Adarsh Vatsa that are not
  on this branch:
  - a "language memoization" / memoized-RLM line of work (`language_memoization.py`
    and `local_llm.py`)
  - a NoLiMa depth sweep
  - paper drafts (`docs/context_management_and_language_memoization.tex`,
    `docs/paper_draft.tex`, `docs/paper_scope.md` and
    `docs/workshop_venues_2026.md`)

  Both lines change `semantic_cache_system.py` and `README.md`, so a merge will
  have conflicts.
- **All 210 unit tests pass**, in about 2 seconds on a laptop. They use mocked
  clients and synthetic fixtures. They check the plumbing, not model quality.
- **Hardware:** runs from June to August used the Jarvis cluster at Stevens
  (Slurm, L40S GPUs). Since 2026-09-20, runs happen on a standalone Linux GPU
  server (Neselab) without Slurm, using the scripts in [`linux/`](../linux/).
- **Models:**
  - executor: `Qwen/Qwen3.6-35B-A3B` behind vLLM's OpenAI-compatible API,
    temperature 0, thinking disabled
  - grader and cache verifier: `Qwen/Qwen3.5-35B-A3B`
  - retrieval: `Qwen3-Embedding-0.6B` embeddings with a FAISS index
- **Benchmarks with results:** LongBench-v2, AA-LCR and MRCR v2. Each has open
  problems, described below.

## How the system works now

The core is `execution.pipeline.Pipeline` in [`execution/`](../execution/),
about 700 lines. AA-LCR, MRCR v2 and the direct/hybrid path of LongBench-v2
each provide a small adapter, under 20 lines, that turns one example into a
`Task`. A `Task` holds the documents, the question and a `render` function that
builds the prompt. The pipeline never branches on benchmark names.

For each example, the pipeline renders the full prompt and counts its tokens
with the executor's tokenizer. It then picks one of these routes:

| Route | When | What happens |
|---|---|---|
| `direct_fit` | The full prompt fits the input budget (either mode) | The whole prompt is sent |
| `middle_truncated` | Direct mode, the prompt doesn't fit, and `--direct-overflow middle` is set | The start and end are kept and the middle is cut |
| `unsupported_context` | Direct mode, the prompt doesn't fit, default `common` profile | The example is recorded with no model call |
| `dense_child_packed` | Hybrid mode, the prompt doesn't fit | Retrieve and pack; see below |

The `dense_child_packed` route works like this:

1. Split the source into 7,500-token chunks with 750 tokens of overlap.
2. Embed the chunks and rank them by similarity to the question.
3. Add chunks in ranked order until the next one would exceed the budget.
4. Render the selected chunks in their original source order.

Direct and hybrid mode behave differently only when the source is larger than
the budget. For that reason, the experiments deliberately use budgets smaller
than the sources, or sources larger than the served context.

Answer caching (exact or semantic, with an LLM verifier) has its own switch.
It is off by default in the `common` profile and was off for every AA-LCR and
MRCR run. Scoring happens after each prediction is saved. Gold answers never
enter the `Task`, the cache or the verifier.

What is **not** on the shared pipeline:

- [`semantic_cache_system.py`](../semantic_cache_system.py) (4,448 lines) is
  the original prototype. It contains the embeddings, FAISS, the reranker, the
  cache controller with dragnet/sniper verification, knowledge-triple
  extraction and the iterative reader.
- The LongBench-v2 cache runner
  ([`long_bench_v2/run_benchmark.py`](../long_bench_v2/run_benchmark.py)) and
  the iterative and RLM baselines still call that prototype directly. **The
  August LongBench results came from this runner, not from the shared
  pipeline.**

## Results so far

All results are single runs with Qwen3.6-35B-A3B as the executor. None has been
repeated, so treat small differences as noise.

### LongBench-v2 (August 2026, older cache runner)

Setup: all 503 LongBench-v2 questions, a 262,144-token context and a
240,000-token input budget. To exercise the cache, the suite also includes 503
paraphrases of the questions, plus 503 exact repeats in the second experiment.
Full report:
[longbench_v2_four_run_comparison_20260813/report.html](reports/longbench_v2_four_run_comparison_20260813/report.html).

| Measure | Cache + retrieval | Full-context baseline | Difference |
|---|---:|---:|---:|
| Accuracy on originals + paraphrases (1,006 rows) | 48.0% | 44.6% | +3.4 pp (95% CI +1.2 to +5.6) |
| Total tokens on those rows | 60.8M | 120.0M | 49.3% fewer |
| Wall time on those rows | 2.02 h | 2.48 h | 1.23× faster |
| Accuracy on all traffic, including exact repeats (1,509 rows) | 48.8% | 44.7% | +4.2 pp |
| Tokens on all traffic | | | 66.2% fewer |

Where the effects come from:

- **The accuracy gain comes from retrieval on long inputs.** 406 of the 503
  questions fit the budget, and both systems sent them in full. For the other
  107, the baseline cut the middle of the prompt while the hybrid retrieved
  evidence; the hybrid scored 56.1% on those 107. The gap is +13.4 pp on rows
  where the baseline truncated and +1.7 pp on the rest.
- **The token savings come from the semantic cache.** Original questions cost
  about the same in both systems: 59.6M versus 60.0M input tokens. The cache
  answered 493 of the 503 paraphrases after a short verification call. The
  paraphrases used 1.2M tokens in the cache run versus 60.0M in the baseline.
- **Run-to-run noise:** two hybrid runs with the same routes and token counts
  gave different answers on 52 of 1,006 questions. This shifted accuracy by
  0.8 pp.

### AA-LCR (run 30 August 2026, investigated 8 September)

AA-LCR has 100 questions over 30 sets of long documents (229 documents in
total). Answers are free-form and graded by an LLM, Qwen3.5-35B-A3B. Report:
[aa_lcr_four_run_results_20260830.md](reports/aa_lcr_four_run_results_20260830/aa_lcr_four_run_results_20260830.md).

| Run | Route (all 100 questions) | Accuracy |
|---|---|---:|
| `direct_262k` | `direct_fit` | 44% |
| `hybrid_262k` | `direct_fit` | 44% (same answers) |
| `direct_64k` | `middle_truncated` | 21% |
| `hybrid_64k` | `dense_child_packed` | 41% |

With a 60,000-token input budget, retrieval beat truncation by 20 pp (95% CI
+9.3 to +32.2, clustered by document set). It came within 3 pp of the
full-context score. At 262K every prompt (88K to 122K tokens) fits, so the two
262K runs are effectively the same experiment. No cache hits occurred.

**The absolute scores are known to be too low.** Artificial Analysis lists
about 64% for this model. The
[8 September investigation](aa_lcr_accuracy_investigation_20260908.md) found
four problems:

- **Answer limit:** the runner hard-coded a 512-token answer limit. 35 answers
  hit it and 29 of those were graded wrong; several stop partway through a
  calculation.
- **Grader errors:** the local grader marked at least 4 correct answers wrong,
  for example "9 percentage points" against the reference 0.09.
- **Old answer keys:** the runs used dataset v1.0.0. Version 1.1, released in
  September 2026, revised 16 answer keys.
- **Different method:** Artificial Analysis allows 16,384 output tokens, uses a
  different checker model and prompt, and averages 3 repeats.

The code for a corrected rerun is done:

- a configurable output budget, with `finish_reason` recorded
- v1.1 pinned alongside v1.0
- versioned grader prompts
- [`aa_lcr/regrade.py`](../aa_lcr/regrade.py) and
  [`aa_lcr/compare_conditions.py`](../aa_lcr/compare_conditions.py)

See the [rerun plan](aa_lcr_rerun_plan_20260908.md) and the
[rerun commands](aa_lcr_rerun_commands.md). **No regrading or rerun has been
done yet.** The 64K experiment also needs redesigning: a 16K answer allowance
leaves only about 49K input tokens in a 64K window.

### MRCR v2 (21 September 2026, first runs on the Neselab server)

MRCR (multi-round coreference resolution) hides 8 matching responses in a long
synthetic conversation. The model must find the requested one (say, the second
moon poem) and copy it exactly. Scoring is deterministic, with no grader model.
Each run below used 10 examples. Overview: [mrcr_v2.md](mrcr_v2.md).

| Source length (executor tokens) | Input budget | Direct (middle cut) | Hybrid (retrieval) |
|---|---|---|---|
| 100K to 200K | 60,000 | score 0.31, 3/10 exact | score 0.41, 4/10 exact |
| 524K to 600K | 240,000 | score 0.22, 2/10 exact | score 0.39, 3/10 exact |

Hybrid is ahead, as it was on AA-LCR, but 10 examples is far too few to draw
conclusions. The hybrid runs computed embeddings on the CPU, and this took
longer than generation: 871 s versus 269 s for the 524K to 600K runs.

Artifacts are in
[`benchmark_artifacts/mrcr_v2/`](../benchmark_artifacts/mrcr_v2/). That folder
also holds two earlier attempts from 00:25 UTC the same day. They are not
results:

- The hybrid attempt failed on every row, because embeddings were set to CUDA
  and CUDA was unavailable.
- The direct attempt recorded every row as unsupported, because
  `--direct-overflow middle` was not set.

The benchmark runbook recommends one more run as the reference for the 60K
pair: a full-context direct run at 240K on the 100K to 200K sources. It has not
been run yet.

### Figures not to cite

- **"96.7% cost reduction" in the README:** this comes from a five-call demo in
  the original prototype, not from a benchmark.
- **LongBench 46.9% versus 13.3%** (the
  [3 August note](longbench_v2_baseline_comparison_20260802.md)): the baseline
  in that run failed on 321 rows with context-length errors.
- **AA-LCR 44% as the model's capability:** the problems are listed above.
- **The MRCR runs from 00:25 UTC on 21 September.**

## Open work, roughly in priority order

1. **Rerun the AA-LCR baseline.** The tools are ready and only the runs remain:
   - regrade the saved answers (conditions A to C in the plan)
   - generate new answers with 16K output tokens (condition D)
   - repeat 3 times

   Decide whether to use the official checker model or report the results as
   local-grader numbers. The grader needs a 32K context window.
2. **Scale up MRCR.** Use more than 10 examples, add the full-context reference
   run and compute confidence intervals. If a second GPU is free, move the
   embeddings onto it to speed up retrieval.
3. **Run LongBench-v2 on the shared pipeline** (`--execution-profile common`).
   The August results come from before the pipeline was unified. On the new
   server, only route audits have been run. The Linux launcher defaults
   LongBench to a 64K window, a 60K input budget and original questions only.
4. **Test the cache on the shared pipeline.** The cache claims currently rest
   on the older LongBench runner, and AA-LCR and MRCR ran with caching off.
   There is no experiment design yet for cache hits under the `common` profile.
5. **Test AA-LCR retrieval at a scale where it matters.** Options are extending
   the document sets beyond 240K tokens or forcing retrieval at 262K.
6. **Housekeeping:**
   - merge this branch with `main`, and decide what happens to the memoization
     work
   - rewrite `README.md`. It still describes the March prototype (Claude models,
     plus `epstein_search.py` and `paper_draft.tex`, which are not on this
     branch). [system_architecture.md](system_architecture.md) and
     [semantic_cache_concept_guide.md](semantic_cache_concept_guide.md) also
     describe that original design.
7. **Add more benchmarks.** [benchmarks.md](benchmarks.md) and
   [Benchmarks.xlsx - Benchmarks.csv](Benchmarks.xlsx%20-%20Benchmarks.csv)
   compare candidates such as OOLONG, CorpusQA, LongMemEval and
   MemoryAgentBench.

## Getting started

1. Check out the working branch:

   ```bash
   git checkout shared-execution-linux-runners
   ```

2. Read these, in order:
   1. [AGENTS.md](../AGENTS.md): short working rules for the repository
   2. this file
   3. [shared_execution_architecture.md](shared_execution_architecture.md)
   4. [task_adapters.md](task_adapters.md)
   5. the benchmark overviews, [aa_lcr.md](aa_lcr.md) and
      [mrcr_v2.md](mrcr_v2.md)
   6. the three result documents linked above

3. Set up a local environment and run the tests. Setup needs `uv` and creates
   `.venv` with Python 3.13:

   ```bash
   bash linux/setup.sh client
   .venv/bin/python -m unittest discover -s test
   ```

   Tests should use mocks and synthetic fixtures, never downloaded model
   weights or live model services.

4. On a GPU server, follow [linux/SETUP_RUNBOOK.md](linux/SETUP_RUNBOOK.md) and
   then [linux/BENCHMARK_RUNBOOK.md](linux/BENCHMARK_RUNBOOK.md). Two options
   let you check a run before starting inference:
   - `--preflight-only` reports each example's route without calling the model.
   - `DRY_RUN=1` prints the command without running it.

   For Jarvis, start at [jarvis/README.md](jarvis/README.md).

Questions to ask early:

- How do I get access to the Neselab server and Jarvis, and where are the model
  weights and Hugging Face cache stored?
- Where is the LongBench-v2 `data.json`? It is 465 MB and not in git. The
  runbook also gives a pinned download.
- Do we have access to the official AA-LCR checker model, if we want scores
  comparable with the leaderboard?
- What are the paper plans? The drafts on `main` date from May, use the
  memoization framing, and include none of the results above.

## Pitfalls that have already caused problems

- **Hybrid embeddings default to CUDA.** On a single-GPU server where vLLM holds
  the GPU, set `SEMANTIC_CACHE_EMBEDDING_DEVICE=cpu`, or every row fails.
- **A fully failed run can look complete.** Without `--fail-fast`, a run in
  which every example fails still reports `status: complete` with a score of 0.
- **Direct mode skips long examples by default.** Under the `common` profile,
  examples over the budget are recorded as unsupported unless you pass
  `--direct-overflow middle`.
- **Budget flags don't change the server.** vLLM sets the served context at
  launch. Keep the runner's budget within it.
- **Don't run both 35B models at once on a single 96 GB GPU.** The runbook
  covers this case (an RTX PRO 6000): each service reserves 90% of GPU memory.
  Run with `--execution-only` and grade afterwards.
- **MRCR data preparation is slow.** It counts tokens for every row in a
  download band, however few rows you later select. Each preparation needs its
  own output directory.
- **The prototype still defaults to Claude.** `semantic_cache_system.py`
  defaults to the Anthropic provider and Claude model IDs. The benchmark
  launchers switch it to `openai_compatible`; check the run manifest to see
  which provider was used.
- **Treat `benchmark_artifacts/` as read-only history.** New runs get new
  timestamped directories. Keep each run directory whole, because its manifests
  record the settings needed to compare runs.
- **The "evaluator" service can play two roles.** It can be the AA-LCR grader,
  the semantic cache verifier, or both, and these roles are separate.

## Repository map

| Path | Contents |
|---|---|
| [`execution/`](../execution/) | Shared pipeline: routing, token counting, chunking and retrieval, packing, cache hooks, HTTP client, output files |
| [`aa_lcr/`](../aa_lcr/) | AA-LCR: dataset preparation (v1.0 and v1.1 pinned), adapter, runner, LLM grading, regrading, comparisons |
| [`mrcr_v2/`](../mrcr_v2/) | MRCR v2: dataset preparation by token band, adapter, runner, deterministic scoring |
| [`long_bench_v2/`](../long_bench_v2/) | LongBench-v2: older cache runner, API baseline, RLM baseline, CSV tools |
| [`semantic_cache_system.py`](../semantic_cache_system.py) | Original prototype: embeddings, FAISS, reranker, semantic cache, iterative reader |
| [`linux/`](../linux/) | Scripts for a server without Slurm: setup, vLLM services, benchmark launcher |
| [`jarvis/`](../jarvis/) | Slurm scripts for the Jarvis cluster |
| [`test/`](../test/) | Unit tests |
| [`docs/`](.) | Design notes, runbooks, investigations, reports |
| `benchmark_data/`, `benchmark_artifacts/` | Prepared inputs and every run's outputs, committed to git |

## Timeline

| When (2026) | What happened |
|---|---|
| March | Adarsh Vatsa's initial commit of the Two-Stage Semantic Cache. RULER v2 implemented. |
| April | NoLiMa, data-scope hashing, a cache benchmark, and a fix for an embedding pooling bug that inflated false cache hits. |
| May | Legal benchmark and RLM tests. After surveying benchmarks, LongBench-v2 became the main one, with API and RLM baselines. On 18 May, the memoization work was committed to `main`. |
| June–July | Moved to Jarvis with vLLM and local Qwen models. Built the iterative reader and evidence ledger for LongBench-v2. |
| 8–14 August | Context raised from 60K to 256K. Built the "fast hybrid" route (send in full when it fits, otherwise retrieve and pack). LongBench four-run comparison. |
| 24 Aug–1 Sep | Added AA-LCR and ran the four-run experiment. |
| 8–10 September | AA-LCR accuracy investigation and rerun tools. |
| 17 September | Added MRCR v2. Put all three benchmarks on the shared direct/hybrid pipeline. Added the Linux runners. |
| 20–21 September | Moved to the Neselab server. First MRCR runs. |
