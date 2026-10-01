# Project status: 1 October 2026

An orientation for someone joining the project: what the system does, which
results hold up, what is unfinished and where to start. It replaces the
[21 September status](archive/project_status_20260921.md). When a document
disagrees with the code, trust the code; see [AGENTS.md](../AGENTS.md).

## The project in brief

We are building an execution layer for long-context reasoning with LLMs. Each
question comes with source text. If the source fits the model's input budget,
the system sends all of it. If not, it retrieves the most relevant parts and
packs them into the budget. Optionally, it reuses verified answers from a
semantic cache when a question is repeated or reworded.

The research question: **compared with sending the full context to the same
model, does this keep or improve accuracy while using fewer tokens and less
time?** We test it on three long-context benchmarks with an open-weight model
served locally by vLLM.

The project began in March 2026 as a "Two-Stage Semantic Cache" prototype on
Claude models. The focus has since moved from the cache to retrieval under a
context budget, and all current experiments use local Qwen models.

## Current state

- **Branch:** work happens on `shared-execution-linux-runners`, 143 commits
  ahead of `main`. `main` was last updated on 2026-05-18 with three commits by
  Adarsh Vatsa that are not on this branch: a memoized-RLM line of work
  (`language_memoization.py`, `local_llm.py`), a NoLiMa depth sweep and paper
  drafts. Both lines change `semantic_cache_system.py` and `README.md`, so a
  merge will conflict.
- **Tests:** 215 unit tests pass in about 3 seconds. They use mocked clients and
  synthetic fixtures and check plumbing, not model quality.
- **Servers:**
  - **Neselab** (one 96 GB RTX PRO 6000, no Slurm) runs AA-LCR and MRCR v2, with
    the scripts in [`linux/`](../linux/) and the [Linux runbooks](linux/README.md).
  - **Jarvis** (Slurm, 4× L40S nodes) runs LongBench-v2, with the scripts in
    [`jarvis/`](../jarvis/) and the [Jarvis runbooks](jarvis/README.md).
- **Models:**
  - executor: `Qwen/Qwen3.6-35B-A3B`, temperature 0, thinking disabled, full
    262,144-token window
  - AA-LCR grader: the same model, for now (see open work)
  - retrieval: `Qwen3-Embedding-0.6B` with a FAISS index
- **Results:** AA-LCR and MRCR v2 have current results on the shared pipeline.
  LongBench-v2 has only August results from the older runner; its shared-pipeline
  runs are ready to start on Jarvis.

## How the system works

The core is `execution.pipeline.Pipeline` in [`execution/`](../execution/).
Each benchmark has a small adapter that turns one example into a `Task`: its
documents, its question and a `render` function that builds the prompt. The
pipeline never branches on benchmark names. See the
[shared execution architecture](shared_execution_architecture.md) and
[task adapters](task_adapters.md).

The pipeline renders the full prompt, counts its tokens with the executor's
tokenizer and picks a route:

| Route | When | What happens |
|---|---|---|
| `direct_fit` | The prompt fits the input budget (either mode) | The whole prompt is sent |
| `middle_truncated` | Direct mode, too long, `--direct-overflow middle` | An equal head and tail are kept |
| `unsupported_context` | Direct mode, too long, no overflow policy | Recorded without a model call |
| `dense_child_packed` | Hybrid mode, too long | Chunks are embedded, ranked against the question, packed up to the budget and shown in source order |

The input budget is the model window minus an output allowance, counted exactly
as vLLM counts it. Direct and hybrid differ only on prompts that do not fit, so
experiments use budgets smaller than the prompts, or prompts longer than the
served window.

Answer caching (exact or semantic, with a verifier model) has its own switches
and is off by default. It was off for every current AA-LCR and MRCR run. Scoring
happens after each prediction is saved, and gold answers never enter the
`Task`, the cache or the verifier.

Not on the shared pipeline: [`semantic_cache_system.py`](../semantic_cache_system.py),
the original prototype, which also supplies the embeddings, FAISS and cache
controller; and the LongBench-v2 iterative and RLM runners.

## Results

All runs use Qwen3.6-35B-A3B as the executor and are single runs per setting.

### AA-LCR (30 September, Neselab)

100 free-form questions over 30 document sets, dataset release 1.1. Prompts are
88K–122K tokens. [Report](reports/aa_lcr_results_20260930/aa_lcr_results_20260930.md).

| Run | Window / input budget | Accuracy |
|---|---|---:|
| Full context | 262,144 / 258,048 | 61% |
| Hybrid | 65,536 / 61,440 | 54% |
| Middle truncation | 65,536 / 61,440 | 37% |

- With the same 64K window, retrieval beat truncation by **+17 points** (95%
  interval +8.4 to +25.9, clustered by document set).
- Hybrid trails full context by 7 points, which is not significant (−16.7 to
  +2.0); the gap is on the longest prompts.
- **Caveat:** the executor grades its own answers. A hand check of about 55
  grades found five clear errors in both directions. Correcting them changes no
  conclusion, but the absolute scores need an independent grader.

### MRCR v2 (29–30 September, Neselab)

Synthetic conversations of 133K, 267K, 533K and 1.06M tokens, one per length,
4-needle release, 30 questions each. The input budget is 258,048 tokens.
Scoring is deterministic. [Report](reports/mrcr_results_20260930/mrcr_results_20260930.md).

| Conversation | Direct (middle truncation) | Hybrid |
|---:|---:|---:|
| 133K | 63% | 63% (identical requests) |
| 267K | 53% | 50% |
| 533K | 27% | **60%** |
| 1.06M | 10% | **30%** |
| All 120 | 38% | **51%** (20 gained, 5 lost, p = 0.004) |

- Hybrid gains only once the conversation exceeds the window, because retrieval
  finds the needles about twice as often as truncation keeps them.
- Most remaining errors copy the wrong occurrence of a real needle, even when
  everything is visible: the model's counting is the limit.
- The gain over the earlier 8-needle runs mixes three changes: needle count,
  retrieval query and chunk size (3,500 instead of 7,500 tokens). A planned
  rerun separates the last two.
- A 40-question repeat gave identical answers in both modes.

### LongBench-v2 (August, older runner, Jarvis)

503 multiple-choice questions at a 240,000-token budget, from before the shared
pipeline. On the 107 questions that did not fit, hybrid scored 56.1% against
41.1% for direct truncation (21 gained, 5 lost, p = 0.0025). The semantic cache
answered 493 of 503 paraphrased questions and roughly halved total tokens.
Reports are in [`reports/archive/`](reports/archive/).

### Figures not to cite

- The "96.7% cost reduction" in the README, from a five-call prototype demo.
- AA-LCR scores from August (44%), superseded and affected by a 512-token answer
  limit, grader errors and old answer keys.
- MRCR results from 21 September (8 needles, 10 questions), now in
  `benchmark_artifacts/mrcr_v2/*/archive/`.
- LongBench 46.9% against 13.3% from the
  [3 August note](archive/longbench_v2_baseline_comparison_20260802.md); that
  baseline failed on 321 rows.

## Open work, roughly in priority order

1. **Run LongBench-v2 on the shared pipeline on Jarvis.** Direct on all 503
   questions with middle truncation, then hybrid on the 100 that exceed the
   262,136-token budget. Commands are in the
   [Jarvis LongBench runbook](jarvis/HPC_RUNBOOK_EXPERIMENT.md). An optional
   answer-cache experiment with repeated and reworded questions is in its §7.
2. **Grade AA-LCR independently.** Regrade the saved answers with a model other
   than the executor (`aa_lcr.regrade`) before reporting absolute scores.
3. **Separate the MRCR chunk-size effect.** Rerun hybrid on the same 120
   questions with 7,500-token chunks. Then consider more questions per
   conversation and the 1M–2M file.
4. **Test the answer cache on the shared pipeline.** It is tested only with
   mocks; the cache results so far come from the older LongBench runner.
5. **Merge with `main`** and decide on the memoization work.
6. **More benchmarks:** candidates are compared in [benchmarks.md](benchmarks.md).

## Getting started

1. Check out `shared-execution-linux-runners`.
2. Read, in order: [AGENTS.md](../AGENTS.md), this file,
   [shared_execution_architecture.md](shared_execution_architecture.md),
   [task_adapters.md](task_adapters.md), the overviews
   ([aa_lcr.md](aa_lcr.md), [mrcr_v2.md](mrcr_v2.md)) and the two September
   reports.
3. Set up a local environment and run the tests (needs `uv`):

   ```bash
   bash linux/setup.sh client
   .venv/bin/python -m unittest discover -s test
   ```

4. On a GPU server, follow the [Linux runbooks](linux/README.md) or the
   [Jarvis runbooks](jarvis/README.md). `--preflight-only` reports each
   example's route without inference, and the launchers' dry-run variables print
   the command without running it.

## Pitfalls that have already caused problems

- **Hybrid embeddings default to CUDA.** On the single-GPU server, where vLLM
  holds the GPU, set `SEMANTIC_CACHE_EMBEDDING_DEVICE=cpu`.
- **A failed run can look complete.** Without `--fail-fast`, a run in which
  every request fails still finishes with a score of 0.
- **Direct mode skips long prompts by default.** Pass `--direct-overflow middle`
  to truncate them instead of recording them as unsupported.
- **Budget flags don't change the server.** vLLM fixes the served window at
  launch; keep the runner's budget within it.
- **Two 35B models don't fit on one 96 GB GPU.** Run AA-LCR with
  `--execution-only` and grade afterwards.
- **MRCR preparation is slow.** It counts tokens for every row in a downloaded
  file, however few rows you later select.
- **The prototype defaults to Claude.** The launchers switch it to the local
  OpenAI-compatible endpoint; check the run manifest.
- **"Evaluator" can mean two roles:** the AA-LCR grader or the semantic cache
  verifier. They are configured separately.
- **Treat `benchmark_artifacts/` as read-only history.** Keep each run directory
  whole; its manifest records the settings needed to compare runs.

## Timeline

| When (2026) | What happened |
|---|---|
| March | Initial Two-Stage Semantic Cache prototype; RULER v2. |
| April–May | NoLiMa, data-scope hashing, legal benchmark, RLM tests. LongBench-v2 chosen as the main benchmark. Memoization work committed to `main` on 18 May. |
| June–July | Moved to Jarvis with vLLM and local Qwen models; iterative reader for LongBench-v2. |
| 8–14 August | 256K context; "fast hybrid" route; LongBench four-run comparison. |
| 24 Aug–10 Sep | AA-LCR added, first runs, accuracy investigation and rerun tools. |
| 17 September | MRCR v2 added; all three benchmarks moved to the shared pipeline; Linux runners. |
| 20–21 September | Moved AA-LCR and MRCR to the Neselab server; first MRCR runs. |
| 28–30 September | Full-window MRCR (4 needles, 100K–1.2M) and AA-LCR (release 1.1) runs and reports. |
| 1 October | Jarvis scripts and runbooks updated for LongBench-v2 on the shared pipeline. |
