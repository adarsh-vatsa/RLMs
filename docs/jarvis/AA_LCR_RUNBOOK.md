# AA-LCR Reasoning Benchmark

This package runs the 100-question Artificial Analysis Long Context Reasoning reasoning track as four internally comparable experiments. It uses Qwen3.6-35B-A3B for answers and Qwen3.5-35B-A3B for equality grading, so its scores are not official Artificial Analysis leaderboard scores.

## Prepare the dataset

Download the pinned CSV and extracted-text archive, verify their SHA-256 hashes, extract the documents, and validate all official references:

```bash
cd /home/edogu/adarsh-rlms
uv run python -m aa_lcr.prepare_dataset
```

Local source material is written under `benchmark_data/aa_lcr/` and is intentionally ignored by Git. The preparation manifest records revision `bdae010bbce259820c0e34c1d7cce210d966fb75`, source URLs, hashes, and validated counts.

## Tokenizer preflight

Run all four token-only preflights before using model services:

```bash
cd /home/edogu/adarsh-rlms
uv run python -m aa_lcr.run_benchmark --experiment direct_262k --preflight-only
uv run python -m aa_lcr.run_benchmark --experiment hybrid_262k --preflight-only
uv run python -m aa_lcr.run_benchmark --experiment direct_64k --preflight-only
uv run python -m aa_lcr.run_benchmark --experiment hybrid_64k --preflight-only
```

The 262K direct preflight fails instead of truncating an unexpectedly over-budget prompt. The 64K direct cell uses deterministic middle truncation. Hybrid cells classify over-budget prompts as `dense_child_packed`.

With the pinned dataset and Qwen3.6 tokenizer, the verified full-prompt range is 87,847–122,112 tokens (median 107,127). Consequently, both 262K cells classify all 100 questions as `direct_fit`; `direct_64k` truncates all 100 to at most 59,999 rendered tokens, and `hybrid_64k` routes all 100 through packed retrieval.

## Start the Jarvis services

Start the executor and evaluator with the same validated Qwen production profile
used in [`HPC_RUNBOOK_SETUP.md`](HPC_RUNBOOK_SETUP.md#10-start-the-two-production-services):

```bash
cd /home/edogu/adarsh-rlms

MODULES="cuda12.8/toolkit/12.8.1" \
EXECUTOR_MODEL=Qwen/Qwen3.6-35B-A3B \
EXECUTOR_TP_SIZE=4 \
EXECUTOR_MAX_MODEL_LEN=262144 \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
VLLM_EXTRA_ARGS="--reasoning-parser qwen3 --language-model-only --max-num-seqs 1 --enable-chunked-prefill --max-num-batched-tokens 8192" \
SYNC_BACK_MODELS=1 VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash jarvis/run.sh submit executor

MODULES="cuda12.8/toolkit/12.8.1" \
EVALUATOR_MODEL=Qwen/Qwen3.5-35B-A3B \
EVALUATOR_TP_SIZE=2 \
EVALUATOR_MAX_MODEL_LEN=16384 \
EVALUATOR_PORT=8001 \
VLLM_EXTRA_ARGS="--reasoning-parser qwen3 --language-model-only" \
SYNC_BACK_MODELS=1 VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash jarvis/run.sh submit evaluator
```

Port `8001` is explicit here so the evaluator URL matches the examples below;
using another free port is also valid. Read the authoritative endpoint URLs from
the corresponding Jarvis `.url` files and export them on the login node. Existing
services can be reused only when they were started with the same models, context
lengths, and vLLM arguments.

## Smoke tests and full runs

Validate each launch without submitting:

```bash
cd /home/edogu/adarsh-rlms

OPENAI_COMPAT_EXECUTOR_BASE_URL=http://executor-host:8000/v1 \
OPENAI_COMPAT_EVALUATOR_BASE_URL=http://evaluator-host:8001/v1 \
AA_LCR_LAUNCH_DRY_RUN=1 \
bash jarvis/run_aa_lcr.sh direct_262k
```

Set `AA_LCR_MAX_ROWS=2` for a real two-question smoke test. Run the complete cells independently so they do not compete for the same services:

```bash
cd /home/edogu/adarsh-rlms

OPENAI_COMPAT_EXECUTOR_BASE_URL=http://executor-host:8000/v1 \
OPENAI_COMPAT_EVALUATOR_BASE_URL=http://evaluator-host:8001/v1 \
bash jarvis/run_aa_lcr.sh direct_262k

OPENAI_COMPAT_EXECUTOR_BASE_URL=http://executor-host:8000/v1 \
OPENAI_COMPAT_EVALUATOR_BASE_URL=http://evaluator-host:8001/v1 \
bash jarvis/run_aa_lcr.sh hybrid_262k

OPENAI_COMPAT_EXECUTOR_BASE_URL=http://executor-host:8000/v1 \
OPENAI_COMPAT_EVALUATOR_BASE_URL=http://evaluator-host:8001/v1 \
bash jarvis/run_aa_lcr.sh direct_64k

OPENAI_COMPAT_EXECUTOR_BASE_URL=http://executor-host:8000/v1 \
OPENAI_COMPAT_EVALUATOR_BASE_URL=http://evaluator-host:8001/v1 \
bash jarvis/run_aa_lcr.sh hybrid_64k
```

Direct cells use a CPU client job. Hybrid cells use one client GPU for Qwen embeddings and FAISS. Every hybrid cell starts from an empty cache and saves its final cache state only inside that run directory.

Artifacts are written to:

```text
benchmark_artifacts/aa_lcr/<experiment>/<run_id>/
```

Each run contains predictions, JSONL/CSV bridge rows, a manifest, the equality evaluation report, and hybrid cache state where applicable.

## Compare four complete runs

The comparison rejects mismatched data, row order, prompts, models, budgets, API errors, or invalid grades:

```bash
cd /home/edogu/adarsh-rlms

uv run python -m aa_lcr.compare_runs \
  --direct-262k benchmark_artifacts/aa_lcr/direct_262k/<run_id> \
  --hybrid-262k benchmark_artifacts/aa_lcr/hybrid_262k/<run_id> \
  --direct-64k benchmark_artifacts/aa_lcr/direct_64k/<run_id> \
  --hybrid-64k benchmark_artifacts/aa_lcr/hybrid_64k/<run_id>
```

The output includes per-cell accuracy, paired deltas, 64K reasoning loss, hybrid retention, category and route breakdowns, operational totals, document-set-clustered 95% bootstrap intervals, and every accepted semantic hit for manual audit.
