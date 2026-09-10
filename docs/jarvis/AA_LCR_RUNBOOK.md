# AA-LCR Reasoning Benchmark

Follow the sections through **Full baseline runs** for the new 100-question
AA-LCR v1.1 baseline: Qwen3.6-35B-A3B, non-thinking, full context, and a
16,384-token answer allowance. These commands use local Qwen3.5-35B-A3B grading
with the v1.1 prompt; the resulting accuracy is a local evaluation. Hosted
regrading and the controlled accuracy investigation are linked below.

Push the implementation and pull it on HPC before starting. Commands assume
the existing [HPC setup](HPC_RUNBOOK_SETUP.md) and run from the repository root.
The historical four-cell comparison at the end is optional.

## Prepare the dataset

Download the pinned CSV and extracted-text archive, verify their SHA-256 hashes, extract the documents, and validate all official references:

```bash
cd /home/edogu/adarsh-rlms
uv run python -m aa_lcr.prepare_dataset --dataset-version 1.1
```

This writes to `benchmark_data/aa_lcr/v1.1/`, pinned to revision
`9a77ef56b717057ade24ceab4d273712a0b4f19e`. This new download directory is ignored
by Git and can be recreated from the committed pins. Both versions verify the pinned CSV
and archive hashes. An explicit `--data-dir` is supported, but preparation
refuses to replace another revision or a different existing CSV.

Legacy source material under `benchmark_data/aa_lcr/` remains unchanged. The
preparation command without `--dataset-version 1.1` still selects v1.0.0.

## Tokenizer preflight

Select all three v1.1 paths together. This checks generation inputs without
contacting model services:

```bash
uv run python -m aa_lcr.run_benchmark --experiment direct_262k \
  --questions-csv benchmark_data/aa_lcr/v1.1/AA-LCR_Dataset.csv \
  --documents-root benchmark_data/aa_lcr/v1.1/extracted_text/lcr \
  --dataset-manifest benchmark_data/aa_lcr/v1.1/dataset_manifest.json \
  --max-output-tokens 16384 --preflight-only
```

Preflight and run metadata identify the dataset version. Existing manifests
without that field are recognized by pinned revision. Selecting v1.1 changes
the answer keys independently of the grader. The runner defaults to 16,384
output tokens for 262K cells and 512 for 64K cells; the local legacy grader
remains the default. Select `--grader-prompt-version aa_lcr_equality_v1.1` for
the published v1.1 grading instructions. The launcher accepts
`AA_LCR_DATA_DIR=benchmark_data/aa_lcr/v1.1` and forwards additional runner flags.
See the [rerun commands](../aa_lcr_rerun_commands.md) for hosted grading,
saved-answer regrading, repeat aggregation, and the controlled comparisons.

The 262K direct preflight fails instead of truncating an unexpectedly
over-budget prompt. Check that all 100 questions are `direct_fit`. The verified
full-prompt range is 87,847–122,112 tokens (median 107,127); the largest prompt
plus 16,384 output tokens fits within the executor's 262,144-token window.

## Start the Jarvis services

Use the executor profile from
[`HPC_RUNBOOK_SETUP.md`](HPC_RUNBOOK_SETUP.md#10-start-the-two-production-services).
The evaluator below uses 32K instead of the earlier 16K window to accommodate
long candidate answers plus grading instructions; verify its startup capacity:

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
EVALUATOR_MAX_MODEL_LEN=32768 \
EVALUATOR_PORT=8001 \
VLLM_EXTRA_ARGS="--reasoning-parser qwen3 --language-model-only" \
SYNC_BACK_MODELS=1 VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash jarvis/run.sh submit evaluator
```

Port `8001` is explicit here; using another free port is also valid. Read the
authoritative endpoint URLs from the corresponding Jarvis `.url` files and
export them on the login node. Existing
services can be reused only when they were started with the same models, context
lengths, and vLLM arguments.

## Select endpoints and run settings

Replace the two placeholder URLs with the values from your services' `.url`
files, then keep these exports in the login shell used for submission:

```bash
cd /home/edogu/adarsh-rlms

export OPENAI_COMPAT_EXECUTOR_BASE_URL="http://YOUR_EXECUTOR_HOST:8000/v1"
export OPENAI_COMPAT_EVALUATOR_BASE_URL="http://YOUR_EVALUATOR_HOST:8001/v1"
export AA_LCR_DATA_DIR=benchmark_data/aa_lcr/v1.1
export AA_LCR_MAX_OUTPUT_TOKENS=16384
unset AA_LCR_MAX_ROWS AA_LCR_RUN_ID AA_LCR_LAUNCH_DRY_RUN
```

Record actual serving details in a JSON file accessible to the client job and
export `AA_LCR_SERVING_METADATA` to its absolute path. Include model weights
revision, vLLM version, dtype, tensor parallelism, launch arguments, and effective
generation settings. This is provenance supplied by you, not automatic server
verification; missing details remain unverified. Do not include credentials.

## Smoke test

First inspect the launch command without submitting:

```bash
AA_LCR_LAUNCH_DRY_RUN=1 \
bash jarvis/run_aa_lcr.sh direct_262k \
  --question-ids 8,44,4 \
  --grader-prompt-version aa_lcr_equality_v1.1 \
  --grader-context-window 32768
```

Confirm that the printed command contains the three v1.1 paths and
`--max-output-tokens 16384`, then submit the same smoke test:

```bash
bash jarvis/run_aa_lcr.sh direct_262k \
  --question-ids 8,44,4 \
  --grader-prompt-version aa_lcr_equality_v1.1 \
  --grader-context-window 32768
```

Wait for completion. IDs 8 and 44 previously ended mid-answer; ID 4 had the
largest prompt. Inspect the saved answers and manifest: three selected rows,
v1.1 dataset and grader prompt, 16,384 output allowance, non-thinking executor,
full inputs, zero API errors, and zero invalid grades. Review termination
reasons and `output_length_terminated_count`; length stops are distinct from
input truncation. Confirm `grader_length_terminated_count` is zero and use
observed latency to check that the client job walltime suits a full run.

## Full baseline runs

Submit the first full pass without the smoke-test question filter:

```bash
AA_LCR_REPEAT_ID=1 AA_LCR_MAX_ROWS=0 \
bash jarvis/run_aa_lcr.sh direct_262k \
  --grader-prompt-version aa_lcr_equality_v1.1 \
  --grader-context-window 32768
```

Wait for this job to finish and verify all 100 rows, zero API errors, and zero
invalid grades before interpreting its score. For three independent passes,
repeat that command with `AA_LCR_REPEAT_ID=2`, then `3`, waiting between jobs so
they do not compete for the same services. Retain every pass and average them;
do not select the best run. Automatic run IDs prevent the smoke test and repeats
from overwriting one another. Direct cells use a CPU client job.

Artifacts are written to:

```text
benchmark_artifacts/aa_lcr/<experiment>/<run_id>/
```

Each run contains predictions, JSONL/CSV bridge rows, a manifest, the equality evaluation report, and hybrid cache state where applicable.

Default run IDs include dataset version, output allowance, repeat ID, and a
timestamp. `AA_LCR_MAX_OUTPUT_TOKENS`, `AA_LCR_RUN_ID`, `AA_LCR_REPEAT_ID`, and
`AA_LCR_SERVING_METADATA` are forwarded to the runner. Termination reasons and
length stops are reported separately from API errors and input truncation.

## Regrade and investigate the accuracy gap

The baseline above answers the generation question with a local grader. To
separate grader, prompt, answer-key, and output-budget effects, follow
[Regrade saved answers first](../aa_lcr_rerun_commands.md) and the condition
matrix in the [rerun plan](../aa_lcr_rerun_plan_20260908.md). Start with the four
percentage-equivalence cases, then grade all 100 historical answers and the
new answers under matching conditions. Hosted grading requires API access and
credentials; no hosted request is made by the commands above.

Use `aa_lcr.compare_conditions` for these comparisons and repeat aggregation.
If the old executor's serving settings cannot be recovered, also run a fresh
512-token control on the same service: repeat the full baseline command with
`AA_LCR_MAX_OUTPUT_TOKENS=512` prefixed. Grade both output budgets with the same
keys, prompt, and checker. Historical serving differences otherwise remain a
possible confounder. Keep 64K retrieval experiments for the later budget
redesign; the existing 60,000-input budget cannot accommodate 16K output.

## Optional: compare the historical four complete runs

The comparison rejects mismatched data, row order, prompts, models, budgets, API errors, or invalid grades:

The historical runs below share v1.0.0 keys, the legacy local grader prompt,
and a 512-token output budget. New defaults give the 262K and 64K cells different
output budgets, so their results will not pass this strict four-cell comparison.
Use the diagnostic comparator for deliberately different conditions.

```bash
cd /home/edogu/adarsh-rlms

uv run python -m aa_lcr.compare_runs \
  --direct-262k benchmark_artifacts/aa_lcr/direct_262k/20260830T235804Z \
  --hybrid-262k benchmark_artifacts/aa_lcr/hybrid_262k/20260830T191732Z \
  --direct-64k benchmark_artifacts/aa_lcr/direct_64k/20260830T193428Z \
  --hybrid-64k benchmark_artifacts/aa_lcr/hybrid_64k/20260830T194827Z
```

The output includes per-cell accuracy, paired deltas, 64K reasoning loss, hybrid retention, category and route breakdowns, operational totals, document-set-clustered 95% bootstrap intervals, and every accepted semantic hit for manual audit.
