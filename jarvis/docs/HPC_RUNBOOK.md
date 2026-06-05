# Jarvis HPC Step-By-Step Runbook

This is the operational checklist for running the local vLLM services and the
benchmark client on Jarvis. Run these commands from the Jarvis login node unless
the step explicitly says otherwise. The commands below assume your current
directory is the parent directory that contains the `adarsh-rlms` repo.

## 0. Push Locally, Pull Remotely

On your local machine, commit and push the changes:

```bash
git status
git add semantic_cache_system.py long_bench_v2/run_benchmark.py test jarvis
git commit -m "Add Jarvis vLLM local provider run scripts"
git push
```

On Jarvis:

```bash
ssh edogu@jarvis.stevens.edu
cd /path/to/parent-directory
git -C adarsh-rlms pull
```

If you instead `cd` directly into the repo, replace
`adarsh-rlms/jarvis/run.sh` with `jarvis/run.sh` in the commands below.

## 1. Choose Scratch Storage

Your account currently cannot create `/mmfs1/project/llm_caching`, so use
Jarvis scratch mode. This keeps large model files under `/local` and keeps only
small Slurm logs and endpoint URL files under your home directory.

```bash
export JARVIS_STORAGE_MODE=scratch
export PROJECT_LOG_DIR=/home/edogu/adarsh-rlms-logs
mkdir -p "$PROJECT_LOG_DIR"
```

The scripts will create these paths inside each Slurm job:

```text
/local/$USER/llm_caching/hf_cache                 node-local Hugging Face cache
/local/$USER/llm_caching/vllm_cache               node-local vLLM cache
/local/$USER/$SLURM_JOB_ID/adarsh-rlms            per-job active scratch
/home/edogu/adarsh-rlms-logs                      small logs and endpoint files
```

Important tradeoff: `/local` is node-local scratch, not shared cluster storage.
A model downloaded on one node is not guaranteed to exist on another node. This
is acceptable for first experiments, but it means the first startup on each GPU
node may download model weights again.

The cleanup trap removes only the per-job
`/local/$USER/$SLURM_JOB_ID/adarsh-rlms` directory. It does not delete
`/local/$USER/llm_caching`, so a later job on the same node can reuse files if
Jarvis has not purged that scratch area.

In scratch mode, `JARVIS_CACHE_STATE_ROOT` defaults to the per-job `/local`
directory and is cleaned when the client job exits. If you need semantic-cache
state to survive across separate client jobs, set a small persistent path before
submitting the client:

```bash
export JARVIS_CACHE_STATE_ROOT=/home/edogu/adarsh-rlms-cache-state
mkdir -p "$JARVIS_CACHE_STATE_ROOT"
```

Do this only for benchmark cache state, not model weights.

The scripts now check free space inside the allocated node before starting work
that can download models. These thresholds are not the model sizes; they include
the model weights, temporary/partial download files, tokenizer/config files,
vLLM/Hugging Face cache overhead, and a scratch buffer. Defaults:

```text
executor:           250 GB free required
small-smoke:         60 GB free required
evaluator/smoke:    160 GB free required
download-all:       350 GB free required
client:              30 GB free required
```

Override only if you know the model is already cached or you intentionally want a
lower threshold:

```bash
MIN_LOCAL_FREE_GB=180 bash adarsh-rlms/jarvis/run.sh submit evaluator
```

Disable only for debugging:

```bash
SKIP_LOCAL_SPACE_CHECK=1 bash adarsh-rlms/jarvis/run.sh submit smoke
```

## 2. Prepare Authentication

The default models may require Hugging Face access approval. Export `HF_TOKEN`
before submitting jobs so Slurm passes it through:

```bash
export HF_TOKEN=<your-hugging-face-token>
```

Confirm the token is present without printing it:

```bash
test -n "$HF_TOKEN" && echo "HF_TOKEN is set"
```

## 3. Install vLLM In Your User Environment

Use a private environment. This does not change system Python or affect other
users.

First check whether Jarvis exposes useful module names. Module names are
cluster-specific, and `python/3.12` or `cuda` may not exist:

```bash
module avail 2>&1 | grep -Ei 'python|cuda|gcc|anaconda|conda' || true
```

On the current Jarvis module listing, useful names include:

```text
python/3.11.10
cuda12.4/toolkit/12.4.1
cuda12.8/toolkit/12.8.1
gcc/13.1.0
```

The preferred path is still a user-local Python 3.12 with `uv`. This does not
change the system Python:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
if [ -f "$HOME/.local/bin/env" ]; then
  source "$HOME/.local/bin/env"
else
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

uv python install 3.12
mkdir -p /home/edogu/.venvs
uv venv /home/edogu/.venvs/adarsh-vllm --python 3.12 --seed
source /home/edogu/.venvs/adarsh-vllm/bin/activate
uv pip install --upgrade pip
uv pip install vllm --torch-backend=auto

python -c "import sys, vllm; print(sys.version); print(vllm.__version__)"
```

If `uv python install 3.12` is not available or you prefer the site module,
`python/3.11.10` is a reasonable vLLM environment base:

```bash
module load python/3.11.10
mkdir -p /home/edogu/.venvs
uv venv /home/edogu/.venvs/adarsh-vllm --python 3.11 --seed
source /home/edogu/.venvs/adarsh-vllm/bin/activate
uv pip install --upgrade pip
uv pip install vllm --torch-backend=auto

python -c "import sys, vllm; print(sys.version); print(vllm.__version__)"
```

Do not pass `MODULES="cuda"`; that module does not exist. If you created the
venv from the `python/3.11.10` module, load that same module inside Slurm before
the venv is activated. For GPU vLLM service jobs, use:

```bash
MODULES="python/3.11.10 cuda12.4/toolkit/12.4.1" \
DRY_RUN=1 VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash adarsh-rlms/jarvis/run.sh submit small-smoke
```

For CPU download jobs created from the same module-based venv, only the Python
module is needed:

```bash
MODULES="python/3.11.10" \
VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash adarsh-rlms/jarvis/run.sh submit download-small-smoke
```

If you used `uv python install 3.12` rather than the site Python module, you can
omit `python/3.11.10` from `MODULES`. The vLLM/PyTorch wheels may also work with
the NVIDIA driver on GPU nodes without a CUDA module.

## 4. Prepare The Client Environment

The vLLM service environment and benchmark client environment can be different.
The service environment needs `vllm`; the client environment needs this repo's
dependencies.

From the parent directory, prepare the client environment inside the repo:

```bash
cd adarsh-rlms
uv python install 3.13
uv sync --python 3.13
uv pip install transformers torch faiss-cpu sentence-transformers anthropic python-dotenv numpy scikit-learn
cd ..
```

Quick import check:

```bash
cd adarsh-rlms
uv run python - <<'PY'
import semantic_cache_system
print("semantic_cache_system import ok")
PY
cd ..
```

## 5. Run A Slurm Dry Run

This verifies Slurm submission, path creation, logs, environment export, and
cleanup without starting vLLM:

```bash
MODULES="python/3.11.10 cuda12.4/toolkit/12.4.1" \
DRY_RUN=1 VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash adarsh-rlms/jarvis/run.sh submit small-smoke
```

The command prints a Slurm job id. Watch it:

```bash
squeue -u "$USER"
tail -f "$PROJECT_LOG_DIR"/rlms-small-smoke-<job_id>.out
```

The dry-run log should show `/local/$USER/<job_id>/adarsh-rlms`, the selected
model, the endpoint file path, and the vLLM command that would have run.
It should also print a line like:

```text
[JARVIS] local free space at /local/...: 700 GB available; 60 GB required
```

## 6. Optional Hugging Face Access Check

In scratch mode, CPU download jobs do not seed the future GPU nodes because
`/local` is node-local. Use this only as an authentication/network check. The
real vLLM service jobs may still download weights when they start on their GPU
nodes.

```bash
MODULES="python/3.11.10" \
VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash adarsh-rlms/jarvis/run.sh submit download-small-smoke
```

Monitor:

```bash
squeue -u "$USER"
tail -f "$PROJECT_LOG_DIR"/rlms-download-<job_id>.out
```

The service commands below set `SYNC_BACK_MODELS=1`. With the default scratch
settings, vLLM writes model weights directly to the node-local
`/local/$USER/llm_caching` cache; if you later switch to a per-job cache layout,
that flag preserves the same sync-back behavior before cleanup.

## 7. Run The 7B Small-Smoke Service

Start one vLLM service with `Qwen/Qwen2.5-7B-Instruct`:

```bash
MODULES="python/3.11.10 cuda12.4/toolkit/12.4.1" \
SYNC_BACK_MODELS=1 VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash adarsh-rlms/jarvis/run.sh submit small-smoke
```

Find the endpoint after the job starts:

```bash
ls -ltr "$PROJECT_LOG_DIR"/*small-smoke*.url
cat "$PROJECT_LOG_DIR"/small-smoke-<job_id>.url
tail -f "$PROJECT_LOG_DIR"/rlms-small-smoke-<job_id>.out
```

The endpoint will look like:

```text
http://<small-smoke-node>:8000/v1
```

Submit a client smoke test against that endpoint:

```bash
SMALL_SMOKE_URL=http://<small-smoke-node>:8000/v1

LLM_PROVIDER=openai_compatible \
OPENAI_COMPAT_BASE_URL="$SMALL_SMOKE_URL" \
OPENAI_COMPAT_EXECUTOR_BASE_URL="$SMALL_SMOKE_URL" \
OPENAI_COMPAT_EVALUATOR_BASE_URL="$SMALL_SMOKE_URL" \
OPENAI_COMPAT_EXECUTOR_MODEL=Qwen/Qwen2.5-7B-Instruct \
OPENAI_COMPAT_EVALUATOR_MODEL=Qwen/Qwen2.5-7B-Instruct \
WAIT_FOR_ENDPOINTS=1 \
CLIENT_CMD="uv run python -m unittest discover -s test -p test_semantic_cache_llm_provider.py" \
  bash adarsh-rlms/jarvis/run.sh submit client
```

Monitor the client:

```bash
squeue -u "$USER"
tail -f "$PROJECT_LOG_DIR"/rlms-client-<job_id>.out
```

Run a tiny LongBench-v2 plumbing check against the same endpoint:

```bash
LLM_PROVIDER=openai_compatible \
OPENAI_COMPAT_EXECUTOR_BASE_URL="$SMALL_SMOKE_URL" \
OPENAI_COMPAT_EVALUATOR_BASE_URL="$SMALL_SMOKE_URL" \
WAIT_FOR_ENDPOINTS=1 \
CLIENT_CMD='uv run python long_bench_v2/run_benchmark.py \
  --llm-provider openai_compatible \
  --executor-model Qwen/Qwen2.5-7B-Instruct \
  --evaluator-model Qwen/Qwen2.5-7B-Instruct \
  --mode cache \
  --cache-state-root "$JARVIS_CACHE_STATE_ROOT" \
  --row-types original,exact,semantic \
  --max-rows 5 \
  --output-dir benchmark_artifacts \
  --manifest-note jarvis-l40s-small-smoke' \
  bash adarsh-rlms/jarvis/run.sh submit client
```

Treat this run as plumbing validation, not benchmark-quality accuracy.

Stop the small-smoke service when the provider test and tiny LongBench check are
done:

```bash
scancel <small_smoke_job_id>
```

## 8. Optional 24B One-Service Smoke Test

After `small-smoke` passes, you can run the existing Mistral 24B one-service
smoke path before starting both production services:

```bash
MODULES="python/3.11.10 cuda12.4/toolkit/12.4.1" \
SYNC_BACK_MODELS=1 VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash adarsh-rlms/jarvis/run.sh submit smoke
```

## 9. Start The Two Production Services

Start the executor service:

```bash
MODULES="python/3.11.10 cuda12.4/toolkit/12.4.1" \
SYNC_BACK_MODELS=1 VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash adarsh-rlms/jarvis/run.sh submit executor
```

Start the evaluator service:

```bash
MODULES="python/3.11.10 cuda12.4/toolkit/12.4.1" \
SYNC_BACK_MODELS=1 VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash adarsh-rlms/jarvis/run.sh submit evaluator
```

Watch both jobs:

```bash
squeue -u "$USER"
tail -f "$PROJECT_LOG_DIR"/rlms-executor-<executor_job_id>.out
tail -f "$PROJECT_LOG_DIR"/rlms-evaluator-<evaluator_job_id>.out
```

Read the endpoint files:

```bash
EXECUTOR_URL=$(cat "$PROJECT_LOG_DIR"/executor-<executor_job_id>.url)
EVALUATOR_URL=$(cat "$PROJECT_LOG_DIR"/evaluator-<evaluator_job_id>.url)

echo "$EXECUTOR_URL"
echo "$EVALUATOR_URL"
```

The expected shape is:

```text
executor:  http://<executor-node>:8000/v1
evaluator: http://<evaluator-node>:8001/v1
```

## 10. Run A Small Benchmark Client Job

Start with a capped LongBench-v2 run:

```bash
LLM_PROVIDER=openai_compatible \
OPENAI_COMPAT_EXECUTOR_BASE_URL="$EXECUTOR_URL" \
OPENAI_COMPAT_EVALUATOR_BASE_URL="$EVALUATOR_URL" \
WAIT_FOR_ENDPOINTS=1 \
CLIENT_CMD='uv run python long_bench_v2/run_benchmark.py \
  --llm-provider openai_compatible \
  --mode cache \
  --cache-state-root "$JARVIS_CACHE_STATE_ROOT" \
  --row-types original,exact,semantic \
  --max-rows 5 \
  --output-dir benchmark_artifacts \
  --manifest-note jarvis-l40s-small' \
  bash adarsh-rlms/jarvis/run.sh submit client
```

Monitor:

```bash
squeue -u "$USER"
tail -f "$PROJECT_LOG_DIR"/rlms-client-<client_job_id>.out
```

When this finishes, inspect the generated artifact paths printed in the client
log. They should point under `benchmark_artifacts/longbench_v2/...`.

## 11. Run The Full Benchmark

Use the same service URLs and remove the row cap:

```bash
LLM_PROVIDER=openai_compatible \
OPENAI_COMPAT_EXECUTOR_BASE_URL="$EXECUTOR_URL" \
OPENAI_COMPAT_EVALUATOR_BASE_URL="$EVALUATOR_URL" \
WAIT_FOR_ENDPOINTS=1 \
CLIENT_CMD='uv run python long_bench_v2/run_benchmark.py \
  --llm-provider openai_compatible \
  --mode cache \
  --cache-state-root "$JARVIS_CACHE_STATE_ROOT" \
  --row-types original,exact,semantic \
  --output-dir benchmark_artifacts \
  --manifest-note jarvis-l40s-full' \
  bash adarsh-rlms/jarvis/run.sh submit client
```

For a cold-cache rerun, add `--cache-reset` to `CLIENT_CMD`. Do not delete
`/local/$USER/llm_caching` unless you intentionally want to force model
downloads again on that node.

## 12. Stop Services After The Experiment

The vLLM service jobs are long-running servers. They do not stop automatically
when a client job finishes.

```bash
scancel <executor_job_id>
scancel <evaluator_job_id>
```

Confirm:

```bash
squeue -u "$USER"
```

## 13. Clean Node-Local Scratch

`/local` is per node, so cleanup must run on the node you want to clean. The
cleanup script defaults to dry-run behavior and removes nothing unless
`CONFIRM_CLEANUP=1` is set.

Inspect stale per-job scratch on a GPU node:

```bash
JARVIS_STORAGE_MODE=scratch \
CLEANUP_NODE=<gpu-node> \
  bash adarsh-rlms/jarvis/run.sh submit cleanup
```

Watch the cleanup log:

```bash
squeue -u "$USER"
tail -f "$PROJECT_LOG_DIR"/rlms-cleanup-<cleanup_job_id>.out
```

Remove stale per-job scratch after reviewing the dry run:

```bash
JARVIS_STORAGE_MODE=scratch \
CLEANUP_NODE=<gpu-node> \
CONFIRM_CLEANUP=1 \
  bash adarsh-rlms/jarvis/run.sh submit cleanup
```

Delete the node-local model cache too only when you intentionally want to force
future model downloads on that node:

```bash
JARVIS_STORAGE_MODE=scratch \
CLEANUP_NODE=<gpu-node> \
CONFIRM_CLEANUP=1 \
CLEAN_NODE_CACHE=1 \
  bash adarsh-rlms/jarvis/run.sh submit cleanup
```

The cleanup script is guarded to target only this project's paths:

```text
/local/$USER/<job_id>/adarsh-rlms
/local/$USER/llm_caching
```

Stop active vLLM jobs before deleting node-local model cache on their node.

## 14. Common Failure Checks

If a service never becomes reachable:

```bash
tail -n 200 "$PROJECT_LOG_DIR"/rlms-executor-<job_id>.out
tail -n 200 "$PROJECT_LOG_DIR"/rlms-evaluator-<job_id>.out
```

Likely causes:

- Missing `HF_TOKEN` or gated model access not approved.
- vLLM environment was not passed with `VLLM_VENV`.
- A site-specific CUDA or compiler module is required inside the Slurm job; use
  only the exact module name shown by `module avail` or recommended by admins.
- Local scratch has too little free space; run the cleanup script on that node
  or lower `MIN_LOCAL_FREE_GB` if the model is already cached.
- The model context length is too high for available GPU memory; lower
  `EXECUTOR_MAX_MODEL_LEN` or `EVALUATOR_MAX_MODEL_LEN`.
- The endpoint URL points to `127.0.0.1` from a different Slurm job. Use the
  hostname URL written to the `.url` file.

If the benchmark imports fail, fix the client environment and rerun only the
client job. The vLLM service jobs can stay running.
