# Jarvis HPC Setup and Validation Runbook

This is the operational checklist for preparing local vLLM services and the
benchmark client on Jarvis. Run these commands from the Jarvis login node unless
the step explicitly says otherwise. The commands below assume your current
directory is the parent directory that contains the `adarsh-rlms` repo.

After the production services pass the checks here, continue with
[`HPC_RUNBOOK_EXPERIMENT.md`](HPC_RUNBOOK_EXPERIMENT.md).

## 0. Push Locally, Pull Remotely

On your local machine, commit and push the changes:

```bash
git status --short
git diff --check
git add \
  long_bench_v2/qwen_prompt.py \
  long_bench_v2/run_benchmark.py \
  long_bench_v2/run_api_benchmark.py \
  test/test_long_bench_v2_hybrid.py \
  test/test_long_bench_v2_api_benchmark.py \
  test/test_semantic_cache_llm_provider.py \
  docs/system_architecture.md \
  docs/longbench_v2_hierarchical_retrieval_plan_20260808.md \
  long_bench_v2/docs/longbench_v2.md \
  docs/jarvis/README.md \
  docs/jarvis/jarvis.md \
  docs/jarvis/HPC_RUNBOOK_SETUP.md \
  docs/jarvis/HPC_RUNBOOK_EXPERIMENT.md
git add -u docs/jarvis
git diff --cached --check
git commit -m "Enforce decoder choices and split Jarvis runbooks"
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
client-gpu:          30 GB free required
```

The `client` line above is a scratch free-space check, not the Slurm RAM
allocation. The dispatcher submits benchmark client jobs with 32 GB RAM. Smaller
token chunks and higher overlap increase chunk count, duplicated chunk text,
tokenizer offset maps, embeddings, metadata, and FAISS state held by the client.
If a smaller chunk profile is OOM-killed, set
`SEMANTIC_CACHE_EMBEDDING_BATCH_SIZE=1`, use `submit client-gpu` for CUDA
embeddings, raise the client Slurm memory, or use a less aggressive chunk-count
profile.

Override only if you know the model is already cached or you intentionally want a
lower threshold:

```bash
MIN_LOCAL_FREE_GB=180 bash adarsh-rlms/jarvis/run.sh submit evaluator
```

Disable only for debugging:

```bash
SKIP_LOCAL_SPACE_CHECK=1 bash adarsh-rlms/jarvis/run.sh submit smoke
```

### Project Storage If Access Is Granted

If you later get permission to create `/mmfs1/project/llm_caching`, you can
switch back to durable project storage. In that mode, model caches, vLLM caches,
logs, and semantic-cache state live under `/mmfs1/project/llm_caching`, while
active per-job scratch still uses `/local`.

Create the project directories once:

```bash
mkdir -p /mmfs1/project/llm_caching/{hf_cache,vllm_cache,logs,cache_state}
```

Then use project mode before submitting jobs:

```bash
export JARVIS_STORAGE_MODE=project
export PROJECT_CACHE_ROOT=/mmfs1/project/llm_caching
unset PROJECT_LOG_DIR
unset JARVIS_CACHE_STATE_ROOT
```

With project mode, the same `adarsh-rlms/jarvis/run.sh submit ...` commands work.
Do not run the node-local cleanup command against `/mmfs1/project/llm_caching`;
that cleanup script is intended for guarded `/local` paths only.

## 2. Prepare Authentication

The default models may require Hugging Face access approval. Best practice on a
shared cluster is to keep tokens out of committed files. This repo's `.gitignore`
already excludes `.env`, and the Jarvis scripts can read `HF_TOKEN` from the repo
`.env` without sourcing the whole file as shell code.

Add this to `adarsh-rlms/.env`:

```bash
HF_TOKEN=<your-hugging-face-token>
```

Then confirm the scripts can see it without printing the token:

```bash
test -n "${HF_TOKEN:-}" && echo "HF_TOKEN is exported" || grep -q '^HF_TOKEN=' adarsh-rlms/.env && echo "HF_TOKEN is in .env"
```

If both are set, the exported shell variable wins. To disable `.env` loading for
Jarvis scripts, set `JARVIS_LOAD_DOTENV=0`.

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

uv pip install "vllm==0.19.1" --torch-backend=cu128

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

uv pip install "vllm==0.19.1" --torch-backend=cu128

python -c "import sys, vllm; print(sys.version); print(vllm.__version__)"
```

Do not pass `MODULES="cuda"`; that module does not exist. The commands below
assume the preferred user-local Python 3.12 venv, so they load only the CUDA
module. If you created the venv from the `python/3.11.10` module instead, prepend
`python/3.11.10` to `MODULES`.

If you previously installed vLLM with `uv pip install vllm --torch-backend=auto`
and see `ImportError: libcudart.so.13`, remove and recreate
`/home/edogu/.venvs/adarsh-vllm` with the pinned CUDA 12.8 install command
above. Recent vLLM releases default to CUDA 13 builds; Jarvis currently exposes
CUDA 12.x modules.

## 4. Run A Slurm Dry Run

This verifies Slurm submission, path creation, logs, environment export, and
cleanup without starting vLLM:

```bash
MODULES="cuda12.8/toolkit/12.8.1" \
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

If you used the site Python module fallback, use
`MODULES="python/3.11.10 cuda12.8/toolkit/12.8.1"` instead.

## 5. Optional Hugging Face Access Check

In scratch mode, CPU download jobs do not seed the future GPU nodes because
`/local` is node-local. Use this only as an authentication/network check. The
real vLLM service jobs may still download weights when they start on their GPU
nodes.

```bash
VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash adarsh-rlms/jarvis/run.sh submit download-small-smoke
```

If you used the site Python module fallback, add
`MODULES="python/3.11.10"` to this CPU download command.

Monitor:

```bash
squeue -u "$USER"
tail -f "$PROJECT_LOG_DIR"/rlms-download-<job_id>.out
```

The service commands below set `SYNC_BACK_MODELS=1`. With the default scratch
settings, vLLM writes model weights directly to the node-local
`/local/$USER/llm_caching` cache; if you later switch to a per-job cache layout,
that flag preserves the same sync-back behavior before cleanup.

## 6. Run The 7B Small-Smoke Service

Start one vLLM service with `Qwen/Qwen2.5-7B-Instruct`:

```bash
MODULES="cuda12.8/toolkit/12.8.1" \
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

## 7. Prepare The Client Environment

The vLLM service environment and benchmark client environment can be different.
The service environment needs `vllm`; the client environment needs this repo's
dependencies. Prepare this before running client smoke tests or benchmarks.

From the parent directory, prepare the client environment inside the repo:

```bash
deactivate # if the vLLM service is still running, e.g. you see something like (adarsh-vllm) in the terminal

cd adarsh-rlms

uv sync --python /home/edogu/.local/bin/python3.13
uv pip install transformers torch faiss-cpu sentence-transformers anthropic python-dotenv numpy scikit-learn

uv run python - <<'PY'
import semantic_cache_system
print("semantic_cache_system import ok")
PY

cd ..
```

## 8. Run Small-Smoke Client Checks

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

Run a tiny LongBench-v2 plumbing check against the same endpoint. The client
command first creates a bounded random source-linked sample; this is preferable
to `--max-rows` on the full CSV because the first LongBench rows can be very
large.

```bash
LLM_PROVIDER=openai_compatible \
OPENAI_COMPAT_EXECUTOR_BASE_URL="$SMALL_SMOKE_URL" \
OPENAI_COMPAT_EVALUATOR_BASE_URL="$SMALL_SMOKE_URL" \
WAIT_FOR_ENDPOINTS=1 \
CLIENT_CMD='export SEMANTIC_CACHE_SEARCH_MODE=iterative
export SEMANTIC_CACHE_EMBEDDING_QUERY_INSTRUCTION="Given a long-context multiple-choice question, retrieve chunks containing evidence, demonstrations, mappings, or facts needed to answer it."
export SEMANTIC_CACHE_EMBEDDING_BATCH_SIZE=1
export SEMANTIC_CACHE_EMBEDDING_MAX_LENGTH=8192
export SEMANTIC_CACHE_DOC_CHUNK_SIZE=10000
export SEMANTIC_CACHE_DOC_CHUNK_OVERLAP=1000
export SEMANTIC_CACHE_SCAN_MIN_CHUNK_RATIO=0.30
export SEMANTIC_CACHE_SCAN_MAX_CHUNK_RATIO=1.0
export SEMANTIC_CACHE_SCAN_MIN_CHUNKS=3
export SEMANTIC_CACHE_SCAN_MAX_CHUNKS=0
export SEMANTIC_CACHE_SCAN_MAX_TOKENS=768
export SEMANTIC_CACHE_SCAN_EMPTY_LEDGER_FALLBACK_RATIO=1.0
export SEMANTIC_CACHE_ITERATIVE_PACKED_FALLBACK_INPUT_TOKEN_BUDGET=60000
export SEMANTIC_CACHE_ITERATIVE_MEMORY_MAX_CHARS=16000
export SEMANTIC_CACHE_ITERATIVE_BATCH_MAX_CHUNKS=3
export SEMANTIC_CACHE_ITERATIVE_BATCH_INPUT_TOKEN_BUDGET=50000
export SEMANTIC_CACHE_MCQ_SYNTHESIS_MAX_TOKENS=8

uv run python long_bench_v2/sample_csv.py \
  --input-path benchmark_data/long_bench_v2/data_cache_suite.csv \
  --output-path benchmark_artifacts/longbench_v2_samples/small_smoke.csv \
  --sample-size 1 \
  --max-token-count 75000 \
  --selection-strategy random \
  --seed 0 && \
uv run python long_bench_v2/run_benchmark.py \
  --suite-csv benchmark_artifacts/longbench_v2_samples/small_smoke.csv \
  --llm-provider openai_compatible \
  --executor-model Qwen/Qwen2.5-7B-Instruct \
  --evaluator-model Qwen/Qwen2.5-7B-Instruct \
  --mode cache \
  --cache-state-root "$JARVIS_CACHE_STATE_ROOT" \
  --row-types original,exact,semantic \
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

## 9. Optional 24B One-Service Smoke Test

After `small-smoke` passes, you can run the Mistral 24B one-service smoke path
before starting both production services. This path now uses
`mistralai/Mistral-Small-24B-Instruct-2501` because the newer
`mistralai/Mistral-Small-3.2-24B-Instruct-2506` resolves as a Pixtral/multimodal
architecture in vLLM and failed on Jarvis with a
`MistralCommonPixtralProcessor` startup error.

```bash
MODULES="cuda12.8/toolkit/12.8.1" \
SYNC_BACK_MODELS=1 VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash adarsh-rlms/jarvis/run.sh submit smoke
```

Wait until the log shows:

```text
Application startup complete.
```

Then read the endpoint URL and verify `/v1/models`:

```bash
SMOKE_URL=$(cat "$PROJECT_LOG_DIR"/smoke-<job_id>.url)
echo "$SMOKE_URL"
curl "$SMOKE_URL/models"
```

The model response should include:

```text
mistralai/Mistral-Small-24B-Instruct-2501
```

Run the provider smoke test with both roles pointing at the one 24B endpoint:

```bash
LLM_PROVIDER=openai_compatible \
OPENAI_COMPAT_BASE_URL="$SMOKE_URL" \
OPENAI_COMPAT_EXECUTOR_BASE_URL="$SMOKE_URL" \
OPENAI_COMPAT_EVALUATOR_BASE_URL="$SMOKE_URL" \
OPENAI_COMPAT_EXECUTOR_MODEL=mistralai/Mistral-Small-24B-Instruct-2501 \
OPENAI_COMPAT_EVALUATOR_MODEL=mistralai/Mistral-Small-24B-Instruct-2501 \
WAIT_FOR_ENDPOINTS=1 \
CLIENT_CMD="uv run python -m unittest discover -s test -p test_semantic_cache_llm_provider.py" \
  bash adarsh-rlms/jarvis/run.sh submit client
```

That job should report:

```text
Ran 9 tests
OK
```

Stop the one-service smoke job after these checks unless you want to run another
tiny benchmark against it:

```bash
scancel <smoke_job_id>
```

If you explicitly override `SMOKE_MODEL` or `EVALUATOR_MODEL` back to the 3.2
model and it fails with `MistralCommonImageProcessor` or
`MistralCommonPixtralProcessor`, it is a Mistral/vLLM processor dependency
issue, not a Jarvis storage or Slurm issue. First update the Mistral processor
dependency inside the vLLM venv:

```bash
source /home/edogu/.venvs/adarsh-vllm/bin/activate
uv pip install --upgrade "mistral_common>=1.6.2"
python -c "import mistral_common; print(mistral_common.__version__)"
```

Then resubmit with an explicit override only if you still want to test 3.2:

```bash
MODULES="cuda12.8/toolkit/12.8.1" \
SMOKE_MODEL=mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
SYNC_BACK_MODELS=1 VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash adarsh-rlms/jarvis/run.sh submit smoke
```

Step 9 is optional; if `small-smoke` and the tiny client benchmark already
passed, you can skip it.

## 10. Start The Two Production Services

Start the executor service:

```bash
MODULES="cuda12.8/toolkit/12.8.1" \
EXECUTOR_MODEL=Qwen/Qwen3.6-35B-A3B \
EXECUTOR_TP_SIZE=4 \
EXECUTOR_MAX_MODEL_LEN=262144 \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
VLLM_EXTRA_ARGS="--reasoning-parser qwen3 --language-model-only --max-num-seqs 1 --enable-chunked-prefill --max-num-batched-tokens 8192" \
SYNC_BACK_MODELS=1 VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash adarsh-rlms/jarvis/run.sh submit executor
```

Start the evaluator service:

```bash
MODULES="cuda12.8/toolkit/12.8.1" \
EVALUATOR_MODEL=Qwen/Qwen3.5-35B-A3B \
EVALUATOR_TP_SIZE=2 \
EVALUATOR_MAX_MODEL_LEN=16384 \
EVALUATOR_PORT=8011 \
VLLM_EXTRA_ARGS="--reasoning-parser qwen3 --language-model-only" \
SYNC_BACK_MODELS=1 VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash adarsh-rlms/jarvis/run.sh submit evaluator
```

This Qwen profile is explicit on purpose: it does not change the repository
defaults. `Qwen/Qwen3.6-35B-A3B` is the executor candidate, and
`Qwen/Qwen3.5-35B-A3B` is the evaluator candidate. Both are run as text-only
non-thinking services for the LongBench-v2 MCQ path. If the evaluator has
serving or output-format issues, use `Qwen/Qwen3-30B-A3B-Instruct-2507` as the
fallback evaluator with the same `EVALUATOR_MAX_MODEL_LEN`.

The executor uses Qwen3.6's native `EXECUTOR_MAX_MODEL_LEN=262144` window. Keep
the evaluator at `16384`; evaluator calls are short semantic-equivalence and
routing checks, not full synthesis prompts. This command affects only a newly
submitted executor job; it does not alter an already-running service.

Watch both jobs:

```bash
squeue -u "$USER"
tail -f "$PROJECT_LOG_DIR"/rlms-executor-<executor_job_id>.out
tail -f "$PROJECT_LOG_DIR"/rlms-evaluator-<evaluator_job_id>.out
```

Before using the executor, verify its startup log reports all of the following:

```text
Available KV cache memory: ...
GPU KV cache size: ... tokens
Maximum concurrency for 262,144 tokens per request: X.XXx
Application startup complete.
```

Require `X.XX` to be at least `1.00` and reject a startup with a CUDA OOM or
engine-initialization failure. For Qwen's hybrid attention layout, use the
explicit maximum-concurrency line as the capacity check rather than dividing
the displayed GPU KV cache token count by the context length.

Read the endpoint files:

```bash
EXECUTOR_URL=$(cat "$PROJECT_LOG_DIR"/executor-<executor_job_id>.url)
EVALUATOR_URL=$(cat "$PROJECT_LOG_DIR"/evaluator-<evaluator_job_id>.url)

echo "$EXECUTOR_URL"
echo "$EVALUATOR_URL"
```

Example:

```bash
EXECUTOR_URL=$(cat "$PROJECT_LOG_DIR"/executor-1123664.url)
EVALUATOR_URL=$(cat "$PROJECT_LOG_DIR"/evaluator-1123665.url)

echo "$EXECUTOR_URL"
echo "$EVALUATOR_URL"
```

The expected shape is:

```text
executor:  http://<executor-node>:8000/v1
evaluator: http://<evaluator-node>:8001/v1
```

## 11. Stop Services After The Experiment

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

## 12. Clean Node-Local Scratch

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

## 13. Common Failure Checks

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

If a benchmark client job is OOM-killed:

```bash
sacct -j <client_job_id> --format=JobID,JobName,State,ExitCode,MaxRSS,ReqMem,Elapsed
tail -n 200 "$PROJECT_LOG_DIR"/rlms-client-<client_job_id>.out
```

Likely causes:

- The client Slurm allocation is still the dispatcher default of 32 GB.
- The local embedding model is still using the default embedding batch size of
  16 chunks per CPU forward pass.
- `SEMANTIC_CACHE_DOC_CHUNK_TOKENS` is low and
  `SEMANTIC_CACHE_DOC_CHUNK_OVERLAP_TOKENS` is high, increasing chunk count and
  duplicated text.
- The sampled suite includes 50k-200k token contexts, so ingest, tokenizer
  offsets, embeddings, metadata, and FAISS state are all larger.

Rerun only the client job with `SEMANTIC_CACHE_EMBEDDING_BATCH_SIZE=1`,
`SEMANTIC_CACHE_EMBEDDING_DEVICE=cuda` through `submit client-gpu`, a larger
`CLIENT_MEM` value, or a less chunk-heavy profile such as `12000/3000` or
`20000/4000`. If the traceback still points inside `EmbeddingEngine.encode` after
batch size 1 on GPU, lower `SEMANTIC_CACHE_EMBEDDING_MAX_LENGTH` to `4096` as a
memory tradeoff. The vLLM service jobs can stay running.

If the benchmark imports fail, fix the client environment and rerun only the
client job. The vLLM service jobs can stay running.
