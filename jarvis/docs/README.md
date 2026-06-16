# Jarvis L40S Runbook

This directory contains Jarvis-specific scripts for serving local models with
vLLM and running semantic-cache benchmark clients against those services.

## Script Layout

```text
jarvis/run.sh             User-facing dispatcher. Calls sbatch with the right resources.
jarvis/lib/env.sh         Shared paths, cache env vars, venv activation, and cleanup.
jarvis/serve_vllm.sh      Starts small-smoke/executor/evaluator/smoke vLLM services inside GPU jobs.
jarvis/run_client.sh      Runs benchmark/client commands inside CPU jobs.
jarvis/download_models.sh Optional model prefetch job for Hugging Face weights.
jarvis/cleanup_local.sh   Inspects or cleans guarded node-local scratch paths.
```

Use `adarsh-rlms/jarvis/run.sh` from the login node when your current directory
is the parent directory that contains the repo. If you `cd` directly into the
repo, use `jarvis/run.sh` instead. The other scripts are role scripts that
`run.sh` submits or delegates to inside Slurm allocations.

For the exact first-run sequence on the cluster, use
[`jarvis/docs/HPC_RUNBOOK.md`](HPC_RUNBOOK.md).

## Storage Policy

There are two storage modes. Scratch mode is the default because this account
cannot currently create `/mmfs1/project/llm_caching`.

Use project mode only if you have permission to create the project directory:

```text
JARVIS_STORAGE_MODE=project
Persistent cache:  /mmfs1/project/llm_caching
Runtime scratch:   /local/$USER/$SLURM_JOB_ID/adarsh-rlms
```

If you cannot create `/mmfs1/project/llm_caching`, use scratch mode:

```text
JARVIS_STORAGE_MODE=scratch
Node-local cache: /local/$USER/llm_caching
Runtime scratch:  /local/$USER/$SLURM_JOB_ID/adarsh-rlms
Small logs:       /home/edogu/adarsh-rlms-logs
```

In default scratch mode, the scripts point Hugging Face and vLLM caches at
`/local/$USER/llm_caching`, then clean only the per-job
`/local/$USER/$SLURM_JOB_ID/adarsh-rlms` directory on exit. This lets a later
job on the same node reuse model files.

In project mode, or if `SCRATCH_SHARED_NODE_CACHE=0`, the scripts can stage
runtime cache files under the per-job `/local` directory and sync back to the
configured cache root during cleanup.

In scratch mode, `/local/$USER/llm_caching` is node-local. It may be reused by
later jobs on the same node, but it is not shared across nodes and may be purged
by cluster policy.

In scratch mode, benchmark cache state defaults to per-job `/local` and is
cleaned with the client job. For warm cache reuse across separate client jobs,
set `JARVIS_CACHE_STATE_ROOT` to a small persistent path such as
`/home/edogu/adarsh-rlms-cache-state`. Do not use `/home` for model weights.

Before model-serving or download work starts, the scripts check free space on the
allocated node. Default minimums are 250 GB for the executor, 60 GB for
`small-smoke`, 160 GB for the evaluator/smoke service, 350 GB for
`download-all`, and 30 GB for client jobs.
These thresholds are larger than the model weights because first startup can
also need partial download files and cache overhead. Override with
`MIN_LOCAL_FREE_GB=<gb>` only when the model is already cached or you have
inspected the node manually.

Expected persistent storage for the current explicit Qwen LongBench-v2 profile:

```text
Qwen3.6 35B-A3B executor:   about 72 GB, https://huggingface.co/Qwen/Qwen3.6-35B-A3B
Qwen3.5 35B-A3B evaluator:  about 72 GB, https://huggingface.co/Qwen/Qwen3.5-35B-A3B
Qwen3 30B-A3B fallback:     about 61 GB, https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
Qwen2.5 7B smoke model:     about 15-25 GB, https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
Qwen embed/reranker:        about 2-4 GB, https://huggingface.co/Qwen/Qwen3-Embedding-0.6B and https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
Qwen executor+evaluator:    about 145 GB before cache overhead
Comfortable cache size:     about 300 GB
Room for fallback variants: about 500 GB
```

Do not place model weights under `/home`, because `/home` is backed by
`/mmfs1/home` and Jarvis best practices warn against large permanent files there.
Small Slurm logs and endpoint URL files are fine under `/home`.

## Install vLLM

Use a user-level virtual environment. Do not use `sudo`, and do not install into
the system Python. If Jarvis admins prefer project storage for Python
environments, use `/mmfs1/project/llm_caching/venvs/adarsh-vllm` instead of the
`/home/edogu` path below.

Installing the vLLM package under `/home/edogu` is acceptable only as a
user-owned software environment. Keep model weights and Hugging Face caches out
of `/home`; `jarvis/run.sh` points those caches at `/local` during jobs.

Python 3.9.18 is not enough for the current setup:

- Current vLLM releases require Python 3.10+.
- This repository currently has `.python-version` set to Python 3.13.

Recommended setup with `uv` and Python 3.12. This path does not require a
system Python module:

```bash
ssh edogu@jarvis.stevens.edu
module avail 2>&1 | grep -Ei 'python|cuda|gcc|anaconda|conda' || true

# Installs uv under your user account if it is not already available.
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

On the current Jarvis module listing, `python/3.11.10` is available and is a
reasonable vLLM environment base if `uv python install 3.12` is unavailable or
you prefer a site module:

```bash
ssh edogu@jarvis.stevens.edu
module load python/3.11.10
mkdir -p /home/edogu/.venvs
uv venv /home/edogu/.venvs/adarsh-vllm --python 3.11 --seed
source /home/edogu/.venvs/adarsh-vllm/bin/activate
uv pip install --upgrade pip

uv pip install "vllm==0.19.1" --torch-backend=cu128

python -c "import sys, vllm; print(sys.version); print(vllm.__version__)"
```

Fallback setup if `uv` is not available but a Python 3.10+ module is available:

```bash
ssh edogu@jarvis.stevens.edu
# Example only; replace with the exact module name shown by `module avail`.
module load <python-3.10-or-newer-module>

python -m venv /home/edogu/.venvs/adarsh-vllm
source /home/edogu/.venvs/adarsh-vllm/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install "vllm==0.19.1" --extra-index-url https://download.pytorch.org/whl/cu128

python -c "import sys, vllm; print(sys.version); print(vllm.__version__)"
```

vLLM should be installed in a Python 3.12 environment unless Jarvis provides
another vLLM-compatible Python 3.10+ stack. The vLLM service and the benchmark
client do not need to run from the same Python environment. The benchmark client
should use the repository's Python 3.13 environment, or another Python 3.10+
environment with the repository dependencies installed.

Do not pass `MODULES="cuda"`; that module does not exist. If you created the
venv from the `python/3.11.10` module, load that same module inside Slurm before
the venv is activated. If a CUDA toolkit module is needed inside the Slurm job,
use exact Jarvis module names such as
`MODULES="python/3.11.10 cuda12.8/toolkit/12.8.1"`. If you used
`uv python install 3.12`, omit `python/3.11.10` from `MODULES`. The vLLM/PyTorch
wheels may also work with the NVIDIA driver on GPU nodes without a CUDA module.
If you see `ImportError: libcudart.so.13`, recreate the vLLM venv with the CUDA
12.8 install above. Recent vLLM releases default to CUDA 13 builds; Jarvis
currently exposes CUDA 12.x modules.

When submitting vLLM service jobs, point the dispatcher at the environment:

```bash
VLLM_VENV=/home/edogu/.venvs/adarsh-vllm bash adarsh-rlms/jarvis/run.sh submit executor
VLLM_VENV=/home/edogu/.venvs/adarsh-vllm bash adarsh-rlms/jarvis/run.sh submit evaluator
```

## Remote Preflight

Before the first Slurm run on Jarvis, use scratch mode if you cannot create the
project directory:

```bash
cd /path/to/parent-directory
export JARVIS_STORAGE_MODE=scratch
export PROJECT_LOG_DIR=/home/edogu/adarsh-rlms-logs
mkdir -p "$PROJECT_LOG_DIR"
```

Make sure the Slurm job can read gated Hugging Face models. The scripts can load
`HF_TOKEN` from `adarsh-rlms/.env` if it is not already exported:

```bash
HF_TOKEN=<your-hugging-face-token>
```

Exporting `HF_TOKEN` in the shell still works and takes precedence. Set
`JARVIS_LOAD_DOTENV=0` to disable `.env` loading.

Also prepare a client environment for the benchmark code. The service venv only
needs vLLM; the client needs this repository's runtime dependencies:

```bash
cd adarsh-rlms
uv python install 3.13
uv sync --python 3.13
uv pip install transformers torch faiss-cpu sentence-transformers anthropic python-dotenv numpy scikit-learn
cd ..
```

If you use a manually activated client venv instead of `uv run`, install the same
packages there and set `CLIENT_CMD` to start with `python ...`.

## vLLM Serving Defaults

The scripts set explicit context limits so vLLM does not try to reserve memory
for the full advertised context window of the model:

```text
executor:  --max-model-len 32768
evaluator: --max-model-len 16384
small-smoke: --max-model-len 8192
smoke:     --max-model-len 8192
```

Override these only when the benchmark needs it and the service has enough KV
cache headroom:

```bash
EXECUTOR_MAX_MODEL_LEN=65536 bash adarsh-rlms/jarvis/run.sh submit executor
EVALUATOR_MAX_MODEL_LEN=32768 bash adarsh-rlms/jarvis/run.sh submit evaluator
```

The shared vLLM defaults are:

```text
VLLM_DTYPE=auto
VLLM_GPU_MEMORY_UTILIZATION=0.90
```

## Modes

From the parent directory that contains the repo, use `adarsh-rlms/jarvis/run.sh`
as a dispatcher:

```bash
bash adarsh-rlms/jarvis/run.sh submit small-smoke
bash adarsh-rlms/jarvis/run.sh submit executor
bash adarsh-rlms/jarvis/run.sh submit evaluator
bash adarsh-rlms/jarvis/run.sh submit smoke
bash adarsh-rlms/jarvis/run.sh submit client
bash adarsh-rlms/jarvis/run.sh submit download-small-smoke
bash adarsh-rlms/jarvis/run.sh submit download-all
bash adarsh-rlms/jarvis/run.sh submit cleanup
```

The submit helper applies the expected Slurm resources:

```text
small-smoke: gpu-l40s, gpu:l40s:1, Qwen2.5 7B, tensor parallel 1
executor:  gpu-l40s, gpu:l40s:4, Llama 3.3 70B, tensor parallel 4
evaluator: gpu-l40s, gpu:l40s:2, Mistral Small 24B, tensor parallel 2
smoke:     gpu-l40s, gpu:l40s:2, one shared endpoint
client:    compute-short, no GPU, benchmark/client command only
download:  compute-short, no GPU, prefetch model weights into the configured cache root
cleanup:   gpu-l40s by default, inspect or clean node-local Jarvis scratch
```

For Mistral Small services, `serve_vllm.sh` adds the Mistral tokenizer/config
flags recommended by the Hugging Face model card. The Jarvis default is
`mistralai/Mistral-Small-24B-Instruct-2501`; the newer 3.2 model resolved as a
Pixtral/multimodal architecture on Jarvis and failed during processor startup.

Prefer the dispatcher above. If you submit role scripts manually, pass
`JARVIS_SCRIPT_DIR`; otherwise Slurm's spool copy of the script cannot find
`lib/env.sh`:

```bash
sbatch --partition=gpu-l40s --gres=gpu:l40s:1 \
  --export=ALL,JARVIS_SCRIPT_DIR="$PWD/adarsh-rlms/jarvis",MODE=small-smoke \
  adarsh-rlms/jarvis/serve_vllm.sh

sbatch --partition=compute-short \
  --export=ALL,JARVIS_SCRIPT_DIR="$PWD/adarsh-rlms/jarvis",MODE=client \
  adarsh-rlms/jarvis/run_client.sh
```

## Endpoint Wiring

Each vLLM service writes its advertised URL to:

```text
$PROJECT_LOG_DIR/<mode>-<job_id>.url
```

For serious benchmark runs, start two services and point the client at both:

```bash
export LLM_PROVIDER=openai_compatible
export OPENAI_COMPAT_EXECUTOR_MODEL=meta-llama/Llama-3.3-70B-Instruct
export OPENAI_COMPAT_EVALUATOR_MODEL=mistralai/Mistral-Small-24B-Instruct-2501
export OPENAI_COMPAT_EXECUTOR_BASE_URL=http://<executor-node>:8000/v1
export OPENAI_COMPAT_EVALUATOR_BASE_URL=http://<evaluator-node>:8001/v1
```

For a one-service smoke test, point both roles at the same endpoint:

```bash
export OPENAI_COMPAT_BASE_URL=http://<smoke-node>:8000/v1
export OPENAI_COMPAT_EXECUTOR_BASE_URL=$OPENAI_COMPAT_BASE_URL
export OPENAI_COMPAT_EVALUATOR_BASE_URL=$OPENAI_COMPAT_BASE_URL
```

For `small-smoke`, also point both roles at the served Qwen model:

```bash
export OPENAI_COMPAT_EXECUTOR_MODEL=Qwen/Qwen2.5-7B-Instruct
export OPENAI_COMPAT_EVALUATOR_MODEL=Qwen/Qwen2.5-7B-Instruct
```

Then submit a client job with the command you want to run:

```bash
WAIT_FOR_ENDPOINTS=1 \
CLIENT_CMD="uv run python -m unittest discover -s test -p test_semantic_cache_llm_provider.py" \
  bash adarsh-rlms/jarvis/run.sh submit client
```

`WAIT_FOR_ENDPOINTS=1` makes the client poll `/v1/models` on the executor and
evaluator endpoints before it starts the command. For benchmark runs, set
`CLIENT_CMD` to the desired benchmark command.

## Cache Hydration

By default in scratch mode, the role scripts use the node-local cache directly:
`/local/$USER/llm_caching/hf_cache`. That cache survives per-job cleanup and can
be reused by later jobs on the same node.

In project mode, or when `SCRATCH_SHARED_NODE_CACHE=0`, the scripts copy already
cached Hugging Face files into the per-job `/local` runtime cache with
`rsync --ignore-existing`.

In project mode, the durable source is `/mmfs1/project/llm_caching/hf_cache`.
In scratch mode, the reusable source is `/local/$USER/llm_caching/hf_cache` on
the same node. This does not hydrate jobs that land on different nodes.

vLLM can download models automatically on first service startup, so a separate
download job is not required for correctness. In scratch mode, CPU download jobs
are useful only for checking Hugging Face auth/network access because they write
to `/local` on the CPU node, not the later GPU node.

Prefetch both default models:

```bash
VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash adarsh-rlms/jarvis/run.sh submit download-all
```

Prefetch one model:

```bash
VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash adarsh-rlms/jarvis/run.sh submit download-executor

VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash adarsh-rlms/jarvis/run.sh submit download-evaluator
```

`download_models.sh` sets `SYNC_BACK_MODELS=1` by default. With the default
scratch settings, downloads go directly to the node-local cache root. If you
switch to a per-job runtime cache layout, the same flag syncs downloaded files
back to the configured cache root during cleanup.

You can also let a service job download and sync on first startup:

```bash
SYNC_BACK_MODELS=1 VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash adarsh-rlms/jarvis/run.sh submit executor
```

## Dry Runs

Dry runs validate path creation, environment variables, endpoint files, and cleanup
without starting vLLM:

```bash
DRY_RUN=1 MODE=executor PROJECT_CACHE_ROOT=/tmp/llm_caching_test \
  LOCAL_BASE=/tmp/llm_caching_local/adarsh-rlms \
  bash adarsh-rlms/jarvis/serve_vllm.sh
```

The cleanup trap removes the dry-run `LOCAL_BASE` only if it matches the guarded
scratch path pattern.

## Cleanup

Cleanup is node-local. Run it on the node you want to inspect:

```bash
JARVIS_STORAGE_MODE=scratch CLEANUP_NODE=<gpu-node> \
  bash adarsh-rlms/jarvis/run.sh submit cleanup
```

The cleanup script is a dry run by default. After reviewing the log, remove stale
per-job scratch with:

```bash
JARVIS_STORAGE_MODE=scratch CLEANUP_NODE=<gpu-node> CONFIRM_CLEANUP=1 \
  bash adarsh-rlms/jarvis/run.sh submit cleanup
```

To delete the node-local model cache too, add `CLEAN_NODE_CACHE=1`. Do this only
when you want future jobs on that node to download models again.
