# Jarvis L40S Runbook

This directory contains Jarvis-specific scripts for serving local models with
vLLM and running semantic-cache benchmark clients against those services.

## Script Layout

```text
jarvis/run.sh             User-facing dispatcher. Calls sbatch with the right resources.
jarvis/lib/env.sh         Shared paths, cache env vars, venv activation, and cleanup.
jarvis/serve_vllm.sh      Starts executor/evaluator/smoke vLLM services inside GPU jobs.
jarvis/run_client.sh      Runs benchmark/client commands inside CPU jobs.
jarvis/download_models.sh Optional model prefetch job for Hugging Face weights.
```

Use `jarvis/run.sh` from the login node. The other scripts are role scripts that
`run.sh` submits or delegates to inside Slurm allocations.

For the exact first-run sequence on the cluster, use
[`jarvis/HPC_RUNBOOK.md`](HPC_RUNBOOK.md).

## Storage Policy

Use project storage for durable model cache files and node-local scratch for
active serving:

```text
Persistent cache: /mmfs1/project/llm_caching
Runtime scratch:  /local/$USER/$SLURM_JOB_ID/adarsh-rlms
```

The run script creates:

```text
/mmfs1/project/llm_caching/hf_cache
/mmfs1/project/llm_caching/vllm_cache
/mmfs1/project/llm_caching/logs
```

It exports Hugging Face and vLLM cache variables to the per-job `/local` path,
then cleans only `/local/$USER/$SLURM_JOB_ID/adarsh-rlms` on exit. It never
deletes `/mmfs1/project/llm_caching`.

Expected persistent storage:

```text
Llama 3.3 70B BF16:      about 141 GB
Mistral Small 24B:       about 50 GB minimal, about 100 GB if full repo files are cached
Qwen embed/reranker:     about 2-4 GB
Comfortable cache size:  about 300 GB
Room for variants:       about 500 GB
```

Do not place these model caches under `/home`, because `/home` is backed by
`/mmfs1/home` and Jarvis best practices warn against large permanent files there.

## Install vLLM

Use a user-level virtual environment. Do not use `sudo`, and do not install into
the system Python. If Jarvis admins prefer project storage for Python
environments, use `/mmfs1/project/llm_caching/venvs/adarsh-vllm` instead of the
`/home/edogu` path below.

Installing the vLLM package under `/home/edogu` is acceptable only as a
user-owned software environment. Keep model weights and Hugging Face caches out
of `/home`; `jarvis/run.sh` points those caches at `/local` during jobs and can
sync model files to `/mmfs1/project/llm_caching`.

Python 3.9.18 is not enough for the current setup:

- Current vLLM releases require Python 3.10+.
- This repository currently has `.python-version` set to Python 3.13.

Recommended setup with `uv` and Python 3.12:

```bash
ssh edogu@jarvis.stevens.edu
module avail python
module avail cuda

# Load the closest available Python 3.12 and CUDA modules on Jarvis if present.
# Exact module names are cluster-specific. If no Python 3.12 module exists,
# use `uv python install 3.12` below.
module load python/3.12
module load cuda

mkdir -p /home/edogu/.venvs
uv venv /home/edogu/.venvs/adarsh-vllm --python 3.12 --seed
source /home/edogu/.venvs/adarsh-vllm/bin/activate
uv pip install --upgrade pip
uv pip install vllm --torch-backend=auto

python -c "import torch, vllm; print(torch.__version__); print(vllm.__version__); print(torch.cuda.is_available())"
```

If Jarvis only exposes Python 3.9.18, install a user-local Python 3.12 with
`uv` first:

```bash
ssh edogu@jarvis.stevens.edu
module load cuda

# Installs uv under your user account if it is not already available.
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

uv python install 3.12
mkdir -p /home/edogu/.venvs
uv venv /home/edogu/.venvs/adarsh-vllm --python 3.12 --seed
source /home/edogu/.venvs/adarsh-vllm/bin/activate
uv pip install --upgrade pip
uv pip install vllm --torch-backend=auto

python -c "import sys, torch, vllm; print(sys.version); print(torch.__version__); print(vllm.__version__); print(torch.cuda.is_available())"
```

Fallback setup if `uv` is not available but a Python 3.10+ module is available:

```bash
ssh edogu@jarvis.stevens.edu
module load python/3.12
module load cuda

python -m venv /home/edogu/.venvs/adarsh-vllm
source /home/edogu/.venvs/adarsh-vllm/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install vllm --torch-backend=auto

python -c "import torch, vllm; print(torch.__version__); print(vllm.__version__); print(torch.cuda.is_available())"
```

vLLM should be installed in a Python 3.12 environment unless Jarvis provides
another vLLM-compatible Python 3.10+ stack. The vLLM service and the benchmark
client do not need to run from the same Python environment. The benchmark client
should use the repository's Python 3.13 environment, or another Python 3.10+
environment with the repository dependencies installed.

When submitting vLLM service jobs, point the dispatcher at the environment:

```bash
VLLM_VENV=/home/edogu/.venvs/adarsh-vllm bash jarvis/run.sh submit executor
VLLM_VENV=/home/edogu/.venvs/adarsh-vllm bash jarvis/run.sh submit evaluator
```

## Remote Preflight

Before the first Slurm run on Jarvis:

```bash
cd /path/to/adarsh-rlms
mkdir -p /mmfs1/project/llm_caching/{hf_cache,vllm_cache,logs}
```

Make sure the Slurm job can read gated Hugging Face models. Prefer exporting
`HF_TOKEN` before `sbatch`, because the scripts redirect `HF_HOME` to `/local`
inside jobs:

```bash
export HF_TOKEN=<your-hugging-face-token>
```

Also prepare a client environment for the benchmark code. The service venv only
needs vLLM; the client needs this repository's runtime dependencies:

```bash
uv python install 3.13
uv sync --python 3.13
uv pip install transformers torch faiss-cpu sentence-transformers anthropic python-dotenv numpy scikit-learn
```

If you use a manually activated client venv instead of `uv run`, install the same
packages there and set `CLIENT_CMD` to start with `python ...`.

## vLLM Serving Defaults

The scripts set explicit context limits so vLLM does not try to reserve memory
for the full advertised context window of the model:

```text
executor:  --max-model-len 32768
evaluator: --max-model-len 16384
smoke:     --max-model-len 8192
```

Override these only when the benchmark needs it and the service has enough KV
cache headroom:

```bash
EXECUTOR_MAX_MODEL_LEN=65536 bash jarvis/run.sh submit executor
EVALUATOR_MAX_MODEL_LEN=32768 bash jarvis/run.sh submit evaluator
```

The shared vLLM defaults are:

```text
VLLM_DTYPE=auto
VLLM_GPU_MEMORY_UTILIZATION=0.90
```

## Modes

Use `jarvis/run.sh` as a dispatcher:

```bash
bash jarvis/run.sh submit executor
bash jarvis/run.sh submit evaluator
bash jarvis/run.sh submit smoke
bash jarvis/run.sh submit client
bash jarvis/run.sh submit download-all
```

The submit helper applies the expected Slurm resources:

```text
executor:  gpu-l40s, gpu:l40s:4, Llama 3.3 70B, tensor parallel 4
evaluator: gpu-l40s, gpu:l40s:2, Mistral Small 24B, tensor parallel 2
smoke:     gpu-l40s, gpu:l40s:2, one shared endpoint
client:    compute-short, no GPU, benchmark/client command only
download:  compute-short, no GPU, prefetch model weights into project cache
```

You can also submit manually:

```bash
MODE=executor sbatch --partition=gpu-l40s --gres=gpu:l40s:4 jarvis/serve_vllm.sh
MODE=evaluator sbatch --partition=gpu-l40s --gres=gpu:l40s:2 jarvis/serve_vllm.sh
MODE=client sbatch --partition=compute-short jarvis/run_client.sh
MODE=download-all sbatch --partition=compute-short jarvis/download_models.sh
```

## Endpoint Wiring

Each vLLM service writes its advertised URL to:

```text
/mmfs1/project/llm_caching/logs/<mode>-<job_id>.url
```

For serious benchmark runs, start two services and point the client at both:

```bash
export LLM_PROVIDER=openai_compatible
export OPENAI_COMPAT_EXECUTOR_MODEL=meta-llama/Llama-3.3-70B-Instruct
export OPENAI_COMPAT_EVALUATOR_MODEL=mistralai/Mistral-Small-3.2-24B-Instruct-2506
export OPENAI_COMPAT_EXECUTOR_BASE_URL=http://<executor-node>:8000/v1
export OPENAI_COMPAT_EVALUATOR_BASE_URL=http://<evaluator-node>:8001/v1
```

For a one-service smoke test, point both roles at the same endpoint:

```bash
export OPENAI_COMPAT_BASE_URL=http://<smoke-node>:8000/v1
export OPENAI_COMPAT_EXECUTOR_BASE_URL=$OPENAI_COMPAT_BASE_URL
export OPENAI_COMPAT_EVALUATOR_BASE_URL=$OPENAI_COMPAT_BASE_URL
```

Then submit a client job with the command you want to run:

```bash
WAIT_FOR_ENDPOINTS=1 \
CLIENT_CMD="uv run python -m unittest discover -s test -p test_semantic_cache_llm_provider.py" \
  bash jarvis/run.sh submit client
```

`WAIT_FOR_ENDPOINTS=1` makes the client poll `/v1/models` on the executor and
evaluator endpoints before it starts the command. For benchmark runs, set
`CLIENT_CMD` to the desired benchmark command.

## Cache Hydration

By default, the role scripts copy already cached Hugging Face files from
`/mmfs1/project/llm_caching/hf_cache` into `/local` with `rsync --ignore-existing`.
If the project cache is empty, Hugging Face/vLLM downloads into `/local`.

vLLM can download models automatically on first service startup, so a separate
download job is not required for correctness. It is still recommended before a
serious run because it gives cleaner logs for gated-model auth, quota, and
network issues.

Prefetch both default models:

```bash
VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash jarvis/run.sh submit download-all
```

Prefetch one model:

```bash
VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash jarvis/run.sh submit download-executor

VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash jarvis/run.sh submit download-evaluator
```

`download_models.sh` sets `SYNC_BACK_MODELS=1` by default, so files downloaded
into `/local` are synced back to `/mmfs1/project/llm_caching/hf_cache` during
cleanup. For first-time population, run one download job at a time to avoid
multiple jobs downloading and syncing the same large model concurrently.

You can also let a service job download and sync on first startup:

```bash
SYNC_BACK_MODELS=1 VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash jarvis/run.sh submit executor
```

## Dry Runs

Dry runs validate path creation, environment variables, endpoint files, and cleanup
without starting vLLM:

```bash
DRY_RUN=1 MODE=executor PROJECT_CACHE_ROOT=/tmp/llm_caching_test \
  LOCAL_BASE=/tmp/llm_caching_local/adarsh-rlms \
  bash jarvis/serve_vllm.sh
```

The cleanup trap removes the dry-run `LOCAL_BASE` only if it matches the guarded
scratch path pattern.
