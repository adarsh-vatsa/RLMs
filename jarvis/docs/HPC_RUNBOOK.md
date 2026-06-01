# Jarvis HPC Step-By-Step Runbook

This is the operational checklist for running the local vLLM services and the
benchmark client on Jarvis. Run these commands from the Jarvis login node unless
the step explicitly says otherwise.

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
cd /path/to/adarsh-rlms
git pull
```

## 1. Create Project Storage

Create the persistent cache directories once:

```bash
mkdir -p /mmfs1/project/llm_caching/{hf_cache,vllm_cache,logs,cache_state}
```

These directories are durable. Do not delete them after each experiment:

```text
/mmfs1/project/llm_caching/hf_cache      Hugging Face model cache
/mmfs1/project/llm_caching/vllm_cache    vLLM persistent cache
/mmfs1/project/llm_caching/logs          Slurm logs and endpoint URL files
/mmfs1/project/llm_caching/cache_state   Semantic cache benchmark state
```

Per-job scratch is created under `/local/$USER/$SLURM_JOB_ID/adarsh-rlms` and is
cleaned automatically when the job exits.

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

If Jarvis has a Python 3.12 module:

```bash
module load python/3.12
module load cuda

mkdir -p /home/edogu/.venvs
uv venv /home/edogu/.venvs/adarsh-vllm --python 3.12 --seed
source /home/edogu/.venvs/adarsh-vllm/bin/activate
uv pip install --upgrade pip
uv pip install vllm --torch-backend=auto

python -c "import sys, torch, vllm; print(sys.version); print(torch.__version__); print(vllm.__version__); print(torch.cuda.is_available())"
```

If Jarvis only exposes Python 3.9.18, install a user-local Python with `uv`:

```bash
module load cuda

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

If Jarvis requires modules inside the Slurm job, pass them when submitting:

```bash
MODULES="cuda" VLLM_VENV=/home/edogu/.venvs/adarsh-vllm bash jarvis/run.sh submit smoke
```

## 4. Prepare The Client Environment

The vLLM service environment and benchmark client environment can be different.
The service environment needs `vllm`; the client environment needs this repo's
dependencies.

From the repo root:

```bash
uv python install 3.13
uv sync --python 3.13
uv pip install transformers torch faiss-cpu sentence-transformers anthropic python-dotenv numpy scikit-learn
```

Quick import check:

```bash
uv run python - <<'PY'
import semantic_cache_system
print("semantic_cache_system import ok")
PY
```

## 5. Run A Slurm Dry Run

This verifies Slurm submission, path creation, logs, environment export, and
cleanup without starting vLLM:

```bash
DRY_RUN=1 VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash jarvis/run.sh submit smoke
```

The command prints a Slurm job id. Watch it:

```bash
squeue -u "$USER"
tail -f /mmfs1/project/llm_caching/logs/rlms-smoke-<job_id>.out
```

The dry-run log should show `/local/$USER/<job_id>/adarsh-rlms`, the selected
model, the endpoint file path, and the vLLM command that would have run.

## 6. Prefetch Model Weights

For the first real run, prefetch one model at a time. This makes gated-model and
network failures show up before you reserve GPU jobs.

```bash
VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash jarvis/run.sh submit download-evaluator
```

After it finishes:

```bash
VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash jarvis/run.sh submit download-executor
```

Monitor:

```bash
squeue -u "$USER"
tail -f /mmfs1/project/llm_caching/logs/rlms-download-<job_id>.out
du -sh /mmfs1/project/llm_caching/hf_cache
```

The download job syncs files from `/local` back to
`/mmfs1/project/llm_caching/hf_cache` during cleanup.

## 7. Run One-Service Smoke Test

Start one vLLM service with the smoke model:

```bash
VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash jarvis/run.sh submit smoke
```

Find the endpoint after the job starts:

```bash
ls -ltr /mmfs1/project/llm_caching/logs/*smoke*.url
cat /mmfs1/project/llm_caching/logs/smoke-<job_id>.url
tail -f /mmfs1/project/llm_caching/logs/rlms-smoke-<job_id>.out
```

The endpoint will look like:

```text
http://<smoke-node>:8000/v1
```

Submit a client smoke test against that endpoint:

```bash
SMOKE_URL=http://<smoke-node>:8000/v1

LLM_PROVIDER=openai_compatible \
OPENAI_COMPAT_BASE_URL="$SMOKE_URL" \
OPENAI_COMPAT_EXECUTOR_BASE_URL="$SMOKE_URL" \
OPENAI_COMPAT_EVALUATOR_BASE_URL="$SMOKE_URL" \
WAIT_FOR_ENDPOINTS=1 \
CLIENT_CMD="uv run python -m unittest discover -s test -p test_semantic_cache_llm_provider.py" \
  bash jarvis/run.sh submit client
```

Monitor the client:

```bash
squeue -u "$USER"
tail -f /mmfs1/project/llm_caching/logs/rlms-client-<job_id>.out
```

Stop the smoke service when the client test is done:

```bash
scancel <smoke_job_id>
```

## 8. Start The Two Production Services

Start the executor service:

```bash
VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash jarvis/run.sh submit executor
```

Start the evaluator service:

```bash
VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash jarvis/run.sh submit evaluator
```

Watch both jobs:

```bash
squeue -u "$USER"
tail -f /mmfs1/project/llm_caching/logs/rlms-executor-<executor_job_id>.out
tail -f /mmfs1/project/llm_caching/logs/rlms-evaluator-<evaluator_job_id>.out
```

Read the endpoint files:

```bash
EXECUTOR_URL=$(cat /mmfs1/project/llm_caching/logs/executor-<executor_job_id>.url)
EVALUATOR_URL=$(cat /mmfs1/project/llm_caching/logs/evaluator-<evaluator_job_id>.url)

echo "$EXECUTOR_URL"
echo "$EVALUATOR_URL"
```

The expected shape is:

```text
executor:  http://<executor-node>:8000/v1
evaluator: http://<evaluator-node>:8001/v1
```

## 9. Run A Small Benchmark Client Job

Start with a capped LongBench-v2 run:

```bash
LLM_PROVIDER=openai_compatible \
OPENAI_COMPAT_EXECUTOR_BASE_URL="$EXECUTOR_URL" \
OPENAI_COMPAT_EVALUATOR_BASE_URL="$EVALUATOR_URL" \
WAIT_FOR_ENDPOINTS=1 \
CLIENT_CMD="uv run python long_bench_v2/run_benchmark.py \
  --llm-provider openai_compatible \
  --mode cache \
  --cache-state-root /mmfs1/project/llm_caching/cache_state \
  --row-types original,exact,semantic \
  --max-rows 5 \
  --output-dir benchmark_artifacts \
  --manifest-note jarvis-l40s-small" \
  bash jarvis/run.sh submit client
```

Monitor:

```bash
squeue -u "$USER"
tail -f /mmfs1/project/llm_caching/logs/rlms-client-<client_job_id>.out
```

When this finishes, inspect the generated artifact paths printed in the client
log. They should point under `benchmark_artifacts/longbench_v2/...`.

## 10. Run The Full Benchmark

Use the same service URLs and remove the row cap:

```bash
LLM_PROVIDER=openai_compatible \
OPENAI_COMPAT_EXECUTOR_BASE_URL="$EXECUTOR_URL" \
OPENAI_COMPAT_EVALUATOR_BASE_URL="$EVALUATOR_URL" \
WAIT_FOR_ENDPOINTS=1 \
CLIENT_CMD="uv run python long_bench_v2/run_benchmark.py \
  --llm-provider openai_compatible \
  --mode cache \
  --cache-state-root /mmfs1/project/llm_caching/cache_state \
  --row-types original,exact,semantic \
  --output-dir benchmark_artifacts \
  --manifest-note jarvis-l40s-full" \
  bash jarvis/run.sh submit client
```

For a cold-cache rerun, add `--cache-reset` to `CLIENT_CMD`. Do not delete
`/mmfs1/project/llm_caching/hf_cache`.

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

## 12. Common Failure Checks

If a service never becomes reachable:

```bash
tail -n 200 /mmfs1/project/llm_caching/logs/rlms-executor-<job_id>.out
tail -n 200 /mmfs1/project/llm_caching/logs/rlms-evaluator-<job_id>.out
```

Likely causes:

- Missing `HF_TOKEN` or gated model access not approved.
- vLLM environment was not passed with `VLLM_VENV`.
- CUDA module is required inside the Slurm job; retry with `MODULES="cuda"`.
- The model context length is too high for available GPU memory; lower
  `EXECUTOR_MAX_MODEL_LEN` or `EVALUATOR_MAX_MODEL_LEN`.
- The endpoint URL points to `127.0.0.1` from a different Slurm job. Use the
  hostname URL written to the `.url` file.

If the benchmark imports fail, fix the client environment and rerun only the
client job. The vLLM service jobs can stay running.

