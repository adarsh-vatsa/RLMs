# Linux setup and model services

These scripts run processes directly. They reuse the existing AA-LCR, MRCR v2,
and LongBench-v2 runners and default to the `common` execution profile. Jarvis
scripts remain separate and unchanged. Run the examples from the repository root.

| Script | Purpose |
|---|---|
| `setup.sh client` | Install benchmark/tokenizer/embedding dependencies |
| `setup.sh server` | Install vLLM in a separate environment |
| `serve_vllm.sh executor\|evaluator` | Start a foreground model service |
| `run_benchmark.sh BENCHMARK direct\|hybrid` | Run a benchmark directly |

All scripts support `DRY_RUN=1`. Dry runs do not install packages, contact
endpoints, download models, or launch inference. Arguments after the mode are
passed to the underlying runner or vLLM as individual arguments, preserving quoting.

## Set up the server

Copy/clone the repository onto the Linux machine. With `uv` installed and a
working NVIDIA driver, create the environments:

```bash
bash linux/setup.sh client
bash linux/setup.sh server
```

The client uses `.venv` with Python 3.13; vLLM uses
`$HOME/.venvs/adarsh-vllm` with Python 3.12. Override `BENCHMARK_VENV` and
`VLLM_VENV` to reuse existing environments. Scripts call their Python binaries
directly, so activation and `uv run` are unnecessary. Setup installs into the
selected environments; it does not delete or recreate existing environments.

The default vLLM pin is `0.19.1`, matching the existing project setup.
`VLLM_VERSION` overrides it; `TORCH_BACKEND` defaults to `auto` and can be set to
the server's required backend (for example `cu128`). Standard `HF_HOME` and
`VLLM_CACHE_ROOT` settings control model caches. No cluster modules, scratch
staging, synchronization, or automatic cleanup are used.

## Start model services, or reuse endpoints

Start each service in a separate terminal or `tmux` session. The GPU lists below
are examples: choose devices and tensor parallelism for your server. Keep executor,
evaluator, and hybrid embedding devices separate when running concurrently.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 EXECUTOR_TP_SIZE=4 \
  bash linux/serve_vllm.sh executor \
  --reasoning-parser qwen3 --language-model-only \
  --enable-chunked-prefill

# Needed only for AA-LCR grading during the run or later regrading.
CUDA_VISIBLE_DEVICES=4,5 EVALUATOR_TP_SIZE=2 \
  bash linux/serve_vllm.sh evaluator \
  --reasoning-parser qwen3 --language-model-only
```

Defaults: executor `Qwen/Qwen3.6-35B-A3B` on port 8000; evaluator
`Qwen/Qwen3.5-35B-A3B` on port 8001 with 32,768 context tokens. Set
`EXECUTOR_MODEL`, `EVALUATOR_MODEL`, the corresponding `_PORT`, `_TP_SIZE`, or
`_MAX_MODEL_LEN` variables to change them. Omitting `EXECUTOR_MAX_MODEL_LEN`
leaves the context limit to vLLM's model configuration; set it to `65536` for a
64K service. Tensor parallelism defaults to 1 unless configured.

Services bind to `127.0.0.1` by default. Set `HOST` when clients must reach this
server over the network. They run in the foreground; Ctrl-C stops the service.
Use `tmux` to keep services and benchmark runs alive across SSH disconnections.
The scripts do not start services implicitly, kill other processes, or manage jobs.

Wait for `Application startup complete`, then check the endpoints:

```bash
export OPENAI_COMPAT_EXECUTOR_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_COMPAT_EXECUTOR_MODEL=Qwen/Qwen3.6-35B-A3B
export OPENAI_COMPAT_EVALUATOR_BASE_URL=http://127.0.0.1:8001/v1
export OPENAI_COMPAT_EVALUATOR_MODEL=Qwen/Qwen3.5-35B-A3B
curl -fsS "$OPENAI_COMPAT_EXECUTOR_BASE_URL/models"
# Only if using the evaluator:
curl -fsS "$OPENAI_COMPAT_EVALUATOR_BASE_URL/models"
```

To use services already running locally or remotely, skip local service startup
and set those URLs/model names to the actual endpoints. They need not be on
Jarvis. For authenticated endpoints, set `OPENAI_COMPAT_API_KEY_ENV` to the name
of an environment variable containing the key; AA-LCR also accepts its separate
grader credential options. No readiness polling is imposed on offline preflight.

Continue with the [benchmark runbook](BENCHMARK_RUNBOOK.md) for dataset preparation, preflight, and execution.
