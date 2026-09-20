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

The server setup is only needed when hosting models on this machine. If you use
existing remote endpoints, install the client environment only.

The client uses `.venv` with Python 3.13; vLLM uses
`$HOME/.venvs/adarsh-vllm` with Python 3.12. Override `BENCHMARK_VENV` and
`VLLM_VENV` to reuse existing environments. Scripts call their Python binaries
directly, so activation and `uv run` are unnecessary. Setup installs into the
selected environments; it does not delete or recreate existing environments.

The default vLLM pin is `0.19.1`. Server setup uses `TORCH_BACKEND=cu129` to
match its default CUDA 12.9 wheel and checks that the CUDA extension imports
after installation. Client setup still defaults to `TORCH_BACKEND=auto`.
Changing `VLLM_VERSION` or `TORCH_BACKEND` requires matching the vLLM wheel and
PyTorch CUDA build; changing the PyTorch backend alone does not select a different
vLLM wheel. See the [vLLM 0.19.1 installation guide](https://docs.vllm.ai/en/v0.19.1/getting_started/installation/gpu/).
Standard `HF_HOME` and `VLLM_CACHE_ROOT` settings control model caches. No cluster
modules, scratch staging, synchronization, or automatic cleanup are used.

### Repair a CUDA runtime mismatch

If vLLM reports missing `libcudart.so.12` while PyTorch is `+cu130` and only
`libcudart.so.13` is installed, the CUDA builds are mismatched. On a driver that
supports CUDA 12.9, create a separate environment with matching packages:

```bash
export VLLM_VENV="$HOME/.venvs/adarsh-vllm-cu129"
uv venv "$VLLM_VENV" --python 3.12
uv pip install --python "$VLLM_VENV/bin/python" \
  'vllm==0.19.1' --torch-backend=cu129
"$VLLM_VENV/bin/python" -c \
  'import torch; import vllm._C; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Use a new directory for this repair; the original environment remains intact.
Keep `VLLM_VENV` exported in each service terminal. A CUDA-13-capable driver can
run CUDA 12 applications through [driver backward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
The CUDA version in `nvidia-smi` describes driver support, not which runtime the
Python environment has installed.

## Start model services, or reuse endpoints

Start each service in a separate terminal or `tmux` session. For a server with
one GPU, use device 0 and tensor parallelism 1:

```bash
CUDA_VISIBLE_DEVICES=0 EXECUTOR_TP_SIZE=1 \
  bash linux/serve_vllm.sh executor \
  --reasoning-parser qwen3 --language-model-only \
  --enable-chunked-prefill
```

On the single 96 GB RTX PRO 6000 server, run AA-LCR with `--execution-only` and
defer grading, or use a remote grader. Do not start both default 35B services
concurrently on that GPU: each service reserves 90% of GPU memory by default.
For later local grading, stop the executor and start the evaluator with
`CUDA_VISIBLE_DEVICES=0 EVALUATOR_TP_SIZE=1`. Hybrid runs can use
`SEMANTIC_CACHE_EMBEDDING_DEVICE=cpu` to leave GPU memory for the executor.

For a server with multiple GPUs, the following lists
are examples: choose devices and tensor parallelism for your server. Keep executor,
evaluator, and hybrid embedding devices separate when running concurrently.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 EXECUTOR_TP_SIZE=4 \
  bash linux/serve_vllm.sh executor \
  --reasoning-parser qwen3 --language-model-only \
  --enable-chunked-prefill

# Needed for AA-LCR grading or optional semantic cache verification.
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
Keep each service running in its own terminal; use a third terminal for the
benchmark. Export custom environment/cache paths in each terminal that needs them.

Runner budget flags do not resize an already-running service. Choose a runner
context no larger than the served limit, with input plus output allowance within
that context. AA-LCR can discover the executor limit automatically; MRCR and
LongBench retain explicit defaults. If you change `EVALUATOR_MAX_MODEL_LEN` for
AA-LCR grading, pass the matching `--grader-context-window` to the runner.

Services bind to `127.0.0.1` by default. Set `HOST` when clients must reach this
server over the network. They run in the foreground; Ctrl-C stops the service.
Use `tmux` to keep services and benchmark runs alive across SSH disconnections.
The scripts do not start services implicitly, kill other processes, or manage jobs.

The evaluator service can serve as the answer grader, the semantic cache verifier,
or both. Dense document retrieval uses embeddings and FAISS, not this service.
See [service roles and cache options](SHARED_EXECUTION_RUNBOOK.md#service-roles)
for the flags that enable verification. AA-LCR's `--execution-only` skips grading
but does not disable an explicitly enabled cache verifier.

Wait for `Application startup complete`, then export these settings and check
the endpoints in the **benchmark terminal**. Settings entered in a service
terminal are not automatically shared with another terminal.

```bash
export OPENAI_COMPAT_EXECUTOR_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_COMPAT_EXECUTOR_MODEL=Qwen/Qwen3.6-35B-A3B
export OPENAI_COMPAT_EVALUATOR_BASE_URL=http://127.0.0.1:8001/v1
export OPENAI_COMPAT_EVALUATOR_MODEL=Qwen/Qwen3.5-35B-A3B
curl -fsS "$OPENAI_COMPAT_EXECUTOR_BASE_URL/models"
# Only if using this service for grading or cache verification:
curl -fsS "$OPENAI_COMPAT_EVALUATOR_BASE_URL/models"
```

To use services already running locally or remotely, skip local service startup
and set those URLs/model names to the actual endpoints. They need not be on
Jarvis. For authenticated endpoints, set `OPENAI_COMPAT_API_KEY_ENV` to the name
of an environment variable containing the key; AA-LCR also accepts its separate
grader credential options. Authenticated `curl` checks also need an Authorization
header; they do not read the runner's key-variable setting automatically.
The Linux launcher does not wait for services, so check readiness before actual
execution. Explicit-context preflight does not need model services.

Continue with the [benchmark runbook](BENCHMARK_RUNBOOK.md) for dataset preparation, preflight, and execution.
