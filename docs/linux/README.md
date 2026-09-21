# Linux runbooks

Run benchmarks on a Linux server without Slurm:

1. [Environment setup and model services](SETUP_RUNBOOK.md)
2. [Dataset preparation, preflight, and execution](BENCHMARK_RUNBOOK.md)

Optional reference: [runner options, defaults, and cache settings](SHARED_EXECUTION_RUNBOOK.md).
This is not a third execution step.

The [scripts](../../linux/) launch the existing benchmark runners directly.
For Slurm-based execution, use the [Jarvis runbooks](../jarvis/README.md).

## Server Configuration

```bash
# OS, CPU, and RAM
cat /etc/os-release
uname -m
lscpu
free -h

# GPUs, driver, memory, and current processes
nvidia-smi

# Filesystem capacity and model-cache size
df -h
du -sh "${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}" 2>/dev/null

# Serving environment and CUDA availability
"${VLLM_VENV:-$HOME/.venvs/adarsh-vllm}/bin/python" - <<'PY'
import sys
import importlib.metadata as metadata
import torch

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("vLLM:", metadata.version("vllm"))
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())

for index in range(torch.cuda.device_count()):
    gpu = torch.cuda.get_device_properties(index)
    print(
        f"GPU {index}: {gpu.name}, "
        f"{gpu.total_memory / 1024**3:.1f} GiB, "
        f"compute capability {gpu.major}.{gpu.minor}"
    )
PY

# Running executor: served model and advertised context limit
curl -fsS "${OPENAI_COMPAT_EXECUTOR_BASE_URL:-http://127.0.0.1:8000/v1}/models"
```

If you’re using the repaired environment, first set:

```bash
export VLLM_VENV="$HOME/.venvs/adarsh-vllm-cu129"
```