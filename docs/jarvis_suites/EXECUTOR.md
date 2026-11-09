# The executor

The executor serves Qwen3.6-35B-A3B with vLLM on 4 L40S GPUs. It uses the full
262,144-token window and answers one request at a time. Every suite uses the
same executor settings.

Start a new executor for each suite, so that its 24-hour limit covers the whole
suite. Steps 1 to 4 take place before you submit the suite, and step 5 after
the suite job ends.

## 1. Start it

```bash
cd ~
MODULES="cuda12.8/toolkit/12.8.1" \
EXECUTOR_MODEL=Qwen/Qwen3.6-35B-A3B \
EXECUTOR_TP_SIZE=4 \
EXECUTOR_MAX_MODEL_LEN=262144 \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
VLLM_EXTRA_ARGS="--reasoning-parser qwen3 --language-model-only --max-num-seqs 1 --enable-chunked-prefill --max-num-batched-tokens 8192" \
SYNC_BACK_MODELS=1 VLLM_VENV="$HOME/.venvs/adarsh-vllm" \
  bash adarsh-rlms/jarvis/run.sh submit executor
```

This prints `Submitted batch job <id>`. Keep the ID for the steps below:

```bash
EXECUTOR_JOB=<id>
```

## 2. Wait until it is ready

```bash
squeue -u "$USER"                                          # ST: PD = waiting for GPUs, R = running
tail -f "$PROJECT_LOG_DIR/rlms-executor-$EXECUTOR_JOB.out"
```

The first start on a node downloads the model, about 70 GB, to that node's
local disk. It is ready when the log shows:

```text
Maximum concurrency for 262,144 tokens per request: X.XXx
Application startup complete.
```

`X.XX` must be at least `1.00`. If the log shows a CUDA out-of-memory or engine
initialization error instead, cancel the job and start it again. Press Ctrl-C
to stop following the log; the executor keeps running.

## 3. Point the suites at it

```bash
export OPENAI_COMPAT_EXECUTOR_BASE_URL=$(cat "$PROJECT_LOG_DIR/executor-$EXECUTOR_JOB.url")
echo "$OPENAI_COMPAT_EXECUTOR_BASE_URL"   # http://<node>:8000/v1
```

This holds only in the current terminal. In a new terminal, set
`EXECUTOR_JOB` and run this step again.

## 4. Record its setup

Every run copies this record into its results, so that each result shows the
hardware and software it ran on. Record it once for each new executor:

```bash
mkdir -p adarsh-rlms/.cache
VLLM_VENV="$HOME/.venvs/adarsh-vllm" EXECUTOR_TP_SIZE=4 \
adarsh-rlms/.venv/bin/python - "$OPENAI_COMPAT_EXECUTOR_BASE_URL" <<'EOF'
import datetime, json, os, subprocess, sys, urllib.request

base = sys.argv[1].rstrip("/")
with urllib.request.urlopen(base + "/models", timeout=10) as response:
    served = json.load(response)["data"]
vllm_python = os.path.join(os.environ["VLLM_VENV"], "bin", "python")
vllm = subprocess.run([vllm_python, "-c", "import importlib.metadata as m; print(m.version('vllm'))"],
                      capture_output=True, text=True, check=True).stdout.strip()
tensor_parallel = int(os.environ.get("EXECUTOR_TP_SIZE", "4"))
metadata = {
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "cluster": "Jarvis",
    "executor_base_url": base,
    "gpus": f"{tensor_parallel} x NVIDIA L40S (gpu-l40s partition)",
    "tensor_parallel_size": tensor_parallel,
    "vllm_version": vllm,
    "served_models": [{"id": m["id"], "max_model_len": m.get("max_model_len")} for m in served],
}
with open("adarsh-rlms/.cache/serving_metadata.json", "w") as handle:
    handle.write(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
EOF
```

The output must show `"max_model_len": 262144`. The suite refuses to start if
this file is missing. It cannot tell, though, whether the file belongs to an
earlier executor, so record it again for every new executor.

Now submit the suite, as described in its page.

## 5. Stop it

When the suite job has ended:

```bash
scancel "$EXECUTOR_JOB"
```

## If something goes wrong

- **The executor never becomes ready.** Read the end of its log:
  `tail -n 200 "$PROJECT_LOG_DIR/rlms-executor-$EXECUTOR_JOB.out"`. Common
  causes are a missing `HF_TOKEN` in `adarsh-rlms/.env`, or too little free
  space on the node's local disk. The log reports free space on a line that
  starts with `[JARVIS] local free space`.
- **The executor stops partway through a suite.** Start a new one and repeat
  steps 2 to 4, then resubmit the same suite command. Finished runs are kept.
