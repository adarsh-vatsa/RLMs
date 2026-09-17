#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

case "${1:-}" in
  executor)
    MODEL="${EXECUTOR_MODEL:-Qwen/Qwen3.6-35B-A3B}"
    PORT="${EXECUTOR_PORT:-8000}"
    TP_SIZE="${EXECUTOR_TP_SIZE:-1}"
    CONTEXT="${EXECUTOR_MAX_MODEL_LEN:-}"
    ;;
  evaluator)
    MODEL="${EVALUATOR_MODEL:-Qwen/Qwen3.5-35B-A3B}"
    PORT="${EVALUATOR_PORT:-8001}"
    TP_SIZE="${EVALUATOR_TP_SIZE:-1}"
    CONTEXT="${EVALUATOR_MAX_MODEL_LEN:-32768}"
    ;;
  help|-h|--help|"")
    echo 'Usage: bash linux/serve_vllm.sh <executor|evaluator> [vLLM arguments...]'
    echo 'Runs in the foreground. Use CUDA_VISIBLE_DEVICES and role-specific TP_SIZE variables.'
    echo 'EXECUTOR_MAX_MODEL_LEN is optional; omit to use the model configuration limit.'
    echo 'DRY_RUN=1 previews the command without loading models.'
    exit 0 ;;
  *) echo "Unknown service: $1" >&2; exit 2 ;;
esac
shift
cd "$REPO_ROOT"
COMMAND=("$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server
  --host "${HOST:-127.0.0.1}" --port "$PORT" --model "$MODEL" --served-model-name "$MODEL"
  --tensor-parallel-size "$TP_SIZE" --dtype "${VLLM_DTYPE:-auto}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}")
if [[ -n "$CONTEXT" ]]; then
  COMMAND+=(--max-model-len "$CONTEXT")
fi
COMMAND+=("$@")
echo "[LINUX] service=http://${HOST:-127.0.0.1}:$PORT/v1 (wait for startup before running benchmarks)"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  run_command "${COMMAND[@]}"
else
  exec "${COMMAND[@]}"
fi
