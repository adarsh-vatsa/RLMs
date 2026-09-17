#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

case "${1:-}" in
  client)
    if [[ ! -x "$BENCHMARK_VENV/bin/python" ]]; then
      run_command uv venv "$BENCHMARK_VENV" --python "${CLIENT_PYTHON_VERSION:-3.13}"
    fi
    run_command uv pip install --python "$BENCHMARK_VENV/bin/python" \
      --torch-backend "${TORCH_BACKEND:-auto}" \
      transformers torch faiss-cpu python-dotenv numpy
    ;;
  server)
    if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
      run_command uv venv "$VLLM_VENV" --python "${SERVER_PYTHON_VERSION:-3.12}"
    fi
    run_command uv pip install --python "$VLLM_VENV/bin/python" \
      "vllm==${VLLM_VERSION:-0.19.1}" --torch-backend "${TORCH_BACKEND:-auto}"
    ;;
  help|-h|--help|"")
    echo 'Usage: bash linux/setup.sh <client|server>'
    echo 'Requires uv. Separate benchmark and vLLM environments; DRY_RUN=1 previews commands.'
    ;;
  *) echo "Unknown setup target: $1" >&2; exit 2 ;;
esac
