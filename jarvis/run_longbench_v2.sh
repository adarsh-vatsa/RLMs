#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-}"
SCRIPT_DIR="${JARVIS_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
case "$TARGET" in
  direct) MODULE=long_bench_v2.run_api_benchmark; SUBMIT_MODE=client ;;
  hybrid) MODULE=long_bench_v2.run_benchmark; SUBMIT_MODE=client-gpu ;;
  help|-h|--help|"")
    echo 'Usage: bash jarvis/run_longbench_v2.sh <direct|hybrid> [runner arguments...]'
    echo 'Uses the common profile, original rows, and the executor endpoint.'
    echo 'LONGBENCH_LAUNCH_DRY_RUN=1 prints without submitting.'
    exit 0 ;;
  *) echo "Unknown mode: $TARGET" >&2; exit 2 ;;
esac
shift
source "$SCRIPT_DIR/lib/execution_services.sh"
jarvis_execution_services 0 0 --execution-profile common "$@"
CLIENT_ARGS=(uv run python -m "$MODULE" --execution-profile common --row-types original
  --context-window-tokens 65536 --max-input-tokens 60000 --max-output-tokens 8)
if [[ "$TARGET" == direct ]]; then
  CLIENT_ARGS+=(--api-provider openai_compatible
    --api-model "${OPENAI_COMPAT_EXECUTOR_MODEL:-Qwen/Qwen3.6-35B-A3B}"
    --api-base-url "${OPENAI_COMPAT_EXECUTOR_BASE_URL:-http://127.0.0.1:8000/v1}")
  export CLIENT_MEM="${CLIENT_MEM:-32G}"
else
  export SEMANTIC_CACHE_SEARCH_MODE=hybrid
  export SEMANTIC_CACHE_EMBEDDING_DEVICE="${SEMANTIC_CACHE_EMBEDDING_DEVICE:-cuda}"
  export SEMANTIC_CACHE_EMBEDDING_DTYPE="${SEMANTIC_CACHE_EMBEDDING_DTYPE:-auto}"
  CLIENT_ARGS+=(--mode baseline --llm-provider openai_compatible
    --executor-model "${OPENAI_COMPAT_EXECUTOR_MODEL:-Qwen/Qwen3.6-35B-A3B}"
    --openai-compat-executor-base-url "${OPENAI_COMPAT_EXECUTOR_BASE_URL:-http://127.0.0.1:8000/v1}")
  export CLIENT_MEM="${CLIENT_MEM:-96G}"
fi
CLIENT_ARGS+=("$@")
printf -v CLIENT_CMD '%q ' "${CLIENT_ARGS[@]}"
export CLIENT_CMD
echo "[LONGBENCH] allocation=$SUBMIT_MODE command=$CLIENT_CMD"
if [[ "${LONGBENCH_LAUNCH_DRY_RUN:-0}" == 1 ]]; then
  exit 0
fi
exec bash "$SCRIPT_DIR/run.sh" submit "$SUBMIT_MODE"
