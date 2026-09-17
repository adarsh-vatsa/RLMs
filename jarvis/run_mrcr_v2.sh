#!/usr/bin/env bash
# Submit MRCR through the existing Jarvis client allocation.
set -euo pipefail

TARGET="${1:-}"
SCRIPT_DIR="${JARVIS_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

usage() {
  cat <<'EOF'
Usage: bash jarvis/run_mrcr_v2.sh <direct|hybrid> [runner arguments...]

Requires OPENAI_COMPAT_EXECUTOR_BASE_URL. No evaluator service is used.
MRCR_DATA_DIR defaults to benchmark_data/mrcr_v2.
Pass --min-source-tokens and --max-source-tokens to narrow prepared source bounds.
Pass --max-input-tokens, --max-output-tokens, and --context-window-tokens for executor budgets.
All additional arguments are forwarded to python -m mrcr_v2.run_benchmark.
MRCR_LAUNCH_DRY_RUN=1 prints the command without submitting a job.
EOF
}

case "$TARGET" in
  direct)
    SUBMIT_MODE="client"
    CLIENT_MEM="${CLIENT_MEM:-32G}"
    ;;
  hybrid)
    SUBMIT_MODE="client-gpu"
    CLIENT_MEM="${CLIENT_MEM:-96G}"
    export SEMANTIC_CACHE_EMBEDDING_DEVICE="${SEMANTIC_CACHE_EMBEDDING_DEVICE:-cuda}"
    export SEMANTIC_CACHE_EMBEDDING_DTYPE="${SEMANTIC_CACHE_EMBEDDING_DTYPE:-auto}"
    ;;
  help|-h|--help|"") usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

: "${OPENAI_COMPAT_EXECUTOR_BASE_URL:?Set OPENAI_COMPAT_EXECUTOR_BASE_URL to the running executor service}"
CLIENT_ARGS=(uv run python -m mrcr_v2.run_benchmark
  --mode "$TARGET"
  --data-dir "${MRCR_DATA_DIR:-benchmark_data/mrcr_v2}"
  --executor-model "${OPENAI_COMPAT_EXECUTOR_MODEL:-Qwen/Qwen3.6-35B-A3B}"
  --executor-base-url "$OPENAI_COMPAT_EXECUTOR_BASE_URL")
shift
CLIENT_ARGS+=("$@")
printf -v CLIENT_CMD '%q ' "${CLIENT_ARGS[@]}"
export CLIENT_CMD CLIENT_MEM
echo "[MRCR] allocation=$SUBMIT_MODE mem=$CLIENT_MEM"
echo "[MRCR] command=$CLIENT_CMD"
if [[ "${MRCR_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
exec bash "$SCRIPT_DIR/run.sh" submit "$SUBMIT_MODE"
