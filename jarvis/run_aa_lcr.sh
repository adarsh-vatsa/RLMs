#!/usr/bin/env bash
# Submit one AA-LCR benchmark cell through the shared Jarvis client runner.

set -euo pipefail

TARGET="${1:-}"
SCRIPT_DIR="${JARVIS_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

usage() {
  cat <<'EOF'
Usage: bash jarvis/run_aa_lcr.sh <direct_262k|hybrid_262k|direct_64k|hybrid_64k> [runner arguments...]

Required environment:
  OPENAI_COMPAT_EXECUTOR_BASE_URL=http://executor-host:8000/v1
  OPENAI_COMPAT_EVALUATOR_BASE_URL=http://evaluator-host:8001/v1

Optional smoke test:
  AA_LCR_MAX_ROWS=2 bash jarvis/run_aa_lcr.sh direct_262k

AA_LCR_DATA_DIR selects the prepared dataset directory (default benchmark_data/aa_lcr).
AA_LCR_MAX_OUTPUT_TOKENS defaults to 16384 for 262K cells and 512 for 64K cells.
AA_LCR_RUN_ID, AA_LCR_REPEAT_ID, and AA_LCR_SERVING_METADATA identify the run.
Additional runner arguments can select grader prompts, API style, and credentials.

Set AA_LCR_LAUNCH_DRY_RUN=1 to print the allocation and command without submitting.
EOF
}

case "$TARGET" in
  direct_262k|direct_64k)
    SUBMIT_MODE="client"
    CLIENT_MEM="${CLIENT_MEM:-32G}"
    ;;
  hybrid_262k|hybrid_64k)
    SUBMIT_MODE="client-gpu"
    CLIENT_MEM="${CLIENT_MEM:-96G}"
    export SEMANTIC_CACHE_EMBEDDING_DEVICE="${SEMANTIC_CACHE_EMBEDDING_DEVICE:-cuda}"
    export SEMANTIC_CACHE_EMBEDDING_DTYPE="${SEMANTIC_CACHE_EMBEDDING_DTYPE:-auto}"
    ;;
  help|-h|--help|"")
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

: "${OPENAI_COMPAT_EXECUTOR_BASE_URL:?Set OPENAI_COMPAT_EXECUTOR_BASE_URL to the running executor service}"
: "${OPENAI_COMPAT_EVALUATOR_BASE_URL:?Set OPENAI_COMPAT_EVALUATOR_BASE_URL to the running evaluator service}"

EXECUTOR_MODEL="${OPENAI_COMPAT_EXECUTOR_MODEL:-Qwen/Qwen3.6-35B-A3B}"
EVALUATOR_MODEL="${OPENAI_COMPAT_EVALUATOR_MODEL:-Qwen/Qwen3.5-35B-A3B}"
DATA_DIR="${AA_LCR_DATA_DIR:-benchmark_data/aa_lcr}"
case "$TARGET" in
  *_262k) OUTPUT_TOKENS="${AA_LCR_MAX_OUTPUT_TOKENS:-16384}" ;;
  *_64k) OUTPUT_TOKENS="${AA_LCR_MAX_OUTPUT_TOKENS:-512}" ;;
esac
CLIENT_ARGS=(uv run python -m aa_lcr.run_benchmark
  --experiment "$TARGET"
  --executor-model "$EXECUTOR_MODEL" --evaluator-model "$EVALUATOR_MODEL"
  --executor-base-url "$OPENAI_COMPAT_EXECUTOR_BASE_URL"
  --evaluator-base-url "$OPENAI_COMPAT_EVALUATOR_BASE_URL"
  --questions-csv "$DATA_DIR/AA-LCR_Dataset.csv"
  --documents-root "$DATA_DIR/extracted_text/lcr"
  --dataset-manifest "$DATA_DIR/dataset_manifest.json"
  --max-output-tokens "$OUTPUT_TOKENS"
  --repeat-id "${AA_LCR_REPEAT_ID:-1}")
if [[ -n "${AA_LCR_MAX_ROWS:-}" ]]; then
  if ! [[ "$AA_LCR_MAX_ROWS" =~ ^[0-9]+$ ]]; then
    echo "AA_LCR_MAX_ROWS must be a non-negative integer" >&2
    exit 2
  fi
  CLIENT_ARGS+=(--max-rows "$AA_LCR_MAX_ROWS")
fi
if [[ -n "${AA_LCR_RUN_ID:-}" ]]; then
  CLIENT_ARGS+=(--run-id "$AA_LCR_RUN_ID")
fi
if [[ -n "${AA_LCR_SERVING_METADATA:-}" ]]; then
  CLIENT_ARGS+=(--serving-metadata "$AA_LCR_SERVING_METADATA")
fi
shift
CLIENT_ARGS+=("$@")

printf -v CLIENT_CMD '%q ' "${CLIENT_ARGS[@]}"
export CLIENT_CMD CLIENT_MEM

echo "[AA-LCR] target=$TARGET"
echo "[AA-LCR] allocation=$SUBMIT_MODE mem=$CLIENT_MEM"
echo "[AA-LCR] command=$CLIENT_CMD"

if [[ "${AA_LCR_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

exec bash "$SCRIPT_DIR/run.sh" submit "$SUBMIT_MODE"
