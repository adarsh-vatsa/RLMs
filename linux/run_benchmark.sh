#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() {
  echo 'Usage: bash linux/run_benchmark.sh <aa_lcr|mrcr_v2|longbench_v2> <direct|hybrid> [runner arguments...]'
  echo 'Uses the common execution profile. All runner arguments are forwarded unchanged.'
  echo 'Set OPENAI_COMPAT_EXECUTOR_BASE_URL/MODEL for an existing local or remote executor.'
  echo 'AA-LCR grading uses OPENAI_COMPAT_EVALUATOR_BASE_URL/MODEL; --execution-only skips grading.'
  echo 'DRY_RUN=1 prints without running; BENCHMARK_VENV selects the Python environment.'
}
BENCHMARK="${1:-}"
case "$BENCHMARK" in help|-h|--help|"") usage; exit 0 ;; esac
TARGET="${2:-}"
case "$TARGET" in direct|hybrid) ;; *) usage >&2; exit 2 ;; esac
shift 2
cd "$REPO_ROOT"
export OPENAI_COMPAT_EXECUTOR_BASE_URL="${OPENAI_COMPAT_EXECUTOR_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_COMPAT_EXECUTOR_MODEL="${OPENAI_COMPAT_EXECUTOR_MODEL:-Qwen/Qwen3.6-35B-A3B}"
if [[ "$TARGET" == hybrid ]]; then
  export SEMANTIC_CACHE_EMBEDDING_DEVICE="${SEMANTIC_CACHE_EMBEDDING_DEVICE:-cuda}"
  export SEMANTIC_CACHE_EMBEDDING_DTYPE="${SEMANTIC_CACHE_EMBEDDING_DTYPE:-auto}"
fi
case "$BENCHMARK" in
  aa_lcr)
    DATA_DIR="${AA_LCR_DATA_DIR:-benchmark_data/aa_lcr/v1.1}"
    COMMAND=("$BENCHMARK_VENV/bin/python" -m aa_lcr.run_benchmark
      --mode "$TARGET" --execution-profile common
      --executor-model "$OPENAI_COMPAT_EXECUTOR_MODEL"
      --executor-base-url "$OPENAI_COMPAT_EXECUTOR_BASE_URL"
      --evaluator-model "${OPENAI_COMPAT_EVALUATOR_MODEL:-Qwen/Qwen3.5-35B-A3B}"
      --evaluator-base-url "${OPENAI_COMPAT_EVALUATOR_BASE_URL:-http://127.0.0.1:8001/v1}"
      --grader-prompt-version aa_lcr_equality_v1.1
      --questions-csv "$DATA_DIR/AA-LCR_Dataset.csv"
      --documents-root "$DATA_DIR/extracted_text/lcr"
      --dataset-manifest "$DATA_DIR/dataset_manifest.json")
    ;;
  mrcr_v2)
    COMMAND=("$BENCHMARK_VENV/bin/python" -m mrcr_v2.run_benchmark
      --mode "$TARGET" --execution-profile common
      --data-dir "${MRCR_DATA_DIR:-benchmark_data/mrcr_v2}"
      --executor-model "$OPENAI_COMPAT_EXECUTOR_MODEL"
      --executor-base-url "$OPENAI_COMPAT_EXECUTOR_BASE_URL")
    ;;
  longbench_v2)
    if [[ "$TARGET" == direct ]]; then
      COMMAND=("$BENCHMARK_VENV/bin/python" -m long_bench_v2.run_api_benchmark
        --api-provider openai_compatible --api-model "$OPENAI_COMPAT_EXECUTOR_MODEL"
        --api-base-url "$OPENAI_COMPAT_EXECUTOR_BASE_URL")
    else
      export SEMANTIC_CACHE_SEARCH_MODE=hybrid
      COMMAND=("$BENCHMARK_VENV/bin/python" -m long_bench_v2.run_benchmark
        --mode baseline --llm-provider openai_compatible
        --executor-model "$OPENAI_COMPAT_EXECUTOR_MODEL"
        --openai-compat-executor-base-url "$OPENAI_COMPAT_EXECUTOR_BASE_URL")
    fi
    COMMAND+=(--execution-profile common --row-types original
      --suite-csv benchmark_data/long_bench_v2/data.csv
      --context-window-tokens 65536 --max-input-tokens 60000 --max-output-tokens 8)
    ;;
  *) echo "Unknown benchmark: $BENCHMARK" >&2; usage >&2; exit 2 ;;
esac
COMMAND+=("$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  run_command "${COMMAND[@]}"
else
  exec "${COMMAND[@]}"
fi
