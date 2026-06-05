#!/usr/bin/env bash
# Shared Jarvis runtime setup. Source this file from role scripts; do not run it.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "jarvis/lib/env.sh must be sourced, not executed." >&2
  exit 2
fi

JARVIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$JARVIS_DIR/.." && pwd)"

MODE="${MODE:-client}"
JARVIS_STORAGE_MODE="${JARVIS_STORAGE_MODE:-scratch}"
LOCAL_BASE="${LOCAL_BASE:-/local/${USER:-user}/${SLURM_JOB_ID:-manual}/adarsh-rlms}"
LOCAL_HF_CACHE="$LOCAL_BASE/hf_cache"
LOCAL_VLLM_CACHE="$LOCAL_BASE/vllm_cache"
LOCAL_TMP="$LOCAL_BASE/tmp"

case "$JARVIS_STORAGE_MODE" in
  project)
    PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT:-/mmfs1/project/llm_caching}"
    PROJECT_LOG_DIR="${PROJECT_LOG_DIR:-$PROJECT_CACHE_ROOT/logs}"
    HYDRATE_FROM_PROJECT_CACHE="${HYDRATE_FROM_PROJECT_CACHE:-1}"
    JARVIS_CACHE_STATE_ROOT="${JARVIS_CACHE_STATE_ROOT:-$PROJECT_CACHE_ROOT/cache_state}"
    ;;
  scratch)
    PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT:-/local/${USER:-user}/llm_caching}"
    PROJECT_LOG_DIR="${PROJECT_LOG_DIR:-${HOME:-$REPO_ROOT}/adarsh-rlms-logs}"
    HYDRATE_FROM_PROJECT_CACHE="${HYDRATE_FROM_PROJECT_CACHE:-1}"
    JARVIS_CACHE_STATE_ROOT="${JARVIS_CACHE_STATE_ROOT:-$LOCAL_BASE/cache_state}"
    ;;
  *)
    echo "[JARVIS] unsupported JARVIS_STORAGE_MODE=$JARVIS_STORAGE_MODE; use project or scratch" >&2
    exit 2
    ;;
esac

PROJECT_HF_CACHE="${PROJECT_HF_CACHE:-$PROJECT_CACHE_ROOT/hf_cache}"
PROJECT_VLLM_CACHE="${PROJECT_VLLM_CACHE:-$PROJECT_CACHE_ROOT/vllm_cache}"
SCRATCH_SHARED_NODE_CACHE="${SCRATCH_SHARED_NODE_CACHE:-1}"

if [[ "$JARVIS_STORAGE_MODE" == "scratch" && "$SCRATCH_SHARED_NODE_CACHE" == "1" ]]; then
  LOCAL_HF_CACHE="$PROJECT_HF_CACHE"
  LOCAL_VLLM_CACHE="$PROJECT_VLLM_CACHE"
fi

EXECUTOR_MODEL="${EXECUTOR_MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
EVALUATOR_MODEL="${EVALUATOR_MODEL:-mistralai/Mistral-Small-3.2-24B-Instruct-2506}"
SMOKE_MODEL="${SMOKE_MODEL:-$EVALUATOR_MODEL}"
SMALL_SMOKE_MODEL="${SMALL_SMOKE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EXECUTOR_PORT="${EXECUTOR_PORT:-8000}"
EVALUATOR_PORT="${EVALUATOR_PORT:-8001}"
SMOKE_PORT="${SMOKE_PORT:-8000}"
SMALL_SMOKE_PORT="${SMALL_SMOKE_PORT:-8000}"
EXECUTOR_MAX_MODEL_LEN="${EXECUTOR_MAX_MODEL_LEN:-32768}"
EVALUATOR_MAX_MODEL_LEN="${EVALUATOR_MAX_MODEL_LEN:-16384}"
SMOKE_MAX_MODEL_LEN="${SMOKE_MAX_MODEL_LEN:-8192}"
SMALL_SMOKE_MAX_MODEL_LEN="${SMALL_SMOKE_MAX_MODEL_LEN:-8192}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
HOST="${HOST:-0.0.0.0}"
KEEP_LOCAL="${KEEP_LOCAL:-0}"
SYNC_BACK_MODELS="${SYNC_BACK_MODELS:-0}"
DRY_RUN="${DRY_RUN:-0}"
VLLM_VENV="${VLLM_VENV:-}"
CLIENT_CMD="${CLIENT_CMD:-python -m unittest discover -s test -p test_semantic_cache_llm_provider.py}"
WAIT_FOR_ENDPOINTS="${WAIT_FOR_ENDPOINTS:-0}"
WAIT_FOR_ENDPOINT_TIMEOUT="${WAIT_FOR_ENDPOINT_TIMEOUT:-1800}"
WAIT_FOR_ENDPOINT_INTERVAL="${WAIT_FOR_ENDPOINT_INTERVAL:-10}"
SKIP_LOCAL_SPACE_CHECK="${SKIP_LOCAL_SPACE_CHECK:-0}"
MIN_LOCAL_FREE_GB="${MIN_LOCAL_FREE_GB:-}"
LOCAL_SPACE_CHECK_PATH="${LOCAL_SPACE_CHECK_PATH:-$LOCAL_BASE}"

jarvis_cleanup() {
  local status=$?
  if [[ "$SYNC_BACK_MODELS" == "1" && -d "$LOCAL_HF_CACHE" && "$LOCAL_HF_CACHE" != "$PROJECT_HF_CACHE" ]]; then
    mkdir -p "$PROJECT_HF_CACHE"
    rsync -a --ignore-existing "$LOCAL_HF_CACHE/" "$PROJECT_HF_CACHE/" || true
  fi
  if [[ "$KEEP_LOCAL" != "1" && -n "$LOCAL_BASE" && -d "$LOCAL_BASE" ]]; then
    case "$LOCAL_BASE" in
      /local/*/adarsh-rlms|/private/tmp/*/adarsh-rlms|/tmp/*/adarsh-rlms)
        rm -rf "$LOCAL_BASE"
        ;;
      *)
        echo "[JARVIS] refusing to clean unexpected LOCAL_BASE=$LOCAL_BASE" >&2
        ;;
    esac
  fi
  return "$status"
}

jarvis_default_min_local_free_gb() {
  if [[ -n "$MIN_LOCAL_FREE_GB" ]]; then
    echo "$MIN_LOCAL_FREE_GB"
    return 0
  fi

  case "$MODE" in
    executor|download-executor)
      echo "${EXECUTOR_MIN_LOCAL_FREE_GB:-250}"
      ;;
    small-smoke|download-small-smoke)
      echo "${SMALL_SMOKE_MIN_LOCAL_FREE_GB:-60}"
      ;;
    evaluator|smoke|download-evaluator|download-smoke)
      echo "${EVALUATOR_MIN_LOCAL_FREE_GB:-160}"
      ;;
    download-all|all|download)
      echo "${DOWNLOAD_ALL_MIN_LOCAL_FREE_GB:-350}"
      ;;
    client)
      echo "${CLIENT_MIN_LOCAL_FREE_GB:-30}"
      ;;
    cleanup)
      echo 0
      ;;
    *)
      echo "${DEFAULT_MIN_LOCAL_FREE_GB:-50}"
      ;;
  esac
}

jarvis_require_local_free_space() {
  if [[ "$SKIP_LOCAL_SPACE_CHECK" == "1" ]]; then
    echo "[JARVIS] local free-space check skipped"
    return 0
  fi

  local required_gb
  required_gb="$(jarvis_default_min_local_free_gb)"
  if ! [[ "$required_gb" =~ ^[0-9]+$ ]]; then
    echo "[JARVIS] MIN_LOCAL_FREE_GB must be an integer, got: $required_gb" >&2
    exit 2
  fi
  if (( required_gb == 0 )); then
    return 0
  fi

  local check_path="$LOCAL_SPACE_CHECK_PATH"
  mkdir -p "$check_path"

  local available_kb available_gb
  available_kb="$(df -Pk "$check_path" | awk 'NR == 2 {print $4}')"
  if ! [[ "$available_kb" =~ ^[0-9]+$ ]]; then
    echo "[JARVIS] unable to read free space for $check_path" >&2
    exit 2
  fi
  available_gb=$((available_kb / 1024 / 1024))

  echo "[JARVIS] local free space at $check_path: ${available_gb} GB available; ${required_gb} GB required"
  if (( available_gb < required_gb )); then
    cat >&2 <<EOF
[JARVIS] insufficient local scratch space on $(jarvis_advertised_host)
[JARVIS] path=$check_path
[JARVIS] available=${available_gb}GB required=${required_gb}GB
[JARVIS] Lower MIN_LOCAL_FREE_GB only if you are sure the model is already cached,
[JARVIS] or run jarvis/cleanup_local.sh on this node.
EOF
    exit 3
  fi
}

jarvis_setup_runtime() {
  mkdir -p "$PROJECT_HF_CACHE" "$PROJECT_VLLM_CACHE" "$PROJECT_LOG_DIR" "$JARVIS_CACHE_STATE_ROOT"
  mkdir -p "$LOCAL_HF_CACHE" "$LOCAL_VLLM_CACHE" "$LOCAL_TMP"
  jarvis_require_local_free_space

  if [[ "$HYDRATE_FROM_PROJECT_CACHE" == "1" && -d "$PROJECT_HF_CACHE" && "$PROJECT_HF_CACHE" != "$LOCAL_HF_CACHE" ]]; then
    rsync -a --ignore-existing "$PROJECT_HF_CACHE/" "$LOCAL_HF_CACHE/" || true
  fi

  export HF_HOME="$LOCAL_HF_CACHE"
  export HUGGINGFACE_HUB_CACHE="$LOCAL_HF_CACHE/hub"
  export TRANSFORMERS_CACHE="$LOCAL_HF_CACHE/transformers"
  export VLLM_CACHE_ROOT="$LOCAL_VLLM_CACHE"
  export TMPDIR="$LOCAL_TMP"
  export LLM_PROVIDER="${LLM_PROVIDER:-openai_compatible}"
  export OPENAI_COMPAT_EXECUTOR_MODEL="${OPENAI_COMPAT_EXECUTOR_MODEL:-$EXECUTOR_MODEL}"
  export OPENAI_COMPAT_EVALUATOR_MODEL="${OPENAI_COMPAT_EVALUATOR_MODEL:-$EVALUATOR_MODEL}"
  export JARVIS_STORAGE_MODE PROJECT_CACHE_ROOT PROJECT_HF_CACHE PROJECT_VLLM_CACHE PROJECT_LOG_DIR
  export SMALL_SMOKE_MODEL SMALL_SMOKE_PORT SMALL_SMOKE_MAX_MODEL_LEN
  export LOCAL_BASE LOCAL_HF_CACHE LOCAL_VLLM_CACHE LOCAL_TMP JARVIS_CACHE_STATE_ROOT

  if [[ -n "${MODULES:-}" ]] && type module >/dev/null 2>&1; then
    # MODULES should be a space-separated list, for example: MODULES="cuda python"
    # shellcheck disable=SC2086
    module load $MODULES
  fi

  if [[ -n "$VLLM_VENV" ]]; then
    if [[ ! -f "$VLLM_VENV/bin/activate" ]]; then
      echo "[JARVIS] VLLM_VENV does not contain bin/activate: $VLLM_VENV" >&2
      exit 2
    fi
    # shellcheck disable=SC1091
    source "$VLLM_VENV/bin/activate"
  fi

  trap 'jarvis_cleanup' EXIT
  trap 'trap - EXIT; jarvis_cleanup; exit 130' INT
  trap 'trap - EXIT; jarvis_cleanup; exit 143' TERM
}

jarvis_advertised_host() {
  hostname -f 2>/dev/null || hostname
}

jarvis_print_runtime() {
  local host_name="${1:-$(jarvis_advertised_host)}"
  cat <<EOF
[JARVIS] mode=$MODE
[JARVIS] repo root=$REPO_ROOT
[JARVIS] storage mode=$JARVIS_STORAGE_MODE
[JARVIS] project cache root=$PROJECT_CACHE_ROOT
[JARVIS] project log dir=$PROJECT_LOG_DIR
[JARVIS] cache state root=$JARVIS_CACHE_STATE_ROOT
[JARVIS] local base=$LOCAL_BASE
[JARVIS] local space check path=$LOCAL_SPACE_CHECK_PATH
[JARVIS] HF_HOME=${HF_HOME:-<not set>}
[JARVIS] VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-<not set>}
[JARVIS] VLLM_VENV=${VLLM_VENV:-<not set>}
[JARVIS] host=$host_name
EOF
}
