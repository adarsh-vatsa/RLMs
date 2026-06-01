#!/usr/bin/env bash
# Shared Jarvis runtime setup. Source this file from role scripts; do not run it.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "jarvis/lib/env.sh must be sourced, not executed." >&2
  exit 2
fi

JARVIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$JARVIS_DIR/.." && pwd)"

MODE="${MODE:-client}"
PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT:-/mmfs1/project/llm_caching}"
PROJECT_HF_CACHE="${PROJECT_HF_CACHE:-$PROJECT_CACHE_ROOT/hf_cache}"
PROJECT_VLLM_CACHE="${PROJECT_VLLM_CACHE:-$PROJECT_CACHE_ROOT/vllm_cache}"
PROJECT_LOG_DIR="${PROJECT_LOG_DIR:-$PROJECT_CACHE_ROOT/logs}"
LOCAL_BASE="${LOCAL_BASE:-/local/${USER:-user}/${SLURM_JOB_ID:-manual}/adarsh-rlms}"
LOCAL_HF_CACHE="$LOCAL_BASE/hf_cache"
LOCAL_VLLM_CACHE="$LOCAL_BASE/vllm_cache"
LOCAL_TMP="$LOCAL_BASE/tmp"

EXECUTOR_MODEL="${EXECUTOR_MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
EVALUATOR_MODEL="${EVALUATOR_MODEL:-mistralai/Mistral-Small-3.2-24B-Instruct-2506}"
SMOKE_MODEL="${SMOKE_MODEL:-$EVALUATOR_MODEL}"
EXECUTOR_PORT="${EXECUTOR_PORT:-8000}"
EVALUATOR_PORT="${EVALUATOR_PORT:-8001}"
SMOKE_PORT="${SMOKE_PORT:-8000}"
EXECUTOR_MAX_MODEL_LEN="${EXECUTOR_MAX_MODEL_LEN:-32768}"
EVALUATOR_MAX_MODEL_LEN="${EVALUATOR_MAX_MODEL_LEN:-16384}"
SMOKE_MAX_MODEL_LEN="${SMOKE_MAX_MODEL_LEN:-8192}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
HOST="${HOST:-0.0.0.0}"
KEEP_LOCAL="${KEEP_LOCAL:-0}"
HYDRATE_FROM_PROJECT_CACHE="${HYDRATE_FROM_PROJECT_CACHE:-1}"
SYNC_BACK_MODELS="${SYNC_BACK_MODELS:-0}"
DRY_RUN="${DRY_RUN:-0}"
VLLM_VENV="${VLLM_VENV:-}"
CLIENT_CMD="${CLIENT_CMD:-python -m unittest discover -s test -p test_semantic_cache_llm_provider.py}"
WAIT_FOR_ENDPOINTS="${WAIT_FOR_ENDPOINTS:-0}"
WAIT_FOR_ENDPOINT_TIMEOUT="${WAIT_FOR_ENDPOINT_TIMEOUT:-1800}"
WAIT_FOR_ENDPOINT_INTERVAL="${WAIT_FOR_ENDPOINT_INTERVAL:-10}"

jarvis_cleanup() {
  local status=$?
  if [[ "$SYNC_BACK_MODELS" == "1" && -d "$LOCAL_HF_CACHE" ]]; then
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

jarvis_setup_runtime() {
  mkdir -p "$PROJECT_HF_CACHE" "$PROJECT_VLLM_CACHE" "$PROJECT_LOG_DIR"
  mkdir -p "$LOCAL_HF_CACHE" "$LOCAL_VLLM_CACHE" "$LOCAL_TMP"

  if [[ "$HYDRATE_FROM_PROJECT_CACHE" == "1" && -d "$PROJECT_HF_CACHE" ]]; then
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
[JARVIS] project cache root=$PROJECT_CACHE_ROOT
[JARVIS] local base=$LOCAL_BASE
[JARVIS] HF_HOME=${HF_HOME:-<not set>}
[JARVIS] VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-<not set>}
[JARVIS] VLLM_VENV=${VLLM_VENV:-<not set>}
[JARVIS] host=$host_name
EOF
}
