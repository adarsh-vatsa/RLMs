#!/usr/bin/env bash
# Start one vLLM OpenAI-compatible service inside a Slurm GPU allocation.

#SBATCH --job-name=rlms-vllm
#SBATCH --time=24:00:00

set -euo pipefail

MODE="${MODE:-${1:-smoke}}"
SCRIPT_DIR="${JARVIS_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=jarvis/lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"

jarvis_setup_runtime
advertised_host="$(jarvis_advertised_host)"
jarvis_print_runtime "$advertised_host"

run_vllm() {
  local model="$1"
  local port="$2"
  local tp_size="$3"
  local max_model_len="$4"
  local url_file="$PROJECT_LOG_DIR/${MODE}-${SLURM_JOB_ID:-manual}.url"
  local url="http://${advertised_host}:${port}/v1"
  echo "$url" > "$url_file"
  echo "[JARVIS] endpoint=$url"
  echo "[JARVIS] endpoint file=$url_file"
  echo "[JARVIS] model=$model"
  echo "[JARVIS] tensor parallel size=$tp_size"
  echo "[JARVIS] max model len=$max_model_len"
  echo "[JARVIS] vLLM dtype=$VLLM_DTYPE"
  echo "[JARVIS] vLLM gpu memory utilization=$VLLM_GPU_MEMORY_UTILIZATION"

  local default_extra_args=()
  case "$model" in
    mistralai/Mistral-Small-*)
      default_extra_args+=(
        --tokenizer_mode mistral
        --config_format mistral
        --load_format mistral
        --tool-call-parser mistral
        --enable-auto-tool-choice
      )
      ;;
  esac
  case "$model" in
    mistralai/Mistral-Small-3.*)
      default_extra_args+=(--limit-mm-per-prompt '{"image":10}')
      ;;
  esac

  local extra_args=()
  if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_args <<< "$VLLM_EXTRA_ARGS"
  fi
  local all_extra_args=()
  if [[ ${#default_extra_args[@]} -gt 0 ]]; then
    all_extra_args+=("${default_extra_args[@]}")
  fi
  if [[ ${#extra_args[@]} -gt 0 ]]; then
    all_extra_args+=("${extra_args[@]}")
  fi

  if [[ ${#all_extra_args[@]} -gt 0 ]]; then
    echo "[JARVIS] vLLM extra args=${all_extra_args[*]}"
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    local dry_run_cmd="python -m vllm.entrypoints.openai.api_server --host $HOST --port $port --model $model --served-model-name $model --tensor-parallel-size $tp_size --max-model-len $max_model_len --dtype $VLLM_DTYPE --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION --download-dir $LOCAL_HF_CACHE"
    if [[ ${#all_extra_args[@]} -gt 0 ]]; then
      echo "[JARVIS] dry run: $dry_run_cmd ${all_extra_args[*]}"
    else
      echo "[JARVIS] dry run: $dry_run_cmd"
    fi
    return 0
  fi

  python -m vllm.entrypoints.openai.api_server \
    --host "$HOST" \
    --port "$port" \
    --model "$model" \
    --served-model-name "$model" \
    --tensor-parallel-size "$tp_size" \
    --max-model-len "$max_model_len" \
    --dtype "$VLLM_DTYPE" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --download-dir "$LOCAL_HF_CACHE" \
    ${all_extra_args[@]+"${all_extra_args[@]}"}
}

case "$MODE" in
  executor)
    run_vllm "$EXECUTOR_MODEL" "$EXECUTOR_PORT" "${EXECUTOR_TP_SIZE:-4}" "$EXECUTOR_MAX_MODEL_LEN"
    ;;
  evaluator)
    run_vllm "$EVALUATOR_MODEL" "$EVALUATOR_PORT" "${EVALUATOR_TP_SIZE:-2}" "$EVALUATOR_MAX_MODEL_LEN"
    ;;
  small-smoke)
    run_vllm "$SMALL_SMOKE_MODEL" "$SMALL_SMOKE_PORT" "${SMALL_SMOKE_TP_SIZE:-1}" "$SMALL_SMOKE_MAX_MODEL_LEN"
    ;;
  smoke)
    run_vllm "$SMOKE_MODEL" "$SMOKE_PORT" "${SMOKE_TP_SIZE:-2}" "$SMOKE_MAX_MODEL_LEN"
    ;;
  *)
    echo "Unsupported serve_vllm MODE: $MODE" >&2
    exit 2
    ;;
esac
