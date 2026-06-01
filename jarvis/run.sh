#!/usr/bin/env bash
# User-facing Jarvis dispatcher. Use this from the login node to submit jobs.

#SBATCH --job-name=adarsh-rlms
#SBATCH --time=24:00:00

set -euo pipefail

MODE="${MODE:-${1:-help}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=jarvis/lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"

usage() {
  cat <<'EOF'
Usage:
  bash jarvis/run.sh submit <executor|evaluator|smoke|client|download-executor|download-evaluator|download-all>

Common commands:
  VLLM_VENV=/home/edogu/.venvs/adarsh-vllm bash jarvis/run.sh submit executor
  VLLM_VENV=/home/edogu/.venvs/adarsh-vllm bash jarvis/run.sh submit evaluator
  VLLM_VENV=/home/edogu/.venvs/adarsh-vllm bash jarvis/run.sh submit download-all
  CLIENT_CMD="uv run python ..." bash jarvis/run.sh submit client

Direct sbatch is also supported if you pass resources yourself:
  MODE=executor sbatch --partition=gpu-l40s --gres=gpu:l40s:4 jarvis/run.sh
  MODE=client sbatch --partition=compute-short jarvis/run.sh

Do not run service modes directly on the login node.
EOF
}

submit_mode() {
  local submit_mode="${1:-client}"
  mkdir -p "$PROJECT_LOG_DIR"

  case "$submit_mode" in
    executor)
      sbatch \
        --partition=gpu-l40s \
        --gres=gpu:l40s:4 \
        --cpus-per-task=32 \
        --mem=220G \
        --time=24:00:00 \
        --job-name=rlms-executor \
        --output="$PROJECT_LOG_DIR/%x-%j.out" \
        --export=ALL,MODE=executor \
        "$SCRIPT_DIR/serve_vllm.sh"
      ;;
    evaluator)
      sbatch \
        --partition=gpu-l40s \
        --gres=gpu:l40s:2 \
        --cpus-per-task=16 \
        --mem=140G \
        --time=24:00:00 \
        --job-name=rlms-evaluator \
        --output="$PROJECT_LOG_DIR/%x-%j.out" \
        --export=ALL,MODE=evaluator \
        "$SCRIPT_DIR/serve_vllm.sh"
      ;;
    smoke)
      sbatch \
        --partition=gpu-l40s \
        --gres=gpu:l40s:2 \
        --cpus-per-task=16 \
        --mem=140G \
        --time=04:00:00 \
        --job-name=rlms-smoke \
        --output="$PROJECT_LOG_DIR/%x-%j.out" \
        --export=ALL,MODE=smoke \
        "$SCRIPT_DIR/serve_vllm.sh"
      ;;
    client)
      sbatch \
        --partition=compute-short \
        --cpus-per-task=8 \
        --mem=32G \
        --time=12:00:00 \
        --job-name=rlms-client \
        --output="$PROJECT_LOG_DIR/%x-%j.out" \
        --export=ALL,MODE=client \
        "$SCRIPT_DIR/run_client.sh"
      ;;
    download-executor|download-evaluator|download-smoke|download-all)
      sbatch \
        --partition=compute-short \
        --cpus-per-task=8 \
        --mem=32G \
        --time=24:00:00 \
        --job-name=rlms-download \
        --output="$PROJECT_LOG_DIR/%x-%j.out" \
        --export=ALL,MODE="$submit_mode" \
        "$SCRIPT_DIR/download_models.sh"
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

delegate_inside_slurm() {
  case "$MODE" in
    executor|evaluator|smoke)
      exec bash "$SCRIPT_DIR/serve_vllm.sh"
      ;;
    client)
      exec bash "$SCRIPT_DIR/run_client.sh"
      ;;
    download-executor|download-evaluator|download-smoke|download-all)
      exec bash "$SCRIPT_DIR/download_models.sh"
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

case "${1:-}" in
  submit)
    submit_mode "${2:-client}"
    ;;
  help|-h|--help)
    usage
    ;;
  "")
    if [[ -n "${SLURM_JOB_ID:-}" && "$MODE" != "help" ]]; then
      delegate_inside_slurm
    else
      usage
      if [[ "$MODE" != "help" ]]; then
        exit 2
      fi
    fi
    ;;
  *)
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
      delegate_inside_slurm
    else
      usage
      exit 2
    fi
    ;;
esac
