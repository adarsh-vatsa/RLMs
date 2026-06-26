#!/usr/bin/env bash
# User-facing Jarvis dispatcher. Use this from the login node to submit jobs.

#SBATCH --job-name=adarsh-rlms
#SBATCH --time=24:00:00

set -euo pipefail

MODE="${MODE:-${1:-help}}"
SCRIPT_DIR="${JARVIS_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export JARVIS_SCRIPT_DIR="$SCRIPT_DIR"
# shellcheck source=jarvis/lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"

usage() {
  cat <<'EOF'
Usage:
  bash jarvis/run.sh submit <small-smoke|executor|evaluator|smoke|client|download-small-smoke|download-executor|download-evaluator|download-all|cleanup>

Common commands:
  JARVIS_STORAGE_MODE=scratch PROJECT_LOG_DIR=/home/edogu/adarsh-rlms-logs bash jarvis/run.sh submit small-smoke
  JARVIS_STORAGE_MODE=scratch PROJECT_LOG_DIR=/home/edogu/adarsh-rlms-logs bash jarvis/run.sh submit smoke
  JARVIS_STORAGE_MODE=scratch CLEANUP_NODE=g101 bash jarvis/run.sh submit cleanup
  VLLM_VENV=/home/edogu/.venvs/adarsh-vllm bash jarvis/run.sh submit executor
  VLLM_VENV=/home/edogu/.venvs/adarsh-vllm bash jarvis/run.sh submit evaluator
  VLLM_VENV=/home/edogu/.venvs/adarsh-vllm bash jarvis/run.sh submit download-all
  CLIENT_CMD="uv run python ..." bash jarvis/run.sh submit client
  CLIENT_MEM=96G CLIENT_CMD="uv run python ..." bash jarvis/run.sh submit client

Direct sbatch is also supported if you pass resources and JARVIS_SCRIPT_DIR yourself:
  sbatch --partition=gpu-l40s --gres=gpu:l40s:4 --export=ALL,JARVIS_SCRIPT_DIR=/path/to/adarsh-rlms/jarvis,MODE=executor /path/to/adarsh-rlms/jarvis/run.sh
  sbatch --partition=compute-short --export=ALL,JARVIS_SCRIPT_DIR=/path/to/adarsh-rlms/jarvis,MODE=client /path/to/adarsh-rlms/jarvis/run.sh

Do not run service modes directly on the login node.
EOF
}

submit_mode() {
  local submit_mode="${1:-client}"
  mkdir -p "$PROJECT_LOG_DIR"

  case "$submit_mode" in
    small-smoke)
      sbatch \
        --partition=gpu-l40s \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=04:00:00 \
        --job-name=rlms-small-smoke \
        --output="$PROJECT_LOG_DIR/%x-%j.out" \
        --export=ALL,JARVIS_SCRIPT_DIR="$SCRIPT_DIR",MODE=small-smoke \
        "$SCRIPT_DIR/serve_vllm.sh"
      ;;
    executor)
      sbatch \
        --partition=gpu-l40s \
        --gres=gpu:l40s:4 \
        --cpus-per-task=32 \
        --mem=220G \
        --time=24:00:00 \
        --job-name=rlms-executor \
        --output="$PROJECT_LOG_DIR/%x-%j.out" \
        --export=ALL,JARVIS_SCRIPT_DIR="$SCRIPT_DIR",MODE=executor \
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
        --export=ALL,JARVIS_SCRIPT_DIR="$SCRIPT_DIR",MODE=evaluator \
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
        --export=ALL,JARVIS_SCRIPT_DIR="$SCRIPT_DIR",MODE=smoke \
        "$SCRIPT_DIR/serve_vllm.sh"
      ;;
    client)
      sbatch \
        --partition=compute-short \
        --cpus-per-task=8 \
        --mem="${CLIENT_MEM:-32G}" \
        --time=12:00:00 \
        --job-name=rlms-client \
        --output="$PROJECT_LOG_DIR/%x-%j.out" \
        --export=ALL,JARVIS_SCRIPT_DIR="$SCRIPT_DIR",MODE=client \
        "$SCRIPT_DIR/run_client.sh"
      ;;
    download-small-smoke|download-executor|download-evaluator|download-smoke|download-all)
      sbatch \
        --partition=compute-short \
        --cpus-per-task=8 \
        --mem=32G \
        --time=24:00:00 \
        --job-name=rlms-download \
        --output="$PROJECT_LOG_DIR/%x-%j.out" \
        --export=ALL,JARVIS_SCRIPT_DIR="$SCRIPT_DIR",MODE="$submit_mode" \
        "$SCRIPT_DIR/download_models.sh"
      ;;
    cleanup)
      local cleanup_args=()
      if [[ -n "${CLEANUP_NODE:-}" ]]; then
        cleanup_args+=(--nodelist="$CLEANUP_NODE")
      fi
      sbatch \
        --partition="${CLEANUP_PARTITION:-gpu-l40s}" \
        "${cleanup_args[@]}" \
        --cpus-per-task=1 \
        --mem=4G \
        --time=00:30:00 \
        --job-name=rlms-cleanup \
        --output="$PROJECT_LOG_DIR/%x-%j.out" \
        --export=ALL,JARVIS_SCRIPT_DIR="$SCRIPT_DIR",MODE=cleanup \
        "$SCRIPT_DIR/cleanup_local.sh"
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

delegate_inside_slurm() {
  case "$MODE" in
    executor|evaluator|smoke|small-smoke)
      exec bash "$SCRIPT_DIR/serve_vllm.sh"
      ;;
    client)
      exec bash "$SCRIPT_DIR/run_client.sh"
      ;;
    download-small-smoke|download-executor|download-evaluator|download-smoke|download-all)
      exec bash "$SCRIPT_DIR/download_models.sh"
      ;;
    cleanup)
      exec bash "$SCRIPT_DIR/cleanup_local.sh"
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
