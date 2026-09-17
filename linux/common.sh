#!/usr/bin/env bash

LINUX_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$LINUX_SCRIPT_DIR/.." && pwd)"
BENCHMARK_VENV="${BENCHMARK_VENV:-$REPO_ROOT/.venv}"
VLLM_VENV="${VLLM_VENV:-$HOME/.venvs/adarsh-vllm}"
export PYTHONUNBUFFERED=1

run_command() {
  printf '[LINUX] command:'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    "$@"
  fi
}
