#!/usr/bin/env bash
# Inspect or clean Jarvis node-local scratch. Defaults to dry-run behavior.

#SBATCH --job-name=rlms-cleanup
#SBATCH --time=00:30:00

set -euo pipefail

MODE="${MODE:-cleanup}"
JARVIS_STORAGE_MODE="${JARVIS_STORAGE_MODE:-scratch}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=jarvis/lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"

CONFIRM_CLEANUP="${CONFIRM_CLEANUP:-0}"
CLEAN_NODE_CACHE="${CLEAN_NODE_CACHE:-0}"
CLEAN_JOB_SCRATCH="${CLEAN_JOB_SCRATCH:-1}"

safe_rm_dir() {
  local target="$1"
  if [[ -z "$target" || ! -e "$target" ]]; then
    return 0
  fi

  case "$target" in
    /local/*/llm_caching|/local/*/*/adarsh-rlms|/private/tmp/*|/tmp/*)
      ;;
    *)
      echo "[JARVIS] refusing to remove unguarded path: $target" >&2
      return 2
      ;;
  esac

  if [[ "$CONFIRM_CLEANUP" == "1" ]]; then
    echo "[JARVIS] removing $target"
    rm -rf "$target"
  else
    echo "[JARVIS] dry run: would remove $target"
  fi
}

print_usage() {
  cat <<EOF
[JARVIS] cleanup host=$(jarvis_advertised_host)
[JARVIS] CONFIRM_CLEANUP=$CONFIRM_CLEANUP
[JARVIS] CLEAN_JOB_SCRATCH=$CLEAN_JOB_SCRATCH
[JARVIS] CLEAN_NODE_CACHE=$CLEAN_NODE_CACHE
[JARVIS] node cache root=$PROJECT_CACHE_ROOT
[JARVIS] user local root=/local/${USER:-user}
EOF
}

active_job_ids() {
  if command -v squeue >/dev/null 2>&1; then
    squeue -h -u "${USER:-}" -o "%i" 2>/dev/null || true
  fi
}

is_active_job_dir() {
  local path="$1"
  local job_id
  job_id="$(basename "$(dirname "$path")")"
  active_job_ids | grep -Fxq "$job_id"
}

print_usage
df -h /local 2>/dev/null || true

if [[ -d "$PROJECT_CACHE_ROOT" ]]; then
  du -sh "$PROJECT_CACHE_ROOT" 2>/dev/null || true
fi

if [[ "$CLEAN_JOB_SCRATCH" == "1" && -d "/local/${USER:-user}" ]]; then
  while IFS= read -r path; do
    if is_active_job_dir "$path"; then
      echo "[JARVIS] keeping active job scratch: $path"
      continue
    fi
    safe_rm_dir "$path"
  done < <(find "/local/${USER:-user}" -mindepth 2 -maxdepth 2 -type d -name adarsh-rlms 2>/dev/null | sort)
fi

if [[ "$CLEAN_NODE_CACHE" == "1" ]]; then
  safe_rm_dir "$PROJECT_CACHE_ROOT"
fi

if [[ "$CONFIRM_CLEANUP" != "1" ]]; then
  echo "[JARVIS] dry run only. Re-run with CONFIRM_CLEANUP=1 to delete listed paths."
fi
