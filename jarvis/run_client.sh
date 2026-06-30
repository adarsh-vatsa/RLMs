#!/usr/bin/env bash
# Run benchmark/client commands inside a Slurm client allocation.

#SBATCH --job-name=rlms-client
#SBATCH --time=12:00:00

set -euo pipefail

MODE="${MODE:-client}"
SCRIPT_DIR="${JARVIS_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=jarvis/lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"

jarvis_setup_runtime
jarvis_print_runtime "$(jarvis_advertised_host)"

cd "$REPO_ROOT"

export OPENAI_COMPAT_BASE_URL="${OPENAI_COMPAT_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_COMPAT_EXECUTOR_BASE_URL="${OPENAI_COMPAT_EXECUTOR_BASE_URL:-$OPENAI_COMPAT_BASE_URL}"
export OPENAI_COMPAT_EVALUATOR_BASE_URL="${OPENAI_COMPAT_EVALUATOR_BASE_URL:-$OPENAI_COMPAT_BASE_URL}"

echo "[JARVIS] executor url=$OPENAI_COMPAT_EXECUTOR_BASE_URL"
echo "[JARVIS] evaluator url=$OPENAI_COMPAT_EVALUATOR_BASE_URL"
echo "[JARVIS] client command=$CLIENT_CMD"

check_endpoint_once() {
  local models_url="${1%/}/models"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 5 "$models_url" >/dev/null
    return $?
  fi

  python - "$models_url" <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
        sys.exit(0 if 200 <= response.status < 500 else 1)
except Exception:
    sys.exit(1)
PY
}

wait_for_endpoint() {
  local label="$1"
  local base_url="$2"
  local deadline=$((SECONDS + WAIT_FOR_ENDPOINT_TIMEOUT))

  echo "[JARVIS] waiting for $label endpoint: ${base_url%/}/models"
  until check_endpoint_once "$base_url"; do
    if (( SECONDS >= deadline )); then
      echo "[JARVIS] timed out waiting for $label endpoint: $base_url" >&2
      return 1
    fi
    sleep "$WAIT_FOR_ENDPOINT_INTERVAL"
  done
  echo "[JARVIS] $label endpoint is reachable"
}

if [[ "$WAIT_FOR_ENDPOINTS" == "1" ]]; then
  wait_for_endpoint "executor" "$OPENAI_COMPAT_EXECUTOR_BASE_URL"
  if [[ "$OPENAI_COMPAT_EVALUATOR_BASE_URL" != "$OPENAI_COMPAT_EXECUTOR_BASE_URL" ]]; then
    wait_for_endpoint "evaluator" "$OPENAI_COMPAT_EVALUATOR_BASE_URL"
  fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[JARVIS] dry run: client command skipped"
  exit 0
fi

bash -lc "$CLIENT_CMD"
