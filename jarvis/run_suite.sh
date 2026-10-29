#!/usr/bin/env bash
# Submit one workshop suite (linux/run_<benchmark>_suite.sh) as a Jarvis job.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  echo 'Usage: bash jarvis/run_suite.sh <aa_lcr|mrcr|longbench> [suite-id]'
  echo 'From the login node, submits one job that runs that benchmark'"'"'s workshop suite against'
  echo 'the executor at OPENAI_COMPAT_EXECUTOR_BASE_URL; the job waits for the executor to come up.'
  echo 'Submit again with the same suite ID to resume. The suite'"'"'s own variables pass through:'
  echo 'ONLY, AA_LCR_RECIPE, MRCR_RECIPE, LONGBENCH_RECIPE, MRCR_ROWS_PER_SOURCE, AA_LCR_GRADER.'
  echo 'AA_LCR_GRADER=luna submits a CPU job that regrades through OpenRouter and needs no executor.'
  echo 'DRY_RUN=1 prints the suite'"'"'s commands on this node without submitting.'
}
case "${1:-}" in
  aa_lcr) METADATA="${AA_LCR_SERVING_METADATA:-.cache/serving_metadata.json}" ;;
  mrcr) METADATA="${MRCR_SERVING_METADATA:-.cache/serving_metadata.json}" ;;
  longbench) METADATA="${LONGBENCH_SERVING_METADATA:-.cache/serving_metadata.json}" ;;
  help|-h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
SUITE_SCRIPT="linux/run_$1_suite.sh"
SUITE="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
cd "$REPO_ROOT"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  exec bash "$SUITE_SCRIPT" "$SUITE"
fi

export SUITE_RESUME="bash $(basename "$REPO_ROOT")/jarvis/run_suite.sh $1 $SUITE"
export CLIENT_CMD="bash $SUITE_SCRIPT $SUITE"
if [[ "$1" == aa_lcr && "${AA_LCR_GRADER:-}" == luna ]]; then
  MODE=client
else
  if [[ -z "${OPENAI_COMPAT_EXECUTOR_BASE_URL:-}" ]]; then
    echo "Set OPENAI_COMPAT_EXECUTOR_BASE_URL to the executor job's endpoint" >&2
    exit 1
  fi
  if [[ ! -f "$METADATA" ]]; then
    echo "Serving metadata not found: $METADATA (record it once per executor job)" >&2
    exit 1
  fi
  source "$SCRIPT_DIR/lib/execution_services.sh"
  jarvis_execution_services 0 0 --execution-profile common
  export WAIT_FOR_ENDPOINTS=1
  MODE=client-gpu
fi
echo "[JARVIS SUITE] Submitting $1 suite $SUITE as a $MODE job: $CLIENT_CMD"
echo "[JARVIS SUITE] Resume or repeat with: $SUITE_RESUME"
exec bash "$SCRIPT_DIR/run.sh" submit "$MODE"
