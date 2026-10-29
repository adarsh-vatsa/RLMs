#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source "$LINUX_SCRIPT_DIR/suite_common.sh"

usage() {
  echo 'Usage: bash linux/run_longbench_suite.sh [suite-id]'
  echo 'Runs every LongBench-v2 workshop condition in turn; answers are scored by letter as they run.'
  echo 'On Jarvis, submit it with: bash jarvis/run_suite.sh longbench [suite-id]'
  echo 'Rerun with the same suite ID to resume: finished runs are skipped, and an'
  echo 'unfinished folder is moved aside before it is retried.'
  echo 'ONLY=name,... runs a subset. LONGBENCH_RECIPE="<hybrid flags>" adds the upgraded-recipe runs.'
  echo 'DRY_RUN=1 prints the commands without running them.'
}
case "${1:-}" in help|-h|--help) usage; exit 0 ;; esac
cd "$REPO_ROOT"

SUITE="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
SUITE_LABEL="LONGBENCH SUITE"
ROOT="benchmark_artifacts/longbench_v2/suites/$SUITE"
LOG_DIR="$ROOT/logs"
SERVING_METADATA="${LONGBENCH_SERVING_METADATA:-.cache/serving_metadata.json}"
EXECUTOR_URL="${OPENAI_COMPAT_EXECUTOR_BASE_URL:-http://127.0.0.1:8000/v1}"
ONLY="${ONLY:-}"

# Original rows whose full prompt is over the 64K budget but fits the 262K window,
# so each has a full-context reference; every condition uses the same 221 questions.
COMMON=(--output-dir "$ROOT" --min-source-tokens 65529 --max-source-tokens 262136 --fail-fast
  --serving-metadata "$SERVING_METADATA")
B262="--context-window-tokens 262144 --max-input-tokens 262136"
B64="--context-window-tokens 65536 --max-input-tokens 65528"
B32="--context-window-tokens 32768 --max-input-tokens 32760"
B128="--context-window-tokens 131072 --max-input-tokens 131064"
LADDER="--evidence-selection fusion --child-tokens 1024 --child-overlap-tokens 128"
condition full          direct "$B262"
condition middle        direct "$B64 --direct-overflow middle"
condition dense         hybrid "$B64"
condition position      hybrid "$B64 --evidence-selection position"
condition random0       hybrid "$B64 --evidence-selection random --selection-seed 0"
condition middle_marked direct "$B64 --direct-overflow middle-marked"
condition head          direct "$B64 --direct-overflow head"
condition bm25          hybrid "$B64 --evidence-selection bm25"
condition fusion        hybrid "$B64 --evidence-selection fusion"
condition ladder1       hybrid "$B64 $LADDER"
condition ladder2       hybrid "$B64 $LADDER --neighbor-children 1"
condition ladder3       hybrid "$B64 $LADDER --neighbor-children 1 --packing fill"
condition ladder4       hybrid "$B64 $LADDER --neighbor-children 1 --packing fill --document-floor"
condition middle_32k    direct "$B32 --direct-overflow middle"
condition position_32k  hybrid "$B32 --evidence-selection position"
condition middle_128k   direct "$B128 --direct-overflow middle"
condition position_128k hybrid "$B128 --evidence-selection position"
if [[ -n "${LONGBENCH_RECIPE:-}" ]]; then
  condition upgraded      hybrid "$B64 $LONGBENCH_RECIPE"
  condition upgraded_32k  hybrid "$B32 $LONGBENCH_RECIPE"
  condition upgraded_128k hybrid "$B128 $LONGBENCH_RECIPE"
fi
select_conditions

# The direct and hybrid runners write to differently named folders, each with
# manifest.json written last.
run_path() {
  if [[ "$1" == direct ]]; then echo "$ROOT/longbench_v2_api/${SUITE}_$2"; else echo "$ROOT/longbench_v2/${SUITE}_$2"; fi
}
finished() { [[ -f "$1/manifest.json" ]]; }

RESUME="bash linux/run_longbench_suite.sh $SUITE"
start_suite "executor $EXECUTOR_URL"
check_executor
run_conditions longbench_v2 "${COMMON[@]}"

step "Results table:"
run_command "$BENCHMARK_VENV/bin/python" -m long_bench_v2.results_table "$ROOT" \
  || step "Results table failed; the runs are unaffected"
finish_suite
