#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source "$LINUX_SCRIPT_DIR/suite_common.sh"

usage() {
  echo 'Usage: bash linux/run_mrcr_suite.sh [suite-id]'
  echo 'Runs every MRCR v2 workshop condition in turn; MRCR scores each answer as it runs.'
  echo 'On Jarvis, submit it with: bash jarvis/run_suite.sh mrcr [suite-id]'
  echo 'Rerun with the same suite ID to resume: finished runs are skipped, and an'
  echo 'unfinished folder is moved aside before it is retried.'
  echo 'MRCR_ROWS_PER_SOURCE=N sets the questions per conversation (default 30; 0 = all 953 questions).'
  echo 'Use one value per suite ID: a different N selects different questions.'
  echo 'ONLY=name,... runs a subset. MRCR_RECIPE="<hybrid flags>" adds the upgraded-recipe runs.'
  echo 'DRY_RUN=1 prints the commands without running them.'
}
case "${1:-}" in help|-h|--help) usage; exit 0 ;; esac
cd "$REPO_ROOT"

SUITE="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
SUITE_LABEL="MRCR SUITE"
ROOT="benchmark_artifacts/mrcr_v2/suites/$SUITE"
LOG_DIR="$ROOT/logs"
SERVING_METADATA="${MRCR_SERVING_METADATA:-.cache/serving_metadata.json}"
EXECUTOR_URL="${OPENAI_COMPAT_EXECUTOR_BASE_URL:-http://127.0.0.1:8000/v1}"
ONLY="${ONLY:-}"
export MRCR_DATA_DIR="${MRCR_DATA_DIR:-benchmark_data/mrcr_v2_4needle_100k_1200k}"
ROWS_PER_SOURCE="${MRCR_ROWS_PER_SOURCE:-30}"
COMMON=(--output-root "$ROOT" --max-output-tokens 4096 --max-rows-per-source "$ROWS_PER_SOURCE" --fail-fast
  --serving-metadata "$SERVING_METADATA")

B262="--context-window-tokens 262144 --max-input-tokens 258048"
B32="--context-window-tokens 32768 --max-input-tokens 28672"
B64="--context-window-tokens 65536 --max-input-tokens 61440"
B128="--context-window-tokens 131072 --max-input-tokens 126976"
CHUNKS="--child-tokens 3500 --child-overlap-tokens 350"
LADDER="--evidence-selection fusion --child-tokens 1024 --child-overlap-tokens 128"
condition middle         direct "$B262 --direct-overflow middle"
condition dense          hybrid "$B262 $CHUNKS"
condition position       hybrid "$B262 $CHUNKS --evidence-selection position"
condition random0        hybrid "$B262 $CHUNKS --evidence-selection random --selection-seed 0"
condition middle_marked  direct "$B262 --direct-overflow middle-marked"
condition head           direct "$B262 --direct-overflow head"
condition bm25           hybrid "$B262 $CHUNKS --evidence-selection bm25"
condition fusion         hybrid "$B262 $CHUNKS --evidence-selection fusion"
condition dense_7500     hybrid "$B262 --child-tokens 7500 --child-overlap-tokens 750"
condition dense_question hybrid "$B262 $CHUNKS --retrieval-query question"
condition ladder1        hybrid "$B262 $LADDER"
condition ladder2        hybrid "$B262 $LADDER --neighbor-children 1"
condition ladder3        hybrid "$B262 $LADDER --neighbor-children 1 --packing fill"
condition ladder4        hybrid "$B262 $LADDER --neighbor-children 1 --packing fill --document-floor"
condition middle_32k     direct "$B32 --direct-overflow middle"
condition position_32k   hybrid "$B32 $CHUNKS --evidence-selection position"
condition middle_64k     direct "$B64 --direct-overflow middle"
condition position_64k   hybrid "$B64 $CHUNKS --evidence-selection position"
condition middle_128k    direct "$B128 --direct-overflow middle"
condition position_128k  hybrid "$B128 $CHUNKS --evidence-selection position"
if [[ -n "${MRCR_RECIPE:-}" ]]; then
  condition upgraded      hybrid "$B262 $MRCR_RECIPE"
  condition upgraded_32k  hybrid "$B32 $MRCR_RECIPE"
  condition upgraded_64k  hybrid "$B64 $MRCR_RECIPE"
  condition upgraded_128k hybrid "$B128 $MRCR_RECIPE"
fi
select_conditions

# The runner writes manifest.json at the start and marks it complete at the end.
finished() { grep -q '"status": "complete"' "$1/manifest.json" 2>/dev/null; }

RESUME="bash linux/run_mrcr_suite.sh $SUITE"
start_suite "data $MRCR_DATA_DIR, $ROWS_PER_SOURCE questions per conversation (0 = all)"
check_executor
run_conditions mrcr_v2 "${COMMON[@]}"

step "Results table:"
run_command "$BENCHMARK_VENV/bin/python" -m mrcr_v2.results_table "$ROOT" \
  || step "Results table failed; the runs are unaffected"
finish_suite
