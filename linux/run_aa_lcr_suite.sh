#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source "$LINUX_SCRIPT_DIR/suite_common.sh"

usage() {
  echo 'Usage: bash linux/run_aa_lcr_suite.sh [suite-id]'
  echo 'Runs every AA-LCR workshop condition in turn and grades each with the executor when it finishes.'
  echo 'On Jarvis, submit it with: bash jarvis/run_suite.sh aa_lcr [suite-id]'
  echo 'Rerun with the same suite ID to resume: finished runs and grades are skipped, and an'
  echo 'unfinished folder is moved aside before it is retried.'
  echo 'ONLY=name,... runs a subset. AA_LCR_RECIPE="<hybrid flags>" adds the upgraded-recipe runs.'
  echo 'AA_LCR_GRADER=luna regrades the finished runs with GPT-5.6 Luna through OpenRouter instead.'
  echo 'DRY_RUN=1 prints the commands without running them.'
}
case "${1:-}" in help|-h|--help) usage; exit 0 ;; esac
cd "$REPO_ROOT"

SUITE="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
SUITE_LABEL="AA-LCR SUITE"
ROOT="benchmark_artifacts/aa_lcr/suites/$SUITE"
LOG_DIR="$ROOT/logs"
DATA_DIR="${AA_LCR_DATA_DIR:-benchmark_data/aa_lcr/v1.1}"
SERVING_METADATA="${AA_LCR_SERVING_METADATA:-.cache/serving_metadata.json}"
EXECUTOR_URL="${OPENAI_COMPAT_EXECUTOR_BASE_URL:-http://127.0.0.1:8000/v1}"
EXECUTOR_MODEL="${OPENAI_COMPAT_EXECUTOR_MODEL:-Qwen/Qwen3.6-35B-A3B}"
GRADER="${AA_LCR_GRADER:-executor}"
ONLY="${ONLY:-}"
# Runs are graded afterwards by the executor (then by Luna); naming it here keeps a
# host's default evaluator, such as Jarvis's, out of the run manifests.
COMMON=(--output-root "$ROOT" --execution-only --max-output-tokens 4096 --fail-fast --serving-metadata "$SERVING_METADATA"
  --evaluator-model "$EXECUTOR_MODEL" --evaluator-base-url "$EXECUTOR_URL")

B64="--context-window-tokens 65536 --max-input-tokens 61440"
B32="--context-window-tokens 32768 --max-input-tokens 28672"
B80="--context-window-tokens 81920 --max-input-tokens 77824"
condition full          direct "--context-window-tokens 262144 --max-input-tokens 258048"
condition middle        direct "$B64 --direct-overflow middle"
condition middle_marked direct "$B64 --direct-overflow middle-marked"
condition head          direct "$B64 --direct-overflow head"
condition dense         hybrid "$B64"
condition position      hybrid "$B64 --evidence-selection position"
condition random0       hybrid "$B64 --evidence-selection random --selection-seed 0"
condition random1       hybrid "$B64 --evidence-selection random --selection-seed 1"
condition random2       hybrid "$B64 --evidence-selection random --selection-seed 2"
condition bm25          hybrid "$B64 --evidence-selection bm25"
condition fusion        hybrid "$B64 --evidence-selection fusion"
condition middle_32k    direct "$B32 --direct-overflow middle"
condition position_32k  hybrid "$B32 --evidence-selection position"
condition middle_80k    direct "$B80 --direct-overflow middle"
condition position_80k  hybrid "$B80 --evidence-selection position"
# Regrading needs only the names, so Luna mode covers the recipe runs without the flags.
if [[ -n "${AA_LCR_RECIPE:-}" || "$GRADER" == luna ]]; then
  condition upgraded     hybrid "$B64 ${AA_LCR_RECIPE:-}"
  condition upgraded_32k hybrid "$B32 ${AA_LCR_RECIPE:-}"
  condition upgraded_80k hybrid "$B80 ${AA_LCR_RECIPE:-}"
fi
select_conditions

case "$GRADER" in
  executor)
    GRADER_MODEL="$EXECUTOR_MODEL"
    GRADER_URL="$EXECUTOR_URL"
    GRADER_FLAGS=(--grader-api-style vllm) ;;
  luna)
    GRADER_MODEL=openai/gpt-5.6-luna
    GRADER_URL=https://openrouter.ai/api/v1
    GRADER_FLAGS=(--grader-api-style openai --grader-reasoning-effort medium
      --evaluator-api-key-env OPENROUTER_API_KEY) ;;
  *) fail "AA_LCR_GRADER must be executor or luna" ;;
esac
GRADES="$ROOT/regrades/${GRADER_MODEL##*/}"

# The runner and aa_lcr.regrade both write manifest.json last.
finished() { [[ -f "$1/manifest.json" ]]; }

regrade() {
  local run="$1" output="$2"; shift 2
  run_command "$BENCHMARK_VENV/bin/python" -m aa_lcr.regrade \
    --source-run "$ROOT/$run" --output-dir "$output" \
    --questions-csv "$DATA_DIR/AA-LCR_Dataset.csv" \
    --documents-root "$DATA_DIR/extracted_text/lcr" \
    --dataset-manifest "$DATA_DIR/dataset_manifest.json" \
    --evaluator-model "$GRADER_MODEL" --evaluator-base-url "$GRADER_URL" "${GRADER_FLAGS[@]}" "$@"
}

graded() { finished "$GRADES/$2/${SUITE}_$1"; }

grade() {
  local name="$1" run="$2/${SUITE}_$1"
  set_aside "$GRADES/$run"
  step "  grade: $GRADER_MODEL (log: $LOG_DIR/${name}.grade-${GRADER_MODEL##*/}.log)"
  logged "$LOG_DIR/${name}.grade-${GRADER_MODEL##*/}.log" regrade "$run" "$GRADES/$run"
}

after_run() {
  if graded "$1" "$2"; then
    step "  grade: already done"
  elif grade "$1" "$2"; then
    step "  grade: done"
  else
    step "  grade: FAILED; see $LOG_DIR/$1.grade-${GRADER_MODEL##*/}.log"
    FAILED+=("$1 (grade)")
  fi
}

RESUME="bash linux/run_aa_lcr_suite.sh $SUITE"
[[ "$GRADER" == executor ]] || RESUME="AA_LCR_GRADER=$GRADER $RESUME"
start_suite "grader $GRADER_MODEL"

if [[ "$GRADER" == luna ]]; then
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    if [[ -z "${OPENROUTER_API_KEY:-}" && -f .env ]]; then
      set -a; source .env; set +a
    fi
    [[ -n "${OPENROUTER_API_KEY:-}" ]] || fail "OPENROUTER_API_KEY is not set and not in .env"
  fi
  FINISHED=()
  for i in "${SELECTED[@]}"; do
    if finished "$ROOT/${MODES[$i]}/${SUITE}_${NAMES[$i]}" || [[ "${DRY_RUN:-0}" == 1 ]]; then
      FINISHED+=("$i")
    else
      step "${NAMES[$i]}: no finished run, skipping"
      FAILED+=("${NAMES[$i]} (no finished run)")
    fi
  done
  (( ${#FINISHED[@]} )) || fail "No finished runs to regrade in suite $SUITE"

  # OpenRouter may drop a parameter the provider does not support, so confirm on
  # one question that reasoning is on before grading everything.
  first="${MODES[${FINISHED[0]}]}/${SUITE}_${NAMES[${FINISHED[0]}]}"
  check="$GRADES/check/${SUITE}_${NAMES[${FINISHED[0]}]}-$(date -u +%Y%m%dT%H%M%SZ)"
  step "Checking one Luna call on $first (question 1)"
  logged "$LOG_DIR/luna-check.log" regrade "$first" "$check" --question-ids 1 \
    || fail "The Luna check call failed; see $LOG_DIR/luna-check.log"
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    "$BENCHMARK_VENV/bin/python" - "$check/bridge_rows.jsonl" <<'EOF' || fail "Luna returned no reasoning tokens; reasoning effort was not applied"
import json, sys
usage = json.loads(open(sys.argv[1]).readline())["grader_usage"]
reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
print(f"[AA-LCR SUITE]   {usage.get('completion_tokens')} output tokens, {reasoning} of them reasoning")
sys.exit(0 if reasoning else 1)
EOF
  fi

  step "Regrading ${#FINISHED[@]} runs in parallel"
  PIDS=()
  for i in "${FINISHED[@]}"; do
    step "${NAMES[$i]}"
    if graded "${NAMES[$i]}" "${MODES[$i]}"; then
      step "  grade: already done"
    elif [[ "${DRY_RUN:-0}" == 1 ]]; then
      grade "${NAMES[$i]}" "${MODES[$i]}"
    else
      grade "${NAMES[$i]}" "${MODES[$i]}" & PIDS+=("$!:${NAMES[$i]}")
    fi
  done
  for entry in ${PIDS[@]+"${PIDS[@]}"}; do
    if wait "${entry%%:*}"; then
      step "${entry#*:}: grade done"
    else
      step "${entry#*:}: grade FAILED; see $LOG_DIR/${entry#*:}.grade-${GRADER_MODEL##*/}.log"
      FAILED+=("${entry#*:}")
    fi
  done
else
  check_executor
  run_conditions aa_lcr "${COMMON[@]}"
fi

step "Results table:"
run_command "$BENCHMARK_VENV/bin/python" -m aa_lcr.results_table "$ROOT" --grader "${GRADER_MODEL##*/}" \
  || step "Results table failed; the runs and grades are unaffected"
finish_suite
