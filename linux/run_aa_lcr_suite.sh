#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() {
  echo 'Usage: bash linux/run_aa_lcr_suite.sh [suite-id]'
  echo 'Runs every AA-LCR workshop condition in turn and grades each with the executor when it finishes.'
  echo 'Rerun with the same suite ID to resume: finished runs and grades are skipped, and an'
  echo 'unfinished folder is moved aside before it is retried.'
  echo 'ONLY=name,... runs a subset. AA_LCR_RECIPE="<hybrid flags>" adds the upgraded-recipe runs.'
  echo 'AA_LCR_GRADER=luna regrades the finished runs with GPT-5.6 Luna through OpenRouter instead.'
  echo 'DRY_RUN=1 prints the commands without running them.'
}
case "${1:-}" in help|-h|--help) usage; exit 0 ;; esac
cd "$REPO_ROOT"

SUITE="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="benchmark_artifacts/aa_lcr/suites/$SUITE"
LOG_DIR="$ROOT/logs"
DATA_DIR="${AA_LCR_DATA_DIR:-benchmark_data/aa_lcr/v1.1}"
SERVING_METADATA="${AA_LCR_SERVING_METADATA:-.cache/serving_metadata.json}"
EXECUTOR_URL="${OPENAI_COMPAT_EXECUTOR_BASE_URL:-http://127.0.0.1:8000/v1}"
GRADER="${AA_LCR_GRADER:-executor}"
ONLY="${ONLY:-}"
COMMON=(--output-root "$ROOT" --execution-only --max-output-tokens 4096 --fail-fast --serving-metadata "$SERVING_METADATA")

B64="--context-window-tokens 65536 --max-input-tokens 61440"
B32="--context-window-tokens 32768 --max-input-tokens 28672"
B80="--context-window-tokens 81920 --max-input-tokens 77824"
NAMES=() MODES=() FLAGS=()
condition() { NAMES+=("$1"); MODES+=("$2"); FLAGS+=("$3"); }
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

step() { echo "[AA-LCR SUITE] $*"; }
fail() { echo "[AA-LCR SUITE] ERROR: $*" >&2; exit 1; }
duration() { printf '%dh%02dm' $(($1 / 3600)) $(($1 % 3600 / 60)); }

for name in ${ONLY//,/ }; do
  [[ " ${NAMES[*]} " == *" $name "* ]] || fail "Unknown condition in ONLY: $name (known: ${NAMES[*]})"
done
SELECTED=()
for i in "${!NAMES[@]}"; do
  if [[ -z "$ONLY" || ",$ONLY," == *",${NAMES[$i]},"* ]]; then SELECTED+=("$i"); fi
done

case "$GRADER" in
  executor)
    GRADER_MODEL="${OPENAI_COMPAT_EXECUTOR_MODEL:-Qwen/Qwen3.6-35B-A3B}"
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

# Both the runner and aa_lcr.regrade refuse an existing output folder, and write
# manifest.json last, so a folder without one is an interrupted attempt.
set_aside() {
  if [[ -d "$1" && ! -f "$1/manifest.json" ]]; then
    local target="$1.incomplete-$(date -u +%Y%m%dT%H%M%SZ)"
    step "  moving unfinished $1 to $target"
    [[ "${DRY_RUN:-0}" == 1 ]] || mv "$1" "$target"
  fi
}

logged() {
  local log="$1"; shift
  if [[ "${DRY_RUN:-0}" == 1 ]]; then "$@"; else "$@" > "$log" 2>&1; fi
}

regrade() {
  local run="$1" output="$2"; shift 2
  run_command "$BENCHMARK_VENV/bin/python" -m aa_lcr.regrade \
    --source-run "$ROOT/$run" --output-dir "$output" \
    --questions-csv "$DATA_DIR/AA-LCR_Dataset.csv" \
    --documents-root "$DATA_DIR/extracted_text/lcr" \
    --dataset-manifest "$DATA_DIR/dataset_manifest.json" \
    --evaluator-model "$GRADER_MODEL" --evaluator-base-url "$GRADER_URL" "${GRADER_FLAGS[@]}" "$@"
}

graded() { [[ -f "$GRADES/$2/${SUITE}_$1/manifest.json" ]]; }

grade() {
  local name="$1" run="$2/${SUITE}_$1"
  set_aside "$GRADES/$run"
  step "  grade: $GRADER_MODEL (log: $LOG_DIR/${name}.grade-${GRADER_MODEL##*/}.log)"
  logged "$LOG_DIR/${name}.grade-${GRADER_MODEL##*/}.log" regrade "$run" "$GRADES/$run"
}

step "Suite $SUITE: ${#SELECTED[@]} conditions, grader $GRADER_MODEL"
step "Logs: $LOG_DIR"
RESUME="bash linux/run_aa_lcr_suite.sh $SUITE"
[[ "$GRADER" == executor ]] || RESUME="AA_LCR_GRADER=$GRADER $RESUME"
step "Resume or repeat with: $RESUME"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  mkdir -p "$LOG_DIR"
fi
FAILED=()

if [[ "$GRADER" == luna ]]; then
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    if [[ -z "${OPENROUTER_API_KEY:-}" && -f .env ]]; then
      set -a; source .env; set +a
    fi
    [[ -n "${OPENROUTER_API_KEY:-}" ]] || fail "OPENROUTER_API_KEY is not set and not in .env"
  fi
  FINISHED=()
  for i in "${SELECTED[@]}"; do
    if [[ -f "$ROOT/${MODES[$i]}/${SUITE}_${NAMES[$i]}/manifest.json" || "${DRY_RUN:-0}" == 1 ]]; then
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
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    [[ -f "$SERVING_METADATA" ]] || fail "Serving metadata not found: $SERVING_METADATA"
    curl -fsS "$EXECUTOR_URL/models" > /dev/null || fail "Executor not reachable at $EXECUTOR_URL"
  fi
  step "Embedding device for hybrid runs: ${SEMANTIC_CACHE_EMBEDDING_DEVICE:-cuda}"
  started_suite=$SECONDS
  n=0
  for i in "${SELECTED[@]}"; do
    n=$((n + 1))
    name="${NAMES[$i]}" mode="${MODES[$i]}"
    run_dir="$ROOT/$mode/${SUITE}_$name"
    step "[$n/${#SELECTED[@]}] $name: $mode ${FLAGS[$i]}"
    if [[ -f "$run_dir/manifest.json" ]]; then
      step "  run: already finished"
    else
      set_aside "$run_dir"
      step "  run: started $(date '+%H:%M') (log: $LOG_DIR/$name.log)"
      started=$SECONDS
      read -ra flags <<< "${FLAGS[$i]}"
      if logged "$LOG_DIR/$name.log" bash linux/run_benchmark.sh aa_lcr "$mode" \
          "${flags[@]}" "${COMMON[@]}" --run-id "${SUITE}_$name"; then
        step "  run: done in $(duration $((SECONDS - started)))"
      else
        step "  run: FAILED after $(duration $((SECONDS - started))); see $LOG_DIR/$name.log"
        FAILED+=("$name")
        continue
      fi
    fi
    if graded "$name" "$mode"; then
      step "  grade: already done"
    elif grade "$name" "$mode"; then
      step "  grade: done"
    else
      step "  grade: FAILED; see $LOG_DIR/${name}.grade-${GRADER_MODEL##*/}.log"
      FAILED+=("$name (grade)")
    fi
  done
  step "Finished in $(duration $((SECONDS - started_suite)))"
fi

step "Results table:"
run_command "$BENCHMARK_VENV/bin/python" -m aa_lcr.results_table "$ROOT" --grader "${GRADER_MODEL##*/}" \
  || step "Results table failed; the runs and grades are unaffected"

if (( ${#FAILED[@]} )); then
  step "Failed: ${FAILED[*]}. Fix the cause and rerun the same command to retry them."
  exit 1
fi
step "All ${#SELECTED[@]} conditions complete"
