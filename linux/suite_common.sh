#!/usr/bin/env bash
# Shared by the run_*_suite.sh scripts. A suite sets SUITE, SUITE_LABEL, ROOT, LOG_DIR,
# SERVING_METADATA, EXECUTOR_URL and ONLY, declares its conditions, and defines
# finished <run-dir>; after_run <name> <mode> is optional, and run_path <mode> <name>
# can be redefined where a runner names its folders differently.

NAMES=() MODES=() FLAGS=() SELECTED=() FAILED=()
condition() { NAMES+=("$1"); MODES+=("$2"); FLAGS+=("$3"); }
run_path() { echo "$ROOT/$1/${SUITE}_$2"; }

step() { echo "[$SUITE_LABEL] $*"; }
fail() { echo "[$SUITE_LABEL] ERROR: $*" >&2; exit 1; }
duration() { printf '%dh%02dm' $(($1 / 3600)) $(($1 % 3600 / 60)); }

select_conditions() {
  local name i
  for name in ${ONLY//,/ }; do
    [[ " ${NAMES[*]} " == *" $name "* ]] || fail "Unknown condition in ONLY: $name (known: ${NAMES[*]})"
  done
  for i in "${!NAMES[@]}"; do
    if [[ -z "$ONLY" || ",$ONLY," == *",${NAMES[$i]},"* ]]; then SELECTED+=("$i"); fi
  done
}

# The runners refuse an existing output folder, so an unfinished one is moved
# aside before it is retried.
set_aside() {
  if [[ -d "$1" ]] && ! finished "$1"; then
    local target="$1.incomplete-$(date -u +%Y%m%dT%H%M%SZ)"
    step "  moving unfinished $1 to $target"
    [[ "${DRY_RUN:-0}" == 1 ]] || mv "$1" "$target"
  fi
}

logged() {
  local log="$1"; shift
  if [[ "${DRY_RUN:-0}" == 1 ]]; then "$@"; else "$@" > "$log" 2>&1; fi
}

start_suite() {
  step "Suite $SUITE: ${#SELECTED[@]} conditions${1:+, $1}"
  step "Logs: $LOG_DIR"
  step "Resume or repeat with: $RESUME"
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    mkdir -p "$LOG_DIR"
  fi
}

check_executor() {
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    [[ -f "$SERVING_METADATA" ]] || fail "Serving metadata not found: $SERVING_METADATA"
    curl -fsS "$EXECUTOR_URL/models" > /dev/null || fail "Executor not reachable at $EXECUTOR_URL"
  fi
  step "Embedding device for hybrid runs: ${SEMANTIC_CACHE_EMBEDDING_DEVICE:-cuda}"
}

# run_conditions <benchmark> <flags for every run...>
run_conditions() {
  local benchmark="$1"; shift
  local started_suite=$SECONDS n=0 i name mode run_dir started flags
  for i in "${SELECTED[@]}"; do
    n=$((n + 1))
    name="${NAMES[$i]}" mode="${MODES[$i]}"
    run_dir="$(run_path "$mode" "$name")"
    step "[$n/${#SELECTED[@]}] $name: $mode ${FLAGS[$i]}"
    if finished "$run_dir"; then
      step "  run: already finished"
    else
      set_aside "$run_dir"
      step "  run: started $(date '+%H:%M') (log: $LOG_DIR/$name.log)"
      started=$SECONDS
      read -ra flags <<< "${FLAGS[$i]}"
      if logged "$LOG_DIR/$name.log" bash linux/run_benchmark.sh "$benchmark" "$mode" \
          "${flags[@]}" "$@" --run-id "${SUITE}_$name"; then
        step "  run: done in $(duration $((SECONDS - started)))"
      else
        step "  run: FAILED after $(duration $((SECONDS - started))); see $LOG_DIR/$name.log"
        FAILED+=("$name")
        continue
      fi
    fi
    if declare -F after_run > /dev/null; then
      after_run "$name" "$mode"
    fi
  done
  step "Finished in $(duration $((SECONDS - started_suite)))"
}

finish_suite() {
  if (( ${#FAILED[@]} )); then
    step "Failed: ${FAILED[*]}. Fix the cause and rerun the same command to retry them."
    exit 1
  fi
  step "All ${#SELECTED[@]} conditions complete"
}
