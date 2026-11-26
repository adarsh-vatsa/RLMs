#!/usr/bin/env bash
# Runs Qwen3.6 with thinking on the planned conditions of one or more benchmarks, with one command: it submits
# an executor pinned to a node and one suite job that runs the listed thinking suites in turn against it,
# regrades AA-LCR with Luna, recomputes the intervals and stops the executor. See docs/jarvis_suites/THINKING.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../linux/common.sh"
cd "$REPO_ROOT"

usage() {
  echo 'Usage: bash jarvis/thinking_runs.sh <suite-id> <benchmarks: aa_lcr,mrcr,longbench in any order>'
  echo 'From the login node, submits an executor pinned to EXECUTOR_NODE (default g101) and one suite job that'
  echo 'waits for it, runs the listed benchmarks'"'"' suites with THINKING=1 and the recipe, regrades AA-LCR with'
  echo 'Luna, recomputes the intervals and cancels the executor. Submit the same command again to resume:'
  echo 'finished runs and grades are skipped.'
  echo 'THINKING_ALLOWANCE (required) is the output allowance in tokens, for the reasoning and the answer.'
  echo 'THINKING_TIME (default 24:00:00) is the time limit of both jobs. For a pilot: MAX_ROWS (AA-LCR and'
  echo 'LongBench-v2) and MRCR_ROWS_PER_SOURCE limit the questions, and AA_LCR_ONLY, MRCR_ONLY and'
  echo 'LONGBENCH_ONLY choose the conditions.'
  echo 'DRY_RUN=1 prints every suite command on this node without submitting.'
}

MODE=submit
if [[ "${1:-}" == job ]]; then MODE=job; shift; fi
case "${1:-}" in help|-h|--help) usage; exit 0 ;; esac
SUITE="${1:-}"
BENCHMARKS="${2:-}"
NODE="${EXECUTOR_NODE:-g101}"
TIME_LIMIT="${THINKING_TIME:-24:00:00}"
# One record per node, so executors on g101 and g102 can serve suites at the same time.
METADATA=".cache/serving_metadata-thinking-$NODE.json"
RECIPE="--evidence-selection fusion --child-tokens 1024 --child-overlap-tokens 128 --neighbor-children 1 --packing fill --document-floor"

step() { echo "[THINKING] $*"; }
fail() { echo "[THINKING] ERROR: $*" >&2; exit 1; }
if [[ -z "$SUITE" || -z "$BENCHMARKS" ]]; then usage >&2; exit 2; fi
[[ "${THINKING_ALLOWANCE:-}" =~ ^[1-9][0-9]*$ ]] || fail "Set THINKING_ALLOWANCE, the output allowance in tokens"
export THINKING_ALLOWANCE
for benchmark in ${BENCHMARKS//,/ }; do
  [[ "$benchmark" == aa_lcr || "$benchmark" == mrcr || "$benchmark" == longbench ]] \
    || fail "Unknown benchmark $benchmark; use aa_lcr, mrcr or longbench"
done
# The thinking-off runs came from g101 and g102; g202 answers identical prompts differently.
[[ "$NODE" == g101 || "$NODE" == g102 ]] || fail "EXECUTOR_NODE is $NODE; unset it or name g101 or g102"

RESUME=""
for name in EXECUTOR_NODE THINKING_ALLOWANCE THINKING_TIME MAX_ROWS MRCR_ROWS_PER_SOURCE AA_LCR_ONLY MRCR_ONLY LONGBENCH_ONLY; do
  if [[ -n "${!name:-}" ]]; then RESUME+="$name=$(printf '%q' "${!name}") "; fi
done
RESUME+="bash $(basename "$REPO_ROOT")/jarvis/thinking_runs.sh $SUITE $BENCHMARKS"

# Waits while the executor job is queued or loading, which can include downloading the model to the node;
# the job's time limit bounds the wait.
wait_for_executor() {
  local state
  step "Waiting for executor job $THINKING_EXECUTOR_JOB ($THINKING_EXECUTOR_URL_FILE)"
  until [[ -s "$THINKING_EXECUTOR_URL_FILE" ]] && "$BENCHMARK_VENV/bin/python" -c \
      'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=10)' \
      "$(cat "$THINKING_EXECUTOR_URL_FILE")/models" 2> /dev/null; do
    state="$(squeue -h -j "$THINKING_EXECUTOR_JOB" -o %T 2> /dev/null || true)"
    [[ "$state" == PENDING || "$state" == CONFIGURING || "$state" == RUNNING ]] \
      || fail "Executor job $THINKING_EXECUTOR_JOB ended (${state:-no longer queued}) before it was ready"
    sleep 30
  done
  OPENAI_COMPAT_EXECUTOR_BASE_URL="$(cat "$THINKING_EXECUTOR_URL_FILE")"
  [[ "$OPENAI_COMPAT_EXECUTOR_BASE_URL" == "http://$NODE."* ]] \
    || fail "Executor is at $OPENAI_COMPAT_EXECUTOR_BASE_URL, not on $NODE"
  step "Executor ready at $OPENAI_COMPAT_EXECUTOR_BASE_URL"
}

# The same record as step 4 of docs/jarvis_suites/EXECUTOR.md.
record_serving_metadata() {
  mkdir -p "$(dirname "$METADATA")"
  VLLM_VENV="$VLLM_VENV" "$BENCHMARK_VENV/bin/python" - "$OPENAI_COMPAT_EXECUTOR_BASE_URL" "$METADATA" <<'EOF'
import datetime, json, os, subprocess, sys, urllib.request

base = sys.argv[1].rstrip("/")
with urllib.request.urlopen(base + "/models", timeout=10) as response:
    served = json.load(response)["data"]
vllm_python = os.path.join(os.environ["VLLM_VENV"], "bin", "python")
vllm = subprocess.run([vllm_python, "-c", "import importlib.metadata as m; print(m.version('vllm'))"],
                      capture_output=True, text=True, check=True).stdout.strip()
metadata = {
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "cluster": "Jarvis",
    "executor_base_url": base,
    "gpus": "4 x NVIDIA L40S (gpu-l40s partition)",
    "tensor_parallel_size": 4,
    "vllm_version": vllm,
    "served_models": [{"id": m["id"], "max_model_len": m.get("max_model_len")} for m in served],
}
with open(sys.argv[2], "w") as handle:
    handle.write(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
EOF
}

run_thinking() {
  local failed=() benchmark
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    trap 'scancel "$THINKING_EXECUTOR_JOB"' EXIT
    wait_for_executor
    record_serving_metadata
  fi
  export OPENAI_COMPAT_EXECUTOR_BASE_URL="${OPENAI_COMPAT_EXECUTOR_BASE_URL:-http://$NODE:8000/v1}"
  export AA_LCR_SERVING_METADATA="$METADATA" MRCR_SERVING_METADATA="$METADATA" LONGBENCH_SERVING_METADATA="$METADATA"
  export THINKING=1 SUITE_RESUME="$RESUME"

  for benchmark in ${BENCHMARKS//,/ }; do
    case "$benchmark" in
      aa_lcr)
        ONLY="${AA_LCR_ONLY:-}" AA_LCR_RECIPE="$RECIPE" AA_LCR_GRADER=executor bash linux/run_aa_lcr_suite.sh "$SUITE" \
          || failed+=(AA-LCR)
        ONLY="${AA_LCR_ONLY:-}" AA_LCR_RECIPE="$RECIPE" AA_LCR_GRADER=luna bash linux/run_aa_lcr_suite.sh "$SUITE" \
          || failed+=("AA-LCR Luna regrade")
        run_command "$BENCHMARK_VENV/bin/python" -m aa_lcr.intervals "benchmark_artifacts/aa_lcr/suites/$SUITE" \
          || failed+=("AA-LCR intervals") ;;
      longbench)
        ONLY="${LONGBENCH_ONLY:-}" LONGBENCH_RECIPE="$RECIPE" bash linux/run_longbench_suite.sh "$SUITE" \
          || failed+=(LongBench-v2)
        run_command "$BENCHMARK_VENV/bin/python" -m long_bench_v2.intervals "benchmark_artifacts/longbench_v2/suites/$SUITE" \
          || failed+=("LongBench-v2 intervals") ;;
      mrcr)
        ONLY="${MRCR_ONLY:-}" MRCR_RECIPE="$RECIPE" bash linux/run_mrcr_suite.sh "$SUITE" || failed+=(MRCR)
        run_command "$BENCHMARK_VENV/bin/python" -m mrcr_v2.intervals "benchmark_artifacts/mrcr_v2/suites/$SUITE" \
          || failed+=("MRCR intervals") ;;
    esac
  done

  if (( ${#failed[@]} )); then
    fail "Failed: ${failed[*]}. See the lines above, fix the cause and submit $RESUME again."
  fi
  step "All runs, grades and intervals complete"
}

submit() {
  local output executor_job
  source "$SCRIPT_DIR/lib/env.sh"
  # The executor of the thinking-off runs; thinking is chosen per request. The partition is named so that
  # settings left in the shell for another executor do not move it.
  output="$(EXECUTOR_PARTITION=gpu-l40s EXECUTOR_GRES=gpu:l40s:4 EXECUTOR_MEM=220G EXECUTOR_TIME="$TIME_LIMIT" \
    MODULES="cuda12.8/toolkit/12.8.1" EXECUTOR_MODEL=Qwen/Qwen3.6-35B-A3B EXECUTOR_TP_SIZE=4 \
    EXECUTOR_MAX_MODEL_LEN=262144 VLLM_GPU_MEMORY_UTILIZATION=0.90 \
    VLLM_EXTRA_ARGS="--reasoning-parser qwen3 --language-model-only --max-num-seqs 1 --enable-chunked-prefill --max-num-batched-tokens 8192" \
    SYNC_BACK_MODELS=1 VLLM_VENV="$VLLM_VENV" EXECUTOR_NODE="$NODE" bash "$SCRIPT_DIR/run.sh" submit executor)"
  echo "$output"
  executor_job="$(awk '/Submitted batch job/ {print $4}' <<< "$output")"
  [[ -n "$executor_job" ]] || fail "Could not read the executor's job ID"
  step "Executor job $executor_job on $NODE"
  THINKING_EXECUTOR_JOB="$executor_job" THINKING_EXECUTOR_URL_FILE="$PROJECT_LOG_DIR/executor-$executor_job.url" \
    EXECUTOR_NODE="$NODE" CLIENT_DEPENDENCY="after:$executor_job" CLIENT_TIME="$TIME_LIMIT" \
    CLIENT_CMD="bash jarvis/thinking_runs.sh job $SUITE $BENCHMARKS" bash "$SCRIPT_DIR/run.sh" submit client-gpu
  step "The suite job starts once the executor does; follow it in $PROJECT_LOG_DIR/rlms-client-gpu-<job id>.out"
  step "Resume or repeat with: $RESUME"
}

if [[ "$MODE" == job || "${DRY_RUN:-0}" == 1 ]]; then run_thinking; else submit; fi
