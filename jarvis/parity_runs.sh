#!/usr/bin/env bash
# Adds the runs that make suite 20261029's conditions match across benchmarks, with one command: it submits
# an executor pinned to a node and one suite job that runs AA-LCR, LongBench-v2 and MRCR in turn against it,
# regrades AA-LCR with Luna, recomputes the intervals and stops the executor.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../linux/common.sh"
cd "$REPO_ROOT"

usage() {
  echo 'Usage: bash jarvis/parity_runs.sh'
  echo 'From the login node, submits an executor pinned to EXECUTOR_NODE (default g101) and one suite job'
  echo 'that waits for it. The job runs the 10 runs of docs/jarvis_suites/PARITY.md in suite 20261029,'
  echo 'grades the AA-LCR ones with the executor and then Luna, recomputes the intervals and cancels the'
  echo 'executor. Submit again to resume: finished runs and grades are skipped.'
  echo 'PARITY_TIME (default 08:00:00) is the time limit of both jobs.'
  echo 'DRY_RUN=1 prints every suite command on this node without submitting.'
}

SUITE=20261029
NODE="${EXECUTOR_NODE:-g101}"
# The runs take about 5 hours. A shorter limit lets Slurm start both jobs in a gap before a reserved job.
TIME_LIMIT="${PARITY_TIME:-08:00:00}"
METADATA=.cache/serving_metadata-parity.json
AA_LCR_RUNS=ladder1,ladder2
LONGBENCH_RUNS=random1,random2,bm25_recipe,bm25_recipe_32k,bm25_recipe_128k
MRCR_RUNS=random1,random2,bm25_recipe

step() { echo "[PARITY] $*"; }
fail() { echo "[PARITY] ERROR: $*" >&2; exit 1; }
# Every reported run comes from g101 or g102; g202 answers identical prompts differently.
[[ "$NODE" == g101 || "$NODE" == g102 ]] || fail "EXECUTOR_NODE is $NODE; unset it or name g101 or g102"

# Waits while the executor job is queued or loading, which can include downloading the model to the node;
# the job's time limit bounds the wait.
wait_for_executor() {
  local state
  step "Waiting for executor job $PARITY_EXECUTOR_JOB ($PARITY_EXECUTOR_URL_FILE)"
  until [[ -s "$PARITY_EXECUTOR_URL_FILE" ]] && "$BENCHMARK_VENV/bin/python" -c \
      'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=10)' \
      "$(cat "$PARITY_EXECUTOR_URL_FILE")/models" 2> /dev/null; do
    state="$(squeue -h -j "$PARITY_EXECUTOR_JOB" -o %T 2> /dev/null || true)"
    [[ "$state" == PENDING || "$state" == CONFIGURING || "$state" == RUNNING ]] \
      || fail "Executor job $PARITY_EXECUTOR_JOB ended (${state:-no longer queued}) before it was ready"
    sleep 30
  done
  OPENAI_COMPAT_EXECUTOR_BASE_URL="$(cat "$PARITY_EXECUTOR_URL_FILE")"
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

run_parity() {
  local failed=()
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    trap 'scancel "$PARITY_EXECUTOR_JOB"' EXIT
    wait_for_executor
    record_serving_metadata
  fi
  export OPENAI_COMPAT_EXECUTOR_BASE_URL="${OPENAI_COMPAT_EXECUTOR_BASE_URL:-http://$NODE:8000/v1}"
  export AA_LCR_SERVING_METADATA="$METADATA" MRCR_SERVING_METADATA="$METADATA" LONGBENCH_SERVING_METADATA="$METADATA"
  export SUITE_RESUME="bash $(basename "$REPO_ROOT")/jarvis/parity_runs.sh"

  ONLY="$AA_LCR_RUNS" AA_LCR_GRADER=executor bash linux/run_aa_lcr_suite.sh "$SUITE" || failed+=(AA-LCR)
  # In turn, not alongside the next suites as in the rerun: two runs regrade in minutes, and the log stays in order.
  ONLY="$AA_LCR_RUNS" AA_LCR_GRADER=luna bash linux/run_aa_lcr_suite.sh "$SUITE" || failed+=("AA-LCR Luna regrade")
  ONLY="$LONGBENCH_RUNS" bash linux/run_longbench_suite.sh "$SUITE" || failed+=(LongBench-v2)
  ONLY="$MRCR_RUNS" MRCR_ROWS_PER_SOURCE=30 bash linux/run_mrcr_suite.sh "$SUITE" || failed+=(MRCR)

  step "Intervals:"
  run_command "$BENCHMARK_VENV/bin/python" -m aa_lcr.intervals "benchmark_artifacts/aa_lcr/suites/$SUITE" \
    || failed+=("AA-LCR intervals")
  run_command "$BENCHMARK_VENV/bin/python" -m long_bench_v2.intervals "benchmark_artifacts/longbench_v2/suites/$SUITE" \
    || failed+=("LongBench-v2 intervals")
  run_command "$BENCHMARK_VENV/bin/python" -m mrcr_v2.intervals "benchmark_artifacts/mrcr_v2/suites/$SUITE" \
    || failed+=("MRCR intervals")

  if (( ${#failed[@]} )); then
    fail "Failed: ${failed[*]}. See the lines above, fix the cause and submit $SUITE_RESUME again."
  fi
  step "All runs, grades and intervals complete"
}

submit() {
  local output executor_job
  source "$SCRIPT_DIR/lib/env.sh"
  output="$(EXECUTOR_TIME="$TIME_LIMIT" MODULES="cuda12.8/toolkit/12.8.1" EXECUTOR_MODEL=Qwen/Qwen3.6-35B-A3B EXECUTOR_TP_SIZE=4 \
    EXECUTOR_MAX_MODEL_LEN=262144 VLLM_GPU_MEMORY_UTILIZATION=0.90 \
    VLLM_EXTRA_ARGS="--reasoning-parser qwen3 --language-model-only --max-num-seqs 1 --enable-chunked-prefill --max-num-batched-tokens 8192" \
    SYNC_BACK_MODELS=1 VLLM_VENV="$VLLM_VENV" EXECUTOR_NODE="$NODE" bash "$SCRIPT_DIR/run.sh" submit executor)"
  echo "$output"
  executor_job="$(awk '/Submitted batch job/ {print $4}' <<< "$output")"
  [[ -n "$executor_job" ]] || fail "Could not read the executor's job ID"
  step "Executor job $executor_job on $NODE"
  PARITY_EXECUTOR_JOB="$executor_job" PARITY_EXECUTOR_URL_FILE="$PROJECT_LOG_DIR/executor-$executor_job.url" \
    EXECUTOR_NODE="$NODE" CLIENT_DEPENDENCY="after:$executor_job" CLIENT_TIME="$TIME_LIMIT" \
    CLIENT_CMD="bash jarvis/parity_runs.sh job" bash "$SCRIPT_DIR/run.sh" submit client-gpu
  step "The suite job starts once the executor does; follow it in $PROJECT_LOG_DIR/rlms-client-gpu-<job id>.out"
}

case "${1:-}" in
  help|-h|--help) usage ;;
  job) run_parity ;;
  "") if [[ "${DRY_RUN:-0}" == 1 ]]; then run_parity; else submit; fi ;;
  *) usage >&2; exit 2 ;;
esac
