#!/usr/bin/env bash
# Reruns suite 20261029's g202 runs, and the controls compared with them, on another node with one
# command: it submits an executor pinned to that node and one suite job that runs AA-LCR, LongBench-v2
# and MRCR in turn against it, regrades AA-LCR with Luna, recomputes the intervals and stops the executor.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../linux/common.sh"
cd "$REPO_ROOT"

usage() {
  echo 'Usage: bash jarvis/rerun_g202_runs.sh'
  echo 'From the login node, submits an executor pinned to EXECUTOR_NODE (default g101) and one suite job'
  echo 'that waits for it. The job moves each g202 run, with its grades and logs, to a sibling suite folder'
  echo '<suite>_g202, reruns it and the controls, regrades AA-LCR with Luna, recomputes the intervals and'
  echo 'cancels the executor. Submit again to resume: finished runs and grades are skipped.'
  echo 'RERUN_TIME (default 12:00:00) is the time limit of both jobs.'
  echo 'DRY_RUN=1 prints the moves and every suite command on this node without submitting or moving.'
}

SUITE=20261029
OLD_NODE=g202
NODE="${EXECUTOR_NODE:-g101}"
# The runs take about 10 hours. A shorter limit lets Slurm start both jobs in a gap before a reserved job.
TIME_LIMIT="${RERUN_TIME:-12:00:00}"
METADATA=.cache/serving_metadata-rerun.json
RECIPE="--evidence-selection fusion --child-tokens 1024 --child-overlap-tokens 128 --neighbor-children 1 --packing fill --document-floor"
AA_LCR_RUNS=bm25_32k,bm25_80k,upgraded,upgraded_32k,upgraded_80k
LONGBENCH_RUNS=bm25_32k,bm25_128k,upgraded,upgraded_32k,upgraded_128k
MRCR_RUNS=bm25_32k,bm25_64k,bm25_128k,upgraded,upgraded_32k,upgraded_64k,upgraded_128k
# The controls are compared with the recipe runs, so they follow them to the new node.
AA_LCR_CONTROLS=position_recipe,position_recipe_32k,position_recipe_80k,bm25_recipe,bm25_recipe_32k,bm25_recipe_80k
MRCR_CONTROLS=bm25_recipe_32k,bm25_recipe_64k,bm25_recipe_128k

step() { echo "[RERUN] $*"; }
fail() { echo "[RERUN] ERROR: $*" >&2; exit 1; }
[[ "$NODE" != "$OLD_NODE" ]] || fail "EXECUTOR_NODE is $OLD_NODE; unset it or name another node"

# move_old_runs <suite root> <hybrid folder> <names>: only runs whose manifest names the old node move,
# so a resubmitted job leaves the new runs in place.
move_old_runs() {
  local root="$1" folder="$2" name run path paths
  for name in ${3//,/ }; do
    run="$folder/${SUITE}_$name"
    grep -qs "\"executor_base_url\": \"http://$OLD_NODE" "$root/$run/manifest.json" || continue
    paths=("$run")
    for path in "$root"/regrades/*/"$run" "$root/logs/$name.log" "$root/logs/$name".grade-*.log; do
      if [[ -e "$path" ]]; then paths+=("${path#"$root"/}"); fi
    done
    for path in "${paths[@]}"; do
      step "moving $root/$path to ${root}_$OLD_NODE/$path"
      if [[ "${DRY_RUN:-0}" != 1 ]]; then
        mkdir -p "$(dirname "${root}_$OLD_NODE/$path")"
        mv "$root/$path" "${root}_$OLD_NODE/$path"
      fi
    done
  done
}

# Waits while the executor job is queued or loading, which can include downloading the model to the node;
# the job's time limit bounds the wait.
wait_for_executor() {
  local state
  step "Waiting for executor job $RERUN_EXECUTOR_JOB ($RERUN_EXECUTOR_URL_FILE)"
  until [[ -s "$RERUN_EXECUTOR_URL_FILE" ]] && "$BENCHMARK_VENV/bin/python" -c \
      'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=10)' \
      "$(cat "$RERUN_EXECUTOR_URL_FILE")/models" 2> /dev/null; do
    state="$(squeue -h -j "$RERUN_EXECUTOR_JOB" -o %T 2> /dev/null || true)"
    [[ "$state" == PENDING || "$state" == CONFIGURING || "$state" == RUNNING ]] \
      || fail "Executor job $RERUN_EXECUTOR_JOB ended (${state:-no longer queued}) before it was ready"
    sleep 30
  done
  OPENAI_COMPAT_EXECUTOR_BASE_URL="$(cat "$RERUN_EXECUTOR_URL_FILE")"
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

run_rerun() {
  local failed=() luna
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    trap 'scancel "$RERUN_EXECUTOR_JOB"' EXIT
    wait_for_executor
    record_serving_metadata
  fi
  export OPENAI_COMPAT_EXECUTOR_BASE_URL="${OPENAI_COMPAT_EXECUTOR_BASE_URL:-http://$NODE:8000/v1}"
  export AA_LCR_SERVING_METADATA="$METADATA" MRCR_SERVING_METADATA="$METADATA" LONGBENCH_SERVING_METADATA="$METADATA"
  export SUITE_RESUME="bash $(basename "$REPO_ROOT")/jarvis/rerun_g202_runs.sh"

  move_old_runs "benchmark_artifacts/aa_lcr/suites/$SUITE" hybrid "$AA_LCR_RUNS"
  move_old_runs "benchmark_artifacts/longbench_v2/suites/$SUITE" longbench_v2 "$LONGBENCH_RUNS"
  move_old_runs "benchmark_artifacts/mrcr_v2/suites/$SUITE" hybrid "$MRCR_RUNS"

  ONLY="$AA_LCR_RUNS,$AA_LCR_CONTROLS" AA_LCR_RECIPE="$RECIPE" AA_LCR_GRADER=executor \
    bash linux/run_aa_lcr_suite.sh "$SUITE" || failed+=(AA-LCR)
  # Luna is reached through OpenRouter, so it regrades while the executor serves the other suites.
  ONLY="$AA_LCR_RUNS,$AA_LCR_CONTROLS" AA_LCR_GRADER=luna bash linux/run_aa_lcr_suite.sh "$SUITE" & luna=$!
  ONLY="$LONGBENCH_RUNS" LONGBENCH_RECIPE="$RECIPE" \
    bash linux/run_longbench_suite.sh "$SUITE" || failed+=(LongBench-v2)
  ONLY="$MRCR_RUNS,$MRCR_CONTROLS" MRCR_RECIPE="$RECIPE" MRCR_ROWS_PER_SOURCE=30 \
    bash linux/run_mrcr_suite.sh "$SUITE" || failed+=(MRCR)
  wait "$luna" || failed+=("AA-LCR Luna regrade")

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
  RERUN_EXECUTOR_JOB="$executor_job" RERUN_EXECUTOR_URL_FILE="$PROJECT_LOG_DIR/executor-$executor_job.url" \
    EXECUTOR_NODE="$NODE" SBATCH_DEPENDENCY="after:$executor_job" CLIENT_TIME="$TIME_LIMIT" \
    CLIENT_CMD="bash jarvis/rerun_g202_runs.sh job" bash "$SCRIPT_DIR/run.sh" submit client-gpu
  step "The suite job starts once the executor does; follow it in $PROJECT_LOG_DIR/rlms-client-gpu-<job id>.out"
}

case "${1:-}" in
  help|-h|--help) usage ;;
  job) run_rerun ;;
  "") if [[ "${DRY_RUN:-0}" == 1 ]]; then run_rerun; else submit; fi ;;
  *) usage >&2; exit 2 ;;
esac
