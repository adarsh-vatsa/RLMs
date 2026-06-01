#!/usr/bin/env bash
# Prefetch Hugging Face model weights through Slurm, then sync them to project cache.

#SBATCH --job-name=rlms-download
#SBATCH --time=12:00:00

set -euo pipefail

MODE="${MODE:-${1:-download-all}}"
SYNC_BACK_MODELS="${SYNC_BACK_MODELS:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=jarvis/lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"

jarvis_setup_runtime
jarvis_print_runtime "$(jarvis_advertised_host)"

target="${DOWNLOAD_TARGET:-$MODE}"
models=()
case "$target" in
  download-executor|executor)
    models=("$EXECUTOR_MODEL")
    ;;
  download-evaluator|evaluator)
    models=("$EVALUATOR_MODEL")
    ;;
  download-smoke|smoke)
    models=("$SMOKE_MODEL")
    ;;
  download-all|all|download)
    models=("$EXECUTOR_MODEL" "$EVALUATOR_MODEL")
    ;;
  *)
    echo "Unsupported download target: $target" >&2
    exit 2
    ;;
esac

for model in "${models[@]}"; do
  echo "[JARVIS] prefetch model=$model"
  echo "[JARVIS] local HF cache=$LOCAL_HF_CACHE"
  echo "[JARVIS] project HF cache=$PROJECT_HF_CACHE"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[JARVIS] dry run: python -c 'huggingface_hub.snapshot_download(...)' $model"
    continue
  fi

  python - "$model" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

model_id = sys.argv[1]
cache_dir = os.environ["HF_HOME"]
print(f"[JARVIS] snapshot_download(repo_id={model_id!r}, cache_dir={cache_dir!r})")
snapshot_download(repo_id=model_id, cache_dir=cache_dir)
PY
done

echo "[JARVIS] download step complete; project sync runs in cleanup."
