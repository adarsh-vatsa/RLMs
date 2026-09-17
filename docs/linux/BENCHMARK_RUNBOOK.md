# Linux benchmark runbook

Complete the [Linux setup](SETUP_RUNBOOK.md) first. Run these commands from the
repository root. Preflight validates inputs and budgets; execution runs inference.
All examples use the shared `common` execution profile.

## Prepare data


These are existing preparation commands; preparation does not need model servers.
The tokenizer and requested dataset files may download on first use.

```bash
.venv/bin/python -m aa_lcr.prepare_dataset --dataset-version 1.1

.venv/bin/python -m mrcr_v2.prepare_dataset \
  --download-bands 65536:131072,131072:262144 --needles 8 \
  --min-source-tokens 60000 --max-source-tokens 250000 \
  --data-dir benchmark_data/mrcr_v2_60k_250k

# Use your existing official LongBench JSON; this does not generate new questions.
.venv/bin/python -m long_bench_v2.export_csv \
  --input-path benchmark_data/long_bench_v2/data.json \
  --output-path benchmark_data/long_bench_v2/data.csv
```

Use `$BENCHMARK_VENV/bin/python` instead when you chose a custom client environment.
MRCR requires a fresh prepared output directory. Existing prepared datasets can
be copied from the old server, preserving their manifests and referenced files;
AA-LCR manifests contain paths, so re-prepare if those paths are no longer valid.

## Preflight only

These commands do not generate answers or embed documents. They can load or
download the executor tokenizer. Explicit context limits keep these preflights
independent of model services. `--execution-only` is not a preflight flag: it
skips AA-LCR grading, but still generates answers unless `--preflight-only` is set.

### AA-LCR

AA-LCR defaults to the prepared v1.1 directory. For a different dataset, set
`AA_LCR_DATA_DIR` and pass its matching `--grader-prompt-version`.

```bash
bash linux/run_benchmark.sh aa_lcr hybrid --execution-only \
  --context-window-tokens 65536 --max-output-tokens 4096 --preflight-only

bash linux/run_benchmark.sh aa_lcr direct --execution-only \
  --context-window-tokens 65536 --max-output-tokens 4096 --preflight-only
```

### MRCR v2

```bash
export MRCR_DATA_DIR=benchmark_data/mrcr_v2_60k_250k
bash linux/run_benchmark.sh mrcr_v2 hybrid \
  --min-source-tokens 100000 --max-source-tokens 200000 \
  --context-window-tokens 65536 --max-input-tokens 60000 \
  --max-output-tokens 4096 --max-rows 10 --preflight-only

bash linux/run_benchmark.sh mrcr_v2 direct \
  --min-source-tokens 100000 --max-source-tokens 200000 \
  --context-window-tokens 65536 --max-input-tokens 60000 \
  --max-output-tokens 4096 --max-rows 10 --preflight-only
```

### LongBench v2

```bash
bash linux/run_benchmark.sh longbench_v2 hybrid --route-audit-only
bash linux/run_benchmark.sh longbench_v2 direct --preflight-only
```

## Actual execution

Start or connect to the executor using the [service instructions](SETUP_RUNBOOK.md#start-model-services-or-reuse-endpoints).
These commands generate answers. GPU 6 below is an example embedding device;
choose the device for your server. The budgets and selections match the
preflight examples above.

### AA-LCR

```bash
CUDA_VISIBLE_DEVICES=6 bash linux/run_benchmark.sh aa_lcr hybrid \
  --execution-only --context-window-tokens 65536 --max-output-tokens 4096

bash linux/run_benchmark.sh aa_lcr direct \
  --execution-only --context-window-tokens 65536 --max-output-tokens 4096
```

These runs save answers for later grading. To grade during execution, start or
connect to the evaluator and omit `--execution-only`.

AA-LCR discovers the full served context from `/models` when
`--context-window-tokens` is omitted. Its input allowance defaults to context
minus output allowance (16,384 output tokens by default). To use the full served
context, omit the context option from both the preflight and execution commands.
That preflight needs a reachable executor for metadata discovery, but still makes
no inference calls.

### MRCR v2

```bash
export MRCR_DATA_DIR=benchmark_data/mrcr_v2_60k_250k
CUDA_VISIBLE_DEVICES=6 bash linux/run_benchmark.sh mrcr_v2 hybrid \
  --min-source-tokens 100000 --max-source-tokens 200000 \
  --context-window-tokens 65536 --max-input-tokens 60000 \
  --max-output-tokens 4096 --max-rows 10

bash linux/run_benchmark.sh mrcr_v2 direct \
  --min-source-tokens 100000 --max-source-tokens 200000 \
  --context-window-tokens 65536 --max-input-tokens 60000 \
  --max-output-tokens 4096 --max-rows 10
```

### LongBench v2

```bash
CUDA_VISIBLE_DEVICES=6 bash linux/run_benchmark.sh longbench_v2 hybrid
bash linux/run_benchmark.sh longbench_v2 direct
```

## Budget and routing options

MRCR defaults to context/input/output budgets of 65,536/60,000/4,096;
LongBench defaults to 65,536/60,000/8. Both accept explicit budget overrides.
LongBench uses original rows from `benchmark_data/long_bench_v2/data.csv`, produced
by the export above; pass `--suite-csv` and `--source-json-path` for different
input paths. MRCR and LongBench need no grading service.

Change `hybrid` to `direct` for any benchmark. Direct prompts exceeding the
input budget are recorded as unsupported; hybrid retrieves and packs evidence.
All runner options—including token bounds, credentials, run identifiers, output
paths, and component switches—are forwarded. Hybrid defaults to CUDA embeddings;
set `SEMANTIC_CACHE_EMBEDDING_DEVICE=cpu` to use CPU instead. The common profile
disables answer caching by default and reuses document indexes within source groups.

## Launch preview (dry run)

This prints the command without loading a tokenizer, validating a dataset, or
contacting services. It is distinct from preflight.

```bash
DRY_RUN=1 bash linux/run_benchmark.sh aa_lcr hybrid --execution-only
```

## Execution logs and outputs

Benchmark artifacts use the existing runner output directories and manifests.
Use `--output-root` for AA-LCR/MRCR or `--output-dir` for LongBench to change them.
The following runs inference and saves its console output:

```bash
mkdir -p .cache/linux-logs
# In Bash, preserve the benchmark exit status when piping to tee.
set -o pipefail
bash linux/run_benchmark.sh aa_lcr direct --execution-only \
  --context-window-tokens 65536 --max-output-tokens 4096 \
  2>&1 | tee .cache/linux-logs/aa-lcr-direct.log
```

See the [shared execution architecture](../shared_execution_architecture.md)
for the pipeline and [MRCR overview](../mrcr_v2.md) for task/scoring details.
