# Linux benchmark runbook

Complete the [Linux setup](SETUP_RUNBOOK.md) first. Run these commands from the
repository root. Preflight validates inputs and budgets; execution runs inference.
All examples use the shared `common` execution profile.
For available flags, defaults, and cache settings, consult the optional
[runner options reference](SHARED_EXECUTION_RUNBOOK.md).

## Prepare data

These are existing preparation commands; preparation does not need model servers.
The tokenizer and requested dataset files may download on first use.
Prepare only the benchmarks you intend to run. A failed LongBench export does
not require repeating successful AA-LCR or MRCR preparation.

### AA-LCR

```bash
.venv/bin/python -m aa_lcr.prepare_dataset --dataset-version 1.1
```

### MRCR v2

```bash
.venv/bin/python -m mrcr_v2.prepare_dataset \
  --executor-model "${OPENAI_COMPAT_EXECUTOR_MODEL:-Qwen/Qwen3.6-35B-A3B}" \
  --download-bands 65536:131072,131072:262144 --needles 8 \
  --min-source-tokens 60000 --max-source-tokens 250000 \
  --data-dir benchmark_data/mrcr_v2_60k_250k
```

To exercise retrieval on sources the executor cannot hold, prepare a band above
the served context as a second dataset. Expect this step, not inference, to
dominate the runtime: preparation counts tokens for every CSV row before
`--max-rows` applies, so the whole band is tokenized however few examples you
later select.

```bash
.venv/bin/python -m mrcr_v2.prepare_dataset \
  --executor-model "${OPENAI_COMPAT_EXECUTOR_MODEL:-Qwen/Qwen3.6-35B-A3B}" \
  --download-bands 262144:524288 --needles 8 \
  --min-source-tokens 250000 --max-source-tokens 600000 \
  --data-dir benchmark_data/mrcr_v2_250k_600k
```

MRCR preparation is a one-time step for each output directory. If it reports
`FileExistsError`, check that directory before retrying. A completed preparation
writes `dataset_manifest.json` last; if present, proceed to MRCR preflight to
validate and reuse the dataset. If it is absent and no preparation is still
running, the directory is incomplete. Retry preparation with a new `--data-dir`
and use that same path for preflight/execution. Do not delete a completed dataset
just to rerun preparation.

### LongBench v2

The exporter reads a local JSON file; it does not download the dataset.
`benchmark_data/long_bench_v2/*.json` is excluded from Git, so a fresh clone will
not contain `data.json`. Copy the file from your previous server to preserve the
same dataset, or download the [official dataset](https://huggingface.co/datasets/zai-org/LongBench-v2/blob/main/data.json)
(approximately 465 MB). This command pins revision
`2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9` and keeps an existing JSON file:

```bash
mkdir -p benchmark_data/long_bench_v2
if [ ! -f benchmark_data/long_bench_v2/data.json ]; then
  curl -fL --retry 3 \
    'https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json' \
    -o benchmark_data/long_bench_v2/data.json.download &&
  mv benchmark_data/long_bench_v2/data.json.download benchmark_data/long_bench_v2/data.json
fi
```

After the download succeeds, export the CSV. Keep both files: the CSV references
the full source contexts in `data.json`.

```bash
.venv/bin/python -m long_bench_v2.export_csv \
  --input-path benchmark_data/long_bench_v2/data.json \
  --output-path benchmark_data/long_bench_v2/data.csv
```

Use `$BENCHMARK_VENV/bin/python` instead when you chose a custom client environment.
MRCR requires a fresh prepared output directory. Existing prepared datasets can
be copied from the old server, preserving their manifests and referenced files;
AA-LCR manifests contain paths, so re-prepare if those paths are no longer valid.
MRCR measures source bounds with the executor tokenizer and chat template;
changing either, or widening the prepared bounds, requires a new preparation.
Official download bands select files and do not guarantee executor-token coverage.

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
  --context-window-tokens 65536 --max-input-tokens 60000 \
  --max-output-tokens 4096 --preflight-only

bash linux/run_benchmark.sh aa_lcr direct --execution-only \
  --context-window-tokens 65536 --max-input-tokens 60000 \
  --max-output-tokens 4096 --preflight-only
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

The budget decides the route, so preflight the budget you intend to execute.
This 60,000-token pair reports `dense_child_packed` for hybrid and
`unsupported_context` for direct, or `middle_truncated` when
`--direct-overflow middle` is added. Repeating it with
`--context-window-tokens 262144 --max-input-tokens 240000` reports `direct_fit`
for both modes instead.

### LongBench v2

```bash
bash linux/run_benchmark.sh longbench_v2 hybrid --route-audit-only
bash linux/run_benchmark.sh longbench_v2 direct --preflight-only
```

## Actual execution

Start or connect to the executor using the [service instructions](SETUP_RUNBOOK.md#start-model-services-or-reuse-endpoints).
These commands execute the benchmark; unsupported direct examples are skipped
without an API call. The hybrid commands below place embeddings on the CPU,
which suits a single-GPU server where the executor already reserves most of the
card. On a multi-GPU server, drop `SEMANTIC_CACHE_EMBEDDING_DEVICE` and pick a
free device with `CUDA_VISIBLE_DEVICES` instead; selecting an index that the
server does not have makes `torch.cuda.is_available()` false and fails every
row before any inference. The budgets and selections match the preflight
examples above.

### AA-LCR

```bash
SEMANTIC_CACHE_EMBEDDING_DEVICE=cpu bash linux/run_benchmark.sh aa_lcr hybrid \
  --execution-only --context-window-tokens 65536 --max-input-tokens 60000 \
  --max-output-tokens 4096

bash linux/run_benchmark.sh aa_lcr direct \
  --execution-only --context-window-tokens 65536 --max-input-tokens 60000 \
  --max-output-tokens 4096 --direct-overflow middle
```

These runs save answers for later grading. To grade during execution, start or
connect to the evaluator and omit `--execution-only`. This option does not
disable semantic cache verification if you explicitly enable it.

Every AA-LCR question exceeds a 60,000-token input budget, so the direct command
needs `--direct-overflow middle` to produce a baseline; without it the common
profile records every example as unsupported. The historical `direct_64k`
preset enabled this truncation through `--experiment`, which this launcher does
not use: `--mode` builds its budgets from the flags above instead.

AA-LCR discovers the full served context from `/models` when
`--context-window-tokens` is omitted. Its input allowance defaults to context
minus output allowance (16,384 output tokens by default). To use the full served
context, omit both `--context-window-tokens` and `--max-input-tokens` from the
preflight and execution commands.
That preflight needs a reachable executor for metadata discovery, but still makes
no inference calls.

### MRCR v2

The input budget is the variable under test here, so run both budgets and
compare them. Check the served context first with
`curl -s "$OPENAI_COMPAT_EXECUTOR_BASE_URL/models"`; the commands below assume
the 262,144-token default, which a 100,000–200,000-token selection fits.

The constrained pair uses a 60,000-token budget. Hybrid retrieves and packs
evidence; direct adds `--direct-overflow middle` to keep a head and a tail
slice, preserving roughly `max-input-tokens / full-rendered-tokens` of the
context. Because MRCR needles are spread through the source, that fraction also
approximates the share of needles the direct baseline can still see. Omit the
option to record the examples as unsupported without an API call.

```bash
export MRCR_DATA_DIR=benchmark_data/mrcr_v2_60k_250k
SEMANTIC_CACHE_EMBEDDING_DEVICE=cpu bash linux/run_benchmark.sh mrcr_v2 hybrid \
  --min-source-tokens 100000 --max-source-tokens 200000 \
  --context-window-tokens 65536 --max-input-tokens 60000 \
  --max-output-tokens 4096 --max-rows 10 --fail-fast

bash linux/run_benchmark.sh mrcr_v2 direct \
  --min-source-tokens 100000 --max-source-tokens 200000 \
  --context-window-tokens 65536 --max-input-tokens 60000 \
  --max-output-tokens 4096 --max-rows 10 --direct-overflow middle --fail-fast
```

The full served budget sends the whole prompt with no truncation, which is the
baseline the constrained runs are measured against:

```bash
bash linux/run_benchmark.sh mrcr_v2 direct \
  --min-source-tokens 100000 --max-source-tokens 200000 \
  --context-window-tokens 262144 --max-input-tokens 240000 \
  --max-output-tokens 4096 --max-rows 10 --fail-fast
```

Do not pair that with a hybrid run at the same budget. When the prompt fits,
both modes route `direct_fit` and send an identical request, and hybrid never
builds the embedding backend, so the two runs measure the same thing. The
preflight above already reports that route for both modes without inference.

Sources larger than the served context are the case retrieval exists for, and
they need the dataset prepared above. Here hybrid routes `dense_child_packed`
at either budget, so running both separates retrieval quality from the packing
budget it is given, while direct can only truncate:

```bash
export MRCR_DATA_DIR=benchmark_data/mrcr_v2_250k_600k

# retrieval on a tight budget
SEMANTIC_CACHE_EMBEDDING_DEVICE=cpu bash linux/run_benchmark.sh mrcr_v2 hybrid \
  --min-source-tokens 250000 --max-source-tokens 600000 \
  --context-window-tokens 65536 --max-input-tokens 60000 \
  --max-output-tokens 4096 --max-rows 10 --fail-fast

# retrieval on the full served budget
SEMANTIC_CACHE_EMBEDDING_DEVICE=cpu bash linux/run_benchmark.sh mrcr_v2 hybrid \
  --min-source-tokens 250000 --max-source-tokens 600000 \
  --context-window-tokens 262144 --max-input-tokens 240000 \
  --max-output-tokens 4096 --max-rows 10 --fail-fast

# truncated direct baseline at the full served budget
bash linux/run_benchmark.sh mrcr_v2 direct \
  --min-source-tokens 250000 --max-source-tokens 600000 \
  --context-window-tokens 262144 --max-input-tokens 240000 \
  --max-output-tokens 4096 --max-rows 10 --direct-overflow middle --fail-fast
```

Dropping `--direct-overflow middle` from that last command records every example
as unsupported without an API call, which is the measurement that the selection
does not fit the served context at all.

`--fail-fast` stops on the first execution error and marks the run
`stopped_early`. Without it, a run whose examples all fail still reports
`status: complete` with a mean score of zero, because execution failures count
as supported examples scoring zero.

### LongBench v2

```bash
SEMANTIC_CACHE_EMBEDDING_DEVICE=cpu bash linux/run_benchmark.sh longbench_v2 hybrid
bash linux/run_benchmark.sh longbench_v2 direct
```

## Budget and routing options

MRCR defaults to context/input/output budgets of 65,536/60,000/4,096;
LongBench defaults to 65,536/60,000/8. Both accept explicit budget overrides.
LongBench uses original rows from `benchmark_data/long_bench_v2/data.csv`, produced
by the export above; pass `--suite-csv` and `--source-json-path` for different
input paths. MRCR and LongBench need no answer-grading service; enabling semantic
cache reads adds a verifier requirement. See the
[answer-cache and verifier options](SHARED_EXECUTION_RUNBOOK.md#answer-cache-and-verifier-options).

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
Retain the complete run directory, including manifests, predictions, bridge rows,
and JSON reports, for later comparison documents. Console logs alone do not
contain all the structured settings and per-example results. Keep dataset
selection and budgets identical within each direct/hybrid comparison, and report
supported coverage alongside quality.
The following runs inference and saves its console output:

```bash
mkdir -p .cache/linux-logs
# In Bash, preserve the benchmark exit status when piping to tee.
set -o pipefail
bash linux/run_benchmark.sh aa_lcr direct --execution-only \
  --context-window-tokens 65536 --max-input-tokens 60000 --max-output-tokens 4096 \
  2>&1 | tee .cache/linux-logs/aa-lcr-direct.log
```

See the [shared execution architecture](../shared_execution_architecture.md)
for the pipeline and [MRCR overview](../mrcr_v2.md) for task/scoring details.
