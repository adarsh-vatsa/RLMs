# Shared direct/hybrid execution on Linux

LongBench-v2 (OpenAI-compatible direct and hybrid), AA-LCR, and MRCR v2 now call
`execution.pipeline.Pipeline`. Dataset parsing, task formatting, and scoring
remain in their adapters. Iterative/RLM and other hosted-provider LongBench paths
are still separate architectures.

See the [architecture diagrams](../shared_execution_architecture.md) and
[implementation plan](../shared_execution_pipeline_plan.md).
Commands below run directly from the repository root without Slurm. Complete
the [Linux setup](SETUP_RUNBOOK.md) first. For Jarvis submissions, use the
[Jarvis version](../jarvis/SHARED_EXECUTION_RUNBOOK.md).

## Select the execution profile

Existing Python entry points retain `--execution-profile legacy` as the default
to preserve their historical routing/packing choices. Use
`--execution-profile common` explicitly for the shared quality configuration:

- Direct overflow is unsupported; hybrid overflow uses dense retrieval.
- Exact embedding-tokenizer offsets are required.
- Evidence overlaps merge within each document and render in source order.
- Answer-cache reads/writes and reranking are disabled; document indexes reuse
  within each source group and are not persisted between processes.
- Temperature is zero and thinking is disabled.

The Linux launcher defaults to `common` for all three benchmarks and original
rows for LongBench. The examples still state the profile explicitly where useful.
Calling the Python modules directly retains their `legacy` default unless
overridden. Legacy cache namespaces are versioned by the pipeline, so old
answer-cache snapshots are not silently reused.

All three accept `--min-source-tokens` and `--max-source-tokens`, measured on the
complete rendered executor prompt. For AA-LCR/LongBench, supply both together;
the row limit is applied after filtering. MRCR retains its prepared bounds and
allows narrowing only. Source selection never truncates an example.

## Dataset selection

Use the [Linux dataset preparation instructions](BENCHMARK_RUNBOOK.md#prepare-data)
for AA-LCR, MRCR, and LongBench. Set the prepared dataset paths in the terminal
used to launch runs:

```bash
export AA_LCR_DATA_DIR=benchmark_data/aa_lcr/v1.1
export MRCR_DATA_DIR=benchmark_data/mrcr_v2_60k_250k
```

The MRCR examples assume preparation includes the 100,000–200,000 source-token
interval. The LongBench examples explicitly select the exported `data.csv` and
original rows. Use the same files, source bounds, and row limits for paired runs.
The Linux AA-LCR launcher defaults to v1.1 data and its matching grader prompt,
whereas the Jarvis launcher retains historical defaults. For comparisons across
servers, explicitly match dataset files, grading version, model, and budgets.

## Preflight only

Preflight loads the executor tokenizer but makes no embedding or inference calls. AA-LCR also
reads executor metadata when its context limit is omitted; supply an explicit
limit for offline preflight. With caching enabled, predicted routes assume a cache miss.

```bash
bash linux/run_benchmark.sh mrcr_v2 hybrid \
  --execution-profile common --preflight-only \
  --min-source-tokens 100000 --max-source-tokens 200000 \
  --context-window-tokens 65536 --max-input-tokens 60000 --max-output-tokens 4096

bash linux/run_benchmark.sh aa_lcr hybrid \
  --execution-profile common --execution-only --preflight-only \
  --grader-prompt-version aa_lcr_equality_v1.1 \
  --context-window-tokens 65536 --max-input-tokens 60000 --max-output-tokens 4096

bash linux/run_benchmark.sh longbench_v2 hybrid --route-audit-only \
  --suite-csv benchmark_data/long_bench_v2/data.csv
```

These commands validate inputs in the current process without inference.
For direct preflight,
replace `hybrid` with `direct` for AA-LCR/MRCR. LongBench direct uses a different
validation flag:

```bash
bash linux/run_benchmark.sh longbench_v2 direct --preflight-only \
  --suite-csv benchmark_data/long_bench_v2/data.csv
```

## Actual execution with the same system policy

Start an executor with a 65,536-token context using the
[executor startup instructions](SETUP_RUNBOOK.md#start-model-services-or-reuse-endpoints), then set its endpoint.
The examples deliberately choose the same model and input allowance. Output
allowances follow the task; retain each allowance for its direct/hybrid pair.
Run services in separate terminals. For hybrid runs, select an embedding GPU via
`CUDA_VISIBLE_DEVICES` as in the setup runbook; the launcher defaults to CUDA
embeddings. It does not start model services or allocate GPUs for you.

```bash
export OPENAI_COMPAT_EXECUTOR_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_COMPAT_EXECUTOR_MODEL=Qwen/Qwen3.6-35B-A3B

bash linux/run_benchmark.sh mrcr_v2 hybrid \
  --execution-profile common \
  --min-source-tokens 100000 --max-source-tokens 200000 \
  --context-window-tokens 65536 --max-input-tokens 60000 --max-output-tokens 4096

bash linux/run_benchmark.sh aa_lcr hybrid \
  --execution-profile common --execution-only \
  --grader-prompt-version aa_lcr_equality_v1.1 \
  --context-window-tokens 65536 --max-input-tokens 60000 --max-output-tokens 4096

bash linux/run_benchmark.sh longbench_v2 hybrid \
  --suite-csv benchmark_data/long_bench_v2/data.csv
```

Replace `hybrid` with `direct` for any of the three benchmarks.
In the common profile, all three record unsupported direct
examples without inference. Compare quality on common supported examples and
report oversized hybrid results separately.
The MRCR interval shown here is entirely above the 60,000-token input budget,
so its direct run has zero supported examples. To compare direct and hybrid on
supported MRCR examples, prepare/select an interval that includes prompts within
the input budget, or increase the served context and runner budgets together.

AA-LCR separates mode from context size. With `--mode direct|hybrid` (or the
launcher names above), omitting `--context-window-tokens` reads the selected
executor's `max_model_len` from its `/v1/models` endpoint. Omitting
`--max-input-tokens` uses that window minus the output allowance (16,384 by
default). This uses the served limit, which can be lower than the model's
advertised capacity. If discovery is unavailable, supply the limit explicitly.
MRCR and LongBench retain their existing budget defaults.

```bash
# AA-LCR: full served context, reserving 4,096 tokens for the answer.
bash linux/run_benchmark.sh aa_lcr hybrid \
  --execution-profile common --execution-only --max-output-tokens 4096

# AA-LCR: explicit 64K context; usable input defaults to 61,440 tokens.
bash linux/run_benchmark.sh aa_lcr direct \
  --execution-profile common --execution-only \
  --context-window-tokens 65536 --max-output-tokens 4096
```

Historical AA-LCR `--experiment` presets remain available through the Python
runner and Jarvis launcher. The Linux launcher accepts only `direct` or `hybrid`.

AA-LCR's `--execution-only` saves answers for its existing `aa_lcr.regrade`
workflow. To grade during the run, omit this flag and configure the evaluator
using the [Linux evaluator startup instructions](SETUP_RUNBOOK.md#start-model-services-or-reuse-endpoints). LongBench and MRCR
have deterministic scorers and do not need an answer-grading service.
`--execution-only` still runs inference; it is not a preflight option.

## Service roles

| Role | When needed |
|---|---|
| Executor | Generate answers on cache misses |
| Embedding model + FAISS | Retrieve source evidence for oversized hybrid inputs; also used for semantic cache lookup |
| Cache verifier | Verify semantic answer-cache candidates when cache reads and semantic matching are enabled |
| Answer grader | Grade AA-LCR predictions, unless `--execution-only` defers grading |

Dense document retrieval does not call the evaluator service. Optional reranking
uses the local reranker model, not the grader. The service named `evaluator` can
host the model for grading, cache verification, or both. Those roles have separate
prompts and configuration. MRCR/LongBench can therefore need this service for
semantic caching even though their answer scoring is deterministic.

## Explicit component experiments

The runner options are shared:

| Options | Behavior |
|---|---|
| `--answer-cache-read` / `--no-answer-cache-read` | Enable/disable answer lookup |
| `--answer-cache-write` / `--no-answer-cache-write` | Enable/disable storage of generated answers |
| `--cache-matching exact` | Exact lookup without a verifier |
| `--cache-matching semantic` | Semantic candidate verification in addition to exact lookup |
| `--cache-verifier-model`, `--cache-verifier-base-url`, `--cache-verifier-api-key-env` | Separate cache-verifier service configuration |
| `--pipeline-rerank-top N` | Use the existing reranker; zero disables it |
| `--evidence-order source` / `score` | Evidence presentation policy |
| `--merge-overlaps` / `--no-merge-overlaps` | Source-range overlap policy; merging requires source order |
| `--direct-overflow unsupported` / `middle` / `error` | Explicit direct baseline policy |
| `--child-tokens`, `--child-overlap-tokens` | Shared chunk settings |

LongBench's direct API baseline remains uncached. Run cache experiments through
hybrid mode. The verifier and AA-LCR grader are separate roles and can point to
different services. Hosted AA-LCR grading with semantic caching requires an
explicit verifier endpoint. The Linux launcher does not wait for services: check
`/models` after startup before executing. AA-LCR automatic context discovery still
requires a reachable executor during preflight when no limit is supplied.

### Execute with semantic answer caching

Start a verifier service using the evaluator startup instructions, or reuse a
compatible endpoint. Configure it explicitly so it is independent of grading.
These commands perform inference and cache operations:

```bash
export OPENAI_COMPAT_EVALUATOR_BASE_URL=http://127.0.0.1:8001/v1
export OPENAI_COMPAT_EVALUATOR_MODEL=Qwen/Qwen3.5-35B-A3B

CACHE_ARGS=(--answer-cache-read --answer-cache-write --cache-matching semantic
  --cache-verifier-model "$OPENAI_COMPAT_EVALUATOR_MODEL"
  --cache-verifier-base-url "$OPENAI_COMPAT_EVALUATOR_BASE_URL")

bash linux/run_benchmark.sh aa_lcr hybrid \
  --execution-profile common --execution-only \
  --grader-prompt-version aa_lcr_equality_v1.1 \
  --context-window-tokens 65536 --max-input-tokens 60000 --max-output-tokens 4096 \
  "${CACHE_ARGS[@]}"

bash linux/run_benchmark.sh mrcr_v2 hybrid --execution-profile common \
  --min-source-tokens 100000 --max-source-tokens 200000 \
  --context-window-tokens 65536 --max-input-tokens 60000 --max-output-tokens 4096 \
  "${CACHE_ARGS[@]}"

bash linux/run_benchmark.sh longbench_v2 hybrid \
  --suite-csv benchmark_data/long_bench_v2/data.csv "${CACHE_ARGS[@]}"
```

Reads and writes are independent. Writes populate the cache; reads attempt reuse
within the same source and execution scope. A semantic verifier call happens
only when a candidate needs verification, not for every retrieval. Enabling
caching does not guarantee cache hits on distinct original questions.
`--execution-only` disables AA-LCR grading, not cache verification.

For exact-only reuse, pass `--answer-cache-read --answer-cache-write
--cache-matching exact`; no verifier endpoint is needed. To disable caching,
pass `--no-answer-cache-read --no-answer-cache-write`. For an authenticated
verifier, pass `--cache-verifier-api-key-env NAME` and export the key as `NAME`.

For a rank-order ablation, pass `--evidence-order score --no-merge-overlaps` to
each benchmark. Such a run is a different system configuration; do not pool it
with the primary common-profile results. Original and repeated/paraphrased
LongBench rows should likewise be reported separately.

## Launch preview (dry run)

Dry run prints the resolved Python command without validating datasets, loading
tokenizers/models, or contacting services. It is distinct from preflight.

```bash
DRY_RUN=1 bash linux/run_benchmark.sh aa_lcr hybrid --execution-only
DRY_RUN=1 bash linux/run_benchmark.sh mrcr_v2 hybrid
DRY_RUN=1 bash linux/run_benchmark.sh longbench_v2 hybrid
```

## Artifacts and limitations

Existing prediction/report files remain available under the runner output
directories. Use `--output-root` for AA-LCR/MRCR or `--output-dir` for LongBench
to change them. Each shared run also writes:

- `execution_manifest.json`: resolved pipeline settings and tokenizer identity.
- `embedding_manifest.json`: embedding settings, if a backend was needed.
- `execution.jsonl`: predictions and execution telemetry, flushed before scoring.
- `evaluation.jsonl`: separate scoring outcomes.
- `execution_report.json`: supported coverage, failure counts, quality, timings,
  and summaries by route and source-length band.

The common report counts execution failures as zero, excludes unsupported direct
examples, and leaves quality incomplete while grading is unresolved. Existing
legacy reports retain their historical metric definitions. AA-LCR grading
usage remains separately available in its bridge rows; common execution costs
exclude grading. Failed API attempts may not return token usage.

Cache and index policies are shared, but persistent cache lifecycle stays with
the existing runners: LongBench can reload saved answer-cache state; AA-LCR saves
its run-local cache; MRCR currently uses run-local answer caching only when
explicitly enabled. Cross-run document-index persistence is not implemented.
Real inference and held-out-benchmark validation are separate from mocked tests.
Keep the full run directories for later comparison reports, not only console logs.
Compare direct/hybrid quality on identical supported examples; report oversized
hybrid results and cache-enabled experiments separately.
