# Linux runner options reference

The execution workflow has two steps: follow the [setup runbook](SETUP_RUNBOOK.md),
then the [benchmark runbook](BENCHMARK_RUNBOOK.md). This document is an optional
reference for changing runner settings; it is not another execution step.

Defaults below apply to `linux/run_benchmark.sh`. It forwards additional options
to the existing benchmark runners. Python entry points used directly can have
different defaults. See the [architecture diagrams](../shared_execution_architecture.md)
for the shared pipeline design.

## Benchmark, mode, and profile

| Setting | Values and behavior |
|---|---|
| Benchmark argument | `aa_lcr`, `mrcr_v2`, or `longbench_v2` |
| Mode argument | `direct` sends the full prompt; `hybrid` sends it when it fits and retrieves evidence otherwise |
| `--execution-profile` | `common` (Linux default) or `legacy` (historical per-runner policies) |
| `--direct-overflow` | `unsupported` (common default), `middle` (truncate), or `error` (reject) |

The common profile uses exact tokenizer offsets, merges overlapping evidence,
orders evidence by source position, and disables answer-cache reads/writes and
reranking by default. Generation uses temperature zero with thinking disabled.
Direct overflow is recorded as `unsupported_context` without inference. Hybrid
retrieval indexes the complete source and packs evidence into the input budget.
On fitting prompts, both modes use the full prompt and record `direct_fit`.

The Linux launcher accepts only the mode arguments above. Historical AA-LCR
`--experiment` presets remain available through its Python runner and Jarvis.
LongBench's iterative/RLM and other hosted-provider paths are outside this shared
Linux direct/hybrid workflow.

## Dataset paths and selection

Dataset environment variables are **optional overrides**, not required setup.
AA-LCR already uses the v1.1 directory by default. MRCR's custom prepared directory
in the benchmark runbook differs from its default, so that example sets
`MRCR_DATA_DIR`; `--data-dir` is an equivalent runner override.

| Benchmark | Linux default input | Override |
|---|---|---|
| AA-LCR | `benchmark_data/aa_lcr/v1.1` | `AA_LCR_DATA_DIR`, or all three of `--questions-csv`, `--documents-root`, `--dataset-manifest` |
| MRCR v2 | `benchmark_data/mrcr_v2` | `MRCR_DATA_DIR` or `--data-dir` |
| LongBench v2 | `benchmark_data/long_bench_v2/data.csv` and `data.json` | `--suite-csv` and `--source-json-path` |

Explicit runner path options override the paths inserted by the launcher.
For AA-LCR, keep the CSV, documents, manifest, and grading version consistent.

| Option | Applies to | Behavior |
|---|---|---|
| `--min-source-tokens`, `--max-source-tokens` | All | Inclusive bounds on the complete rendered executor prompt, independent of the executor input budget |
| `--max-rows` | All | Deterministic limit after filtering; `0` means all eligible rows |
| `--question-ids`, `--document-set-ids` | AA-LCR | Comma-separated identifier filters |
| `--source-ids` | LongBench | Comma-separated source identifiers |
| `--row-types` | LongBench | Defaults to `original`; other row types must exist in the selected CSV |
| `--row-order` | LongBench hybrid | `input` or `source_grouped`; controls question order |
| `--tokenizer-revision` | MRCR | Tokenizer revision; defaults to the prepared revision |

AA-LCR and LongBench require both source bounds if either is supplied; otherwise
no source-length filter is imposed. MRCR defaults to its prepared bounds and
allows narrowing only. Bounds must be positive with minimum no greater than
maximum. Selection never truncates a source. Changing MRCR's tokenizer/template
or widening its prepared bounds requires preparation again. Download bands and
needle count are preparation options, covered in the benchmark runbook.

## Executor token budgets

| Option | AA-LCR default | MRCR default | LongBench default |
|---|---|---|---|
| `--context-window-tokens` | Discover served limit from executor `/models` | 65,536 | 65,536 |
| `--max-input-tokens` | Context minus output allowance | 60,000 | 60,000 |
| `--max-output-tokens` | 16,384 | 4,096 | 8 |

All budgets must be positive, and input plus output allowance must not exceed
the context window. Input counts include the rendered chat template. Runner
settings do not resize the model service; the configured context must fit its
served limit.

For AA-LCR, omitting both context and input options uses the full served context,
reserving the output allowance. An explicit input limit remains in effect even
when context is discovered. If discovery is unavailable, supply a context limit.
MRCR and LongBench do not automatically expand their default budgets.

## Preflight, dry run, and grading

| Option | Applies to | Behavior |
|---|---|---|
| `DRY_RUN=1` | Linux launcher | Print the command only; no dataset validation, tokenization, or service calls |
| `--preflight-only` | AA-LCR, MRCR, LongBench direct | Validate selection/budgets and report expected routes without inference or embedding |
| `--route-audit-only` | LongBench hybrid | Hybrid equivalent of preflight |
| `--preflight-output` | AA-LCR, MRCR | Save the preflight JSON to a file |
| `--execution-only` | AA-LCR | Generate and save answers but skip grading; this is not preflight |
| `--grader-prompt-version` | AA-LCR | Linux default `aa_lcr_equality_v1.1`; match the dataset version |
| `--grader-context-window` | AA-LCR | Grader's served context; default 32,768 |
| `--grader-max-output-tokens` | AA-LCR | Grader response allowance; default depends on prompt version and API style |
| `--grader-api-style` | AA-LCR | `vllm` (default) or `openai` |
| `--grader-reasoning-effort` | AA-LCR | Hosted grader reasoning setting; local vLLM grading is non-thinking |

Preflight can load/download the tokenizer. AA-LCR also needs the executor metadata
endpoint when its context limit is omitted. Predicted routes assume a cache miss.
MRCR and LongBench use deterministic scorers and have no answer-grading service.

## Service roles

| Role | Purpose |
|---|---|
| Executor | Generate answers on cache misses |
| Embedding model + FAISS | Retrieve source evidence; support semantic cache lookup |
| Cache verifier | Decide whether a semantic answer-cache candidate is reusable |
| Answer grader | Score AA-LCR predictions against references |

Dense document retrieval does not call the evaluator service. Optional reranking
uses the local reranker. The service named `evaluator` can host the grader, cache
verifier, or both; they use separate prompts and configuration. Disabling AA-LCR
grading does not disable an explicitly enabled verifier.

### Model and endpoint options

The Linux launcher defaults to executor `Qwen/Qwen3.6-35B-A3B` at
`http://127.0.0.1:8000/v1`. `OPENAI_COMPAT_EXECUTOR_MODEL` and
`OPENAI_COMPAT_EXECUTOR_BASE_URL` override those defaults for all benchmarks.
Runner-specific equivalents are:

| Runner | Model | Endpoint | Credential variable name |
|---|---|---|---|
| AA-LCR / MRCR | `--executor-model` | `--executor-base-url` | `--api-key-env` |
| LongBench direct | `--api-model` | `--api-base-url` | `--api-key-env` |
| LongBench hybrid | `--executor-model` | `--openai-compat-executor-base-url` | `--openai-compat-api-key-env` |
| AA-LCR grader | `--evaluator-model` | `--evaluator-base-url` | `--evaluator-api-key-env` |
| Cache verifier | `--cache-verifier-model` | `--cache-verifier-base-url` | `--cache-verifier-api-key-env` |

AA-LCR's Linux grader defaults to `Qwen/Qwen3.5-35B-A3B` at
`http://127.0.0.1:8001/v1`; the launcher accepts `OPENAI_COMPAT_EVALUATOR_MODEL`
and `OPENAI_COMPAT_EVALUATOR_BASE_URL` overrides. These do not automatically
configure the MRCR verifier; use the verifier options explicitly.
`OPENAI_COMPAT_API_KEY_ENV` supplies the executor credential-variable default.
Credential options take an environment variable's **name**, not the secret itself.

## Answer-cache and verifier options

| Option | Common-profile default | Behavior |
|---|---|---|
| `--answer-cache-read` / `--no-answer-cache-read` | Off | Enable/disable reuse of stored answers |
| `--answer-cache-write` / `--no-answer-cache-write` | Off | Enable/disable storing generated answers |
| `--cache-matching` | `semantic` | `exact` requires an identical query; `semantic` adds model-verified semantic matches |
| `--cache-verifier-model` | Runner-dependent fallback | Explicit verification model; required for MRCR semantic reads |
| `--cache-verifier-base-url` | Runner-dependent fallback | Explicit verification endpoint; required for MRCR semantic reads and hosted AA-LCR grading with semantic caching |
| `--cache-verifier-api-key-env` | No explicit override | Environment variable holding verifier credentials |

Semantic matching alone does not enable caching. The relevant option combinations
are:

| Intended behavior | Options |
|---|---|
| Populate without reuse | `--answer-cache-write --no-answer-cache-read` |
| Populate and reuse exact answers | `--answer-cache-read --answer-cache-write --cache-matching exact` |
| Populate and reuse semantic matches | `--answer-cache-read --answer-cache-write --cache-matching semantic`, plus verifier model/endpoint options |
| Disable answer caching | `--no-answer-cache-read --no-answer-cache-write` |

Specify the verifier model and endpoint explicitly for portable configurations.
Reads and writes are independent. Cache verification occurs when a semantic
candidate needs checking, not on every retrieval. Reuse is limited to the same
source and execution scope; enabling caching does not guarantee hits.
LongBench's direct API baseline rejects answer caching; use hybrid for its cache
experiments.

Document-index reuse is independent of answer caching and remains enabled within
source groups. Indexes are not persisted across runs. Answer-cache persistence
varies: LongBench hybrid can reload saved state (`--cache-state-root` selects its
location; `--cache-reset` deletes the selected namespace before running), AA-LCR
saves run-local state, and MRCR keeps answer caching in memory for the run.

## Retrieval and packing options

These affect oversized hybrid inputs, not fitting full-prompt requests.

| Option | Common-profile default | Behavior |
|---|---|---|
| `--child-tokens` | 7,500 | Chunk size measured with the embedding tokenizer |
| `--child-overlap-tokens` | 750 | Overlap between adjacent children; must be smaller than chunk size |
| `--pipeline-rerank-top` | `0` | Positive values enable the local reranker; zero disables it |
| `--evidence-order` | `source` | `source` presents evidence chronologically; `score` follows retrieval ranking |
| `--merge-overlaps` / `--no-merge-overlaps` | On with source order | Merge overlapping/adjoining ranges within each document |
| `SEMANTIC_CACHE_EMBEDDING_DEVICE` | `cuda` in Linux hybrid mode | Embedding device; can be `cpu` or a CUDA device |
| `SEMANTIC_CACHE_EMBEDDING_DTYPE` | `auto` | Embedding precision |

Merging requires source order. For a score-order comparison, use
`--evidence-order score` with `--no-merge-overlaps`. Packing measures the complete
rendered request against the input budget. LongBench direct has no chunk-size
options because it does not retrieve.

## Run control and artifacts

| Option | Applies to | Behavior |
|---|---|---|
| `--max-retries` | AA-LCR, MRCR, LongBench direct | Total request attempts, including the first; local default 5 |
| `--request-timeout-seconds` | AA-LCR, MRCR, LongBench direct | Per-request timeout; defaults to 1,800 / 1,800 / 120 respectively |
| `--fail-fast` | AA-LCR, MRCR, LongBench direct | Stop on a failed request instead of continuing |
| `--run-id`, `--repeat-id` | AA-LCR, MRCR | Run/repeat identifiers; repeat ID labels a run rather than launching repetitions |
| `--serving-metadata` | AA-LCR, MRCR | JSON describing the actual served model and settings |
| `--manifest-note` | LongBench | Free-text annotation for a run |
| `--output-root` | AA-LCR, MRCR | Artifact parent directory; defaults to `benchmark_artifacts/aa_lcr` or `benchmark_artifacts/mrcr_v2` |
| `--output-dir` | LongBench | Artifact parent directory; defaults to `benchmark_artifacts` |

Shared artifacts include `execution_manifest.json`, `execution.jsonl`,
`evaluation.jsonl`, `execution_report.json`, and `embedding_manifest.json` when a
backend is needed. Existing benchmark-specific predictions, bridge rows, and
reports are retained.

The common report counts execution failures as zero, excludes unsupported direct
examples, and leaves quality incomplete while grading is unresolved. Execution
usage excludes AA-LCR answer-grading usage, which remains in its bridge rows.
Compare quality on identical supported examples and report coverage, oversized
hybrid results, and cache/component experiments separately.
