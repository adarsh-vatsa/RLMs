# LongBench options on Jarvis

An optional reference for changing LongBench-v2 run settings on Jarvis. The
workflow is in the [LongBench runbook](HPC_RUNBOOK_EXPERIMENT.md). AA-LCR and
MRCR v2 run on the Neselab Linux server; see the
[Linux runbooks](../linux/README.md). Their Jarvis launchers
(`jarvis/run_aa_lcr.sh`, `jarvis/run_mrcr_v2.sh`) still work but are not
documented here.

LongBench's direct and hybrid runners use the shared execution pipeline, so the
shared options (budgets, overflow, chunking, evidence order, answer caching and
artifacts) behave as described in the
[Linux options reference](../linux/SHARED_EXECUTION_RUNBOOK.md). This page covers
what is specific to LongBench and to Jarvis.

## The launcher

`bash adarsh-rlms/jarvis/run_longbench_v2.sh <direct|hybrid> [runner options...]`
submits one Slurm client job:

| Mode | Runner | Allocation | Default memory |
|---|---|---|---|
| `direct` | `long_bench_v2.run_api_benchmark` | `client` (CPU) | 32 GB |
| `hybrid` | `long_bench_v2.run_benchmark --mode baseline` | `client-gpu` (one L40S) | 96 GB |

It always passes `--execution-profile common --row-types original`,
`--suite-csv benchmark_data/long_bench_v2/data.csv` and the full-window budgets
`--context-window-tokens 262144 --max-input-tokens 262136 --max-output-tokens 8`,
plus the executor model and endpoint. Options given on the command line come
later and override these.

| Variable | Purpose |
|---|---|
| `OPENAI_COMPAT_EXECUTOR_BASE_URL` | Executor endpoint; required except for preflight |
| `OPENAI_COMPAT_EXECUTOR_MODEL` | Served model name; default `Qwen/Qwen3.6-35B-A3B` |
| `LONGBENCH_SERVING_METADATA` | Serving-setup JSON copied into the run manifest |
| `CLIENT_MEM` | Slurm memory for the client job |
| `SEMANTIC_CACHE_EMBEDDING_DEVICE` | Hybrid embedding device; default `cuda` |
| `SEMANTIC_CACHE_EMBEDDING_DTYPE` | Hybrid embedding precision; default `auto` |
| `SEMANTIC_CACHE_EMBEDDING_BATCH_SIZE` | Chunks embedded per batch; default 16, lower it if the GPU runs out of memory |
| `LONGBENCH_LAUNCH_DRY_RUN=1` | Print the job and command without submitting |

The launcher checks which services the options need before submitting. With
answer caching off (the default), only the executor is needed. Semantic cache
reads also need a cache verifier (`--cache-verifier-model` and
`--cache-verifier-base-url`). Preflight and route audits need no service.

## Runner options

| Option | Applies to | Behavior |
|---|---|---|
| `--direct-overflow middle` | Direct | Keep a head and tail of oversized prompts; without it they are recorded as unsupported |
| `--min-source-tokens`, `--max-source-tokens` | Both | Select questions by full rendered prompt length; supply both |
| `--max-rows` | Both | Limit the number of questions; with source bounds, applied after the length filter |
| `--source-ids` | Both | Comma-separated document identifiers |
| `--row-types` | Both | The launcher selects `original`; `data.csv` also holds `exact` repeats |
| `--row-order` | Hybrid | `source_grouped` (default) or `input` |
| `--preflight-only` | Direct | Write `route_audit.json` without inference |
| `--route-audit-only` | Hybrid | Write `route_audit.json` without embedding or inference |
| `--answer-cache-read`, `--answer-cache-write` | Hybrid | Reuse and store answers; off by default |
| `--cache-matching` | Hybrid | `semantic` (default) or `exact` question matching |
| `--cache-verifier-model`, `--cache-verifier-base-url` | Hybrid | Model that confirms semantic matches |
| `--cache-reset` | Hybrid | Start from an empty saved cache |
| `--fail-fast` | Both | Stop on the first failed request |
| `--serving-metadata` | Both | JSON object recorded in the manifest |
| `--manifest-note` | Both | Free-text note recorded in the manifest |
| `--output-dir` | Both | Artifact parent directory; default `benchmark_artifacts` |
| `--child-tokens`, `--child-overlap-tokens` | Hybrid | Chunk size and overlap in embedding tokens; default 7,500 and 750 |
| `--evidence-order`, `--merge-overlaps` | Hybrid | Evidence presentation; default source order with overlaps merged |
| `--max-retries` | Direct | Request attempts per question; default 5 for local endpoints |
| `--request-timeout-seconds` | Direct | Per-request timeout; default 120 |

Direct runs are written under `benchmark_artifacts/longbench_v2_api/` and hybrid
runs under `benchmark_artifacts/longbench_v2/`; hybrid route audits go under
`benchmark_artifacts/longbench_v2_route_audit/`.

## Dry run

```bash
OPENAI_COMPAT_EXECUTOR_BASE_URL=http://executor-host:8000/v1 LONGBENCH_LAUNCH_DRY_RUN=1 \
  bash adarsh-rlms/jarvis/run_longbench_v2.sh hybrid --min-source-tokens 262137 --max-source-tokens 5000000
```

This prints the allocation, the services the run will wait for and the resolved
command. It does not read data or contact services.
