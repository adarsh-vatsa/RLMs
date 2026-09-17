# Shared direct/hybrid execution

LongBench-v2 (OpenAI-compatible direct and hybrid), AA-LCR, and MRCR v2 now call
`execution.pipeline.Pipeline`. Dataset parsing, task formatting, and scoring
remain in their adapters. Iterative/RLM and other hosted-provider LongBench paths
are still separate architectures.

See the [architecture diagrams](../shared_execution_architecture.md) and
[implementation plan](../shared_execution_pipeline_plan.md).

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

The new LongBench launcher below defaults to `common` and original rows.
AA-LCR/MRCR launchers retain their existing defaults. Always record the explicit
profile in an experiment command. Legacy cache namespaces are versioned by the
new pipeline, so old answer-cache snapshots are not silently reused.

All three accept `--min-source-tokens` and `--max-source-tokens`, measured on the
complete rendered executor prompt. For AA-LCR/LongBench, supply both together;
the row limit is applied after filtering. MRCR retains its prepared bounds and
allows narrowing only. Source selection never truncates an example.

## Prepare and preflight

Use the existing [AA-LCR](AA_LCR_RUNBOOK.md), [MRCR](MRCR_V2_RUNBOOK.md), and
[LongBench](HPC_RUNBOOK_EXPERIMENT.md) preparation instructions. Preflight loads
the executor tokenizer but makes no embedding or inference calls. AA-LCR also
reads executor metadata when its context limit is omitted; supply an explicit
limit for offline preflight. With caching enabled, predicted routes assume a cache miss.

```bash
bash jarvis/run_mrcr_v2.sh hybrid \
  --execution-profile common --preflight-only \
  --min-source-tokens 100000 --max-source-tokens 200000 \
  --context-window-tokens 65536 --max-input-tokens 60000 --max-output-tokens 4096

bash jarvis/run_aa_lcr.sh hybrid \
  --execution-profile common --preflight-only \
  --context-window-tokens 65536 --max-input-tokens 60000 --max-output-tokens 4096

bash jarvis/run_longbench_v2.sh hybrid --route-audit-only
```

These commands submit preflight jobs. Set the respective `MRCR_LAUNCH_DRY_RUN`,
`AA_LCR_LAUNCH_DRY_RUN`, or `LONGBENCH_LAUNCH_DRY_RUN` to `1` to print commands
without submitting anything.

## Execute with the same system policy

Start an executor with a 65,536-token context using the
[executor startup instructions](MRCR_V2_RUNBOOK.md), then set its endpoint.
The examples deliberately choose the same model and input allowance. Output
allowances follow the task; retain each allowance for its direct/hybrid pair.

```bash
export OPENAI_COMPAT_EXECUTOR_BASE_URL=http://executor-host:8000/v1
export OPENAI_COMPAT_EXECUTOR_MODEL=Qwen/Qwen3.6-35B-A3B

bash jarvis/run_mrcr_v2.sh hybrid \
  --execution-profile common \
  --context-window-tokens 65536 --max-input-tokens 60000 --max-output-tokens 4096

bash jarvis/run_aa_lcr.sh hybrid \
  --execution-profile common --execution-only \
  --context-window-tokens 65536 --max-input-tokens 60000 --max-output-tokens 4096

bash jarvis/run_longbench_v2.sh hybrid
```

Replace `hybrid` with `direct` for any of the three benchmarks.
In the common profile, all three record unsupported direct
examples without inference. Compare quality on common supported examples and
report oversized hybrid results separately.

AA-LCR separates mode from context size. With `--mode direct|hybrid` (or the
launcher names above), omitting `--context-window-tokens` reads the selected
executor's `max_model_len` from its `/v1/models` endpoint. Omitting
`--max-input-tokens` uses that window minus the output allowance (16,384 by
default). This uses the served limit, which can be lower than the model's
advertised capacity. If discovery is unavailable, supply the limit explicitly.
MRCR and LongBench retain their existing budget defaults.

```bash
# AA-LCR: full served context, reserving 4,096 tokens for the answer.
bash jarvis/run_aa_lcr.sh hybrid \
  --execution-profile common --execution-only --max-output-tokens 4096

# AA-LCR: explicit 64K context; usable input defaults to 61,440 tokens.
bash jarvis/run_aa_lcr.sh direct \
  --execution-profile common --execution-only \
  --context-window-tokens 65536 --max-output-tokens 4096
```

Historical AA-LCR experiment names remain accepted with their original budgets.

AA-LCR's `--execution-only` saves answers for its existing `aa_lcr.regrade`
workflow. To grade during the run, omit this flag and configure the evaluator
using the [AA-LCR grading instructions](AA_LCR_RUNBOOK.md). LongBench and MRCR
have deterministic scorers and do not need an answer-grading service.

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
explicit verifier endpoint. Launcher readiness waits follow enabled roles.

For a rank-order ablation, pass `--evidence-order score --no-merge-overlaps` to
each benchmark. Such a run is a different system configuration; do not pool it
with the primary common-profile results. Original and repeated/paraphrased
LongBench rows should likewise be reported separately.

## Artifacts and limitations

Existing prediction/report files remain available. Each shared run also writes:

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
