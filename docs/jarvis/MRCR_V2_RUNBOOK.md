# MRCR v2 Jarvis Runbook

For a short explanation of the task and scoring, see the
[benchmark overview with examples](../mrcr_v2.md).

Separate direct/hybrid adapter for the released English text-style MRCR v2p1
CSVs. It reuses the existing embedding engine, FAISS retrieval, and
OpenAI-compatible HTTP client. It does not use an evaluator, knowledge lookup,
reranking, or answer-cache reads/writes. Existing benchmark behavior is unchanged.

Commands run from the repository root on Jarvis after completing the
[HPC setup](HPC_RUNBOOK_SETUP.md). The steps below cover data preparation,
executor startup, endpoint readiness, preflight, and benchmark submission.
**MRCR requires only an executor.** Its scorer is deterministic, so do not
start an evaluator for this benchmark. Evaluator startup for AA-LCR is documented
in the [AA-LCR service instructions](AA_LCR_RUNBOOK.md#start-the-jarvis-services).

## Prepare a bounded dataset

Use the existing benchmark Python environment (Python 3.13+, `transformers` and
the executor tokenizer). Hybrid execution additionally needs the existing
semantic-cache dependencies, including `torch`, `numpy`, `faiss-cpu` or the
environment's FAISS GPU installation, and `python-dotenv`. Preparation and
preflight do not load executor or embedding weights. The tokenizer may download
on first use. Jarvis uses the repository's existing client/client-GPU environments.

```bash
uv run python -m mrcr_v2.prepare_dataset \
  --download-bands 65536:131072,131072:262144 \
  --needles 8 \
  --min-source-tokens 60000 --max-source-tokens 250000 \
  --data-dir benchmark_data/mrcr_v2_60k_250k
```

Alternatively pass `--input-csv /path/to/released.csv` (repeatable). Preparation
requires a new output directory; failed preparation leaves an incomplete
directory without a usable manifest. Choose a new directory when retrying.
Downloads are limited to the explicitly listed official bands; use endpoints
from powers of two between 4,096 and 8,388,608. The cumulative `upto_128K` file is
not automatically downloaded. Local CSVs must have the selected needle count.

`--min-source-tokens` and `--max-source-tokens` are **inclusive bounds on the full
rendered executor prompt**, including the chat template, few-shot examples and
final instruction. The official file bands and `context_len` use a Gemini
tokenizer and are not interchangeable with these bounds. Only the supplied
files are searched; narrow filters can produce no matches. No source is
truncated to fit. `--max-rows` defaults to all eligible unique rows; a positive
value takes the first rows in sorted-input-file then CSV-row order, after
filtering. All supplied rows are still checked for conflicting duplicates.

Preparation stores each shared source once and keeps answers and upstream
position metadata in separate question records. The manifest identifies input
file hashes, selected IDs, bounds, tokenizer revision/fingerprint and template.
Use `--tokenizer-revision` to select a particular executor tokenizer revision.

## Start the executor

Use the existing Qwen service environment from the HPC setup. This profile
matches MRCR's default 65,536-token context limit:

```bash
MODULES="cuda12.8/toolkit/12.8.1" \
EXECUTOR_MODEL=Qwen/Qwen3.6-35B-A3B \
EXECUTOR_TP_SIZE=4 \
EXECUTOR_PORT=8000 \
EXECUTOR_MAX_MODEL_LEN=65536 \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
VLLM_EXTRA_ARGS="--reasoning-parser qwen3 --language-model-only --max-num-seqs 1 --enable-chunked-prefill --max-num-batched-tokens 8192" \
SYNC_BACK_MODELS=1 VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash jarvis/run.sh submit executor
```

Adjust `VLLM_VENV` to your service environment. Record the submitted job ID and
wait for `Application startup complete` in its log. With the default scratch
storage profile, logs and endpoint files are under `$HOME/adarsh-rlms-logs`;
use your configured `PROJECT_LOG_DIR` if different. The URL file can appear
before the service is ready.

```bash
export PROJECT_LOG_DIR="${PROJECT_LOG_DIR:-$HOME/adarsh-rlms-logs}"
# Replace 123456 with the executor job ID returned by sbatch.
EXECUTOR_JOB_ID=123456
export OPENAI_COMPAT_EXECUTOR_BASE_URL="$(cat "$PROJECT_LOG_DIR/executor-${EXECUTOR_JOB_ID}.url")"
export OPENAI_COMPAT_EXECUTOR_MODEL=Qwen/Qwen3.6-35B-A3B
export OPENAI_COMPAT_BASE_URL="$OPENAI_COMPAT_EXECUTOR_BASE_URL"
export OPENAI_COMPAT_EVALUATOR_BASE_URL="$OPENAI_COMPAT_EXECUTOR_BASE_URL"
export WAIT_FOR_ENDPOINTS=1
export MRCR_DATA_DIR=benchmark_data/mrcr_v2_60k_250k
curl -fsS "$OPENAI_COMPAT_EXECUTOR_BASE_URL/models"
```

The evaluator URL above is an alias for the shared Jarvis readiness check:
`run_client.sh` otherwise may poll a second endpoint. It does not start an
evaluator or cause evaluator calls. The MRCR adapter only calls the executor.
Confirm `/models` lists `Qwen/Qwen3.6-35B-A3B` before running.

An existing compatible executor can be reused. If you start a larger context
window, pass matching `--context-window-tokens` and appropriate input/output
budgets to the runner. Increasing source bounds alone does not require a larger
executor window: hybrid retrieval still packs evidence into the configured
input budget. Changing service environment variables does not resize an
already-running executor.

## Preflight

```bash
uv run python -m mrcr_v2.run_benchmark \
  --data-dir benchmark_data/mrcr_v2_60k_250k --mode hybrid \
  --min-source-tokens 100000 --max-source-tokens 200000 \
  --context-window-tokens 65536 --max-input-tokens 60000 --max-output-tokens 4096 \
  --max-rows 10 --preflight-only
```

Runtime source bounds default to the prepared bounds and can only narrow them.
Changing the tokenizer/template or widening the available interval requires
re-preparation. `--preflight-output audit.json` saves the preflight report;
preflight validates source hashes, prompt reconstruction, counts and budgets,
but does not load embedding weights, embed documents or contact the executor.

Source bounds are independent of executor budgets. Input plus output allowance
must fit `--context-window-tokens`. Default budgets are 60,000 input and 4,096
output within a 65,536-token context. Choose an output budget large enough to
copy the target response; `finish_reason` exposes output-limit terminations.

- **Direct:** sends the original upstream prompt as one user message when it
  fits, otherwise records `unsupported_context` without an API call.
- **Hybrid:** matches LongBench routing, using `direct_fit` when the prompt fits
  and `dense_child_packed` otherwise. It always disables answer caching.
  Dense retrieval uses the final instruction. It selects chunks in score order,
  merges overlapping/adjoining source ranges and presents them chronologically.
  Omitted regions have explicit separators. Few-shot examples remain outside
  the index and are retained in the request. Packing stops before the next
  candidate would exceed the exact rendered input budget.

Hybrid defaults to 7,500 embedding tokens per child and 750 overlap; adjust with
`--child-tokens` and `--child-overlap-tokens`. Exact tokenizer offsets are
required, and reaching the configured embedding truncation limit is an error.
One transcript index is reused for its questions. There is no benchmark-specific
occurrence solver and no gold answer/position information in execution inputs.

Both modes use temperature 0 and disable thinking. Direct and hybrid execute
the same prompt for `direct_fit`; these rows do not measure a retrieval benefit.
Oversized hybrid rows measure retrieval-assisted system capability, not native
model context capacity. Published upstream evaluation uses different generation
settings; report our settings alongside scores. No truncated baseline is provided.

## Submit benchmark clients

```bash
MRCR_LAUNCH_DRY_RUN=1 bash jarvis/run_mrcr_v2.sh hybrid \
  --min-source-tokens 100000 --max-source-tokens 200000 \
  --max-input-tokens 60000 --max-output-tokens 4096 --context-window-tokens 65536
```

Using the endpoint and dataset exports above, submit either mode with the same
selection and budgets:

```bash
bash jarvis/run_mrcr_v2.sh direct \
  --min-source-tokens 100000 --max-source-tokens 200000 \
  --max-input-tokens 60000 --max-output-tokens 4096 --context-window-tokens 65536 \
  --max-rows 10

bash jarvis/run_mrcr_v2.sh hybrid \
  --min-source-tokens 100000 --max-source-tokens 200000 \
  --max-input-tokens 60000 --max-output-tokens 4096 --context-window-tokens 65536 \
  --max-rows 10
```

With this particular 100K–200K source selection and 60K input budget, every
direct row is reported as unsupported; hybrid uses retrieval. To compare
full-prompt answers, prepare/select shorter examples or use a larger executor
window and input budget. Remove `--max-rows 10` for all eligible examples.

Direct uses a client allocation (32G
default); hybrid uses client-GPU (96G default). Override `CLIENT_MEM` as needed;
these defaults are not a verified 8M-token memory guarantee. All runner flags,
including bounds, `--run-id`, `--repeat-id`, `--serving-metadata`, timeout,
retry and credential options are forwarded. No evaluator service is required.

## Scores and artifacts

The vendored scorer is from upstream commit
`67b7fd29b2205ee0a3226e0d3e5d74140a253b42`, with Apache-2.0 attribution in
[scoring.py](../../mrcr_v2/scoring.py) and [LICENSE](../../mrcr_v2/LICENSE).
The original metric is unchanged: missing the
required marker scores zero, otherwise `SequenceMatcher` compares the reference
body with text following the **last** marker. Despite the upstream docstring,
the implementation allows leading text before the marker.

Results report the official mean similarity (0–1), exact-match accuracy
(`prediction.strip() == target.strip()`), and strict prefix compliance. They
are different metrics: similarity 0.8 does not mean 80% exact answers. Execution
failures receive zero and remain in the supported-example denominator.
Unsupported direct rows are excluded and reported through coverage counts.
If none are supported, quality metrics are null. Reports group results by
route and power-of-two bands of actual executor source length.

Each unique run directory under `benchmark_artifacts/mrcr_v2/<mode>/` contains
`manifest.json`, incrementally flushed `predictions.jsonl` and `bridge_rows.csv`,
and `eval_report.json`. Records include token counts, selected character ranges,
timings, returned API usage, finish reason and failures. Ingestion time is charged
to the first question using each transcript index. Failed HTTP attempts can
incur usage not returned by the endpoint; recorded usage covers returned
completions. An interrupted run retains completed prediction rows and a running
manifest; automatic resume is outside v1. `--fail-fast` saves processed results
and exits with an error. Upstream position metadata is retained for offline
inspection only; its Gemini offsets must not be used as executor offsets.

Run offline validation with:

```bash
uv run python -m unittest discover -s test -p 'test_mrcr_v2.py'
```

The tests use synthetic fixtures and injected clients/tokenizers; they do not
download weights or run a real benchmark.
