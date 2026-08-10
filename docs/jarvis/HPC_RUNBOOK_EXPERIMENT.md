# Jarvis HPC Experiment Runbook

This runbook contains the LongBench-v2 benchmark controls, active hybrid
experiments, decoder-policy validation, and full-run commands. Complete
[`HPC_RUNBOOK_SETUP.md`](HPC_RUNBOOK_SETUP.md) through production-service
startup and endpoint validation before running anything here.

Run these commands from the Jarvis login node unless a section says otherwise.
They assume the current directory is the parent directory containing the
`adarsh-rlms` repository and that `EXECUTOR_URL` and `EVALUATOR_URL` refer
to the running production services.

## 1. Run A Small Comparable Benchmark Client Job

Start with a sampled LongBench-v2 run to verify the services, cache reuse, and
scoring path before submitting a larger job. This command generates a bounded
domain-balanced source-linked suite first, with three source groups per eligible
domain in the token band, then scores the linked `original`, `exact`, and
`semantic` rows. Keep parameter experiments in this one block so each run has a
single command to compare.

```bash
LLM_PROVIDER=openai_compatible \
OPENAI_COMPAT_EXECUTOR_BASE_URL="$EXECUTOR_URL" \
OPENAI_COMPAT_EVALUATOR_BASE_URL="$EVALUATOR_URL" \
WAIT_FOR_ENDPOINTS=1 \
CLIENT_MEM=96G \
CLIENT_CMD='export SEMANTIC_CACHE_SEARCH_MODE=iterative
export SEMANTIC_CACHE_EMBEDDING_QUERY_INSTRUCTION="Given a multiple-choice question, retrieve chunks containing evidence, demonstrations, mappings, or facts needed to answer it."
export SEMANTIC_CACHE_EMBEDDING_DEVICE=cuda
export SEMANTIC_CACHE_EMBEDDING_DTYPE=auto
export SEMANTIC_CACHE_EMBEDDING_BATCH_SIZE=2
export SEMANTIC_CACHE_EMBEDDING_MAX_LENGTH=8192
export SEMANTIC_CACHE_DOC_CHUNK_TOKENS=10000
export SEMANTIC_CACHE_DOC_CHUNK_OVERLAP_TOKENS=2000
export SEMANTIC_CACHE_SCAN_MIN_CHUNK_RATIO=0.50
export SEMANTIC_CACHE_SCAN_MAX_CHUNK_RATIO=0.65
export SEMANTIC_CACHE_SCAN_MIN_CHUNKS=4
export SEMANTIC_CACHE_SCAN_MAX_CHUNKS=0
export SEMANTIC_CACHE_SCAN_MAX_TOKENS=1536
export SEMANTIC_CACHE_SCAN_EMPTY_LEDGER_FALLBACK_RATIO=1.0
export SEMANTIC_CACHE_ITERATIVE_PACKED_FALLBACK_INPUT_TOKEN_BUDGET=60000
export SEMANTIC_CACHE_ITERATIVE_MEMORY_MAX_CHARS=16000
export SEMANTIC_CACHE_ITERATIVE_BATCH_MAX_CHUNKS=3
export SEMANTIC_CACHE_ITERATIVE_BATCH_INPUT_TOKEN_BUDGET=50000
export SEMANTIC_CACHE_MCQ_SYNTHESIS_MAX_TOKENS=8
export SEMANTIC_CACHE_MCQ_PROMPT_STYLE=strict
export OPENAI_COMPAT_EXECUTOR_EXTRA_BODY_JSON="{\"chat_template_kwargs\":{\"enable_thinking\":false}}"
export OPENAI_COMPAT_EVALUATOR_EXTRA_BODY_JSON="{\"chat_template_kwargs\":{\"enable_thinking\":false}}"

uv run python long_bench_v2/sample_csv.py \
  --input-path benchmark_data/long_bench_v2/data_cache_suite.csv \
  --output-path benchmark_artifacts/longbench_v2_samples/jarvis_param_search.csv \
  --sample-size 1 \
  --samples-per-domain 3 \
  --selection-strategy random \
  --min-token-count 50000 \
  --max-token-count 200000 \
  --seed 0 && \
uv run python long_bench_v2/run_benchmark.py \
  --suite-csv benchmark_artifacts/longbench_v2_samples/jarvis_param_search.csv \
  --llm-provider openai_compatible \
  --mode cache \
  --cache-reset \
  --cache-state-root "$JARVIS_CACHE_STATE_ROOT" \
  --executor-model Qwen/Qwen3.6-35B-A3B \
  --evaluator-model Qwen/Qwen3.5-35B-A3B \
  --row-types original,exact,semantic \
  --output-dir benchmark_artifacts \
  --manifest-note jarvis-l40s-param-search-iterative-scan-strict-mcq' \
  bash adarsh-rlms/jarvis/run.sh submit client-gpu
```

`submit client-gpu` gives the benchmark client one L40S for local embeddings.
`SEMANTIC_CACHE_EMBEDDING_DEVICE=cuda` makes a missing CUDA runtime fail clearly
instead of silently falling back to CPU, and `SEMANTIC_CACHE_EMBEDDING_DTYPE=auto`
uses CUDA-friendly reduced precision. `SEMANTIC_CACHE_EMBEDDING_BATCH_SIZE=1` is
the safest fallback for 8192-token embedding forwards if the active batch size
of `2` OOMs. `CLIENT_MEM=96G` keeps enough host RAM for
chunk text, tokenizer offset maps, embeddings, metadata, and FAISS state.
`SEMANTIC_CACHE_EMBEDDING_MAX_LENGTH=8192` keeps the default amount of each chunk
visible to the embedding model. Lower it only if batch size 1 still OOMs, because
lower values can weaken FAISS ranking when the relevant evidence appears late in
a chunk. If Slurm reports `oom_kill`, confirm the memory limit and peak RSS:

```bash
sacct -j <client_job_id> --format=JobID,JobName,State,ExitCode,MaxRSS,ReqMem,Elapsed
```

Use `CLIENT_MEM=64G` first if queue pressure matters; use `CLIENT_MEM=96G` when
testing smaller chunks, higher overlap, or GPU embeddings against 50k-200k token
samples.

Monitor:

```bash
squeue -u "$USER"
tail -f "$PROJECT_LOG_DIR"/rlms-client-<client_job_id>.out
tail -f "$PROJECT_LOG_DIR"/rlms-client-gpu-<client_job_id>.out
```

When this finishes, inspect the generated artifact paths printed in the client
log. They should point under `benchmark_artifacts/longbench_v2/...`.

For domain-targeted diagnostics, add `--domains "<domain name>"` to the
`sample_csv.py` call rather than changing the benchmark runner. For example,
use `--domains "Long In-context Learning"` when isolating long in-context
learning failures.

`--samples-per-domain` overrides `--sample-size`.

The sampled validation command resets only this selected cache namespace. Keep
that reset while testing retrieval or synthesis changes; otherwise exact and
semantic rows can reuse a bad first-write answer from an older run. The active
prompt profile is `strict`, which asks the executor to reject choices that are
too narrow, too broad, partially supported, or unsupported before returning a
single letter.

The profile above is the current balanced LongBench/Jarvis parameter-search path.
FAISS ranks likely chunks first, then the executor inspects chunks in batches of
up to three and carries a cumulative memory ledger forward. The ledger keeps
additive chunk-referenced notes, target facts, option-code mappings, the current
`best_choice`, rationale, and up to five open questions. For many-shot relation
rows, examples are retained in `code_mappings` only when they use one of the
relation codes present in the current answer options. The scan budget is
adaptive:
`SEMANTIC_CACHE_SCAN_MIN_CHUNK_RATIO=0.50` means early stop is not allowed until
at least 50% of chunks have been inspected, while
`SEMANTIC_CACHE_SCAN_MAX_CHUNK_RATIO=0.65` means this profile initially scans at
most 65% of chunks if no answer is found. The reader asks FAISS for the
ratio-based scan budget and inspects those chunks in FAISS-ranked order.
`SEMANTIC_CACHE_SCAN_MAX_CHUNKS=0` leaves the ratio-based maximum uncapped by an
absolute chunk count. If the normal scan produces no useful observations,
`SEMANTIC_CACHE_SCAN_EMPTY_LEDGER_FALLBACK_RATIO=1.0` allows scanning the
remaining chunks. Reduced-budget scans also continue over remaining available
chunks when an ordering row needs chronology coverage or a symbolic-code row has
competing mapped codes/open code questions. If final ledger adjudication is
empty, invalid, asks for more context, lacks required ordering evidence, or lacks
required symbolic-code contrast, the reader falls back to a bounded structured
packed synthesis call under
`SEMANTIC_CACHE_ITERATIVE_PACKED_FALLBACK_INPUT_TOKEN_BUDGET`. Ordering rows
require a chosen sequence, evidence for each numbered narrative, and adjacent
pairwise order evidence keyed by edges such as `2<4`. Symbolic-code rows with
multiple mapped candidate option codes require selected-code evidence and
rejection evidence for each other mapped code before the ledger-only final answer
is accepted. `SEMANTIC_CACHE_ITERATIVE_BATCH_MAX_CHUNKS=3` lets the iterative
inspector pack up to three scan chunks into one executor call, subject to
`SEMANTIC_CACHE_ITERATIVE_BATCH_INPUT_TOKEN_BUDGET=50000`; set the max chunks to
`1` to restore the older one-chunk-per-call behavior. The cumulative memory text
is bounded by `SEMANTIC_CACHE_ITERATIVE_MEMORY_MAX_CHARS`; if it exceeds the cap,
old prose updates are trimmed while structured facts, mappings, best choice, and
parse failures are preserved.

If rows still take too long, lower the ratio band first:

```bash
export SEMANTIC_CACHE_SCAN_MIN_CHUNK_RATIO=0.20
export SEMANTIC_CACHE_SCAN_MAX_CHUNK_RATIO=0.50

uv run python long_bench_v2/run_benchmark.py \
  ...
```

The main knobs to edit in the one command above are the sample token band,
`SEMANTIC_CACHE_DOC_CHUNK_TOKENS`,
`SEMANTIC_CACHE_DOC_CHUNK_OVERLAP_TOKENS`,
`SEMANTIC_CACHE_EMBEDDING_BATCH_SIZE`,
`SEMANTIC_CACHE_EMBEDDING_MAX_LENGTH`,
`SEMANTIC_CACHE_SCAN_MIN_CHUNK_RATIO`, `SEMANTIC_CACHE_SCAN_MAX_CHUNK_RATIO`,
`SEMANTIC_CACHE_SCAN_MIN_CHUNKS`, `SEMANTIC_CACHE_SCAN_MAX_CHUNKS`,
`SEMANTIC_CACHE_SCAN_MAX_TOKENS`,
`SEMANTIC_CACHE_SCAN_EMPTY_LEDGER_FALLBACK_RATIO`, and
`SEMANTIC_CACHE_ITERATIVE_PACKED_FALLBACK_INPUT_TOKEN_BUDGET`,
`SEMANTIC_CACHE_ITERATIVE_MEMORY_MAX_CHARS`. For additional short, medium, and
long random sample examples, see
`long_bench_v2/docs/longbench_v2.md`.

Chunking changes affect client RAM as well as retrieval quality. The effective
chunk step is `DOC_CHUNK_TOKENS - DOC_CHUNK_OVERLAP_TOKENS`, so `10000/2000`
roughly doubles the number of chunks compared with `20000/4000` over the same
source length. If the smaller profile OOMs before producing artifacts, try a
larger client allocation or a middle profile such as `12000/3000` before lowering
chunk size further.

## 2. Run The Historical Full Benchmark

Use the same service URLs and run the intended row set directly:

This command exercises the cache/retrieval benchmark over the full CSV. It is
not a direct "send every full LongBench context to the model" run. The full
document is ingested into token-bounded chunks, FAISS ranks likely chunks, and
the iterative reader inspects an adaptive ratio of chunks with a cumulative
memory ledger. The executor service above starts vLLM with
`EXECUTOR_MAX_MODEL_LEN=262144`, so each chunk-inspection call stays below that
context window while the index still covers the full source document.

```bash
LLM_PROVIDER=openai_compatible \
OPENAI_COMPAT_EXECUTOR_BASE_URL="$EXECUTOR_URL" \
OPENAI_COMPAT_EVALUATOR_BASE_URL="$EVALUATOR_URL" \
WAIT_FOR_ENDPOINTS=1 \
CLIENT_MEM=96G \
CLIENT_CMD='export SEMANTIC_CACHE_SEARCH_MODE=iterative
export SEMANTIC_CACHE_EMBEDDING_QUERY_INSTRUCTION="Given a multiple-choice question, retrieve chunks containing evidence, demonstrations, mappings, or facts needed to answer it."
export SEMANTIC_CACHE_EMBEDDING_DEVICE=cuda
export SEMANTIC_CACHE_EMBEDDING_DTYPE=auto
export SEMANTIC_CACHE_EMBEDDING_BATCH_SIZE=2
export SEMANTIC_CACHE_EMBEDDING_MAX_LENGTH=8192
export SEMANTIC_CACHE_DOC_CHUNK_TOKENS=10000
export SEMANTIC_CACHE_DOC_CHUNK_OVERLAP_TOKENS=2000
export SEMANTIC_CACHE_SCAN_MIN_CHUNK_RATIO=0.50
export SEMANTIC_CACHE_SCAN_MAX_CHUNK_RATIO=0.65
export SEMANTIC_CACHE_SCAN_MIN_CHUNKS=4
export SEMANTIC_CACHE_SCAN_MAX_CHUNKS=0
export SEMANTIC_CACHE_SCAN_MAX_TOKENS=1536
export SEMANTIC_CACHE_SCAN_EMPTY_LEDGER_FALLBACK_RATIO=1.0
export SEMANTIC_CACHE_ITERATIVE_PACKED_FALLBACK_INPUT_TOKEN_BUDGET=60000
export SEMANTIC_CACHE_ITERATIVE_MEMORY_MAX_CHARS=16000
export SEMANTIC_CACHE_ITERATIVE_BATCH_MAX_CHUNKS=3
export SEMANTIC_CACHE_ITERATIVE_BATCH_INPUT_TOKEN_BUDGET=50000
export SEMANTIC_CACHE_MCQ_SYNTHESIS_MAX_TOKENS=8
export SEMANTIC_CACHE_MCQ_PROMPT_STYLE=strict
export OPENAI_COMPAT_EXECUTOR_EXTRA_BODY_JSON="{\"chat_template_kwargs\":{\"enable_thinking\":false}}"
export OPENAI_COMPAT_EVALUATOR_EXTRA_BODY_JSON="{\"chat_template_kwargs\":{\"enable_thinking\":false}}"

uv run python long_bench_v2/run_benchmark.py \
  --llm-provider openai_compatible \
  --mode cache \
  --cache-reset \
  --cache-state-root "$JARVIS_CACHE_STATE_ROOT" \
  --executor-model Qwen/Qwen3.6-35B-A3B \
  --evaluator-model Qwen/Qwen3.5-35B-A3B \
  --row-types original,exact,semantic \
  --output-dir benchmark_artifacts \
  --manifest-note jarvis-l40s-full-iterative-scan-strict-mcq' \
  bash adarsh-rlms/jarvis/run.sh submit client-gpu
```

Keep `--cache-reset` for the first comparable full run. Remove it only when you
intentionally want to resume from an existing benchmark cache namespace. Do not
delete `/local/$USER/llm_caching` unless you intentionally want to force model
downloads again on that node.

## 3. Audit And Smoke The Fast Hybrid v1

Follow `docs/longbench_v2_hierarchical_retrieval_plan_20260808.md` as the
governing architecture and experiment plan. Keep the iterative command above as
the historical control. The constrained rerun starts with a vLLM capability
probe and the five previously invalid sources, followed by the isolated
50-source hybrid gate. Do not start either full run before that gate passes.

No executor startup flag changes. Confirm the active service environment and
then verify the vLLM 0.19.1 choice request itself. The request must return HTTP
200 and exactly one letter:

```bash
/home/edogu/.venvs/adarsh-vllm/bin/python -c \
  'import vllm; print(vllm.__version__); assert vllm.__version__ == "0.19.1"'

python - "$EXECUTOR_URL" <<'PY'
import json
import re
import sys
import urllib.request

url = sys.argv[1].rstrip("/") + "/chat/completions"
payload = {
    "model": "Qwen/Qwen3.6-35B-A3B",
    "messages": [{
        "role": "user",
        "content": "Choose one. A. Alpha B. Beta C. Gamma D. Delta. Return one letter.",
    }],
    "max_tokens": 8,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
    "structured_outputs": {"choice": ["A", "B", "C", "D"]},
}
request = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=120) as response:
    assert response.status == 200, response.status
    body = json.load(response)
content = body["choices"][0]["message"]["content"].strip()
assert re.fullmatch(r"[A-D]", content), repr(content)
print(f"structured choice capability ok: {content}")
PY
```

Do not use the removed `guided_choice` field. A rejected
`structured_outputs.choice` request is a failed capability check; the benchmark
runners do not fall back to unconstrained generation.

The route audit loads only the Qwen3.6 tokenizer. It does not load the embedding
or reranker models, connect to FAISS, read cache state, or call either vLLM
service:

```bash
LLM_PROVIDER=openai_compatible \
WAIT_FOR_ENDPOINTS=0 \
CLIENT_MEM=32G \
CLIENT_CMD='export SEMANTIC_CACHE_SEARCH_MODE=hybrid

uv run python long_bench_v2/run_benchmark.py \
  --llm-provider openai_compatible \
  --mode cache \
  --executor-model Qwen/Qwen3.6-35B-A3B \
  --evaluator-model Qwen/Qwen3.5-35B-A3B \
  --row-types original \
  --context-window-tokens 262144 \
  --max-input-tokens 240000 \
  --max-output-tokens 8 \
  --child-tokens 7500 \
  --child-overlap-tokens 750 \
  --route-audit-only \
  --output-dir benchmark_artifacts' \
  bash adarsh-rlms/jarvis/run.sh submit client-gpu
```

The route audit must record
`mcq_decoder_constraint_version=vllm_structured_choice_abcd_v1`. Then submit a
hybrid smoke for the five sources that produced invalid prompt-only outputs.
The final ID below is the corrected dataset ID (`...35dfe9d`):

```bash
LLM_PROVIDER=openai_compatible \
OPENAI_COMPAT_EXECUTOR_BASE_URL="$EXECUTOR_URL" \
OPENAI_COMPAT_EVALUATOR_BASE_URL="$EVALUATOR_URL" \
WAIT_FOR_ENDPOINTS=1 \
CLIENT_MEM=96G \
CLIENT_CMD='export SEMANTIC_CACHE_SEARCH_MODE=hybrid
export SEMANTIC_CACHE_EMBEDDING_QUERY_INSTRUCTION="Given a multiple-choice question, retrieve chunks containing evidence, demonstrations, mappings, or facts needed to answer it."
export SEMANTIC_CACHE_EMBEDDING_DEVICE=cuda
export SEMANTIC_CACHE_EMBEDDING_DTYPE=auto
export SEMANTIC_CACHE_EMBEDDING_BATCH_SIZE=2
export SEMANTIC_CACHE_EMBEDDING_MAX_LENGTH=8192
export OPENAI_COMPAT_EXECUTOR_EXTRA_BODY_JSON="{\"chat_template_kwargs\":{\"enable_thinking\":false}}"
export OPENAI_COMPAT_EVALUATOR_EXTRA_BODY_JSON="{\"chat_template_kwargs\":{\"enable_thinking\":false}}"

uv run python long_bench_v2/run_benchmark.py \
  --llm-provider openai_compatible \
  --mode cache \
  --cache-reset \
  --cache-state-root "$JARVIS_CACHE_STATE_ROOT" \
  --executor-model Qwen/Qwen3.6-35B-A3B \
  --evaluator-model Qwen/Qwen3.5-35B-A3B \
  --row-types original \
  --source-ids 66fcffd9bb02136c067c94c5,6724631ebb02136c067d7300,66eb873c5a08c7b9b35dd849,6708a096bb02136c067d1789,66ebd0825a08c7b9b35dfe9d \
  --context-window-tokens 262144 \
  --max-input-tokens 240000 \
  --max-output-tokens 8 \
  --child-tokens 7500 \
  --child-overlap-tokens 750 \
  --output-dir benchmark_artifacts \
  --manifest-note jarvis-l40s-fast-hybrid-decoder-v1-smoke' \
  bash adarsh-rlms/jarvis/run.sh submit client-gpu
```

Accept this smoke only when:

- All five rows have `valid_choice=true`; accuracy is not a smoke criterion.
- Every non-cache row has `final_rendered_input_tokens <= 240000`.
- `api_error_count`, `context_length_error_count`, and `invalid_choice_count`
  are zero.
- The manifest, `hybrid_policy`, and every bridge row record
  `vllm_structured_choice_abcd_v1` with ordered choices A-D.
- Manifest call totals reconcile with bridge-row executor and semantic-verifier
  totals.

Before the gate, run the five-source direct smoke at the start of Section 4. Once
both five-source smokes pass, rerun the same 50-source hybrid gate in isolation,
without another client sharing the executor. Use the existing 50-source list
and a fresh cache namespace created by the decoder-versioned `hybrid_policy`:

```bash
  --source-ids 66f4cd2c821e116aacb316ef,66efc5e3821e116aacb23df1,66f50109821e116aacb31f16,66f40b9c821e116aacb30a99,6725d977bb02136c067d8373,6703f73cbb02136c067cd74a,670765abbb02136c067d06b4,66ec4370821e116aacb1c905,6725db01bb02136c067d847f,66ec1eb9821e116aacb1af36,66fb77e7bb02136c067c7db1,66f2d553821e116aacb2bc8f,6708a096bb02136c067d1789,66ecf139821e116aacb1e0e1,66fa50acbb02136c067c6827,66ebd3ba5a08c7b9b35e0446,66ece545821e116aacb1dd77,66f9625fbb02136c067c5456,66ed2c87821e116aacb1f149,672494e5bb02136c067d7697,67039cfabb02136c067cd04e,6724c83fbb02136c067d7962,6728586bbb02136c067d8f4f,6719bc01bb02136c067d43fa,671b3d1bbb02136c067d5283,671b170cbb02136c067d4f4a,67192057bb02136c067d41b4,6719dc46bb02136c067d470b,66f37eb9821e116aacb2d295,66fb6d71bb02136c067c7c34,66f2ad2b821e116aacb2ac0f,66eb873c5a08c7b9b35dd849,66f97dc3bb02136c067c56c8,66f91f2cbb02136c067c4b1e,66ebd0825a08c7b9b35dfe9d,66fa788abb02136c067c6d75,66faa0f5bb02136c067c722c,66f3df1e821e116aacb2f7be,67041f08bb02136c067cdb52,66fcffd9bb02136c067c94c5,66f40e44821e116aacb30b45,672861afbb02136c067d90d8,66f2a414821e116aacb2a3af,66f2a59d821e116aacb2a553,6724cae7bb02136c067d79be,66f55828821e116aacb3363e,66ec17e4821e116aacb1a6a7,6724631ebb02136c067d7300,66f954a5bb02136c067c511a,66ebd34d5a08c7b9b35e035d \
```

Repeat the hybrid command above with that list and manifest note
`jarvis-l40s-fast-hybrid-decoder-v1-gate-50`. Accept only 50/50 valid choices,
at least 24/50 correct versus the historical iterative result's 25/50, zero
API/context errors, 50 compact cache writes, and at least 2x isolated miss-path
speedup. If it remains below 24/50, inspect the remaining A-D disagreements
before changing retrieval. After it passes, continue with the full direct run
in Section 4, then the sampled and full hybrid suites in Sections 5 and 6.

## 4. Run The Direct Qwen3.6 Ablation

This baseline sends each original LongBench-v2 example to the same Qwen3.6
executor in one chat request. It keeps the strict MCQ prompt, temperature `0`,
eight-token output cap, and disabled thinking from the system run, while
bypassing retrieval, embeddings, reranking, cache state, and multi-call
execution. It uses only the 503 `original` rows; `exact` and `semantic` are
cache-behavior fixtures rather than additional official benchmark questions.

Run the same five-source decoder smoke through the direct path. Point the
evaluator URL at the executor too so `run_client.sh` waits for only that one
service:

```bash
OPENAI_COMPAT_EXECUTOR_BASE_URL="$EXECUTOR_URL" \
OPENAI_COMPAT_EVALUATOR_BASE_URL="$EXECUTOR_URL" \
WAIT_FOR_ENDPOINTS=1 \
CLIENT_MEM=32G \
CLIENT_CMD='uv run python long_bench_v2/run_api_benchmark.py \
  --suite-csv benchmark_data/long_bench_v2/data_cache_suite.csv \
  --source-json-path benchmark_data/long_bench_v2/data.json \
  --row-types original \
  --source-ids 66fcffd9bb02136c067c94c5,6724631ebb02136c067d7300,66eb873c5a08c7b9b35dd849,6708a096bb02136c067d1789,66ebd0825a08c7b9b35dfe9d \
  --api-provider openai_compatible \
  --api-base-url "$OPENAI_COMPAT_EXECUTOR_BASE_URL" \
  --api-model Qwen/Qwen3.6-35B-A3B \
  --context-window-tokens 262144 \
  --max-input-tokens 240000 \
  --max-output-tokens 8 \
  --output-dir benchmark_artifacts \
  --manifest-note jarvis-qwen36-direct-decoder-v1-smoke' \
  bash adarsh-rlms/jarvis/run.sh submit client
```

After the client job syncs its artifacts back, validate the newest smoke run:

```bash
SMOKE_RUN_DIR=$(ls -dt adarsh-rlms/benchmark_artifacts/longbench_v2_api/* | head -1)
uv run --project adarsh-rlms python - "$SMOKE_RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
manifest = json.loads((run_dir / "manifest.json").read_text())
rows = [json.loads(line) for line in (run_dir / "bridge_rows.jsonl").read_text().splitlines()]
assert manifest["rows_selected"] == len(rows) == 5
assert manifest["context_window_tokens"] == 262144
assert manifest["max_input_tokens"] == 240000
assert manifest["context_window_safety_margin_tokens"] == 22136
assert manifest["api_error_count"] == 0
assert manifest["valid_choice_count"] == 5
assert manifest["invalid_choice_count"] == 0
assert manifest["mcq_decoder_constraint_version"] == "vllm_structured_choice_abcd_v1"
assert manifest["mcq_allowed_choices"] == ["A", "B", "C", "D"]
assert all(row["api_status"] == "ok" for row in rows)
assert all(row["valid_choice"] for row in rows)
assert all(row["mcq_decoder_constraint_version"] == "vllm_structured_choice_abcd_v1" for row in rows)
assert all(row["prompt_tokens_before_truncation"] not in {0, 2} for row in rows)
assert all(row["prompt_tokens_after_truncation"] <= 240000 for row in rows)
print(f"validated direct decoder smoke: {run_dir}")
PY
```

Accuracy is not a five-row smoke criterion. After the 50-source hybrid gate and
this direct smoke pass, run all 503 original rows:

```bash
OPENAI_COMPAT_EXECUTOR_BASE_URL="$EXECUTOR_URL" \
OPENAI_COMPAT_EVALUATOR_BASE_URL="$EXECUTOR_URL" \
WAIT_FOR_ENDPOINTS=1 \
CLIENT_MEM=32G \
CLIENT_CMD='uv run python long_bench_v2/run_api_benchmark.py \
  --suite-csv benchmark_data/long_bench_v2/data_cache_suite.csv \
  --source-json-path benchmark_data/long_bench_v2/data.json \
  --row-types original \
  --api-provider openai_compatible \
  --api-base-url "$OPENAI_COMPAT_EXECUTOR_BASE_URL" \
  --api-model Qwen/Qwen3.6-35B-A3B \
  --context-window-tokens 262144 \
  --max-input-tokens 240000 \
  --max-output-tokens 8 \
  --output-dir benchmark_artifacts \
  --manifest-note jarvis-qwen36-direct-decoder-v1-full-original' \
  bash adarsh-rlms/jarvis/run.sh submit client
```

The runner writes a new timestamped directory under
`benchmark_artifacts/longbench_v2_api/` and refuses to reuse an existing run
directory. Inputs above 240,000 rendered tokens are tokenized with Qwen's chat
template and truncated from the middle, retaining the beginning and end of the
user prompt while preserving the strict system message. Every request carries
the mandatory choice decoder contract. The 240,000-token input cap plus the
eight-token output cap leaves 22,136 tokens of safety inside the 262,144-token
server window. For the full artifact, require 503 rows, 503 successful API
responses, 503 valid choices, zero context errors, and the expected 107
truncated rows. Check `truncated_row_count`, `valid_choice_count`,
`invalid_choice_count`, `api_error_count`, `total_request_attempts`, and
`answer_accuracy` in
`manifest.json` before comparing the result with the saved system run's 503-row
`original` accuracy.

After the full client job syncs back, run the corresponding full-artifact
checks before using its accuracy in a comparison note:

```bash
FULL_RUN_DIR=$(ls -dt adarsh-rlms/benchmark_artifacts/longbench_v2_api/* | head -1)
uv run --project adarsh-rlms python - "$FULL_RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
manifest = json.loads((run_dir / "manifest.json").read_text())
rows = [json.loads(line) for line in (run_dir / "bridge_rows.jsonl").read_text().splitlines()]
assert manifest["rows_selected"] == len(rows) == 503
assert manifest["total_api_calls"] == 503
assert manifest["context_window_tokens"] == 262144
assert manifest["max_input_tokens"] == 240000
assert manifest["context_window_safety_margin_tokens"] == 22136
assert manifest["api_error_count"] == 0
assert manifest["truncated_row_count"] == sum(row["prompt_truncated"] for row in rows) == 107
assert manifest["valid_choice_count"] == 503
assert manifest["invalid_choice_count"] == 0
assert manifest["mcq_decoder_constraint_version"] == "vllm_structured_choice_abcd_v1"
assert manifest["mcq_allowed_choices"] == ["A", "B", "C", "D"]
assert all(row["api_status"] == "ok" for row in rows)
assert all(row["valid_choice"] for row in rows)
assert all(row["mcq_decoder_constraint_version"] == "vllm_structured_choice_abcd_v1" for row in rows)
assert all(row["prompt_tokens_before_truncation"] not in {0, 2} for row in rows)
assert all(row["prompt_tokens_after_truncation"] <= 240000 for row in rows)
print(f"validated full direct baseline: {run_dir}")
PY
```

## 5. Validate A Source-Linked Hybrid Sample

Before the full hybrid suite, use the existing source-linked sampler to select
two reproducible source groups from each eligible LongBench domain. The six
domains produce 12 source groups and 36 rows: 12 `original`, 12 `exact`, and 12
`semantic`. The explicit `source_grouped` order keeps each original ahead of
its dependent cache rows.

```bash
LLM_PROVIDER=openai_compatible \
OPENAI_COMPAT_EXECUTOR_BASE_URL="$EXECUTOR_URL" \
OPENAI_COMPAT_EVALUATOR_BASE_URL="$EVALUATOR_URL" \
WAIT_FOR_ENDPOINTS=1 \
CLIENT_MEM=96G \
CLIENT_CMD='export SEMANTIC_CACHE_SEARCH_MODE=hybrid
export SEMANTIC_CACHE_EMBEDDING_QUERY_INSTRUCTION="Given a multiple-choice question, retrieve chunks containing evidence, demonstrations, mappings, or facts needed to answer it."
export SEMANTIC_CACHE_EMBEDDING_DEVICE=cuda
export SEMANTIC_CACHE_EMBEDDING_DTYPE=auto
export SEMANTIC_CACHE_EMBEDDING_BATCH_SIZE=2
export SEMANTIC_CACHE_EMBEDDING_MAX_LENGTH=8192
export OPENAI_COMPAT_EXECUTOR_EXTRA_BODY_JSON="{\"chat_template_kwargs\":{\"enable_thinking\":false}}"
export OPENAI_COMPAT_EVALUATOR_EXTRA_BODY_JSON="{\"chat_template_kwargs\":{\"enable_thinking\":false}}"

uv run python long_bench_v2/sample_csv.py \
  --input-path benchmark_data/long_bench_v2/data_cache_suite.csv \
  --output-path benchmark_artifacts/longbench_v2_samples/jarvis_hybrid_decoder_v1_pre_full.csv \
  --samples-per-domain 2 \
  --row-types original,exact,semantic \
  --selection-strategy random \
  --seed 20260809 && \
uv run python long_bench_v2/run_benchmark.py \
  --suite-csv benchmark_artifacts/longbench_v2_samples/jarvis_hybrid_decoder_v1_pre_full.csv \
  --source-json-path benchmark_data/long_bench_v2/data.json \
  --llm-provider openai_compatible \
  --mode cache \
  --cache-reset \
  --cache-state-root "$JARVIS_CACHE_STATE_ROOT" \
  --executor-model Qwen/Qwen3.6-35B-A3B \
  --evaluator-model Qwen/Qwen3.5-35B-A3B \
  --row-types original,exact,semantic \
  --row-order source_grouped \
  --context-window-tokens 262144 \
  --max-input-tokens 240000 \
  --max-output-tokens 8 \
  --child-tokens 7500 \
  --child-overlap-tokens 750 \
  --output-dir benchmark_artifacts \
  --manifest-note jarvis-l40s-fast-hybrid-decoder-v1-pre-full-sample' \
  bash adarsh-rlms/jarvis/run.sh submit client-gpu
```

After the artifact syncs back, validate the stratified 36-row route pattern:

```bash
SAMPLE_RUN_DIR=$(ls -dt adarsh-rlms/benchmark_artifacts/longbench_v2/* | head -1)
uv run --project adarsh-rlms python - "$SAMPLE_RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
manifest = json.loads((run_dir / "manifest.json").read_text())
rows = [json.loads(line) for line in (run_dir / "bridge_rows.jsonl").read_text().splitlines()]
routes = manifest["hybrid_route_counts"]
original_route_count = routes.get("direct_fit", 0) + routes.get("dense_child_packed", 0)

assert manifest["rows_selected"] == len(rows) == 36
assert manifest["row_type_counts"] == {"exact": 12, "original": 12, "semantic": 12}
assert original_route_count == 12
assert routes.get("exact_cache", 0) == 12
assert routes.get("semantic_cache", 0) == 12
assert manifest["executor_answer_calls"] == 12
assert manifest["semantic_verifier_calls"] == 12
assert sum(bool(row["compact_cache_write"]) for row in rows) == 12
assert manifest["valid_choice_count"] == 36
assert manifest["invalid_choice_count"] == 0
assert manifest["api_error_count"] == 0
assert manifest["context_length_error_count"] == 0
assert manifest["mcq_decoder_constraint_version"] == "vllm_structured_choice_abcd_v1"
assert all(row["api_status"] == "ok" for row in rows)
print(f"validated source-linked hybrid sample: {run_dir}")
PY
```

Treat a missing exact or semantic hit as a cache-routing failure to inspect
before the full run. Keep the same embedding batch size in this sample and the
full command so their operational profiles remain comparable.

## 6. Run The Full Hybrid Suite

Only after the constrained direct artifact and the source-linked hybrid sample
pass, run the full 1,509-row hybrid suite in the decoder-versioned namespace:

```bash
LLM_PROVIDER=openai_compatible \
OPENAI_COMPAT_EXECUTOR_BASE_URL="$EXECUTOR_URL" \
OPENAI_COMPAT_EVALUATOR_BASE_URL="$EVALUATOR_URL" \
WAIT_FOR_ENDPOINTS=1 \
CLIENT_MEM=96G \
CLIENT_CMD='export SEMANTIC_CACHE_SEARCH_MODE=hybrid
export SEMANTIC_CACHE_EMBEDDING_QUERY_INSTRUCTION="Given a multiple-choice question, retrieve chunks containing evidence, demonstrations, mappings, or facts needed to answer it."
export SEMANTIC_CACHE_EMBEDDING_DEVICE=cuda
export SEMANTIC_CACHE_EMBEDDING_DTYPE=auto
export SEMANTIC_CACHE_EMBEDDING_BATCH_SIZE=2
export SEMANTIC_CACHE_EMBEDDING_MAX_LENGTH=8192
export OPENAI_COMPAT_EXECUTOR_EXTRA_BODY_JSON="{\"chat_template_kwargs\":{\"enable_thinking\":false}}"
export OPENAI_COMPAT_EVALUATOR_EXTRA_BODY_JSON="{\"chat_template_kwargs\":{\"enable_thinking\":false}}"

uv run python long_bench_v2/run_benchmark.py \
  --llm-provider openai_compatible \
  --mode cache \
  --cache-reset \
  --cache-state-root "$JARVIS_CACHE_STATE_ROOT" \
  --executor-model Qwen/Qwen3.6-35B-A3B \
  --evaluator-model Qwen/Qwen3.5-35B-A3B \
  --row-types original,exact,semantic \
  --context-window-tokens 262144 \
  --max-input-tokens 240000 \
  --max-output-tokens 8 \
  --child-tokens 7500 \
  --child-overlap-tokens 750 \
  --output-dir benchmark_artifacts \
  --manifest-note jarvis-l40s-fast-hybrid-decoder-v1-full' \
  bash adarsh-rlms/jarvis/run.sh submit client-gpu
```

The shared LongBench request helper injects the choice constraint; do not add
`guided_choice` or disable the contract through an extra-body environment
variable. Validate 1,509 rows, the exact/semantic/direct/packed route totals,
zero API/context errors, all valid choices, and the decoder contract in the
manifest, `hybrid_policy`, and bridge rows before producing a comparison note.

## After The Experiments

Use Sections 11-13 of
[`HPC_RUNBOOK_SETUP.md`](HPC_RUNBOOK_SETUP.md) to stop services, clean
node-local scratch, and diagnose common failures.
