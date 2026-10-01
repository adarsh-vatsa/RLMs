# Jarvis LongBench-v2 runbook

Jarvis runs LongBench-v2 on the shared direct/hybrid execution pipeline. AA-LCR
and MRCR v2 run on the Neselab Linux server instead; see the
[Linux runbooks](../linux/README.md).

Complete the [setup runbook](HPC_RUNBOOK_SETUP.md) first. Run these commands on
the Jarvis login node from the parent directory that contains the
`adarsh-rlms` repository.

## Experiment design

LongBench-v2 has 503 multiple-choice questions, each on its own document. Full
prompts range from about 10,000 to 4.5 million executor tokens (median about
103,000). Both modes use the executor's full served window: 262,144 tokens, of
which 8 are reserved for the answer letter, leaving a 262,136-token input
budget. At that budget, 403 questions fit and 100 do not (67 between 262K and 1M
tokens, 33 above 1M).

- **Direct** runs all 503 questions. Prompts that fit are sent whole; the 100
  oversized prompts keep an equal head and tail (`--direct-overflow middle`).
- **Hybrid** runs only the 100 oversized questions. On a prompt that fits, hybrid
  sends exactly the direct request, so running those again adds nothing. It
  splits each document into 7,500-token chunks with 750 tokens of overlap, ranks
  them against the question, fills the budget with the best-ranked chunks and
  presents them in document order. Embeddings run on the GPU of a `client-gpu`
  job.
- Answers are constrained to A–D and scored by letter, so no evaluator service is
  needed. Answer caching is off, and only `original` rows are used. The optional
  [answer-cache experiment](#7-optional-answer-cache-experiment) adds repeated and
  reworded questions with caching on.

The August Jarvis runs, made before the shared pipeline existed, scored 56.1% for
hybrid against 41.1% for direct on 107 oversized questions at a 240,000-token
budget. These runs repeat that comparison on the current pipeline.

## 1. Prepare the data

`benchmark_data/long_bench_v2/data.csv` is in the repository. It references the
full documents in `data.json`, which is excluded from Git. Copy `data.json` from
another server, or download the pinned revision (about 465 MB):

```bash
cd adarsh-rlms
mkdir -p benchmark_data/long_bench_v2
if [ ! -f benchmark_data/long_bench_v2/data.json ]; then
  curl -fL --retry 3 \
    'https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json' \
    -o benchmark_data/long_bench_v2/data.json.download &&
  mv benchmark_data/long_bench_v2/data.json.download benchmark_data/long_bench_v2/data.json
fi
cd ..
```

Regenerate the CSV only if the JSON changes:
`uv run python -m long_bench_v2.export_csv --input-path benchmark_data/long_bench_v2/data.json --output-path benchmark_data/long_bench_v2/data.csv`
from the repository directory.

## 2. Start the executor

Only the executor service is needed. Start it with the full Qwen3.6 window:

```bash
MODULES="cuda12.8/toolkit/12.8.1" \
EXECUTOR_MODEL=Qwen/Qwen3.6-35B-A3B \
EXECUTOR_TP_SIZE=4 \
EXECUTOR_MAX_MODEL_LEN=262144 \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
VLLM_EXTRA_ARGS="--reasoning-parser qwen3 --language-model-only --max-num-seqs 1 --enable-chunked-prefill --max-num-batched-tokens 8192" \
SYNC_BACK_MODELS=1 VLLM_VENV=/home/edogu/.venvs/adarsh-vllm \
  bash adarsh-rlms/jarvis/run.sh submit executor
```

Watch its log (`tail -f "$PROJECT_LOG_DIR"/rlms-executor-<executor_job_id>.out`)
until it shows:

```text
Maximum concurrency for 262,144 tokens per request: X.XXx
Application startup complete.
```

Require `X.XX` to be at least `1.00`, and reject a startup with a CUDA
out-of-memory or engine-initialization error. Qwen's hybrid attention makes the
displayed KV cache token count misleading, so use this line as the capacity
check. Then read its endpoint:

```bash
EXECUTOR_URL=$(cat "$PROJECT_LOG_DIR"/executor-<executor_job_id>.url)
export OPENAI_COMPAT_EXECUTOR_BASE_URL="$EXECUTOR_URL"
```

## 3. Record the serving setup

Each run copies this file into its manifest, so results from Jarvis and from the
Neselab server stay attributable. Record it once per executor job:

```bash
mkdir -p adarsh-rlms/.cache
VLLM_VENV=/home/edogu/.venvs/adarsh-vllm EXECUTOR_TP_SIZE=4 \
adarsh-rlms/.venv/bin/python - "$EXECUTOR_URL" <<'EOF'
import datetime, json, os, subprocess, sys, urllib.request

base = sys.argv[1].rstrip("/")
with urllib.request.urlopen(base + "/models", timeout=10) as response:
    served = json.load(response)["data"]
vllm_python = os.path.join(os.environ["VLLM_VENV"], "bin", "python")
vllm = subprocess.run([vllm_python, "-c", "import importlib.metadata as m; print(m.version('vllm'))"],
                      capture_output=True, text=True, check=True).stdout.strip()
tensor_parallel = int(os.environ.get("EXECUTOR_TP_SIZE", "4"))
metadata = {
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "cluster": "Jarvis",
    "executor_base_url": base,
    "gpus": f"{tensor_parallel} x NVIDIA L40S (gpu-l40s partition)",
    "tensor_parallel_size": tensor_parallel,
    "vllm_version": vllm,
    "served_models": [{"id": m["id"], "max_model_len": m.get("max_model_len")} for m in served],
}
with open("adarsh-rlms/.cache/serving_metadata.json", "w") as handle:
    handle.write(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
EOF
export LONGBENCH_SERVING_METADATA=.cache/serving_metadata.json
```

Client jobs start in the repository directory, so the exported path is relative
to it. Check that `served_models` shows `max_model_len` 262144.

## 4. Preflight

Preflight counts tokens and reports each question's route without inference.
It submits a CPU client job and does not wait for the executor.

```bash
bash adarsh-rlms/jarvis/run_longbench_v2.sh direct --direct-overflow middle --preflight-only

bash adarsh-rlms/jarvis/run_longbench_v2.sh hybrid \
  --min-source-tokens 262137 --max-source-tokens 5000000 --route-audit-only
```

Each job writes a `route_audit.json`: direct under
`benchmark_artifacts/longbench_v2_api/<run id>/`, hybrid under
`benchmark_artifacts/longbench_v2_route_audit/<run id>/`. Count the routes in
each, from the repository directory:

```bash
.venv/bin/python -c 'import collections, json, sys; audit = json.load(open(sys.argv[1])); rows = audit["rows"] if isinstance(audit, dict) else audit; print(collections.Counter(row["route"] for row in rows))' <path to route_audit.json>
```

Direct should show 403 `direct_fit` and 100 `middle_truncated`; hybrid should
show 100 `dense_child_packed`.

## 5. Run

For a smoke test, add `--max-rows 3` to either command; with source bounds, the
limit applies after the length filter.

```bash
bash adarsh-rlms/jarvis/run_longbench_v2.sh direct --direct-overflow middle --fail-fast
```

After the direct job finishes:

```bash
bash adarsh-rlms/jarvis/run_longbench_v2.sh hybrid \
  --min-source-tokens 262137 --max-source-tokens 5000000 --fail-fast
```

Run the two jobs one after the other. Requests that reach one vLLM service at
the same time are batched together, which can change temperature-zero outputs.
`--fail-fast` stops a run on its first failed request instead of recording the
failure as a wrong answer.

The direct job uses a CPU allocation with 32 GB of memory and the hybrid job a
GPU allocation with 96 GB; set `CLIENT_MEM` to change either. If the hybrid job
runs out of GPU memory while embedding, lower `SEMANTIC_CACHE_EMBEDDING_BATCH_SIZE`
(default 16) in the submitting shell. Direct sends about 62 million input tokens
in total; hybrid embeds about 99 million tokens of documents, which dominates
its run time. Neither has been timed on Jarvis yet.

## 6. Compare the results

Direct runs are written under `benchmark_artifacts/longbench_v2_api/<run id>/`
and hybrid runs under `benchmark_artifacts/longbench_v2/<run id>/`, each with
`manifest.json`, `bridge_rows.jsonl` and an evaluation report. Compare the two
modes on the oversized questions, where they differ:

```bash
cd adarsh-rlms
LONGBENCH_HYBRID_RUN=benchmark_artifacts/longbench_v2/<hybrid run id> \
LONGBENCH_DIRECT_RUN=benchmark_artifacts/longbench_v2_api/<direct run id> \
.venv/bin/python - "$LONGBENCH_HYBRID_RUN" "$LONGBENCH_DIRECT_RUN" <<'EOF'
import json, sys
from math import comb

def rows(run):
    return {r["case_id"]: r for r in map(json.loads, open(f"{run}/bridge_rows.jsonl"))
            if r.get("row_type", "original") == "original"}

def correct(row):
    return str(row["answer_correct"]).lower() in ("true", "1")

hybrid, direct = rows(sys.argv[1]), rows(sys.argv[2])
cases = [c for c, r in hybrid.items() if r.get("hybrid_route") == "dense_child_packed"]
h = sum(correct(hybrid[c]) for c in cases)
d = sum(correct(direct[c]) for c in cases)
gained = sum(correct(hybrid[c]) and not correct(direct[c]) for c in cases)
lost = sum(correct(direct[c]) and not correct(hybrid[c]) for c in cases)
n, k = gained + lost, min(gained, lost)
p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
print(f"Oversized questions: {len(cases)}")
print(f"  hybrid {h}/{len(cases)} ({h / len(cases):.1%}), direct {d}/{len(cases)} ({d / len(cases):.1%})")
print(f"  gained {gained}, lost {lost}, paired exact binomial p = {p:.4f}")
print(f"Direct on all {len(direct)} questions: {sum(map(correct, direct.values())) / len(direct):.1%}")
EOF
cd ..
```

The p-value comes from an exact binomial test on the questions where the two
modes disagree. On the August runs (hybrid `20260812T014012Z`, direct
`20260812T152758Z`) this prints 107 oversized questions, 56.1% against 41.1%, 21
gained, 5 lost and p = 0.0025.

## 7. Optional: answer-cache experiment

This tests whether answers can be reused for repeated questions.
`data_cache_suite.csv` holds each of the 503 questions three times: the
`original`, an `exact` repeat and a reworded `semantic` version. Originals are
answered as in the hybrid run and their answers stored; the repeats should be
answered from the cache without calling the executor. An answer is reused only
for a question about the same document, and rows are grouped by document so each
original runs before its repeats.

The suite is in the repository and reads documents from the same `data.json`, so
it needs no preparation. Its reworded questions were generated once with
`long_bench_v2/generate_semantic_questions.ts` and merged with
`long_bench_v2/combine_csv.py`; both are described in the
[LongBench-v2 notes](../../long_bench_v2/docs/longbench_v2.md).

A reworded question is matched by embedding similarity and then confirmed by a
verifier model. These commands use the executor as the verifier, so no second
service is needed. To use a different model, start it with `submit evaluator`
and pass its model and endpoint instead.

Smoke-test on the two shortest documents (six rows) first:

```bash
bash adarsh-rlms/jarvis/run_longbench_v2.sh hybrid \
  --suite-csv benchmark_data/long_bench_v2/data_cache_suite.csv \
  --row-types original,exact,semantic \
  --answer-cache-read --answer-cache-write --cache-reset \
  --cache-verifier-model Qwen/Qwen3.6-35B-A3B \
  --cache-verifier-base-url "$OPENAI_COMPAT_EXECUTOR_BASE_URL" \
  --source-ids 66f37eb9821e116aacb2d295,66f245cc821e116aacb28698 --fail-fast
```

Then run all 1,509 rows with the same command without `--source-ids`. The
originals cost about as much as the direct and hybrid runs together, because the
100 oversized documents are embedded again; the 1,006 repeats add a cache lookup
each, plus an executor call on a miss. To skip the document embedding, limit the
run to the documents that fit with
`--min-source-tokens 1 --max-source-tokens 262136`.

The cache is saved under
`benchmark_artifacts/longbench_v2/cache_state/` and reloaded by a later run with
the same settings; `--cache-reset` starts each run with an empty cache. Do not
run this alongside the other runs, which share the executor.

Each row in `bridge_rows.jsonl` records `from_cache` and `cache_type` next to
`expected_cache_type`. Summarize by row type from the repository directory:

```bash
.venv/bin/python -c 'import json, sys; print(json.dumps(json.load(open(sys.argv[1]))["by_row_type"], indent=2))' \
  benchmark_artifacts/longbench_v2/<run id>/manifest.json
```

`exact` rows should be answered from the cache almost every time. The
`semantic` hit rate shows how often reworded questions are recognized. On a hit,
the row gets the original's answer, so compare each row type's accuracy with the
originals': a lower accuracy on hits means answers were reused for questions
that differ.

## 8. Stop the executor

Service jobs keep running until cancelled:

```bash
scancel <executor_job_id>
```

## Earlier experiments

Earlier procedures on this page covered the pre-unification runners: iterative
reading, the legacy-profile answer-cache runs, and Llama, Mistral and smaller
smoke-test profiles. They are no longer maintained;
their artifacts remain under `benchmark_artifacts/longbench_v2*/`, and the
previous version of this page is in Git history
(`git log -- docs/jarvis/HPC_RUNBOOK_EXPERIMENT.md`).
