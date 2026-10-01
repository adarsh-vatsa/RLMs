# Linux benchmark runbook

Complete the [Linux setup](SETUP_RUNBOOK.md) first. Run these commands from the
repository root. Preflight validates inputs and budgets; execution runs inference.
All examples use the shared `common` execution profile. LongBench-v2 was run on
Jarvis, not on the Linux server; see the [Jarvis runbooks](../jarvis/README.md).
Its results are in `benchmark_artifacts/longbench_v2` and
`benchmark_artifacts/longbench_v2_api`.
For available flags, defaults, and cache settings, consult the optional
[runner options reference](SHARED_EXECUTION_RUNBOOK.md).

## Prepare data

These are existing preparation commands; preparation does not need model servers.
The tokenizer and requested dataset files may download on first use.
Prepare only the benchmarks you intend to run.

### AA-LCR

```bash
.venv/bin/python -m aa_lcr.prepare_dataset --dataset-version 1.1
```

### MRCR v2

One dataset covers sources from about 100K to 1.2M executor tokens: four
released files, each holding one conversation of about 133K, 267K, 534K and
1.07M tokens. See
[released files](../mrcr_v2.md#released-files-from-100k-to-12m-tokens) for the
per-file counts.

The release comes with 2, 4 or 8 needles per question group. Fewer needles
means fewer identical requests for the model to count, so its own ceiling is
higher and retrieval differences show more clearly. The 29 September runs used
8 needles, prepared in `benchmark_data/mrcr_v2_100k_1200k`. Set the count once;
the dataset directory follows it:

```bash
export MRCR_NEEDLES=4
export MRCR_DATA_DIR=benchmark_data/mrcr_v2_${MRCR_NEEDLES}needle_100k_1200k
.venv/bin/python -m mrcr_v2.prepare_dataset \
  --executor-model "${OPENAI_COMPAT_EXECUTOR_MODEL:-Qwen/Qwen3.6-35B-A3B}" \
  --download-bands 65536:131072,131072:262144,262144:524288,524288:1048576 \
  --needles "$MRCR_NEEDLES" --min-source-tokens 100000 --max-source-tokens 1200000 \
  --data-dir "$MRCR_DATA_DIR"
```

This downloads about 2.6 GB for 2 or 4 needles (2.3 GB for 8). Expect
preparation, not inference, to dominate the runtime: it counts tokens for every
question's full prompt, roughly 510 million tokens for the 8-needle files,
regardless of how many questions you later run. A file whose conversation falls
outside the bounds contributes no rows without raising an error, so list each
prepared conversation before running:

```bash
.venv/bin/python - "$MRCR_DATA_DIR" <<'EOF'
import collections, json, sys
from pathlib import Path

lengths = collections.defaultdict(list)
for line in (Path(sys.argv[1]) / "questions.jsonl").open():
    row = json.loads(line)
    lengths[row["source_id"]].append(row["full_rendered_input_tokens"])
for source_id, tokens in sorted(lengths.items(), key=lambda item: min(item[1])):
    band = 1 << (min(tokens).bit_length() - 1)
    print(f"{source_id[:12]}  {min(tokens):>9,}-{max(tokens):,} tokens  "
          f"{len(tokens):>4} questions  report band [{band:,}, {2 * band:,})")
EOF
```

Expect four lines, one per conversation, each in a different report band. The
[released files](../mrcr_v2.md#released-files-from-100k-to-12m-tokens) table
lists the 8-needle values; question counts differ for other needle counts.

MRCR preparation is a one-time step for each output directory. If it reports
`FileExistsError`, check that directory before retrying. A completed preparation
writes `dataset_manifest.json` last; if present, proceed to MRCR preflight to
validate and reuse the dataset. If it is absent and no preparation is still
running, the directory is incomplete. Retry preparation with a new `--data-dir`
and use that same path for preflight/execution. Do not delete a completed dataset
just to rerun preparation.

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

AA-LCR defaults to the prepared v1.1 directory; set `AA_LCR_DATA_DIR` to use
another copy of it.

```bash
bash linux/run_benchmark.sh aa_lcr hybrid --execution-only \
  --context-window-tokens 65536 --max-input-tokens 61440 \
  --max-output-tokens 4096 --preflight-only

bash linux/run_benchmark.sh aa_lcr direct --execution-only \
  --context-window-tokens 65536 --max-input-tokens 61440 \
  --max-output-tokens 4096 --direct-overflow middle --preflight-only
```

Every question should report `dense_child_packed` for hybrid and
`middle_truncated` for direct.

### MRCR v2

```bash
export MRCR_NEEDLES=4
export MRCR_DATA_DIR=benchmark_data/mrcr_v2_${MRCR_NEEDLES}needle_100k_1200k
export MRCR_ROWS_PER_SOURCE=10
bash linux/run_benchmark.sh mrcr_v2 hybrid \
  --context-window-tokens 262144 --max-input-tokens 258048 \
  --max-output-tokens 4096 --max-rows-per-source "$MRCR_ROWS_PER_SOURCE" \
  --preflight-only

bash linux/run_benchmark.sh mrcr_v2 direct \
  --context-window-tokens 262144 --max-input-tokens 258048 \
  --max-output-tokens 4096 --max-rows-per-source "$MRCR_ROWS_PER_SOURCE" \
  --direct-overflow middle --preflight-only
```

Preflight reports each question's route. The 133K conversation fits the
258,048-token input budget and reports `direct_fit` for both modes. The larger
ones report `dense_child_packed` for hybrid and `middle_truncated` for direct,
or `unsupported_context` if `--direct-overflow middle` is omitted.

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

AA-LCR prompts are 87,847–122,112 executor tokens (median 107,136), so the
full served context would fit every one and hybrid would send the same request
as direct. The comparison instead uses a 64K model window: 65,536 tokens minus
the 4,096-token output allowance leaves a 61,440-token input budget, which every
question exceeds. Direct keeps a head and a tail slice of each prompt, a median
of 57%. Record the serving setup first, as in the
[setup runbook](SETUP_RUNBOOK.md#record-the-serving-setup); each command copies it
into its run manifest.

```bash
export AA_LCR_RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
SEMANTIC_CACHE_EMBEDDING_DEVICE=cpu bash linux/run_benchmark.sh aa_lcr hybrid \
  --execution-only --context-window-tokens 65536 --max-input-tokens 61440 \
  --max-output-tokens 4096 --run-id "$AA_LCR_RUN_ID" --fail-fast \
  --serving-metadata .cache/serving_metadata.json

bash linux/run_benchmark.sh aa_lcr direct \
  --execution-only --context-window-tokens 65536 --max-input-tokens 61440 \
  --max-output-tokens 4096 --direct-overflow middle --run-id "$AA_LCR_RUN_ID" --fail-fast \
  --serving-metadata .cache/serving_metadata.json
```

Without `--direct-overflow middle`, the common profile records every example as
unsupported. Hybrid embeds each of the 30 document sets once and reuses the
index for that set's questions. On this server with CPU embeddings, the 30
September runs took about 2.75 hours for hybrid and 14 minutes for direct.

With thinking disabled, the model still reasons inside its answer, and 9 of those
200 answers reached the 4,096-token output limit before giving a final value
(4 hybrid, 5 direct). A larger `--max-output-tokens` needs a matching smaller
`--max-input-tokens` (65,536 minus the output allowance), and both modes must use
the same values.

OPTIONAL: For the full-context reference, run direct once at the full served window,
where every prompt fits. A hybrid run there would send identical requests.

```bash
bash linux/run_benchmark.sh aa_lcr direct \
  --execution-only --context-window-tokens 262144 --max-input-tokens 258048 \
  --max-output-tokens 4096 --run-id "${AA_LCR_RUN_ID}_full" --fail-fast \
  --serving-metadata .cache/serving_metadata.json
```

`--execution-only` saves answers without grading them, so generation and
grading stay separate steps and grading can be repeated with any grader. To
grade during execution instead, omit it; the grader is then the executor, as
below. It does not disable semantic cache verification if you explicitly enable
that.

#### Grade the saved answers

By default the grader is the executor model, reached through the running
executor service, so grading needs no second model and no service restart. Both
modes are graded by the same model, which keeps the hybrid–direct comparison
even. Absolute scores need an independent grader, though: in the 30 September
runs, the executor grading its own answers made 3 errors among 30 grades that
were checked by hand. It accepted a cut-off answer with no final value, and
rejected "$901,170 (in thousands)" against "$901 million" and a ranking with
country names spelled out. Regrading writes to a new directory and
never changes the saved answers. The grader prompt applies the official AA-LCR
equivalence rules, so, for example, `0.09`, `9%` and `9 percentage points` match.
Pass the grader model explicitly, since `aa_lcr.regrade` does not read
`OPENAI_COMPAT_EVALUATOR_MODEL`.

Set the two run folders first. They default to the shared run ID; if the runs
got different IDs, as the 30 September runs did, set them explicitly.

```bash
AA_LCR_HYBRID_RUN=${AA_LCR_HYBRID_RUN:-hybrid/$AA_LCR_RUN_ID}   # e.g. hybrid/20260930T025706Z
AA_LCR_DIRECT_RUN=${AA_LCR_DIRECT_RUN:-direct/$AA_LCR_RUN_ID}   # e.g. direct/20260930T134555Z
AA_LCR_DATA_DIR=${AA_LCR_DATA_DIR:-benchmark_data/aa_lcr/v1.1}
AA_LCR_GRADER_MODEL=${OPENAI_COMPAT_EVALUATOR_MODEL:-${OPENAI_COMPAT_EXECUTOR_MODEL:-Qwen/Qwen3.6-35B-A3B}}
AA_LCR_GRADER_URL=${OPENAI_COMPAT_EVALUATOR_BASE_URL:-${OPENAI_COMPAT_EXECUTOR_BASE_URL:-http://127.0.0.1:8000/v1}}
for run in "$AA_LCR_HYBRID_RUN" "$AA_LCR_DIRECT_RUN"; do
  .venv/bin/python -m aa_lcr.regrade \
    --source-run "benchmark_artifacts/aa_lcr/$run" \
    --output-dir "benchmark_artifacts/aa_lcr/regrades/${AA_LCR_GRADER_MODEL##*/}/$run" \
    --questions-csv "$AA_LCR_DATA_DIR/AA-LCR_Dataset.csv" \
    --documents-root "$AA_LCR_DATA_DIR/extracted_text/lcr" \
    --dataset-manifest "$AA_LCR_DATA_DIR/dataset_manifest.json" \
    --evaluator-model "$AA_LCR_GRADER_MODEL" --evaluator-base-url "$AA_LCR_GRADER_URL"
done
```

The grader's model name in the output path keeps regrades by different graders
apart. For an independent grader, set `OPENAI_COMPAT_EVALUATOR_BASE_URL` and
`OPENAI_COMPAT_EVALUATOR_MODEL` as in the
[setup runbook](SETUP_RUNBOOK.md#start-model-services-or-reuse-endpoints) and run
the loop again. A hosted OpenAI-style grader also needs
`--grader-api-style openai` and `--evaluator-api-key-env`; a local grader served
with fewer than 32,768 context tokens needs a matching `--grader-context-window`.
Set these variables again in a new shell. Add `--validate-only` to check that a
run's answers and sources are intact without calling the grader.

#### Compare the two modes

Compare regrades made by the same grader. The report gives the paired accuracy
difference with a 95% interval that resamples whole document sets, the questions
each mode gained and lost, and how many answers hit the output limit. It refuses
to overwrite an existing report.

```bash
AA_LCR_REGRADES=benchmark_artifacts/aa_lcr/regrades/${AA_LCR_GRADER_MODEL##*/}
.venv/bin/python -m aa_lcr.compare_conditions \
  --baseline "$AA_LCR_REGRADES/$AA_LCR_DIRECT_RUN" \
  --candidate "$AA_LCR_REGRADES/$AA_LCR_HYBRID_RUN" \
  --output "benchmark_artifacts/aa_lcr/comparisons/${AA_LCR_GRADER_MODEL##*/}/${AA_LCR_HYBRID_RUN#*/}.json"
```

For the 30 September runs graded by Qwen3.6, this gives +17 points for hybrid
(37% to 54%; 95% interval +8.4 to +25.9), with 23 questions gained and 6 lost.

### MRCR v2

Check the served context first with
`curl -s "$OPENAI_COMPAT_EXECUTOR_BASE_URL/models"`; the commands below assume
the 262,144-token default. Run each mode once over the whole dataset. The
prepared bounds apply by default; add `--min-source-tokens` and
`--max-source-tokens` to narrow a run to some of the conversations.

`--max-rows-per-source` takes that many evenly spaced questions from each
conversation, spreading them across needle groups; `0` runs all of them.

```bash
export MRCR_NEEDLES=4
export MRCR_DATA_DIR=benchmark_data/mrcr_v2_${MRCR_NEEDLES}needle_100k_1200k
export MRCR_ROWS_PER_SOURCE=30 MRCR_CHILD_TOKENS=3500 MRCR_CHILD_OVERLAP_TOKENS=350
```

Both modes use the full served context: the input budget is the context window
minus the output allowance (262,144 − 4,096 = 258,048), so recompute it if you
change either. The runner counts tokens with the executor's tokenizer and chat
template, and the served input counts have matched it exactly, so no extra
margin is needed. Hybrid retrieves evidence and packs it into that budget.
Direct sends the whole prompt when it fits; otherwise
`--direct-overflow middle` keeps a head and a tail slice, preserving roughly
`max-input-tokens / full-rendered-tokens` of the context. Because MRCR needles
are spread through the source, that fraction also approximates the share of
needles direct can still see.

```bash
SEMANTIC_CACHE_EMBEDDING_DEVICE=cpu bash linux/run_benchmark.sh mrcr_v2 hybrid \
  --context-window-tokens 262144 --max-input-tokens 258048 \
  --max-output-tokens 4096 --max-rows-per-source "$MRCR_ROWS_PER_SOURCE" \
  --child-tokens "$MRCR_CHILD_TOKENS" --child-overlap-tokens "$MRCR_CHILD_OVERLAP_TOKENS" \
  --fail-fast --serving-metadata .cache/serving_metadata.json

bash linux/run_benchmark.sh mrcr_v2 direct \
  --context-window-tokens 262144 --max-input-tokens 258048 \
  --max-output-tokens 4096 --max-rows-per-source "$MRCR_ROWS_PER_SOURCE" \
  --direct-overflow middle --fail-fast --serving-metadata .cache/serving_metadata.json
```

The 133K conversation fits the budget, so direct reads it whole, which makes it
the full-context reference for the larger conversations. Hybrid routes those
questions as `direct_fit` and sends the identical request, so treat its rows on
that conversation as a control. The report's `by_source_length_band` separates
the four conversations. The 267K conversation is just over the budget, so
hybrid retrieves on it while direct drops only about 3% of it.

Hybrid splits each conversation into chunks of `MRCR_CHILD_TOKENS`
embedding-tokenizer tokens that overlap by `MRCR_CHILD_OVERLAP_TOKENS`, and
ranks them against the request the question describes, for example
`poem about stars in a formal style`. The marker, the ordinal and the output
instruction are left out of the retrieval query; the model still receives the
full question, and each row records the query as `retrieval_query`. A needle's
only link to its question is its request line, about 15 tokens, so smaller
chunks give it more weight; the 29 September runs used the 7,500-token default.
The same two flags set the chunk size for AA-LCR hybrid runs, where the
defaults remain 7,500 and 750. Chunks must stay below the embedding
model's 8,192-token input limit, and the overlap below the chunk size.

Each hybrid run builds its own index, embedding each conversation once and
reusing it for every question on that conversation. With 7,500-token chunks on
CPU, embedding took 7.7, 15 and 28.5 minutes for the 267K, 534K and 1.07M
conversations; smaller chunks embed about as many tokens but have not been
timed. Packing re-tokenizes the prompt once per candidate chunk, about 160 ms
each at this budget, so 1,000-token chunks add roughly 40 seconds per question,
against about 6 seconds with 7,500-token chunks. Generation takes about 30
seconds per question at this budget.
To measure how much retrieval depends on the budget, rerun the hybrid command
with `--context-window-tokens 65536 --max-input-tokens 61440`; that repeats the
embedding. Dropping `--direct-overflow middle` records every oversized example
as unsupported without an API call instead.

`--fail-fast` stops on the first execution error and marks the run
`stopped_early`. Without it, a run whose examples all fail still reports
`status: complete` with a mean score of zero, because execution failures count
as supported examples scoring zero.

## Budget and routing options

MRCR defaults to context/input/output budgets of 65,536/60,000/4,096 and accepts
explicit overrides. MRCR needs no answer-grading service; enabling semantic
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
Use `--output-root` to change them.
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
  --context-window-tokens 65536 --max-input-tokens 61440 --max-output-tokens 4096 \
  --direct-overflow middle 2>&1 | tee .cache/linux-logs/aa-lcr-direct.log
```

See the [shared execution architecture](../shared_execution_architecture.md)
for the pipeline and [MRCR overview](../mrcr_v2.md) for task/scoring details.
