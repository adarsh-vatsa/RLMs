# LongBench Client Parameter Reference

This file explains the parameters used in the Jarvis LongBench-v2 client command.
Each item is intentionally short so it can be used while tuning one run at a time.

## Slurm Client Resources

- `bash adarsh-rlms/jarvis/run.sh submit client`: Uses the dispatcher client
  default of 32 GB RAM. This is separate from the runbook's client scratch
  free-space check.

- `bash adarsh-rlms/jarvis/run.sh submit client-gpu`: Uses one L40S for the
  benchmark client. Use this with `SEMANTIC_CACHE_EMBEDDING_DEVICE=cuda` when
  local embedding ingest is the bottleneck.

- `CLIENT_MEM=64G` or `CLIENT_MEM=96G`: Use this on the outer dispatcher command
  when smaller chunk profiles are OOM-killed. The benchmark client holds
  overlapping chunk text, tokenizer offset maps, embeddings, metadata, and FAISS
  state during ingest.

## Semantic Cache Environment Variables

- `SEMANTIC_CACHE_SEARCH_MODE`: Selects the LongBench retrieval path.
  Use `iterative` for the Jarvis FAISS-prioritized cumulative chunk reader; the global default remains `packed`.

- `SEMANTIC_CACHE_EMBEDDING_QUERY_INSTRUCTION`: Instruction prepended before query embedding.
  For LongBench, use a benchmark evidence instruction rather than a legal-domain retrieval instruction.

- `SEMANTIC_CACHE_EMBEDDING_DEVICE`: Device for the local embedding model. The
  default is `cpu`. Use `cuda` only inside a GPU client allocation; CUDA requests
  fail clearly when no CUDA device is available.

- `SEMANTIC_CACHE_EMBEDDING_DTYPE`: Dtype for the local embedding model. The
  default is `float32`. Use `auto` with CUDA to select reduced precision for
  faster ingest and lower GPU memory use.

- `SEMANTIC_CACHE_EMBEDDING_BATCH_SIZE`: Number of chunks the local embedding
  model processes per forward pass during ingest. The default is `16`; use `1`
  or `2` when long chunks OOM. This changes memory and throughput, not intended
  retrieval behavior.

- `SEMANTIC_CACHE_EMBEDDING_MAX_LENGTH`: Maximum token length passed to the
  embedding model for each chunk. The default is `8192`. Lowering it reduces
  memory, but can hurt FAISS ranking when relevant evidence appears late in a
  chunk. This is recorded in the cache namespace and benchmark manifest.

- `SEMANTIC_CACHE_DOC_CHUNK_TOKENS`: Enables token-based document chunking and sets the target chunk size.
  Larger values reduce chunk count and client memory pressure, but chunks that are too large make FAISS evidence selection coarser.

- `SEMANTIC_CACHE_DOC_CHUNK_OVERLAP_TOKENS`: Sets token overlap between adjacent chunks.
  Overlap protects boundary evidence, but higher overlap increases chunk count, duplicated text, embedding cost, and client RAM use.
  The effective step is `DOC_CHUNK_TOKENS - DOC_CHUNK_OVERLAP_TOKENS`; for example, `10000/2000` creates about twice as many chunks as `20000/4000` over the same source.

- `SEMANTIC_CACHE_SCAN_MIN_CHUNK_RATIO`: Fraction of chunks that must be inspected before early stopping is allowed.
  This keeps longer contexts from stopping after the same tiny number of chunks as shorter contexts.

- `SEMANTIC_CACHE_SCAN_MAX_CHUNK_RATIO`: Fraction of chunks the reader may inspect if no high-confidence answer is found.
  This is the ratio-based FAISS retrieval and scan budget.

- `SEMANTIC_CACHE_SCAN_MIN_CHUNKS`: Absolute minimum number of chunks inspected before early stopping is allowed.
  This is a safety floor in addition to `SEMANTIC_CACHE_SCAN_MIN_CHUNK_RATIO`.

- `SEMANTIC_CACHE_SCAN_MAX_CHUNKS`: Optional absolute maximum number of chunks inspected in iterative mode.
  Set it to `0` to rely only on `SEMANTIC_CACHE_SCAN_MAX_CHUNK_RATIO`.

- `SEMANTIC_CACHE_SCAN_MAX_TOKENS`: Maximum output tokens for each chunk-inspection or final-adjudication call.
  The current Jarvis profile uses `1024`; lower caps can truncate the inspector JSON or retain too little evidence.

- `SEMANTIC_CACHE_SCAN_EMPTY_LEDGER_FALLBACK_RATIO`: Extra scan budget used only when the normal scan produced no useful ledger memory.
  `1.0` lets the reader continue through all chunks before giving up on iterative evidence extraction.

- `SEMANTIC_CACHE_ITERATIVE_PACKED_FALLBACK_INPUT_TOKEN_BUDGET`: Input-token budget for the bounded packed fallback.
  This path is used when the ledger is empty, invalid, or still low-confidence after final adjudication.

- `SEMANTIC_CACHE_ITERATIVE_MEMORY_MAX_CHARS`: Character cap for the cumulative iterative memory string.
  When the cap is exceeded, old prose updates are trimmed while structured target facts, code mappings, best choice, and parse failures are preserved.

- `SEMANTIC_CACHE_MCQ_PROMPT_STYLE`: Selects the multiple-choice prompt template.
  `strict` tells the model to compare choices carefully and return only `A`, `B`, `C`, or `D`.

- `OPENAI_COMPAT_EXECUTOR_EXTRA_BODY_JSON`: Extra JSON merged into executor API requests.
  For Qwen, `{"chat_template_kwargs":{"enable_thinking":false}}` disables thinking mode.

- `OPENAI_COMPAT_EVALUATOR_EXTRA_BODY_JSON`: Extra JSON merged into evaluator API requests.
  Keep it consistent with the executor when comparing non-thinking Qwen runs.

## Sampling Command Parameters

- `--input-path`: Source CSV used to build the sampled suite.
  In this workflow it points at `benchmark_data/long_bench_v2/data_cache_suite.csv`.

- `--output-path`: Destination CSV for the sampled suite.
  Reusing this path overwrites the previous sampled parameter-search suite.

- `--sample-size`: Number of source groups to sample before row-type expansion.
  With `original,exact,semantic`, `--sample-size 3` produces 9 benchmark rows.

- `--min-token-count`: Excludes source groups below this estimated token count.
  Use it to focus the sample on long-context rows.

- `--max-token-count`: Excludes source groups above this estimated token count.
  Lowering it avoids extreme outliers while tuning speed and accuracy.

- `--selection-strategy`: Strategy for selecting eligible source groups.
  `random` is the most realistic small-sample strategy, while `shortest` is mainly useful for fast smoke tests.

- `--domains`: Comma-separated LongBench domain values to sample from.
  Use this for targeted diagnostics, such as isolating `Long In-context Learning`.

- `--samples-per-domain`: Domain-balanced sampling mode.
  When greater than `0`, this samples that many source groups from each eligible domain and overrides `--sample-size`.

- `--seed`: Random seed for reproducible sampling.
  Keep it fixed when comparing parameter changes.

## Benchmark Runner Parameters

- `--suite-csv`: Sampled or full benchmark CSV passed to the runner.
  This should match the `--output-path` from `sample_csv.py` for sampled runs.

- `--llm-provider`: LLM backend used by `semantic_cache_system.py`.
  Use `openai_compatible` for the Jarvis vLLM services.

- `--mode`: Benchmark mode.
  `cache` enables persistent cache reuse, while `baseline` disables cache reads.

- `--cache-reset`: Deletes the resolved cache namespace before running.
  Keep this on while tuning so exact and semantic rows do not reuse stale wrong answers.

- `--cache-state-root`: Root directory for persistent benchmark cache state.
  On Jarvis this should usually be `$JARVIS_CACHE_STATE_ROOT`.

- `--executor-model`: Model used for chunk inspection and final answer adjudication.
  In the current Jarvis Qwen profile this is `Qwen/Qwen3.6-35B-A3B`.

- `--evaluator-model`: Model used for evaluator-class cache checks and related verification calls.
  In the current Jarvis Qwen profile this is `Qwen/Qwen3.5-35B-A3B`.

- `--row-types`: Comma-separated row variants to run.
  `original,exact,semantic` tests first-write quality plus exact and semantic cache reuse.
