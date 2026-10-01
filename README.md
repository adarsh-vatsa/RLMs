# Long-context execution with retrieval and semantic caching

A research prototype for answering questions over long sources with an LLM.
When a source fits the model's input budget, the system sends all of it. When it
does not, it retrieves the most relevant passages and packs them into the
budget. Optionally, it reuses verified answers from a semantic cache when a
question is repeated or reworded.

The research question: compared with sending the full context to the same model,
does this keep or improve accuracy while using fewer tokens and less time?
Current experiments use `Qwen/Qwen3.6-35B-A3B` served locally by vLLM.

For an orientation, results and open work, start with the
[project status](docs/project_status_20261001.md).

## How it works

Every benchmark goes through one shared pipeline, `execution.pipeline.Pipeline`.
A small per-benchmark adapter turns each example into a task (documents,
question and prompt renderer). The pipeline counts the full prompt's tokens with
the executor's tokenizer and picks a route:

| Route | When | What is sent |
|---|---|---|
| `direct_fit` | The prompt fits the input budget | The whole prompt |
| `middle_truncated` | Direct mode with `--direct-overflow middle`, the prompt is too long | An equal head and tail |
| `dense_child_packed` | Hybrid mode, the prompt is too long | Retrieved chunks, packed to the budget in source order |

Retrieval uses `Qwen3-Embedding-0.6B` and a FAISS index. Answer caching (exact
or semantic, with a verifier model) is a separate switch, off by default.
Scoring happens after each prediction is saved; gold answers never reach the
solver or the cache. See the
[shared execution architecture](docs/shared_execution_architecture.md) and
[task adapters](docs/task_adapters.md).

## Benchmarks

| Benchmark | Server | Overview | Results |
|---|---|---|---|
| AA-LCR | Neselab (Linux) | [aa_lcr.md](docs/aa_lcr.md) | [30 Sep report](docs/reports/aa_lcr_results_20260930/aa_lcr_results_20260930.md) |
| MRCR v2 | Neselab (Linux) | [mrcr_v2.md](docs/mrcr_v2.md) | [30 Sep report](docs/reports/mrcr_results_20260930/mrcr_results_20260930.md) |
| LongBench-v2 | Jarvis (Slurm) | [longbench_v2.md](long_bench_v2/docs/longbench_v2.md) | Shared-pipeline runs pending; August results in [docs/reports/archive](docs/reports/archive/) |

Candidate benchmarks are compared in [docs/benchmarks.md](docs/benchmarks.md).

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/). Unit tests use mocks and synthetic
fixtures; they never download model weights or call model services.

```bash
uv sync
uv run python -m unittest discover -s test
```

To run benchmarks, follow the [Linux runbooks](docs/linux/README.md) on a GPU
server or the [Jarvis runbooks](docs/jarvis/README.md) on the cluster.

## Repository layout

| Path | Contents |
|---|---|
| `execution/` | Shared pipeline: routing, token counting, chunking, retrieval, packing, cache hooks, HTTP client, artifacts |
| `aa_lcr/`, `mrcr_v2/`, `long_bench_v2/` | Per-benchmark dataset preparation, adapters, runners and scoring |
| `semantic_cache_system.py` | Embeddings, FAISS, reranker and the semantic cache controller; also the original prototype |
| `linux/`, `jarvis/` | Serving and launch scripts for a standalone GPU server and for Slurm |
| `test/` | Unit tests |
| `docs/` | Design notes, runbooks, reports; older material in `docs/archive/` |
| `benchmark_data/`, `benchmark_artifacts/` | Prepared inputs and run outputs; treat existing artifacts as read-only |

The project began in March 2026 as a "Two-Stage Semantic Cache" on Claude models.
That design is described in [docs/system_architecture.md](docs/system_architecture.md)
and still lives in `semantic_cache_system.py`, which defaults to the Anthropic
provider; the benchmark launchers switch it to the local OpenAI-compatible
endpoint. Its five-call demo figures are not benchmark results.

## License

MIT
