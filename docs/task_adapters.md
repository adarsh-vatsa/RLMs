# Task adapters

How a benchmark plugs into `execution.pipeline.Pipeline`. This is the
implementation-level companion to the
[shared execution architecture](shared_execution_architecture.md), which covers
the conceptual boundaries and the request flow.

Applies to the benchmarks on the shared pipeline: AA-LCR, MRCR v2, and
LongBench-v2's OpenAI-compatible direct/hybrid path. LongBench iterative/RLM and
hosted-provider paths do not use it.

## The contract

An adapter is one `solver_task(...)` factory returning a frozen
[`Task`](../execution/contracts.py). Each benchmark has exactly one, and all
three are under twenty lines:
`aa_lcr/adapter.py`, `mrcr_v2/adapter.py`, `long_bench_v2/adapter.py`.

```python
@dataclass(frozen=True)
class Task:
    """Solver input only. References and scoring metadata belong outside this record."""

    case_id: str
    source_id: str
    documents: tuple[Document, ...]
    query: str
    render: Callable[[list[dict] | None], list[dict]]
    prompt_version: str
    choices: tuple[str, ...] = ()
    required_prefix: str = ""
    fixed_instructions: str = ""
    legacy_docs_dir: Path | None = None
```

The pipeline never constructs a prompt itself. It calls back into `render` every
time it needs one, so the adapter owns prompt shape and nothing else.

## `render` is the only behavior

```python
render: Callable[[list[dict] | None], list[dict]]
```

It takes evidence and returns OpenAI chat messages. Two modes:

| Call | Returns | Used by |
|---|---|---|
| `render(None)` | The full prompt, whole documents, nothing omitted | Token accounting, `direct_fit`, `middle_truncated` |
| `render(evidence)` | The prompt rebuilt from selected source fragments | `dense_child_packed` |

`render(None)` must render the complete source. Routing decisions and the
recorded `full_rendered_input_tokens` both derive from its token count, so an
adapter that abbreviates here corrupts every downstream budget decision.

MRCR shows the whole pattern:

```python
def render(evidence):
    context = body if evidence is None else render_document_slices(document, evidence, OMITTED)
    return messages(prefix + context + question)
```

The benchmark-specific knowledge is only *where the context sits in the prompt*.
Fragment stitching is shared: `render_document_slices` in
[`execution/packing.py`](../execution/packing.py) concatenates selected ranges
and inserts a separator wherever it skipped source text. The default separator is
`[... omitted source ...]`; MRCR passes `[... omitted conversation ...]` to match
its transcript framing.

The renderer formats engine-selected evidence. It cannot select, reorder, or
extend it.

## What the three adapters actually differ in

| Adapter | Documents | Prompt shape | Declarative fields |
|---|---|---|---|
| MRCR v2 | One conversation body | `prefix + context + question` | `required_prefix` (the 12-char marker), `fixed_instructions` |
| LongBench-v2 | One context | Strict MCQ system prompt + context + query | `choices=("A","B","C","D")`, `fixed_instructions` |
| AA-LCR | Several ordered documents | Renumbered documents + question | `legacy_docs_dir` for the pre-unification path |

AA-LCR is the only one that groups fragments per document, using `groupby` over
`item["document_id"]` so each document renders as one numbered block with its
internal gaps marked.

MRCR derives its `required_prefix` by parsing the question in the adapter:

```python
marker = question.removeprefix("User: Prepend ").split(" to the ", 1)[0]
```

## What the pipeline does with the declarative fields

These are interpreted generically. No engine code branches on a benchmark name.

| Field | Pipeline behavior |
|---|---|
| `documents` | Chunked with the embedding tokenizer and indexed; the index is reused across a source group |
| `query` | The retrieval query and the answer-cache key |
| `source_id` | Groups examples so one index serves many questions |
| `prompt_version` | Recorded per row; part of the cache scope |
| `choices` | Selects the MCQ vs answer cache store, and constrains cache eligibility |
| `required_prefix` | Constrains cache eligibility |
| `fixed_instructions` | Folded into the cache scope fingerprint |

`choices` and `required_prefix` both feed `_eligible`, which gates answer-cache
reads *and* writes:

```python
@staticmethod
def _eligible(task, prediction):
    text = prediction.strip()
    return bool(text) and (not task.choices or text in task.choices) and (
        not task.required_prefix or text.startswith(task.required_prefix))
```

A malformed generation is therefore never cached and a malformed cache entry is
never served. Note this validates *form*, not correctness — it has no access to
reference answers.

## Packing calls `render` once per candidate

`pack` does not estimate token counts. It re-renders and re-tokenizes the whole
prompt to test each candidate's fit:

```python
for result in results:
    candidate = [*selected, result]
    candidate_evidence = source_ranges(documents, candidate, merge) if order == "source" else [...]
    if chat_token_count(tokenizer, render(candidate_evidence)) > budget:
        stop_reason = "next_candidate_overflow"
        break
    selected, evidence = candidate, candidate_evidence
```

Greedy, exact, no binary search. Two consequences for adapter authors:

- **`render` must be pure and cheap.** It is called once per candidate, plus
  once for the final request. Side effects or I/O inside it multiply.
- **Packing cost scales with candidates times prompt size.** A large budget over
  a large source means many tokenizations of a near-budget-sized prompt, and it
  is CPU-bound. This tends to dominate wall-clock time before generation does.

`source_ranges` verifies that every retrieved chunk still matches
`document.text[char_start:char_end]` and raises otherwise, so retrieval drift
fails loudly rather than silently packing stale text. Overlapping ranges merge
only within a single document.

## The boundary that matters

`Task` is solver input only. Gold answers, needle positions, and scoring
metadata never enter it, the cache, or the verifier. The runner scores *after*
`pipeline.execute` returns, which is why `mrcr_v2/scoring.py` is entirely
separate from `mrcr_v2/adapter.py`.

Keep it that way when adding a benchmark. A reference answer reachable from a
`Task` is reachable from the prompt.

## Adding an adapter

1. Write `solver_task(...)` returning a `Task`. Put it in the benchmark package,
   not in `execution/`.
2. Implement `render` for both modes. Reuse `render_document_slices` rather than
   stitching fragments by hand.
3. Set `choices` or `required_prefix` if the task has a checkable output form.
4. Set `fixed_instructions` to any prompt text outside the retrieval corpus, so
   it participates in the cache scope.
5. Give `prompt_version` a new string whenever prompt shape changes; it
   invalidates caches and labels artifacts.
6. Keep scoring in a separate module that reads the runner's saved rows.

The engine needs no changes. If a benchmark seems to require one, the shared
contract is probably being bypassed.

## Legacy variants

The `legacy=False` parameter and `legacy_docs_dir` on the AA-LCR and LongBench
adapters select the pre-unification ingest path, retained for reproducing older
runs. That path chunks with the *executor* tokenizer and silently falls back to a
word-count approximation if the tokenizer cannot load, so its chunk sizes are not
directly comparable to the current embedding-tokenizer chunking. MRCR has no
legacy variant, having been added after the pipeline was unified.
