# Context Management And Language Memoization

This note distills the current design discussion around Recursive Language
Models, long-context agents, and a stronger cache abstraction for this repo.

## 1. The General Principle

For long-context work, the core problem is not simply search. It is context
management under a token budget.

If an agent is given a question and a context that is larger than its usable
attention window, the agent needs a policy for deciding what to read, what to
delegate, what to compact, and what to carry forward. The basic loop is:

1. Notice that the relevant context exceeds the usable prompt budget.
2. Choose a context-management move:
   - split context and delegate bounded slices to subagents
   - compact prior state
   - inspect a selected region
   - ask a new instance of itself with selected context and state
3. Receive compact, structured outputs.
4. Decide whether the evidence is enough to answer.
5. If not, repeat with a better-targeted prompt or context selection.

The important requirement is not a specific programmatic scaffold. The
orchestrator needs enough feedback and control to keep every reasoning step
inside a bounded attention window.

## 2. RLM As One Formalization

Zhang, Kraska, and Khattab's
[Recursive Language Models](https://arxiv.org/abs/2512.24601) formalize one
version of this loop. Their abstraction treats long prompts as an external
environment and lets the model programmatically examine, decompose, and
recursively call itself over snippets of the prompt.

That is useful and clean, but it does not introduce a fundamentally different
class of operation from a well-built long-context agent runtime. A capable
agent with token-budget feedback, addressable context, isolated subagents,
compaction, and self-reprompting can implement the same general strategy.

In other words:

- RLM is a disciplined substrate for recursive context management.
- A strong agent runtime can be functionally equivalent if it exposes the same
  control over context selection, delegation, and compaction.
- The deeper unsolved problem is not just whether an agent can recurse over a
  context, but whether the system can avoid re-solving work it has already done.

## 3. Why Standard Search Still Matters

A model attending over a bounded chunk can often outperform classical search
inside that chunk. It can interpret paraphrases, reason over local evidence,
and synthesize answers in ways BM25 or vector similarity cannot.

That does not make search obsolete. Search should be treated as a cheap
candidate generator and context-addressing layer, not as the final reasoning
layer.

Useful low-cost signals include:

- token counts
- chunk IDs and source spans
- file/page/section metadata
- headings and structural boundaries
- lexical matches
- vector candidates
- cached summaries
- prior subagent results

The strongest architecture is therefore index-assisted recursive attention:
cheap indexing proposes where to look, bounded model calls read those regions,
and the orchestrator reasons over compact returns.

## 4. The Stronger Cache Framing

The current repo mostly implements semantic caching in the shape:

```text
query + context -> answer
```

That is valuable, but it is not yet the full idea. The sharper framing is
dynamic programming over language-model subproblems.

In normal dynamic programming, a solver avoids recomputing a subproblem once it
has been solved. For language work, equality is fuzzier: two subproblems may be
equivalent, overlapping, composable, or merely useful as a hint. The cache
should therefore store more than final answers.

A richer memoized artifact should capture:

```text
MemoEntry:
  task: the subproblem or question
  context_scope: document, chunk, span, or set of spans
  result: answer, summary, fact, decision, or pointer
  evidence: source spans supporting the result
  dependencies: subcalls used to produce it
  confidence: confidence or verifier status
  reusable_as:
    - exact answer
    - supporting fact
    - search hint
    - aggregation component
    - ruled-out region
```

This turns cache lookup from a binary hit/miss into a reuse decision.

## 5. Reuse Modes

A future recursive call might benefit from prior work in several ways:

- **Replay**: same task and same context scope, so return the prior answer.
- **Verify**: prior answer likely applies, but a cheap model or deterministic
  check should confirm.
- **Compose**: a larger task can reuse solved subregions and only call models
  for missing regions.
- **Narrow**: prior work says where the answer likely lives.
- **Rule out**: prior work says where the answer probably does not live.
- **Aggregate**: prior partial answers can be combined into a larger answer.

The core semantic operations become:

- Are these two tasks equivalent?
- Does the old result entail what the new query needs?
- Does the old context span overlap enough to reuse?
- Can partial results compose into a larger answer?
- Is the prior result trustworthy enough to avoid a model call?

This is where models are useful as semantic equality, entailment, and
composition operators over memoized work.

## 6. A DP-Shaped Solver

At a high level, the target algorithm looks like:

```text
Solve(question Q, context C):
  if memo has an exact solved entry for Q over C:
      return it

  if memo has useful partial entries for Q over regions of C:
      reuse those entries
      identify missing or ambiguous regions
  else:
      identify candidate regions with indexes, summaries, or decomposition

  for each missing region:
      call a bounded subagent with Q and that region
      store the subagent result as a memo entry

  aggregate partial results
  verify answer and evidence
  store Solve(Q, C)
  return answer
```

This is different from ordinary semantic caching because the goal is not only
answer replay. The goal is work avoidance and search guidance throughout a
recursive reasoning process.

## 7. Relationship To This Repo

The existing implementation is a useful starting point:

- `source_chunk_hash` protects direct `cached_query()` reuse.
- `data_scope_hash` protects retrieval `search()` reuse across document sets.
- the LLM Sniper can verify semantic equivalence before replaying an answer.
- knowledge extraction starts moving from answer blobs toward fact-level reuse.
- persistence lets work survive across runs.

But the architecture should evolve from a semantic cache into a language
memoization graph.

The next architectural shift is:

```text
from: cache answers
to: memoize recursive subproblem work
```

That means storing structured intermediate artifacts, evidence spans,
dependency trails, verifier metadata, and reuse modes.

## 8. Research Positioning

The clean distinction from RLM is:

- RLM layer: how a model recursively navigates and processes huge context.
- Memoization layer: how the system remembers and reuses recursive subproblem
  work.
- Provenance layer: how the system knows which subanswers are trustworthy.
- Planning layer: how prior subanswers guide future decomposition.

The stronger claim is not that recursive context management is new. It is that
recursive context management needs persistent, scoped, semantic memoization to
become cheap, deterministic, and auditable.

Put differently:

> RLM gives recursive execution. Language memoization gives reusable recursive
> execution.

## 9. Practical Next Steps

Concrete repo directions:

- Add a first-class `MemoEntry` schema distinct from cached answer entries.
- Store source spans and context scopes for every intermediate result.
- Add reuse labels such as `exact_answer`, `supporting_fact`, `search_hint`,
  `partial_aggregate`, and `ruled_out`.
- Replace binary cache hit logic with a ranked reuse planner.
- Add verifier calls only when reuse would skip meaningful model work.
- Record memo reuse telemetry in benchmark bridge rows.
- Build an ablation comparing:
  - no memoization
  - exact answer replay
  - semantic answer replay
  - fact reuse
  - full memoized subproblem reuse

The target system should let an orchestrator solve a long-context task once,
then reuse the resulting subproblem graph whenever later tasks overlap with
that work.
