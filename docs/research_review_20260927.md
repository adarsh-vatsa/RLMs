# Research review and direction: 27 September 2026

This review covers where the project stands, what the literature says, and what
we should build next. It is written as a handoff for a new working session.
Everything here was produced in one Claude Code session. That session covered
four things:

- a code audit of both lines of work
- a spot-check of key bugs
- a survey of the Recursive Language Models (RLM) research line
- a broader literature survey

Section 8 lists sources and their confidence levels.

---

## 1. TL;DR

- **The repo holds two unmerged lines of work.** Neither one yet supports a
  publishable claim.
  - `main` (Adarsh, May 2026) holds the *language memoization* design and code.
  - `shared-execution-linux-runners` (Engin, April to September 2026) holds a
    clean long-context *execution pipeline* and three benchmarks.
- **Keep about 15% of the code.**
  - Engin's `execution/` Task/adapter pipeline is well built.
  - Adarsh's `ContextScope` interval math and `MemoStore` persistence are the
    right primitives.
  - The remainder should be optimized, used as a baseline, or discarded: most of
    `semantic_cache_system.py`, the iterative reader, knowledge triples,
    consensus verify, the Router, and the pre-warmer.
- **No headline number survives the audit.**
  - LongBench-v2 cache savings are guaranteed by how the traffic was built.
  - The memoization 41 → 0 call result replays an identical run.
  - MRCR uses effectively one haystack per band.
  - AA-LCR 64K (retrieval 41% vs truncation 21%) is the only real but still
    preliminary effect.
- **The field has a clear gap that fits our idea.** No one memoizes recursive
  (RLM-style) sub-call results *across different questions* scoped to evidence
  spans. No benchmark measures *amortized* cost over a stream of questions on the
  same corpus. The RLM authors list caching/memoization as future work.
- **Recommended direction.** Build a system with three parts:
  - span-scoped, evidence-verified memoization shared across questions
  - a cost-escalating planner: replay → compose → read residual → retrieve+pack
    → recursive map
  - an amortized-cost evaluation in which questions arrive as a stream over the
    same corpus

  Use the upstream `rlms` package as the recursive executor rather than
  rebuilding it.

---

## 2. How we got here: the conceptual thread

The direction came out of a design discussion. Each step is recorded here so the
reasoning can be challenged.

1. **The original "Dragnet & Sniper" cache is a memoization layer, not an
   architecture.** It sits inside `rlm_query()`. It does not change how context
   is decomposed. It only avoids paying twice for near-duplicate (query, chunk)
   calls.
2. **Its recall is limited by the weakest component.** A candidate is found only
   if a 0.6B embedding model ranks it in the top-5 *within an MD5 bucket of the
   exact chunk text*. Re-chunking invalidates everything. The cache is also
   invisible to the orchestrator model, which never reasons about it.
3. **Attention over a bounded context is a better search operator than frozen
   embeddings.** It is also expensive. So embeddings/BM25 should *propose where
   to look*, and bounded model calls should *read*.
4. **An orchestrator with strategic dispatch is functionally what RLM already
   is.** The orchestrator sees context metadata, chooses chunks and prompts,
   spawns leaf sub-agents that cannot see the full context or recurse, reads
   their structured returns, and re-dispatches. Zhang, Kraska and Khattab's RLM
   is this design with a Python REPL as the substrate. The only real
   disagreement is how much structure to impose: Zhang argues for minimal
   enforced structure. The novelty therefore cannot be "recursive context
   management".
5. **What RLM lacks is persistence of derived work.** Every query starts from
   zero. The flat REPL scratchpad also gives the orchestrator no structured map
   of what it has already learned about the corpus.
6. **Synthesis: a query optimizer over materialized views.**

   | DB concept | Here | Who built it |
   |---|---|---|
   | Base tables | Raw source spans | — |
   | Indexes | Embeddings, FAISS, lexical, structure | Engin (retrieval) |
   | Materialized views with lineage | Memo entries: task + scope → result + evidence + deps | Adarsh (`MemoEntry`) |
   | Physical operators | Bounded LLM calls | everyone |
   | Planner | Chooses how to answer under budget | Engin: 2-plan router · RLM: model-written code · Adarsh: memo-consulting map-reduce |
   | Cost model | Tokens / latency / accuracy risk | Engin (token budgets) |

   The memo graph and the "wiki / context graph" are the same object:
   - It is the *write side* when recording what was computed.
   - It is the *read side* when shown to the orchestrator as a navigable map.
     For example: "§4 covered; §7 searched for X, nothing; §9–12 unread".

   Showing it to the orchestrator lets attention do the matching. This is better
   than a hidden embedding-threshold cache.

---

## 3. Where our work stands

### 3.1 State of the repo

| | `main` | `shared-execution-linux-runners` |
|---|---|---|
| Last update | 2026-05-18 | 2026-09-21 |
| Author | Adarsh Vatsa | Engin Deniz Dogu (almost all commits) |
| Focus | Language memoization / DP-memo over LLM subproblems | Budgeted long-context execution (direct vs retrieve+pack), benchmarks |
| Models | Claude (prototype), local MLX (`local_llm.py`) | Local Qwen3.6-35B-A3B executor, Qwen3.5-35B-A3B grader via vLLM |
| Key files | `language_memoization.py`, `scripts/run_dp_memo_*`, `docs/context_management_and_language_memoization.md` | `execution/`, `aa_lcr/`, `mrcr_v2/`, `long_bench_v2/`, `docs/project_status_20260921.md` |

A trial merge showed only 3 textual conflicts: `README.md`, `pyproject.toml` and
`semantic_cache_system.py`. The real conflict is conceptual. `main` rewrote the
controller to be memo-first. Engin's pipeline calls `lookup_cached_result`, which
exists only on his branch. A code-level merge was judged pointless. The
recommendation below reuses parts selectively.

The branch `claude/understand-repo-structure-3UiKx` equals `main` plus two
additions: this document and the citation-graph research skill
(`.claude/skills/citation-graph-research/`).

### 3.2 Engin's branch: audit verdicts

Paths are relative to `shared-execution-linux-runners`.

| Subsystem | Verdict | Reason |
|---|---|---|
| `execution/` architecture: Task/adapter, routing, token budgeting, artifacts | **KEEP** | Clean contract. `Task.render(evidence)` is the only behaviour. Adapters are 15–25 lines. Budgets use the executor's own tokenizer and chat template. Gold answers are structurally excluded. Failures become rows. |
| Retrieval (`execution/retrieval.py`, `retrieve()` in the prototype) | **Optimize** | One vector per 7,500-token chunk (Qwen3-Embedding-0.6B), ranked by cosine against the raw question including boilerplate instructions. No BM25, hybrid fusion, query rewriting or reranking. This is behind 2025–26 practice. Move to 512–1K chunks, BM25+dense fusion, a real rerank, neighbour expansion, and query cleanup. |
| Reranker inside the pipeline | **Bug** | With `rerank_top>0`, `retrieve()` truncates to `max(rerank_top, 5)` instead of reordering, which starves the packer (`semantic_cache_system.py:3242`). Off by default, so unnoticed. |
| Packing (`execution/packing.py`) | **Optimize** | Stops at the first overflowing chunk. The MRCR 60K run packed about 53.3K of 60K tokens (about 11% of budget unused). It also re-tokenizes a full render per candidate. Fix: skip-and-continue, trim the last chunk, estimate incrementally, then check exactly once. |
| Cache hook + scope fingerprint | **Optimize** | Scope is sound but very conservative: source content, full config identity, tokenizer, prompt version and embedding identity. It fails closed. See the confirmed bug below. The MCQ verifier compares question intent, not answer letter, which is a hazard when options are reordered. |
| Adapters (AA-LCR, MRCR, LongBench) | **KEEP** | Small and correct. Drop the `legacy` variants once old runs no longer need reproducing. |
| LongBench older cache runner | **DISCARD after migration** | Pre-unification. Move LongBench onto `execution/`. |
| Iterative reader / evidence ledger | **DISCARD** | 46.9% accuracy on 175.7M tokens and 7,438 calls, against 48.8% on 60.8M tokens for the simple hybrid. About 800 lines of LongBench question-type heuristics were tuned on 18–54 rows. |
| RLM baseline wrapper (`long_bench_v2/rlm_helpers.py`) | **Optimize or drop** | A faithful wrapper of upstream `rlms` (depth 1). Its only run is 3 rows on Gemini 3.1 Pro via OpenRouter. Cost records as 0 because pricing only knows Claude. |
| `semantic_cache_system.py` (4,448 lines) | **DISCARD (extract ~400 lines)** | The pipeline needs only `EmbeddingEngine`, `FAISSIndex`, the tokenizer chunker and the compact cache store/lookup. |
| Tests (210, all mocked) | **KEEP + extend** | Good contract tests. Missing: ranking behaviour, reranker path, cross-bucket cache mapping, verifier negatives. `test_data_scope_cache.py` (2.1K lines) mostly tests dead prototype paths. |

**Confirmed bug (verified in this session): cache index mapping.**
`store_compact_answer` sets `cache_idx = get_total_entries()`, which is global
insertion order (`semantic_cache_system.py:2875`). `_find_cache_entry_by_flat_idx`
resolves indices by walking buckets in dict order (`:2258-2266`). Once any entry
is appended to a bucket that is not the last, every later index points at a
neighbouring entry. Scope filtering then hides it as a missed hit. No test
covers interleaved traffic.

**Validity of reported results:**

- **LongBench-v2 (+3.4 pp, −49% tokens), August, old runner:**
  - Each of the 503 sources has exactly one original, one paraphrase (same
    choices, same answer) and one exact repeat. Every cache bucket therefore
    holds one question. The verifier never sees a hard negative, so the 98%
    paraphrase hit rate is guaranteed by construction.
  - The token saving follows from 50% duplicate traffic.
  - The accuracy gain comes entirely from 107 over-budget originals, where
    retrieval scored 56.1% vs about 42% for head/tail truncation. Paraphrase
    rows double-count it: originals alone were 241 vs 225 correct.
  - A second hybrid run differed on 20 of 396 direct-fit originals whose prompts
    should be identical, so serving was nondeterministic.
- **AA-LCR (41% vs 21% at 64K), 30 August, pre-unification:**
  - This is a real paired effect, but against a weak baseline. Middle truncation
    of concatenated documents drops whole documents.
  - Missing controls: random chunks in source order at equal budget, BM25, and
    per-document proportional truncation.
  - Other limits: single run, known grader errors, 512-token output cap, v1.0
    answer keys.
  - Absolute scores are known to be low: the status doc cites about 64% for this
    model on Artificial Analysis.
- **MRCR v2 (10 examples):**
  - All 10 rows in each band share *one* conversation (60K band: source
    `8d4197dd…`; 240K band: `d27bb260…`), so this is effectively n=1.
  - There is no full-context reference.
  - Similarity ranking also cannot solve ordinal queries ("the *second* poem about
    X"), because dropping an earlier match shifts the ordinal.
- The status doc's line "caching off for every AA-LCR run" conflicts with the
  AA-LCR artifacts, which contain cache state and 6 rejected verifier calls.

### 3.3 Adarsh's memoization work (`main`): audit verdicts

| Subsystem | Verdict | Reason |
|---|---|---|
| `ContextScope` interval math (`language_memoization.py:156-289`) | **KEEP** | Correct and tested. The right primitive. |
| Content hashing / invalidation | **Optimize (critical)** | `content_hash` is per document, and an empty hash is a wildcard (`:201`, DuckDB `:1255`). NoLiMa's "source" hash covers test/haystack/needle/depth but not text or length (`scripts/run_dp_memo_nolima.py:125-132`). At depths ≠ D00 it silently reuses stale chunk results across lengths. Fix: hash each chunk's own text, and key windows by the hashes of the chunks they cover. |
| `TaskSpec` keying (`:111-156`) | **Optimize (core gap)** | The signature includes the exact normalized prompt, so compose and rule-out only apply to the *identical* question. **Reuse across different questions, the central research idea, is not implemented.** The one question-agnostic extractor stores its output under the question's key. |
| `plan_reuse` (`:820-855`) | **Optimize** | Replay needs an exact match. Compose uses correct interval subtraction but is restricted to the same task. Rule-out equals replay of a negative entry. **Confirmed leak (verified):** entries marked `exact_answer` count as coverage (`can_cover_scope`, `:406-413`), so stored *final answers* are fed back to the aggregator as partials (`run_dp_memo_nolima.py:425-428`). Nested scopes are not deduplicated. |
| "Narrow" / search hints | **Implement or drop** | `hint_entries` are collected (`:836`) and never consumed. |
| Semantic matching | **Optimize** | `ranked_text_candidates` is word overlap that also tokenizes metadata JSON (`:737`, `:750-794`). The only LLM check (`reuse_verifier`) runs only at identical scope, and no runner passes it. The live smoke's verifier falls back to a hardcoded phrase check. |
| `MemoStore` (JSON) / `DuckDBMemoStore` | **KEEP / optimize** | Good persistence. `save()` is a silent no-op when the DB is already a file elsewhere (`:1459-1465`). Lookups parse full JSON and then re-filter in Python. |
| `solve_with_memo` NoLiMa solver (`semantic_cache_system.py:1130-1391`) | **KEEP as baseline** | Single-level map-reduce over 2×200-word windows. Not DP or recursion. 81 chunks → 41 windows explains the 41 cold calls. It reads the whole haystack with no retrieval step. It duplicates about 100 lines of `memoized_subproblem`. |
| NoLiMa 41 → 0 and 16K→256K overlap claims | **DISCARD as claims** | Warm runs replay the identical question/doc/hash. The overlap chain is n=1 per length, D00 only, relies on the unsound source hash, and the aggregator sees prior final answers. There is no baseline, and prefix KV caching gets the same prefix reuse for free. |
| Report (`docs/dp_memo_benchmark_report.md`) | **Revise** | n=5 depth sweeps (one needle and question each). Prompts were changed after failures and only failing samples were re-run. Scoring uses substring match. `max_kv_size` (default 2048, a rotating window, `local_llm.py:38`) and `max_tokens` are not in manifests. |
| Shared-context benchmark (`scripts/run_dp_memo_shared_context.py`) | **KEEP + scale** | The only cross-question design (question-agnostic extraction, then per-question planning). Tiny: one ~80-word doc, 5 questions. |
| Mutable-workload benchmark | **Optimize** | A good scenario, but invalidation is a hand-picked `invalidate_scope` call with `content_hash=""`. |
| `local_llm.py` | **KEEP** | Thin and fine. Record KV/token limits. |
| Chunk table, lineage, graph edges | **DISCARD for now** | Written but never read. |
| Legacy cache→memo migration (`semantic_cache_system.py:621-703`) | **DISCARD after one-time migration** | Runs on every store/save/search, with quadratic cost. |
| Worklog (`docs/long_horizon_substrate_worklog.md`, 1,658 lines) | **Condense** | Turn it into a short decisions log. |

### 3.4 Original prototype components (both branches)

| Component | Verdict | Reason |
|---|---|---|
| Dragnet & Sniper | **Fold into memo verifier** | The only real semantic-equivalence logic. Currently bucketed by exact-chunk MD5. The parallel path's first-hit-wins is nondeterministic. |
| Knowledge-triple extraction | **DISCARD** | Extracts from the *answer* rather than the source, one LLM call per write, and nothing reads it any more. |
| Grounding regex | **Optimize** | A free numeric check. Answers with no numbers are labelled "GROUNDED". Check against stored evidence spans instead. |
| Consensus verify | **DISCARD or make blocking** | Fails open (errors count as "AGREED") and never blocks the write. |
| Context-collapse guard, recursive summarization | **DISCARD** | Low value with compact memo entries. Lossy, not memoized. |
| Keyword Router | **DISCARD** | Substring match sends most queries to the cheap model. Hardcoded model IDs. |
| CachePreWarmer | **DISCARD** | Superseded by question-agnostic per-chunk extraction memo. |
| "96.7% cost reduction" (README) | **Do not cite** | From a 5-call demo. |

---

## 4. The RLM line since the original paper

Tags follow the key in Section 8.

- **Paper.** "Recursive Language Models", Zhang, Kraska and Khattab, arXiv
  2512.24601. There are at least 3 arXiv versions [snippet]. The repo README
  cites "our NeurIPS 2026 paper" [read], though the broader survey could not
  independently confirm acceptance.
  - Benchmarks: S-NIAH, BrowseComp-Plus, OOLONG, OOLONG-Pairs.
  - Reported results: RLM(GPT-5) 91.3% on BrowseComp-Plus with 1K documents;
    F1 58.0 vs <0.1 for the base model on OOLONG-Pairs [snippet].
  - New since the blog: **RLM-Qwen3-8B**, fine-tuned on 1,000 filtered
    trajectories from Qwen3-Coder-480B, +28.3% over base [snippet].
  - Stated limitations: recursion depth 1; blocking synchronous sub-calls; high
    cost variance; seconds-to-minutes latency; no prefix caching of sub-calls
    [snippet].
- **Official code.** `github.com/alexzhang13/rlm` (`pip install rlms`, about 5.6k
  stars) [read].
  - Sandboxes: local, IPython, Docker, Modal, Prime, Daytona, E2B.
  - Options: `max_depth`, `max_concurrent_subcalls`, `persistent=True`
    multi-turn sessions with versioned context/history, and `compaction`.
  - Also includes a trajectory visualizer and a prime-rl training harness.
  - Releases v0.1.2 (May 28, depth>1) and v0.1.3 (June 26, parallel sub-calls).
  - **No memoization of sub-call results across queries** [read/inferred].
- **DSPy.** `dspy.RLM` runs in a Deno/Pyodide WASM sandbox with `llm_query`,
  `llm_query_batched` and `SUBMIT`. There is no explicit caching, and each forward
  pass starts a fresh interpreter [read].
- **Same group.** *Prime Agent: A Self-Improving RLM Harness* (arXiv 2608.23552,
  Aug 2026, with Alex Zhang) keeps persistent histories, memories, skills and
  subagent specs across trajectories, and reports 95.5% on ARC-AGI-3 [snippet].
  *Headlong* is a bash micro-harness with recursive LLMs [snippet].
- **Critiques and reproductions:**
  - *Think, But Don't Overthink: Reproducing RLMs* (arXiv 2603.02615) [snippet]:
    - Depth 1 helps on complex reasoning.
    - Depth 2 and simple retrieval get *worse*.
    - Latency grows from 3.6 s (plain) to 89.3 s (depth 1) to 344.5 s (depth 2).
    - Failure modes: compounding format errors and redundant loops.
  - *SRLM / RLMs Meet Uncertainty* (Apple, arXiv 2603.15653) [snippet]:
    - Recursion is not the main driver of the gains.
    - Self-reflective program search beats RLM by up to 22% at equal time.
    - RLM hurts on inputs that fit in context.
  - *Chained RLM* (Mitra & Ulukus, arXiv 2608.05124) [snippet]: a sequence of
    fresh roots passing plain-text state (summary, blackboard, artifacts) with
    auditable stored trajectories. This is **the closest relative of our idea,
    but reuse is within one task, not across queries.**
- **Adoptions.** Prime Intellect `RLMEnv` (verifiers), Google ADK, Ax, Symbolica
  ARC-AGI, a clinical C-RLM, and minRLM [snippet/read].

**Implication.** Rebuilding an RLM runtime has no research value. Use `rlms` (or
`dspy.RLM`) as the recursive operator. Our contribution sits *around* it:
memoized, verified reuse of its sub-call work across queries, and planning that
avoids recursion when cheaper plans suffice. The critiques say recursion should
be avoided where it isn't needed.

---

## 5. State of the field: what is broadly agreed

1. **Long context degrades unevenly.** This comes from Context Rot (Chroma, 2025),
   NoLiMa (11 of 13 models below 50% of baseline at 32K), OOLONG and HELMET
   (synthetic NIAH predicts downstream poorly). 2026 frontier models are much
   better: Claude Opus 4.6 scores 76% on MRCR v2 8-needle at 1M [read]. The
   problem has shrunk but not gone.
2. **Neither long context nor retrieval dominates.** "LC vs RAG" (Li et al.)
   found LC 56.3% vs RAG 49.0%, but RAG uniquely solves about 1.3K questions.
   LaRA's verdict is "no silver bullet". **At matched token budgets, simple
   baselines are strong.** DOS-RAG (retrieve, then read in *document order*)
   beats ReadAgent and RAPTOR by 2–8 points (EMNLP'25). GraphRAG's multi-hop
   advantage vanishes at matched budgets. *This validates Engin's source-order
   packing and requires matched-budget comparisons in all our claims.*
3. **Recursion and decomposition help on aggregation-heavy tasks and hurt on
   simple ones** (RLM, the RLM reproduction, SRLM, Chain-of-Agents).
4. **Precomputing structure (GraphRAG, RAPTOR, HippoRAG 2, LightRAG) has real
   build cost** and is sensitive to graph quality (GraphRAG-Bench, ICLR'26).
   Build-once/query-many only pays off at high reuse.
5. **Amortization is the emerging lens:**
   - Sleep-time compute: about 5× less test-time compute; 2.5× cheaper per query
     when amortized across related queries.
   - AnnoIndex: amortized tokens per query = (offline + online) / #queries.
   - Efficiency Frontier (May 2026): expensive preprocessing becomes optimal as
     the reuse count N grows.
   - *Beyond the Context Window* (Mar 2026): fact memory becomes cheaper than
     long context after about 10 interactions.
6. **Verified caching is maturing:**
   - vCache (ICLR'26): learned per-entry thresholds with an error-rate bound.
   - Krites (Apple): an off-path LLM judge.
   - **GroundedCache** (May 2026): reuse only if similarity, overlap with fresh
     evidence, source version and answer-evidence support all pass. Unsafe reuse
     drops to 0% on HotpotQA and 1.5% under document drift, vs up to 51.5% for
     naive caching.
   - Semantic caching of *intermediate summaries* cuts redundant computation by
     50–60%.
7. **Reuse of agent computation is active:**
   - Agentic Plan Caching (NeurIPS'25): −50% cost.
   - Workload-aware caching of agent-DAG intermediates (2026): −64.7% latency.
   - Agent Workflow Memory; Metacognitive Reuse.
   - KV-level: CacheBlend and EPIC reuse non-prefix chunks; prompt-caching
     studies report −41–80% cost.

### Open gaps (where we can contribute)

1. **No benchmark for amortized cost on many questions over one corpus.** Every
   long-context benchmark scores questions independently. Nothing reports cost
   per correct answer as a function of query count, stream order, or overlap
   between questions.
2. **No cross-query memoization of recursive sub-calls tied to evidence spans.**
   The RLM authors list it as future work. Chained RLM is within-task. Plan and
   DAG caching are not span-scoped.
3. **Caching *negative* evidence ("span S does not contain X") appears
   unstudied.**
4. **No provenance-gated reuse of *intermediate* results with span-level
   invalidation.** GroundedCache only covers final answers.
5. **Nobody has measured error propagation from reused sub-results.**

---

## 6. Recommended direction

### 6.1 Thesis

> Long-context systems should do each piece of reading work once. Combine three
> things: span-scoped, evidence-verified memoization of sub-call results that
> works across questions; a planner that escalates from cheap to expensive plans;
> and an RLM-style recursive operator used only when needed. Over a stream of
> questions on the same corpus, the goal is at-least-matched accuracy against
> budgeted retrieval, with sharply falling amortized cost.

This is neither "we invented recursion" nor "we invented semantic caching". It
claims the combination and the evaluation lens (gaps 1–4).

### 6.2 Architecture

```
question + corpus
   │
   ▼
Planner (cheapest plan first; stop when verifier satisfied)
   1. replay     exact task + scope in memo
   2. compose    memo partials cover scope → aggregate, read only gaps
   3. residual   uncovered, not-ruled-out spans fit budget → read directly
   4. retrieve   BM25+dense+rerank over residual, memo hints as priors → pack
   5. recurse    rlms/dspy.RLM map-reduce over residual (aggregation tasks)
   │
   ▼  every plan writes back
Memo store (views with lineage)
   - key: task-class + chunk-content-hashes (auto invalidation)
   - entry kinds: question-agnostic extraction | question-specific finding |
     negative evidence | final answer (never reused as a partial)
   - evidence spans + dependencies + verifier status
   │
   ▼
Verifier (GroundedCache-style: equivalence + evidence support + source version)
   │
   ▼
Orchestrator view: the memo rendered as a navigable map ("what we know,
where, what's ruled out, what's unread"), shown in-context instead of hidden.
```

### 6.3 What to reuse

- `execution/` Task/adapter pipeline, routing, budgeting, artifacts: the
  operator layer *and* the main baseline.
- `ContextScope`, `MemoStore` / `DuckDBMemoStore`: the memo layer, with rekeying.
- Dragnet & Sniper equivalence logic, folded into the verifier.
- Upstream `rlms` as the recursive operator. The current wrapper already calls it.
- Engin's adapters for AA-LCR, MRCR and LongBench-v2.

### 6.4 Work plan

**Phase 0: foundations (1–2 weeks).** Make the baseline honest before building
on it.

1. Extract the roughly 400 needed lines of `semantic_cache_system.py` into
   `execution/`. Freeze the prototype.
2. Fix the cache index bug, the reranker truncation bug and packing
   skip-and-continue. Add tests for each.
3. Upgrade retrieval: 512–1K chunks, BM25+dense fusion, rerank, and query
   cleanup (strip MCQ instructions and MRCR markers).
4. AA-LCR: add random-chunks-in-source-order and BM25 controls at equal budget.
   Apply the v1.1 keys and a 16K output limit, and run 3 repeats. This turns the
   one real result into a solid one.

**Phase 1: the memo layer on the pipeline (2–3 weeks).**

1. Key memo entries by chunk-content hashes. Remove the source-hash and
   empty-hash wildcards.
2. Split entry kinds. Question-agnostic extraction ("what does chunk c say about
   entities/numbers/events") is reusable across questions. Question-specific
   findings and negative evidence are reusable across questions of the same
   *task class*. Final answers are replay-only and never partials.
3. Add a `cache_read` hook in `execution/pipeline.py` that consults the memo
   before routing. Write back from every retrieval call via structured leaf
   output (`relevant? / finding / span`).
4. Verifier: equivalence plus evidence-support plus source version. It must fail
   closed and block writes on failure.

**Phase 2: planner and orchestrator view (2–3 weeks).**

1. Implement the escalation ladder (6.2) with an explicit per-plan cost model.
2. Render the memo map into the orchestrator prompt (read side).
3. Plug `rlms` in as plan 5 for aggregation-type tasks only.

**Phase 3: amortized evaluation (the paper's centrepiece).**

1. **Stream protocol.** Order questions per corpus and report cumulative
   tokens/calls/latency *per correct answer* vs question index, averaged over
   several random orders.
   - Candidate corpora: AA-LCR (about 3.3 questions per document set), OOLONG
     (aggregation, where per-chunk labels should pay off most), LongBench-v2
     documents with several distinct questions, MRCR with multiple
     conversations per band.
2. **Hard negatives.** Include different questions on the same document to
   measure false reuse and error propagation (gap 5).
3. **Drift.** Edit spans mid-stream to test automatic invalidation (gap 4).
4. **Baselines at matched budgets:**
   - full context
   - middle truncation
   - DOS-RAG-style retrieve+pack
   - prefix/prompt caching
   - vCache-style answer caching
   - plain `rlms`
5. **Ablations:**
   - no memo
   - exact replay only
   - + question-agnostic extraction
   - + negative evidence
   - + orchestrator view
   - + verifier

### 6.5 Risks

- **Error propagation.** A wrong memo entry is reused everywhere. Mitigate with
  evidence-gated verification, and measure it explicitly.
- **Cold-start cost.** Question-agnostic extraction is expensive up front.
  Efficiency Frontier / sleep-time results imply a break-even N. We must report
  where it lies.
- **Novelty pressure.** Chained RLM, Prime Agent, GroundedCache and
  workload-aware caching are close neighbours. Re-run the citation-graph walk
  (Section 7) before writing, since the field moves monthly.
- **Nondeterminism.** Serving differed across identical prompts (LongBench
  hybrid). Pin vLLM settings and report repeat variance.

### 6.6 Decisions needed (Adarsh + Engin)

1. Adopt this direction, or keep the "budgeted retrieval" framing from Engin's
   status doc as the main line?
2. Which branch becomes the base? Recommendation: Engin's branch, since it holds
   `execution/`. Port `language_memoization.py` onto it rather than merging
   `main`.
3. Venue and timeline. `docs/workshop_venues_2026.md` on `main` lists options.
   The May paper drafts use the memoization framing and include none of the
   benchmark results.

---

## 7. Research workflow going forward

- Use the citation-graph skill (`.claude/skills/citation-graph-research/`, also
  installable at `~/.claude/skills/`). Seed with RLM (`arXiv:2512.24601`) plus
  one caching paper (GroundedCache or vCache). Walk forward citations with
  keywords `cache memo reuse amortiz persistent`, and treat papers that cite
  both seeds as the core.
- Priority reads to verify (many claims below are snippet-level):
  - the final RLM paper (v3)
  - Chained RLM
  - GroundedCache
  - the RLM reproduction (2603.02615)
  - SRLM (2603.15653)
  - Efficiency Frontier (2605.23071)
  - Sleep-time Compute (2504.13171)
  - DOS-RAG (2506.03989)
  - Workload-aware caching (2607.20495)
  - Prime Agent (2608.23552)

---

## 8. Sources and confidence

**Confidence tags:**

- **[read]** primary page read in-session
- **[snippet]** search-engine summary of the cited URL
- **[inferred]** our reasoning

**Caveats:**

- This session's network proxy blocked arXiv, Semantic Scholar and OpenAlex.
  Most paper claims are **[snippet]** and must be verified before citing.
- Code findings come from reading, not execution. There was no Python
  environment with dependencies, and `benchmark_artifacts/` on `main` is not in
  git. Two findings were verified by directly reading the code: the cache index
  bug and the memo final-answer leak.

**RLM line:**

- arXiv 2512.24601 (v1–v3) https://arxiv.org/abs/2512.24601 [snippet]
- Blog https://alexzhang13.github.io/blog/2025/rlm/ [snippet]
- Code https://github.com/alexzhang13/rlm [read]
- Releases https://github.com/alexzhang13/rlm/releases [read]
- DSPy RLM https://github.com/stanfordnlp/dspy/blob/main/dspy/predict/rlm.py [read]
- Prime Agent https://arxiv.org/abs/2608.23552 [snippet]
- Headlong https://github.com/laude-institute/headlong [snippet]
- Prime Intellect RLMEnv https://www.primeintellect.ai/blog/rlm [snippet]
- Reproduction https://arxiv.org/abs/2603.02615 [snippet]
- SRLM https://arxiv.org/abs/2603.15653 [snippet]
- Chained RLM https://arxiv.org/abs/2608.05124 [snippet]

**Long context vs RAG:**

- Context Rot https://research.trychroma.com/context-rot [snippet]
- NoLiMa https://arxiv.org/abs/2502.05167 [snippet]
- LongBench v2 https://arxiv.org/abs/2412.15204 [snippet]
- HELMET https://arxiv.org/abs/2410.02694 [snippet]
- OOLONG https://arxiv.org/abs/2511.02817 [snippet]
- LOFT https://arxiv.org/abs/2406.13121 [snippet]
- LC vs RAG https://arxiv.org/abs/2501.01880 [snippet]
- LaRA https://arxiv.org/abs/2502.09977 [snippet]
- DOS-RAG https://arxiv.org/abs/2506.03989 [snippet]
- AA-LCR https://artificialanalysis.ai/articles/announcing-aa-lcr [snippet]
- Claude Opus 4.6 https://www.anthropic.com/news/claude-opus-4-6 [read]

**Agentic context:**

- Chain-of-Agents https://research.google/blog/chain-of-agents-large-language-models-collaborating-on-long-context-tasks/ [snippet]
- ReadAgent https://arxiv.org/abs/2402.09727 [snippet]
- MemWalker https://arxiv.org/abs/2310.05029 [snippet]
- Anthropic context engineering https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents [snippet]
- ACE https://arxiv.org/abs/2510.04618 [snippet]
- Recursive-reasoning state/termination https://arxiv.org/abs/2605.06690 [snippet]

**Structured corpora:**

- RAG vs GraphRAG https://arxiv.org/abs/2502.11371 [snippet]
- GraphRAG-Bench https://arxiv.org/abs/2506.05690 [snippet]
- HippoRAG 2 https://arxiv.org/abs/2502.14802 [snippet]
- LightRAG https://github.com/hkuds/lightrag [snippet]
- AnnoIndex https://arxiv.org/abs/2608.13384 [snippet]

**Memory:**

- LongMemEval https://arxiv.org/abs/2410.10813 [snippet]
- MemoryAgentBench https://arxiv.org/abs/2507.05257 [snippet]
- Mem0 https://arxiv.org/abs/2504.19413 [snippet]
- Zep https://arxiv.org/abs/2501.13956 [snippet]
- MemoryOS https://arxiv.org/abs/2506.06326 [snippet]
- A-Mem https://arxiv.org/abs/2502.12110 [snippet]
- Sleep-time Compute https://arxiv.org/abs/2504.13171 [snippet]
- Beyond the Context Window https://arxiv.org/abs/2603.04814 [snippet]

**Reuse and caching:**

- vCache https://arxiv.org/abs/2502.03771 [snippet]
- Krites https://arxiv.org/abs/2602.13165 [snippet]
- GroundedCache https://arxiv.org/abs/2605.27494 [snippet]
- Contextual-summary caching https://arxiv.org/abs/2505.11271 [snippet]
- CacheBlend https://arxiv.org/abs/2405.16444 [snippet]
- EPIC https://arxiv.org/abs/2410.15332 [snippet]
- Prompt caching for agents https://arxiv.org/abs/2601.06007 [snippet]
- Agentic Plan Caching https://arxiv.org/abs/2506.14852 [snippet]
- Workload-aware caching https://arxiv.org/abs/2607.20495 [snippet]
- Agent Workflow Memory https://arxiv.org/abs/2409.07429 [snippet]
- Metacognitive Reuse https://arxiv.org/abs/2509.13237 [snippet]
- KVFlow https://arxiv.org/abs/2507.07400 [snippet]
- SCBench https://arxiv.org/abs/2412.10319 [snippet]
- Efficiency Frontier https://arxiv.org/abs/2605.23071 [snippet]

---

## 9. Handoff notes for the next session

- Check out `claude/understand-repo-structure-3UiKx` (this doc + skill) and
  fetch `origin/shared-execution-linux-runners` (Engin's code). A convenient
  setup: `git worktree add ../rlms-engin origin/shared-execution-linux-runners`.
- Start with Phase 0 items 2–3 on a new branch cut from Engin's branch. They are
  small, testable, and unblock everything else.
- Tests on Engin's branch: `bash linux/setup.sh client` then
  `.venv/bin/python -m unittest discover -s test` (210 tests, mocked).
- Verify the [snippet] citations with the citation-graph skill before using any
  number in writing.
