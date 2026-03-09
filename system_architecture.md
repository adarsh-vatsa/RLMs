# Two-Stage Semantic Cache — System Architecture

> **File**: [`semantic_cache_system.py`](file:///Users/zeitgeist/research/RLMs/semantic_cache_system.py) (1941 lines)
> **Dependencies**: `transformers`, `torch`, `faiss-cpu`, `anthropic`, `python-dotenv`, `numpy`
> **Local Models**: `Qwen3-Embedding-0.6B` (596M params, 1024-dim embeddings), `Qwen3-Reranker-0.6B` (yes/no cross-encoder)
> **API Models**: `claude-sonnet-4-5` (execution/synthesis), `claude-haiku-4-5` (evaluation/sniper/consensus/knowledge extraction)
> **Python**: 3.9+

---

## System Topology

```
┌──────────────────────────────────────────────────────────────────┐
│                     AGENT LAYER (Framework-Agnostic)              │
│  AutonomousAgent.cached_query(query, context)                    │
│  └─ Replaces direct LLM API calls for any framework             │
└────────┬────────────────────────────────────────────────┬────────┘
         │ CACHE HIT                            CACHE MISS │
         ▼                                                 ▼
┌────────────────────────────┐        ┌────────────────────────────┐
│  SemanticCacheController   │        │     Router.select_model()  │
│  ┌──────────────────────┐  │        │  ┌──────────────────────┐  │
│  │ 1. Hash Bucket       │  │        │  │ Keyword heuristic    │  │
│  │ 2. Exact Match       │  │        │  │ Simple → Haiku       │  │
│  │ 3. Vector Dragnet    │  │        │  │ Complex → Sonnet     │  │
│  │    (Qwen3 + FAISS)   │  │        │  └──────────────────────┘  │
│  │ 4. LLM Sniper       │  │        └────────────┬───────────────┘
│  │ 5. Knowledge Lookup  │  │                     │
│  │ 6. Collapse Guard    │  │                     ▼
│  │ 7. Provenance Check  │  │        ┌────────────────────────────┐
│  └──────────────────────┘  │        │     Anthropic API          │
└────────────────────────────┘        │  store() → cache + embed   │
                                      │  + knowledge extraction    │
                                      │  + consensus verify        │
                                      └────────────────────────────┘
```

### Retrieval Pipeline (for document search via domain clients)

```
┌──────────────────────────────────────────────────────────────────┐
│                   SEARCH PIPELINE (search())                      │
│                                                                    │
│   Query ──┬──→ Cache Check (exact / semantic / knowledge fact)     │
│           │     ├─ HIT  → Serve cached answer (with provenance)   │
│           │     └─ MISS ↓                                         │
│           └──→ FAISS Dragnet (top-20, ~130ms on CPU)              │
│                    ↓                                               │
│              Qwen3-Reranker (relevance gate, top-5)               │
│                    ↓                                               │
│              Sonnet Synthesis (grounded answer)                    │
│                    ↓                                               │
│              Grounding Check → Consensus Verify → Cache Store     │
│                    ↓                                               │
│              Knowledge Extraction → Fact FAISS Index              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Core Infrastructure Components

### 0a. EmbeddingEngine (Qwen3-Embedding-0.6B)
**Lines 73–125** · Local 596M-parameter embedding model.

| Property | Value |
|----------|-------|
| Model | `Qwen/Qwen3-Embedding-0.6B` |
| Output dim | 1024 |
| Pooling | CLS token (`last_hidden_state[:, 0, :]`) |
| Normalization | L2-normalized (cosine similarity via inner product) |
| Max length | 8,192 tokens (32K context supported) |
| Device | CPU (MPS has known segfault issues with some architectures) |

**Instruction-aware encoding**: Queries are prefixed with a retrieval instruction; documents are encoded raw. This asymmetric encoding is standard for Qwen3.

```python
# Query encoding (with instruction)
encode_query("What charges did Maxwell face?")
# → "Instruct: Given a legal query, retrieve relevant court documents\nQuery: What charges..."

# Document encoding (no instruction)
encode_documents(["The defendant was charged with..."])
```

**Replaces**: `all-MiniLM-L6-v2` (SentenceTransformers). Qwen3 provides 1024-dim vs 384-dim, 8K context vs 512, and instruction-awareness.

---

### 0b. FAISSIndex
**Lines 132–195** · Vector similarity search replacing brute-force numpy.

- `IndexFlatIP` (inner product on L2-normalized vectors = exact cosine similarity)
- **Lazy import**: FAISS is imported on first use (`_ensure_faiss()`) to avoid segfault conflicts with `transformers` on Apple Silicon
- Parallel metadata array tracks per-vector metadata (filename, chunk index, text, etc.)
- Persistence via `faiss.write_index()` / `faiss.read_index()` + JSON metadata sidecar
- Performance: ~130ms for top-20 search on 1,420 vectors (CPU)

**Replaces**: `sklearn.metrics.pairwise.cosine_similarity` + raw numpy loops.

---

### 0c. Reranker (Qwen3-Reranker-0.6B)
**Lines 202–268** · Cross-encoder that sits between the Vector Dragnet and LLM Synthesis.

- **Architecture**: Generative cross-encoder using `AutoModelForCausalLM`
- **Scoring**: Extracts `yes`/`no` token log-probabilities → `softmax → P(yes)` = relevance score
- **Relevance gate**: Configurable threshold (default 0.5) filters irrelevant documents
- **Chat template**: Uses Qwen3's built-in reranker prompt format with `<Instruct>`, `<Query>`, `<Document>` tags
- **Cost**: $0 (runs locally)
- **Latency**: ~10.9s for 20 candidates on CPU (benefits significantly from GPU)

> **Why a reranker?** FAISS does similarity on 1024-dim embeddings — good for recall but imprecise for ranking. The cross-encoder processes the full query-document pair jointly, achieving much higher ranking precision. This is the industry-standard two-stage retrieval pattern (bi-encoder → cross-encoder).

---

## Cache Components

### 1. ExecutionMetrics
**Lines 274–363** · Tracks every API call, cache event, cost, and provenance result.

| Counter | What |
|---------|------|
| `exact_hits` / `semantic_hits` / `cache_misses` | Cache performance |
| `parallel_chunk_evaluations` | Times parallel Sniper chunking was used |
| `ephemeral_retrievals` | Large results served but flagged "don't persist" |
| `recursive_summarizations` | Oversized results chunked and summarized |
| `grounded_results` / `partial_results` / `inferred_results` | Source provenance verification |

**Cost model**: Real Anthropic pricing — Sonnet at `$3/$15` per MTok, Haiku at `$0.25/$1.25` per MTok.

---

### 2. SemanticCacheController
**Lines 369–1429** · The core engine. Contains 14 distinct mechanisms:

#### 2a. Hash Bucketing (`_get_chunk_hash`)
**Line 429** · MD5 hash of the source data chunk.
- Guarantees cache entries are never compared across different source documents
- Prevents cross-document hallucination

> **Scenario**: An agent sweeps 10,000 court documents asking "Did this party commit fraud?" on each one. Page 42 says "Yes" and Page 99 says "No". Without hash bucketing, the cache might return Page 42's answer for Page 99's query because the *questions* are identical — only the data differs. The hash ensures each page's answers stay isolated.

#### 2b. Exact Match (inside `check()`)
**Line 784** · Free O(N) string scan within the bucket.
- Case-insensitive `.lower()` comparison
- Zero API cost, instant return

> **Scenario**: An RLM writes a Python `for` loop that calls `rlm_query("Classify this entry as ERROR or INFO", chunk)` on 500 log entries. 200 of the entries are identical (duplicated logs). The exact-match layer catches all 200 repeats with zero API cost — no embeddings, no Haiku calls.

#### 2c. Vector Dragnet (`_vector_dragnet`)
**Line 444** · Qwen3-Embedding-0.6B + FAISS for local similarity search.
- 1024-dim embeddings via `encode_single()` (CLS token pooling, instruction-aware)
- Cosine similarity via FAISS `IndexFlatIP` on L2-normalized vectors
- Returns Top-K (default: 5) candidates above 0.3 threshold
- **Cost**: $0 (runs locally)

> **Scenario**: On Monday, the agent asks "What is the MRR?" On Wednesday, a different analyst triggers "Report the Monthly Recurring Revenue figure." Exact match fails (different words), but the Dragnet catches high cosine similarity (~0.91) and passes both to the Sniper for evaluation.

#### 2d. LLM Sniper (`_llm_sniper_evaluate`)
**Line 472** · Sends Top-K candidates to `claude-haiku-4-5` for semantic equivalence.
- Structured JSON prompt: "Does the new query ask for the EXACT same information?"
- Returns `{"hit": true/false, "id": index}`
- **Cost**: ~$0.0001 per evaluation

> **Scenario (Hit)**: "What is the EBITDA?" vs. "What was the EBITDA figure?" — Haiku recognizes these ask for identical information → cache hit, $0.0001.
>
> **Scenario (Critical Rejection)**: "Show logs that INCLUDE timeout errors" vs. "Show logs that EXCLUDE timeout errors" — vector similarity is ~0.95, but Haiku catches the logical inversion → correct miss, prevents serving the opposite answer. This is the scenario that breaks every pure-vector cache.

#### 2e. Parallel Sniper Chunking (`_parallel_sniper_evaluate`)
**Line 539** · Prevents evaluator Context Rot at scale.
- If candidates exceed `PARALLEL_BATCH_SIZE` (5), chunks them into batches
- Fires all batches via `ThreadPoolExecutor` simultaneously
- First batch to return `{"hit": true}` wins; remaining futures cancelled
- **Scaling**: O(1) wall-clock time

> **Scenario**: After 6 months of operation, a financial cache has 100,000 entries. A new query about "Q3 revenue" matches 500 near-miss candidates via vector search. Stuffing 500 options into a single Haiku prompt would cause attention degradation ("Lost in the Middle"). Instead, the system chunks them into 100 batches of 5 and fires 100 parallel Haiku calls — all return in ~500ms with zero accuracy loss.

#### 2f. Context Collapse Guard (`_apply_context_collapse_guard`)
**Line 642** · Two protection tiers:

| Result size | Behavior | Flag |
|-------------|----------|------|
| < 2,000 tokens | Normal return | — |
| 2,000–4,000 tokens | Serve but flag as **ephemeral** | `ephemeral: True` |
| > 4,000 tokens | **Recursive parallel summarization** | `was_summarized: True` |

> **Scenario (Ephemeral)**: An agent asks "Summarize the performance review" and the cached result is a 2,500-token summary. The system serves it but says `ephemeral: True` — the calling agent should use this answer for the current step but NOT append it to its rolling context history, preventing context window bloat.
>
> **Scenario (Terminal Overflow)**: An RLM sweeps 10,000 documents asking for "summarize the legal risks." Each sub-agent returns a paragraph. When the root LLM wakes up, `results` contains ~1M tokens. Without the guard, the root LLM tries to read the terminal and crashes from context overflow. With the guard, each cached result is compressed to ~200 tokens before returning.

#### 2g. Recursive Parallel Summarization (`_recursive_summarize`)
**Line 701** · Logarithmic compression of oversized results.
- Chunks oversized text into ~1000-token pieces
- Summarizes each chunk via **parallel** Haiku sub-agent calls
- **Truly recursive**: checks if joined summary still exceeds threshold → recurses
- Tighter prompts at deeper levels ("condense to 1-2 sentences")
- Safety cap at depth 5 with hard truncation fallback
- **Scaling**: Logarithmic — 100K tokens → 2 levels → ~400 tokens

> **Scenario**: A previous agent run produced a 100K-token due diligence report and cached it. A new query hits the cache and retrieves it. The system splits it into 25 chunks of 4K tokens each. 25 parallel Haiku calls each produce ~200 tokens → 5,000 token joined summary. Still too large. Recurse: split into 2 chunks → 2 parallel calls → ~400 token final summary. Total: 2 recursive levels, all parallel. The agent receives a crisp summary instead of a context-destroying wall of text.

#### 2h. Source Provenance & Grounding (`_grounding_check`)
**Line 586** · Zero-cost fact verification via regex.
- Extracts quantitative facts from result via regex:
  - Dollar amounts (`$12.4M`, `$500K`)
  - Percentages (`4.2%`, `112%`)
  - Large numbers with commas (`2,847`)
- Checks each fact against the original source text
- Tags result as `GROUNDED` / `PARTIAL` / `INFERRED`
- **Cost**: $0 (regex + substring matching)

> **Scenario (GROUNDED)**: The LLM extracts "ARR is $12.4M" from a 10-K filing. The grounding check finds `$12.4M` in the original source text → `GROUNDED`. This fact is now locked in the cache and will never hallucinate on subsequent queries.
>
> **Scenario (INFERRED — hallucination caught)**: The LLM extracts "revenue was $14.2M" but the source text only contains "$12.4M". The grounding check flags this as `INFERRED` with `unverified_facts: ["$14.2M"]` → the downstream system can surface a warning or trigger re-extraction before the wrong number enters the cache.
>
> **Scenario (PARTIAL)**: The LLM says "ARR grew from $12.4M to $15.1M." The source contains `$12.4M` but not `$15.1M` → `PARTIAL`. The first figure is verified; the second needs human review.

#### 2i. Multi-Model Consensus on Write (`consensus_verify`)
**Line 901** · Every cache write is verified by a second model.

- After the primary model (Sonnet) generates an answer, Haiku independently answers the same query
- Quantitative facts are extracted from both answers via regex
- If both models produce the same numbers → `consensus: AGREED`
- If they diverge → `consensus: DISPUTED`, divergent facts surfaced
- **Cost**: ~$0.0001 per write (one extra Haiku call)

**Why consensus must be continuous, not one-time:**

The cache updates results on every subsequent cache miss. If we ran consensus only on the "first write," every subsequent miss would enter the cache unverified. Since the architecture generates new writes constantly — through cold misses, pre-warming sweeps (100+ entries at once), and recursive summarization intermediates — consensus must be a **property of the write path**, not a one-time ceremony. Every entry earns its trust at write-time, whether it's the first query or the ten-thousandth.

> **Scenario (AGREED)**: Sonnet extracts "ARR is $12.4M" from a financial filing. Haiku independently answers the same query and also says "$12.4M." The facts match → `AGREED`. The number is now triple-verified: grounded in source text AND confirmed by two independent models.
>
> **Scenario (DISPUTED)**: Sonnet says "revenue grew by 23%." Haiku says "revenue grew by 18%." The divergent fact `["23%"]` is surfaced → the system can flag for human review, re-extract with a more explicit prompt, or hold the entry in a `DISPUTED` state until resolved.

#### 2j. Store with Provenance (`store`)
**Line 853** · Each cache entry contains:
```python
{
    "query": str,            # The original query text
    "result": str,           # The LLM's answer
    "embedding": ndarray,    # Qwen3 1024-dim vector (CLS token pooling)
    "source_context": str,   # Original source text for verification
    "model_used": str,       # Which model generated this result
    "grounding_info": dict,  # Pre-computed grounding verification
    "sources": list,         # Source document references
}
```

On store, the system also:
1. Embeds the query and adds it to the `_cache_index` (FAISS) for fast similarity lookup
2. Triggers knowledge extraction (see 2k)

#### 2k. Knowledge Extraction on Write (`_extract_facts`)
**Line 983** · Decomposes answers into atomic triples for cross-query reuse.

On every cache write, the evaluator model (Haiku) decomposes the synthesized answer into **atomic (subject, relation, object) triples**. Each triple is independently embedded and indexed in a separate FAISS knowledge index.

**Why this matters**: A monolithic cache stores `query → answer` blobs. If an answer about "charges" also mentions the judge, defense attorney, and sentencing, that knowledge is *buried* and only accessible via the original query. Knowledge extraction makes every fact independently discoverable.

```
Write path:
  LLM Answer → Haiku decomposes → (subject | relation | object) triples
  Each triple → embedded via Qwen3 → indexed in knowledge FAISS

Read path (inside search()):
  Exact match? → Sniper semantic match? → Knowledge fact lookup → Full pipeline miss
```

**Example**: Caching an answer about "What charges did Maxwell face?" extracts:
```
Ghislaine Maxwell | charged with     | sex trafficking of minors
Ghislaine Maxwell | charged under    | 18 U.S.C. § 2423
Ghislaine Maxwell | sentenced to     | 3 + 3 + 5 years
Judge             | Maxwell case     | Alison J. Nathan
Defense attorney  | Maxwell          | Christian R. Everdell
```

Now "Who was Maxwell's lawyer?" hits the knowledge index → serves cached answer at **$0.00** — even though the original cached query was about *charges*, not *attorneys*.

**Cost**: ~$0.0002 per write (one Haiku call for extraction).

---

### Retrieval Layer (New)

#### 2l. Document Ingestion (`ingest`)
**Line 1036** · Chunks documents → embeds → builds FAISS doc index.

- Reads `.txt` files from a directory
- Chunks into ~3000-character pieces with 200-character overlap at sentence boundaries
- Encodes all chunks via `encode_documents()` (Qwen3, no instruction prefix)
- Builds `doc_index` (FAISS) with per-chunk metadata (filename, chunk_index, text, etc.)
- Persists via `save_doc_index()` which writes `index.faiss`, `metadata.json`, `chunks.json`, and `corpus_config.json`

#### 2m. Document Retrieval with Reranker (`retrieve`)
**Line 1077** · FAISS dragnet → Qwen3-Reranker relevance gate.

1. Encode query via `encode_query()` (with instruction prefix)
2. FAISS search for top-20 candidates (~130ms)
3. Extract text from FAISS metadata for each candidate
4. Rerank via Qwen3-Reranker-0.6B with relevance threshold (default 0.5)
5. Return top-5 reranked results with scores and metadata

> **Why reranker has a relevance gate**: Unlike the Sniper (which checks semantic equivalence of cache queries), the Reranker checks *relevance* of documents to a query. The 0.5 threshold ensures completely irrelevant documents (e.g., about bail conditions when asking about charges) are filtered, even if FAISS gave them a high vector similarity score.

#### 2n. Full Search Pipeline (`search`)
**Line 1130** · The main entry point for domain-specific clients.

```
search("What charges did Maxwell face?")
  │
  ├─► Cache Check (check()) → if HIT → return cached answer
  │     ├─ Exact match (free)
  │     ├─ Semantic match via Sniper ($0.0001)
  │     └─ Knowledge fact lookup (free, FAISS)
  │
  ├─► Cache MISS:
  │     ├─ retrieve() → FAISS + Reranker (top-5)
  │     ├─ Sonnet synthesis from top-3 sources
  │     ├─ Grounding check (free)
  │     ├─ Consensus verify ($0.0001)
  │     ├─ store() → cache + embed + fact extract
  │     └─ Return answer with provenance
  │
  └─► Return: {answer, results, timing, grounding, consensus, from_cache, relevant_facts}
```

---

### Persistence Layer (New)

#### 2o. Save (`save`)
**Line 1257** · Persists the full system state:
- `corpus_config.json` — namespace identity (corpus_id, domain, counts, model info, timestamps)
- `cache_entries.json` — all cached query→answer pairs with embeddings
- `knowledge.json` — all extracted (subject, relation, object) triples
- `cache_idx/` — FAISS index of cache query embeddings
- `knowledge_idx/` — FAISS index of knowledge triple embeddings

#### 2p. Load (`load`)
**Line 1306** · Restores full system state with validation:
- **Corpus identity validation**: If the controller has a `corpus_id`, it must match the stored config. Mismatches are **refused** (prevents cross-namespace contamination).
- Loads document FAISS index (supports both `doc_idx/` subdirectory and root-level layout)
- Loads cache entries and rebuilds cache FAISS index if missing
- Loads knowledge triples and rebuilds knowledge FAISS index if missing
- **Fallback**: If `chunks.json` is missing (old index format), extracts chunks from FAISS metadata

#### 2q. Corpus-Level Namespace Isolation
**`__init__` accepts `corpus_id` and `corpus_domain`** · Infrastructure-only namespace tagging.

Each corpus (Epstein docs, finance filings, medical records) gets its own:
- FAISS document index (embeddings)
- Cache entries (query-answer pairs)
- Knowledge graph (extracted triples)
- Configuration (`corpus_config.json`)
- Separate directory on disk

```python
# Epstein legal documents
cache = SemanticCacheController(metrics, embedder, reranker,
    corpus_id="epstein", corpus_domain="legal")
cache.load("epstein_index/")

# Financial records (completely separate namespace)
cache = SemanticCacheController(metrics, embedder, reranker,
    corpus_id="finance_q1", corpus_domain="finance")
cache.load("finance_index/")

# Cross-wiring protection:
cache = SemanticCacheController(metrics, embedder, reranker,
    corpus_id="epstein", corpus_domain="legal")
cache.load("finance_index/")
# → [CORPUS] ⚠ MISMATCH: expected 'epstein', found 'finance_q1'
# → Refusing to load — wrong namespace.
```

The `corpus_id` is purely infrastructural — it **NEVER enters any LLM prompt**, consuming zero tokens and introducing zero context pollution.

---

### 3. CachePreWarmer
**Lines 1433–1488** · Solves the cold-start problem.

- Takes an agent, a corpus (list of chunks), and a list of template queries
- Programmatically sweeps every query × chunk combination
- Populates the cache before any human touches the system

> **Scenario**: A PE firm uploads a 50-page financial model at 8am. The system immediately deploys a pre-warming sweep with template queries: "What is the ARR?", "What is the EBITDA?", "What is the churn rate?", etc., across every page. By 8:10am, the cache is 100% saturated. When the analyst logs in at 8:15am, every question they ask is an instant, deterministic cache hit — zero latency, zero hallucination risk.
>
> **Scenario (Epstein files)**: 10,000 court documents are uploaded. The pre-warmer sweeps with templates: "List all individuals mentioned", "Extract all dates", "Identify financial transactions", "Flag references to locations." The cache is saturated within hours. Every subsequent analyst query about names, dates, or locations hits the cache instantly with grounded, verified facts.

---

### 4. Router
**Lines 1494–1513** · Heterogeneous model dispatch.

- Keyword heuristic: `classify`, `extract`, `find`, `count`, `list` → Haiku ($0.25/MTok)
- Everything else → Sonnet ($3/MTok)
- **Future**: Could be a trained complexity classifier

> **Scenario**: An RLM loop fires 10,000 queries. 8,000 are simple extraction tasks ("Extract the date from this filing") and 2,000 are complex synthesis ("Summarize the legal risk exposure"). Without routing, all 10,000 go to Sonnet at $3/MTok. With routing, 8,000 go to Haiku at $0.25/MTok and 2,000 go to Sonnet. Cost drops by ~70% with no quality loss on the simple tasks.

---

### 5. AutonomousAgent
**Lines 1519–1597** · The transparent cache interceptor.

- `cached_query(query, context)` replaces direct API calls
- Framework-agnostic: works for RLMs, LangChain, AutoGen, or any custom agent
- On cache miss: routes to model, stores result with provenance, applies collapse guard
- Returns a rich dict: `{result, from_cache, ephemeral, was_summarized, grounding, verified_facts, unverified_facts}`

> **Scenario (Drop-in replacement)**: A LangChain agent currently calls `llm.invoke(prompt)` directly. By swapping to `agent.cached_query(query, context)`, the entire caching, routing, collapse protection, and provenance system activates with zero changes to the agent's logic. The agent doesn't need to know about caching — it just calls the function and gets a richer response.

---

### 6. Domain Client Pattern (epstein_search.py)

The library is designed for domain-agnostic reuse. Each domain client is a thin wrapper (~150–600 lines) that imports the core system and adds domain-specific config:

```python
# epstein_search.py — thin client
class EpsteinSearchEngine:
    def __init__(self):
        self.embedder = EmbeddingEngine()
        self.reranker = Reranker()
        self.faiss_index = FAISSIndex()
        self.cache = SemanticCache(self.embedder)
        self.faiss_index.load(INDEX_DIR)

    def search(self, query):
        # Cache check → FAISS Dragnet → Reranker → Sonnet synthesis → Cache store
```

The same library can serve: legal filings, financial documents, medical records, compliance reports — each in an isolated corpus namespace.

---

## Configuration Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Local embedding model |
| `RERANKER_MODEL` | `Qwen/Qwen3-Reranker-0.6B` | Local cross-encoder reranker |
| `EMBEDDING_DIM` | 1024 | Embedding vector dimension |
| `EXECUTOR_MODEL` | `claude-sonnet-4-5` | Primary synthesis model |
| `EVALUATOR_MODEL` | `claude-haiku-4-5` | Sniper, consensus, knowledge extraction |
| `TOP_K_CANDIDATES` | 5 | Max vector search results for cache lookup |
| `PARALLEL_BATCH_SIZE` | 5 | Max candidates per Sniper evaluation call |
| `EPHEMERAL_TOKEN_THRESHOLD` | 2,000 | Flag results above this as ephemeral |
| `MAX_CONTEXT_TOKENS` | 4,000 | Trigger recursive summarization above this |
| `MAX_RECURSION_DEPTH` | 5 | Safety cap on summarization recursion |

---

## What the Return Value Looks Like

Every `cached_query()` call returns:

```python
{
    "result": "The extracted answer text...",
    "from_cache": True,               # Was this a cache hit?
    "ephemeral": False,               # Should the agent avoid persisting this?
    "was_summarized": False,          # Was the result condensed?
    "grounding": "GROUNDED",          # GROUNDED | PARTIAL | INFERRED
    "verified_facts": ["$12.4M"],     # Facts found in source
    "unverified_facts": [],           # Facts NOT in source (⚠ hallucination risk)
}
```

Every `search()` call returns:

```python
{
    "query": "What charges did Maxwell face?",
    "answer": "Based on the source documents...",
    "results": [                       # Reranked source documents
        {"text": "...", "score": 0.87, "metadata": {"filename": "DOJ-OGR-00001229.txt"}},
    ],
    "timing": {"cache_ms": 2.1, "dragnet_ms": 130, "rerank_ms": 10900, "synthesize_ms": 3200},
    "from_cache": False,
    "grounding": {"grounding": "GROUNDED", "verified_facts": [...], "unverified_facts": []},
    "consensus": {"consensus": "AGREED", "divergent_facts": []},
    "relevant_facts": [                # Knowledge triples that matched
        {"subject": "Maxwell", "relation": "charged with", "object": "sex trafficking"},
    ],
}
```

---

## Use Cases

### Core Capabilities (What the architecture gives you)

#### 1. Infinite Context Processing
Any corpus, any size. The agent only ever sees bounded context windows, but the cache + recursive summarization means it can process and retrieve information from arbitrarily large datasets without context overflow. Combined with any model — Sonnet, Gemini, GPT — the cache sits *underneath* the LLM and makes the context window irrelevant. This is not "bigger context windows" (the hardware war) — it's "virtual memory for LLMs" (the systems war).

#### 2. Framework-Agnostic Drop-In
`cached_query(query, context)` replaces any direct LLM API call. Works with RLMs, LangChain, AutoGen, CrewAI, bare API scripts, or any custom agent. The agent doesn't need to know about caching. One function swap and the entire system activates.

#### 3. Hallucination Firewall (Provenance + Grounding)
Every cached result is verified at write-time against its source text. Numbers, dollar amounts, and percentages must appear in the original document or they get flagged `INFERRED`. In high-stakes domains, this turns a probabilistic system into a deterministic lookup — the cache doesn't guess, it serves verified facts.

**Human-in-the-Loop Extension**: At the end of each pre-warming run, the system can surface all `INFERRED` and `PARTIAL` results to a human reviewer via the terminal for manual verification before they enter the production cache. One review pass → permanent correctness.

#### 4. Cost Collapse (O(N) → O(1))
The cache eliminates redundant API calls entirely. The Sniper evaluator uses Haiku (~$0.0001/call) to prevent Sonnet calls (~$0.01). A single Sonnet call costs ~100 Haiku evaluations. And honestly, Haiku might be overkill — a fine-tuned tiny model or even a deterministic classifier could replace the Sniper at near-zero cost for well-defined domains.

#### 5. Deterministic Reproducibility
For regulated industries (finance, healthcare, government), you need to prove your AI gave the same answer when asked the same question twice. Without a cache, LLMs are non-deterministic even at temperature=0 (due to floating-point batching). The cache guarantees identical outputs for equivalent queries — producing an auditable, reproducible record of every computation.

#### 6. Cold-Start Elimination
Standard caches start empty and take weeks/months to saturate via organic user queries. Pre-warming eliminates this: the system is profitable from Minute 1. An RLM pre-warming sweep generates the redundancy needed to saturate the cache in a single session.

#### 7. Context Rot Immunity
Even models with 1M token windows (Gemini) suffer from attention degradation on information buried in the middle. This architecture guarantees that every sub-query operates on a small, focused chunk with full attention fidelity — regardless of how large the total corpus is.

#### 8. Cross-Query Knowledge Reuse (New)
Traditional caches only match queries that ask the *same* question. Knowledge extraction enables matches for queries that ask about *related* information. A query about "Who was Maxwell's attorney?" can be answered from a cache entry originally about "What charges did Maxwell face?" — because the attorney's name was extracted as an independent $(s, r, o)$ triple.

#### 9. Multi-Domain Deployment (New)
Corpus-level namespace isolation means one installation can serve legal, financial, medical, and engineering teams simultaneously. Each corpus has its own FAISS indices, cache, and knowledge graph. No cross-contamination, no token pollution.

---

### Domain Applications

#### Finance & Private Equity
- **10-K / Annual Report Sweeps**: Pre-warm with "ARR?", "EBITDA?", "Churn?", "Gross Margin?" across every page. Analysts get instant, grounded, deterministic answers.
- **Due Diligence**: Sweep a target company's filings. Lock in every number. The cache becomes a verified, queryable database of the target's financials.
- **Multi-Portfolio Consistency**: A PE firm analyzing 50 portfolio companies. Same template queries across all 50 → massive cache saturation. Cross-company comparisons use identical extraction methodology.
- **Hallucination-Proof Numerical Stability**: The grounding check ensures "$12.4M" actually appears in the source. The cache turns the LLM into a deterministic database lookup for quantitative data.

#### Legal & Investigation
- **Mass Document Review**: Sweeping 10,000+ court filings, contracts, or depositions with template queries. Extract parties, dates, obligations, risk clauses.
- **Epstein Files / FOIA Dumps**: Pre-warm with "List individuals mentioned", "Extract dates", "Identify financial transactions", "Flag travel references." The cache becomes an instant, searchable index of the entire corpus.
- **Contract Comparison**: Same extraction queries across hundreds of vendor contracts. The cache catches identical boilerplate and only pays for genuinely unique clauses.
- **E-Discovery**: Reduce review costs by 90%+ when thousands of documents contain similar language. The Sniper prevents false matches on legally critical distinctions.

#### Medical & Clinical Research
- **Clinical Trial Data**: Sweeping patient records with "What was the dosage?", "What adverse events occurred?", "What was the outcome?" The grounding check is critical — you **cannot** hallucinate a drug dosage.
- **Literature Review**: Processing thousands of papers. Pre-warm with "What methodology was used?", "What dataset?", "What were the results?" The cache ensures consistent extraction across the entire corpus.
- **Diagnostic Support**: Repeated queries about symptoms and conditions. The cache locks in verified diagnostic criteria, preventing hallucination on medical facts.

#### Software Engineering
- **Large Codebase Analysis**: An agent sweeping a 100K+ line codebase for security vulnerabilities, code smells, or architecture violations. The same patterns (SQL injection, XSS, buffer overflow) appear across hundreds of files — the cache catches the redundancy.
- **Code Review at Scale**: Template queries like "Does this function handle errors?", "Is there input validation?", "Are there hardcoded secrets?" across every file in a repo.
- **Migration Planning**: Sweeping a legacy codebase with "What framework is this file using?", "What dependencies does it import?", "Is this compatible with Python 3.12?" Thousands of files, highly redundant answers.
- **CI/CD Test Determinism**: When running automated test suites that rely on LLM outputs, the cache ensures identical results between runs. No more flaky tests from LLM non-determinism.

#### Compliance & Audit
- **Regulatory Scanning**: Sweep policy documents against compliance checklists ("Does this policy address data retention?", "Is there a breach notification clause?"). Pre-warm the cache with the full regulatory checklist → instant compliance reports for new policies.
- **Audit Trail**: Every cache entry records the model used, the source text, and the grounding status. This creates a permanent, verifiable audit trail of every AI-generated conclusion.
- **Cross-Jurisdiction Consistency**: The same compliance queries applied to policies from 50 different jurisdictions. The cache ensures methodological consistency across all reviews.

#### Intelligence & OSINT
- **Open Source Intelligence**: Processing massive amounts of public records, social media posts, or news articles with extraction templates. High redundancy → rapid cache saturation.
- **Pattern Detection**: Pre-warm with entity extraction queries across a large corpus. The cached, grounded entity graph becomes a searchable intelligence database.

#### Multi-Tenant / Organizational
- **Shared Org-Level Cache**: Every user in an organization benefits from every other user's queries. The first analyst who asks "What is Q3 revenue?" pays full price. Every subsequent analyst across the firm gets it free and grounded.
- **Agent Swarm Memoization**: In multi-agent architectures, Agent A's cached results are available to Agent B. One agent's cognitive work feeds the entire swarm. A 10-agent system processing the same corpus doesn't do 10x the API calls — it does 1x, then 9 free cache hits.
- **Cross-Session Memory**: Unlike ephemeral context that dies when the process ends, the cache persists to disk. An agent's work on Monday is instantly available to any agent on Friday.

#### Education & Content
- **Tutoring at Scale**: Thousands of students asking variations of "Explain photosynthesis" or "What is the quadratic formula?" → massive semantic overlap → near-100% cache hit rate.
- **Content Moderation**: Processing millions of user posts. Similar content (spam, hate speech patterns) triggers cache hits. Grounding ensures moderation decisions are traceable to specific policy clauses.
- **Localization QA**: Checking translations across large document sets. The cache ensures consistent terminology — the same technical term gets the same translation every time.

---

## Contributions Summary

| # | Contribution | Novel? |
|---|-------------|--------|
| 1 | Two-Stage Dragnet + Sniper (vector net → LLM gate) | Extends Cortex/GPTCache |
| 2 | Parallel Top-K Chunking for evaluator Context Rot immunity | **Novel** |
| 3 | Context Collapse Guard (ephemeral + recursive summarization) | **Novel** |
| 4 | Truly recursive parallel summarization (logarithmic compression) | **Novel** |
| 5 | Programmatic Cache Pre-Warming (Day-1 saturation via agent sweep) | **Novel** |
| 6 | Source Provenance & Grounding Verification (zero-cost regex check) | **Novel** |
| 7 | Multi-Model Consensus on Write (continuous verification via Haiku) | **Novel** |
| 8 | Knowledge Extraction & Fact Indexing ($(s,r,o)$ triples in FAISS) | **Novel** |
| 9 | Qwen3 + FAISS + Reranker retrieval pipeline (bi-encoder → cross-encoder) | Established pattern, our integration |
| 10 | Corpus-Level Namespace Isolation (multi-domain deployment) | **Novel** |
| 11 | Full Persistence (cache + knowledge + FAISS indices to disk) | Architectural contribution |
| 12 | Heterogeneous Model Routing (cost-aware dispatch) | Established pattern, our integration |
| 13 | Framework-agnostic interceptor design | Architectural contribution |
