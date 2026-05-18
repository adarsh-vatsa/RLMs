# Cache Entries, Semantic Hits, and Knowledge Hits

This note summarizes how a cache entry is used in the repository without requiring a full read of the architecture documentation.

## What a Cache Entry Stores

When `SemanticCacheController.store()` writes a new result, it creates a cache entry with the generated answer and the metadata needed to reuse that answer safely:

```python
entry = {
    "query": query,
    "result": result,
    "embedding": embedding,
    "source_context": context,
    "model_used": model_used,
    "grounding_info": grounding_info,
    "data_scope_hash": data_scope_hash,
}
```

It then attaches optional extracted facts:

```python
entry["facts"] = facts
```

The key point is that the cache entry has two roles:

1. It is a reusable answer for exact and semantic cache hits.
2. It is the parent record for extracted knowledge triples used by knowledge cache hits.

## Field Usage

| Field | Purpose | How it is used later |
| --- | --- | --- |
| `query` | Original query that produced the answer. | Used for exact string matches and shown to the semantic Sniper when checking whether a new query is equivalent to a previous query. |
| `result` | Generated answer. | Returned directly when the cache hits. Exact, semantic, and knowledge hits all ultimately return a cached `result`. |
| `embedding` | Embedding of the original query. | Supports semantic lookup, especially in the bucketed `check()` path. In `search()`, the cache FAISS index separately stores query embeddings with cache metadata. |
| `source_context` | Source text or chunk used to produce the answer. | Used for provenance and grounding checks. It lets the system verify whether quantitative facts in the answer appear in the source. |
| `model_used` | Model that generated the answer. | Provides traceability for persisted cache entries. It is metadata, not a primary cache-hit gate. |
| `grounding_info` | Result of the write-time grounding check. | Returned with cache hits so downstream code can see whether the answer was `GROUNDED`, `PARTIAL`, or `INFERRED`. |
| `data_scope_hash` | Hash of the active corpus or document scope. | Prevents reuse across the wrong corpus. Exact, semantic, and knowledge hits are all scoped. |
| `facts` | Extracted `(subject, relation, object)` triples. | Feeds the separate knowledge index. These facts make knowledge cache hits possible. |

## Stores and Indexes

There are two cache-related records and two cache-related vector indexes:

| Store or index | In-memory field | Persisted location | What it contains | Used for |
| --- | --- | --- | --- | --- |
| Cache entries | `self.cache` | `cache_entries.json` | Query, answer, source context, query embedding, grounding metadata, scope metadata, and extracted facts. | Exact hits and final answer lookup for semantic and knowledge hits. |
| Semantic cache index | `self._cache_index` | `cache_idx/` | FAISS vectors for cached query embeddings, with metadata pointing back to cache entries. | Semantic hits. |
| Knowledge records | `self.knowledge` | `knowledge.json` | Flattened extracted triples, with metadata pointing back to source cache entries. | Knowledge hit metadata and source-entry lookup. |
| Knowledge index | `self.knowledge_index` | `knowledge_idx/` | FAISS vectors for extracted triple embeddings. | Knowledge hits. |

There is also a document retrieval index, `self.doc_index`, persisted under `doc_idx/` with `chunks.json`. That is not a cache store. It is used after a cache miss to retrieve source document chunks for synthesis.

The original query vector is stored twice for different reasons:

1. In the cache entry as `entry["embedding"]`, so the serialized cache entry is self-contained.
2. In `self._cache_index`, so semantic lookup can do fast nearest-neighbor search over cached queries.

If `cache_idx/` is missing when loading persisted state, the system can rebuild the semantic cache index from the saved `entry["embedding"]` values in `cache_entries.json`.

## Semantic Hits

A semantic hit is query-level reuse.

The system asks:

> Is the new query asking for the same information as a previous cached query?

The read path is:

1. Embed the new query.
2. Compare that new query embedding against cached query embeddings in the semantic cache FAISS index.
3. Filter candidates by score and active `data_scope_hash`.
4. Ask the evaluator model, used as the Sniper, whether the new query is semantically identical to one cached query.
5. If accepted, return that cache entry's `result` with `cache_type: "semantic"`.

This means semantic hits depend mainly on the cached `query`, query embedding, `result`, `grounding_info`, and `data_scope_hash`.

Example:

```text
Previous query: What was the EBITDA?
New query:      What EBITDA figure is reported?
```

If the Sniper decides these ask for the same information, the cached answer is reused.

## Knowledge Hits

A knowledge hit is fact-level reuse.

The system asks:

> Does the new query match a fact extracted from any previous cached answer?

On every cache write, the generated answer is sent to the evaluator model with an instruction to extract one triple per line:

```text
subject | relation | object
```

The parser splits each line on `|` and keeps the first three fields. Each fact is then stored with metadata that links it back to the source cache entry:

```python
fact["source_cache_idx"] = cache_idx
fact["source_chunk_hash"] = chunk_hash
fact["data_scope_hash"] = data_scope_hash
```

Each triple is embedded as a simple concatenation of its three fields:

```python
fact_text = f"{fact['subject']} {fact['relation']} {fact['object']}"
```

That vector goes into a separate knowledge FAISS index.

The read path is:

1. Embed the new query.
2. Compare that new query embedding against embedded triples in the knowledge FAISS index.
3. Keep facts whose source cache entry matches the active corpus scope.
4. Require the fact score to pass the knowledge threshold.
5. Use margin and lexical-support checks to decide whether an additional knowledge verifier is needed.
6. If accepted, follow the fact back to its parent cache entry and return that entry's `result` with `cache_type: "knowledge"` and `relevant_facts`.

This means knowledge hits depend mainly on `facts`, fact embeddings, `source_cache_idx`, `result`, `grounding_info`, and `data_scope_hash`.

Example:

```text
Cached query: What charges did Maxwell face?
Extracted fact: Defense attorney | Maxwell | Christian R. Everdell
New query: Who was Maxwell's lawyer?
```

The new query is not semantically equivalent to the cached query, but it can still match the extracted attorney fact. That creates a knowledge hit.

## Main Difference

| Cache mode | Matching unit | Reuse question | Returned answer |
| --- | --- | --- | --- |
| Exact | Cached query string | Is the query text the same? | Cached `result` |
| Semantic | Cached query meaning | Is this a paraphrase of a previous query? | Cached `result` |
| Knowledge | Extracted fact triple | Does a fact inside a previous answer answer this query? | Parent cache entry's `result` plus `relevant_facts` |

Semantic hits reuse prior work at the query-answer level. Knowledge hits make facts inside prior answers independently discoverable, then route back to the original cache entry for the returned answer and provenance metadata.
