# 300K Token Execution Flow

Simple view of how one ~300K-token document moves through the current semantic
cache pipeline.

## Example Setup

```text
Document size:        ~300,000 tokens
Chunk size:           10,000 tokens
Chunk overlap:         2,000 tokens
Chunk stride:          8,000 tokens
Approx chunk count:       38 chunks

Search mode:          iterative
Embedding model:      Qwen3-Embedding-0.6B
Embedding max length: 8,192 tokens per chunk
Reranker:             disabled in iterative mode
```

## Main Flow

```text
300K-token document
        |
        v
Write source text
context.txt
        |
        v
Ingest document
        |
        v
Chunk with overlap
~38 chunks of 10K tokens
        |
        v
Embed each chunk locally
Qwen3-Embedding, 1024-d vectors
        |
        v
Build FAISS document index
chunk_index + offsets + data_scope_hash
        |
        v
Build query
question + A/B/C/D choices
        |
        v
Search cache
exact -> semantic -> knowledge
        |
        +-- cache hit ----------------------+
        |                                   |
        v                                   v
      miss                          return cached answer
        |
        v
FAISS ranks chunks for query
        |
        v
Iterative reader scans chunks
one executor call per inspected chunk
        |
        v
Evidence ledger grows
memory_updates, target_facts, code_mappings, open_questions
        |
        v
Can answer safely?
        |
        +-- yes --> return answer
        |
        +-- no
             |
             v
Final adjudication from ledger only
             |
             v
Still invalid / needs context / ungrounded symbolic code?
             |
             +-- yes --> packed fallback over inspected chunks
             |
             +-- no
                  |
                  v
               return answer
        |
        v
Consensus verify
        |
        v
Store cache entry + compact source context
        |
        v
Persist benchmark artifacts and cache state
```

## Scan Budget For 300K Example

With the current Jarvis-style scan settings:

```text
total_chunks = 38
min ratio    = 0.50
max ratio    = 1.0

early-stop minimum = ceil(38 * 0.50) = 19 chunks
scan budget        = ceil(38 * 1.0)  = 38 chunks
```

So the system may inspect the full document, but never as one 300K-token prompt.
It reads chunk by chunk and carries a compact ledger forward.

## Task-Specific Branches

```text
Symbolic code task
event/relation type with choices like aba, aai, aaz
        |
        v
Keep only current option-code mappings
Require explicit mapping before accepting answer


Ordering task
choices like 1432 / 4123 / 2431
        |
        v
FAISS selects relevant chunks
Selected chunks are scanned in document order


Generic QA
        |
        v
Use normal evidence-ledger scan
```

## Packed Mode Contrast

```text
cache miss
   |
   v
FAISS top_k chunks
   |
   v
optional reranker
   |
   v
pack small chunk set into one prompt
   |
   v
single synthesis call
   |
   v
consensus + store
```

Packed mode is cheaper in LLM calls. Iterative mode is better when evidence may
be spread across a very large document.

## Key Artifact Fields

```text
ingested_chunks
search_mode
reranker_disabled
total_dataset_context_token_estimate
total_input_tokens
input_token_savings_vs_context
input_token_savings_percent
iterative_scan_total_chunks
iterative_scan_budget
iterative_scan_visited_chunk_count
iterative_scan_selected_chunk_indices
iterative_scan_supporting_chunk_indices
iterative_scan_extra_scan_used
iterative_scan_extra_scan_reason
iterative_scan_extra_scan_chunk_count
iterative_scan_stop_reason
iterative_scan_packed_fallback_reason
iterative_scan_parse_failure_count
```
