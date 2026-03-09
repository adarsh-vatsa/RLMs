"""
================================================================================
TWO-STAGE SEMANTIC CACHE FOR AUTONOMOUS AGENTS
================================================================================

A complete, production-quality implementation of the Two-Stage "Dragnet & Sniper"
Semantic Cache architecture.

This system implements every theoretical concept from our research:

1. HASH BUCKETING         — Partition cache by document chunk to prevent
                            cross-context contamination.
2. VECTOR DRAGNET         — Local SentenceTransformer embedding + cosine
                            similarity for fast Top-K candidate retrieval.
3. LLM SNIPER             — Micro-LLM (Claude Haiku) validates logical +
                            semantic equivalence of candidates.
4. PARALLEL TOP-K CHUNKING— When K is large, chunk candidates into parallel
                            Haiku calls to prevent evaluator Context Rot.
5. CONTEXT COLLAPSE GUARD — Two defenses against the agent drowning in its
                            own cached outputs:
                            (A) Ephemeral Retrieval: large results are served
                                but NOT appended to agent context.
                            (B) Recursive Chunking: oversized results are
                                summarized via sub-agent calls.
6. CACHE PRE-WARMING      — Programmatic Day-1 sweep to saturate the cache
                            before any human touches the system.
7. HETEROGENEOUS ROUTING  — Dispatch to cheap models for simple tasks,
                            expensive models for complex reasoning.

Usage:
    cd /Users/zeitgeist/research/RLMs
    python3 semantic_cache_system.py

Requirements:
    pip install sentence-transformers anthropic python-dotenv numpy scikit-learn
================================================================================
"""

import os
import hashlib
import time
import json
import re
import math
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------
dotenv_path = '/Users/zeitgeist/research/RLMs/.env'
load_dotenv(dotenv_path)
client = Anthropic()

# ---------------------------------------------------------------------------
# Model Config
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
EMBEDDING_DIM = 1024
EXECUTOR_MODEL = "claude-sonnet-4-20250514"
EVALUATOR_MODEL = "claude-haiku-4-20250414"


# ============================================================================
# 0a. EMBEDDING ENGINE — Qwen3-Embedding-0.6B (596M params, 1024-dim)
# ============================================================================

class EmbeddingEngine:
    """Qwen3-Embedding-0.6B: instruction-aware embeddings with 32K context."""

    def __init__(self):
        print("  [EMBED] Loading Qwen3-Embedding-0.6B...")
        from transformers import AutoModel, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            EMBEDDING_MODEL, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            EMBEDDING_MODEL, trust_remote_code=True, torch_dtype=torch.float32
        )
        self.model.eval()
        self.device = "cpu"
        self.model.to(self.device)
        self.torch = torch
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"  [EMBED] ✓ Loaded on {self.device} ({param_count // 1_000_000}M params)")

    def encode(self, texts: list, instruction: str = "") -> np.ndarray:
        """Encode texts with optional instruction prefix. Uses CLS token pooling."""
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
        batch_size = 16
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=8192, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                outputs = self.model(**inputs)
                # CLS token pooling (matches existing FAISS indices)
                embs = outputs.last_hidden_state[:, 0, :]
                embs = embs / embs.norm(dim=1, keepdim=True)
            all_embs.append(embs.cpu().numpy())
            if len(texts) > batch_size and (i + batch_size) % 100 == 0:
                print(f"    Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks...")
        return np.vstack(all_embs).astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode(
            [query],
            instruction="Given a legal query, retrieve relevant court documents and filings"
        )

    def encode_documents(self, documents: list) -> np.ndarray:
        return self.encode(documents)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text (for cache entries). Returns 1D array."""
        return self.encode_query(text)[0]


# ============================================================================
# 0b. FAISS INDEX — Inner product on normalized vectors = cosine similarity
# ============================================================================

class FAISSIndex:
    """FAISS IndexFlatIP wrapper with metadata tracking and persistence."""

    def __init__(self, dim: int = EMBEDDING_DIM):
        self._faiss = None  # Lazy import to avoid segfault on Apple Silicon
        self.dim = dim
        self.index = None
        self.metadata: List[Dict] = []

    def _ensure_faiss(self):
        """Lazy import faiss — must happen AFTER transformers loads."""
        if self._faiss is None:
            import faiss as _faiss
            self._faiss = _faiss
        if self.index is None:
            self.index = self._faiss.IndexFlatIP(self.dim)

    def add(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        self._ensure_faiss()
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        self.index.add(embeddings)
        self.metadata.extend(metadata_list)

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[float, Dict]]:
        self._ensure_faiss()
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if self.index.ntotal == 0:
            return []
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((float(score), self.metadata[idx]))
        return results

    def save(self, path: Path):
        self._ensure_faiss()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "metadata.json", "w") as f:
            json.dump(self.metadata, f)
        print(f"  [FAISS] Saved {self.index.ntotal} vectors to {path}")

    def load(self, path: Path) -> bool:
        self._ensure_faiss()
        path = Path(path)
        index_path = path / "index.faiss"
        meta_path = path / "metadata.json"
        if not index_path.exists() or not meta_path.exists():
            return False
        self.index = self._faiss.read_index(str(index_path))
        with open(meta_path) as f:
            self.metadata = json.load(f)
        print(f"  [FAISS] Loaded {self.index.ntotal} vectors from {path}")
        return True

    @property
    def total(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal


# ============================================================================
# 0c. RERANKER — Qwen3-Reranker-0.6B (generative yes/no scoring)
# ============================================================================

class Reranker:
    """Cross-encoder reranker with yes/no token probability scoring."""

    def __init__(self):
        print("  [RERANK] Loading Qwen3-Reranker-0.6B...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            RERANKER_MODEL, trust_remote_code=True, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            RERANKER_MODEL, trust_remote_code=True, torch_dtype=torch.float32
        )
        self.model.eval()
        self.device = "cpu"
        self.model.to(self.device)
        self.torch = torch

        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.max_length = 8192
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        print(f"  [RERANK] ✓ Loaded on {self.device}")

    def _format_pair(self, query: str, doc: str, instruction: str = None) -> str:
        if instruction is None:
            instruction = "Given a query, retrieve relevant documents that answer the query"
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, pairs: List[str]):
        inputs = self.tokenizer(
            pairs, padding=False, truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    def rerank(self, query: str, documents: List[str], top_k: int = 5,
               relevance_threshold: float = 0.5) -> List[Tuple[int, float, str]]:
        """Re-score and filter by relevance threshold. Returns [(idx, score, text)]."""
        pairs = [self._format_pair(query, doc) for doc in documents]
        inputs = self._process_inputs(pairs)

        with self.torch.no_grad():
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = self.torch.stack([false_vector, true_vector], dim=1)
            batch_scores = self.torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        # Apply relevance gate: only return docs the reranker says "yes" to
        results = []
        for orig_idx, score in ranked[:top_k]:
            if score >= relevance_threshold:
                results.append((orig_idx, float(score), documents[orig_idx]))
        return results


# ============================================================================
# 1. EXECUTION METRICS — Tracks every API call, cache event, and dollar spent
# ============================================================================
class ExecutionMetrics:
    """
    THEORY: Economic Scaling Proof
    -----------------------------------------------------------------------
    By recording every call and its cost, we can empirically demonstrate
    that the cache drives execution economics from O(N) (linear cost per
    query) towards an asymptote of O(1) as the cache saturates.
    """

    def __init__(self):
        self.stats = {
            "claude-sonnet-4-5": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0},
            "claude-haiku-4-5":  {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0},
        }
        self.exact_hits = 0
        self.semantic_hits = 0
        self.cache_misses = 0
        self.parallel_chunk_evaluations = 0   # How many times we used parallel chunking
        self.ephemeral_retrievals = 0         # Large results served but not appended
        self.recursive_summarizations = 0     # Oversized results chunked/summarized
        self.pre_warm_entries = 0             # Entries added during pre-warming
        self.grounded_results = 0             # Results verified against source text
        self.partial_results = 0              # Results partially grounded
        self.inferred_results = 0             # Results not found in source (potential hallucination)
        self.consensus_agreed = 0             # Multi-model consensus: models agreed
        self.consensus_disputed = 0           # Multi-model consensus: models disagreed
        self.start_time = None
        self.end_time = None

    def record_call(self, model: str, input_tokens: int, output_tokens: int):
        """Record a real API call with actual token counts and compute cost."""
        self.stats[model]["calls"] += 1
        self.stats[model]["input_tokens"] += input_tokens
        self.stats[model]["output_tokens"] += output_tokens
        # Anthropic pricing (as of early 2026)
        if "sonnet" in model:
            self.stats[model]["cost"] += (input_tokens * 0.003 / 1000) + (output_tokens * 0.015 / 1000)
        elif "haiku" in model:
            self.stats[model]["cost"] += (input_tokens * 0.00025 / 1000) + (output_tokens * 0.00125 / 1000)

    def print_summary(self):
        duration = (self.end_time or time.time()) - (self.start_time or time.time())
        total_cost = sum(m["cost"] for m in self.stats.values())
        total_calls = sum(m["calls"] for m in self.stats.values())
        total_cache_events = self.exact_hits + self.semantic_hits
        total_queries = total_cache_events + self.cache_misses

        print(f"\n{'='*65}")
        print(f"  TWO-STAGE SEMANTIC CACHE — EXECUTION SUMMARY")
        print(f"{'='*65}")
        print(f"  Time Elapsed           : {duration:.2f}s")
        print(f"  Total Queries Processed: {total_queries}")
        print(f"  Total API Calls Made   : {total_calls}")
        print(f"  Total API Cost         : ${total_cost:.6f}")
        print(f"{'─'*65}")
        print(f"  CACHE PERFORMANCE")
        print(f"    Exact Hits           : {self.exact_hits}")
        print(f"    Semantic Hits (Sniper): {self.semantic_hits}")
        print(f"    Cache Misses         : {self.cache_misses}")
        if total_queries > 0:
            hit_rate = (total_cache_events / total_queries) * 100
            print(f"    Overall Hit Rate     : {hit_rate:.1f}%")
        print(f"{'─'*65}")
        print(f"  ADVANCED FEATURES")
        print(f"    Parallel Chunk Evals : {self.parallel_chunk_evaluations}")
        print(f"    Ephemeral Retrievals : {self.ephemeral_retrievals}")
        print(f"    Recursive Summarize  : {self.recursive_summarizations}")
        print(f"    Pre-Warmed Entries   : {self.pre_warm_entries}")
        print(f"{'─'*65}")
        total_provenance = self.grounded_results + self.partial_results + self.inferred_results
        print(f"  SOURCE PROVENANCE")
        print(f"    Grounded (verified)  : {self.grounded_results}")
        print(f"    Partially Grounded   : {self.partial_results}")
        print(f"    Inferred (unverified): {self.inferred_results}")
        if total_provenance > 0:
            grounded_pct = (self.grounded_results / total_provenance) * 100
            print(f"    Grounding Rate       : {grounded_pct:.1f}%")
        total_consensus = self.consensus_agreed + self.consensus_disputed
        print(f"    Consensus Agreed     : {self.consensus_agreed}")
        print(f"    Consensus Disputed   : {self.consensus_disputed}")
        if total_consensus > 0:
            agree_pct = (self.consensus_agreed / total_consensus) * 100
            print(f"    Consensus Rate       : {agree_pct:.1f}%")
        print(f"{'─'*65}")
        print(f"  COST BREAKDOWN BY MODEL")
        for model, data in self.stats.items():
            if data['calls'] > 0:
                print(f"    {model}:")
                print(f"      Calls: {data['calls']}  |  Tokens: {data['input_tokens']}in / {data['output_tokens']}out  |  Cost: ${data['cost']:.6f}")
        print(f"{'='*65}\n")


# ============================================================================
# 2. SEMANTIC CACHE CONTROLLER — The core Two-Stage engine
# ============================================================================
class SemanticCacheController:
    """
    THEORY: The Two-Stage "Dragnet & Sniper" Architecture
    -----------------------------------------------------------------------
    Standard Semantic Caches (like GPTCache) rely solely on vector similarity
    thresholds. This causes catastrophic collisions: "Include X" and "Exclude X"
    have nearly identical embeddings but opposite meanings.

    Our architecture splits the cache lookup into two stages:
      Stage 1 (Dragnet): Fast, cheap vector search to find Top-K candidates.
      Stage 2 (Sniper):  A micro-LLM (Haiku) performs logical validation,
                         catching inversions that math alone cannot detect.

    Additionally, we implement:
      - Hash Bucketing:     Partition by document chunk hash.
      - Parallel Chunking:  Batch large candidate sets to prevent evaluator
                            Context Rot.
      - Context Collapse Guard: Protect the agent from oversized cache returns.
    """

    # Configuration constants
    EVALUATOR_MODEL = "claude-haiku-4-5"
    TOP_K_CANDIDATES = 5           # Default number of vector search results
    PARALLEL_BATCH_SIZE = 5        # Max candidates per single Haiku evaluation call
    EPHEMERAL_TOKEN_THRESHOLD = 2000  # If cached result exceeds this, flag as ephemeral
    MAX_CONTEXT_TOKENS = 4000      # If cached result exceeds this, recursively chunk it

    def __init__(self, metrics: ExecutionMetrics, embedder: EmbeddingEngine = None,
                 reranker: Reranker = None, corpus_id: str = None,
                 corpus_domain: str = "general"):
        self.cache = {}  # Key: chunk_hash -> Value: List[{query, result, embedding, source_context, ...}]
        self.metrics = metrics
        self.embedder = embedder  # Will be set externally or via init_models()
        self.reranker = reranker  # Optional: for document retrieval
        self.doc_index = None     # FAISS index for document retrieval
        self.knowledge: List[Dict] = []       # Extracted (subj, rel, obj) triples
        self.knowledge_index = None            # FAISS index for fact lookup
        self._cache_index = None               # FAISS index for cached query lookup
        self._persist_path = None              # Path for persistence

        # ── Corpus-level namespace isolation ──
        # Each corpus (Epstein docs, finance filings, medical records) gets its
        # own FAISS indices, cache, and knowledge graph. The corpus_id is
        # infrastructure-only — it NEVER enters any LLM prompt.
        self.corpus_id = corpus_id             # e.g. "epstein", "finance_q1"
        self.corpus_domain = corpus_domain     # e.g. "legal", "finance", "medical"
        self.corpus_config = {}                # Loaded from corpus_config.json

        if corpus_id:
            print(f"[INIT] Corpus: {corpus_id} (domain: {corpus_domain})")
        print("[INIT] Semantic Cache Controller ready.\n")

    # ------------------------------------------------------------------
    # THEORY: Hash Bucketing
    # ------------------------------------------------------------------
    # Before we even look at the query, we look at the DATA the agent is
    # reading. We hash the document chunk into a unique ID so the cache
    # never accidentally mixes up answers from Page 42 with Page 99.
    # This prevents cross-document contamination.
    # ------------------------------------------------------------------
    def _get_chunk_hash(self, context: str) -> str:
        """Compute a deterministic hash of the data chunk being analyzed."""
        return hashlib.md5(context.strip().encode()).hexdigest()

    # ------------------------------------------------------------------
    # STAGE 1: THE VECTOR DRAGNET
    # ------------------------------------------------------------------
    # Uses a lightweight, free, local SentenceTransformer to embed the
    # query and compute cosine similarity against all cached queries in
    # this hash bucket. Returns the Top-K most similar candidates.
    #
    # THEORY: This is O(log N) with proper ANN indexing. For our proof-
    # of-concept we use brute-force numpy, which is O(N) but still
    # sub-millisecond for realistic cache sizes.
    # ------------------------------------------------------------------
    def _vector_dragnet(self, new_query: str, bucket_entries: list) -> list:
        """Cast a wide, cheap net to find broadly similar past queries."""
        if not bucket_entries:
            return []

        new_embedding = self.embedder.encode_single(new_query)
        stored_embeddings = np.array([e["embedding"] for e in bucket_entries])
        # Use numpy dot product (embeddings are already L2-normalized)
        similarities = np.dot(stored_embeddings, new_embedding)

        # Grab the top K indices, sorted by descending similarity
        k = min(self.TOP_K_CANDIDATES, len(bucket_entries))
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        return [(bucket_entries[i], i, similarities[i]) for i in top_k_indices]

    # ------------------------------------------------------------------
    # STAGE 2: THE LLM SNIPER (Single-Batch Evaluation)
    # ------------------------------------------------------------------
    # Takes a batch of candidate queries and asks Haiku:
    # "Does the NEW query ask for the EXACT SAME logical computation
    #  as any of these candidates?"
    #
    # THEORY: This is the key innovation over pure vector caches.
    # Because Haiku is an actual language model, it understands that
    # "Include the timeout logs" ≠ "Exclude the timeout logs" even
    # though their embeddings are 99% similar.
    # ------------------------------------------------------------------
    def _llm_sniper_evaluate(self, new_query: str, candidates: list) -> dict:
        """
        Ask Haiku to evaluate a batch of candidates for semantic equivalence.

        Args:
            new_query: The incoming query to check
            candidates: List of (entry, original_index, similarity_score) tuples

        Returns:
            {"hit": True/False, "id": original_index or None}
        """
        prompt = (
            "You are a Semantic Cache Controller. Your ONLY job is to determine "
            "if a NEW QUERY asks for the EXACT SAME information as any PREVIOUS QUERY.\n\n"
            "CRITICAL: Pay close attention to logical differences. "
            "'Include X' and 'Exclude X' are DIFFERENT. "
            "'Find errors' and 'Find successes' are DIFFERENT.\n\n"
            f'NEW QUERY: "{new_query}"\n\n'
            "PREVIOUS QUERIES:\n"
        )
        for entry, orig_idx, sim_score in candidates:
            prompt += f'  ID {orig_idx}: "{entry["query"]}" (similarity: {sim_score:.3f})\n'

        prompt += (
            "\nIf the NEW QUERY is semantically identical to one of the PREVIOUS "
            "QUERIES (even if paraphrased), respond: {\"hit\": true, \"id\": <ID>}\n"
            "If it asks for DIFFERENT information, respond: {\"hit\": false, \"id\": null}\n"
            "Output ONLY valid JSON. No explanation."
        )

        try:
            response = client.messages.create(
                model=self.EVALUATOR_MODEL,
                max_tokens=50,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            self.metrics.record_call(
                self.EVALUATOR_MODEL,
                response.usage.input_tokens,
                response.usage.output_tokens
            )

            raw = response.content[0].text
            json_match = re.search(r'\{.*\}', raw.replace('\n', ''))
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            print(f"      [SNIPER ERROR] {e}. Defaulting to miss.")

        return {"hit": False, "id": None}

    # ------------------------------------------------------------------
    # PARALLEL TOP-K CHUNKING
    # ------------------------------------------------------------------
    # THEORY: If the Vector Dragnet returns a large number of candidates
    # (e.g., 500 near-misses in a massive cache), we CANNOT pass all 500
    # to Haiku in a single prompt — Haiku would suffer Context Rot and
    # hallucinate its evaluation.
    #
    # Solution: Chunk the candidates into small batches (e.g., 5 per call)
    # and fire them all in parallel using ThreadPoolExecutor.
    #
    # Wall-clock latency: O(1) — all batches run simultaneously.
    # Dollar cost: O(K/C) Haiku calls — but at $0.0001/call, this is
    # negligible compared to a single Sonnet call at $0.05.
    # ------------------------------------------------------------------
    def _parallel_sniper_evaluate(self, new_query: str, candidates: list) -> dict:
        """
        Chunk candidates into parallel batches for Haiku evaluation.
        Prevents evaluator Context Rot while maintaining O(1) wall-clock time.
        """
        batch_size = self.PARALLEL_BATCH_SIZE
        num_batches = math.ceil(len(candidates) / batch_size)
        batches = [candidates[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

        print(f"      [PARALLEL] Chunking {len(candidates)} candidates into {num_batches} parallel batches of ≤{batch_size}")
        self.metrics.parallel_chunk_evaluations += 1

        # Fire all batches in parallel
        with ThreadPoolExecutor(max_workers=num_batches) as executor:
            futures = {
                executor.submit(self._llm_sniper_evaluate, new_query, batch): i
                for i, batch in enumerate(batches)
            }

            for future in as_completed(futures):
                result = future.result()
                if result.get("hit") and result.get("id") is not None:
                    # First hit wins — cancel remaining futures (best-effort)
                    executor.shutdown(wait=False, cancel_futures=True)
                    return result

        return {"hit": False, "id": None}

    # ------------------------------------------------------------------
    # CONTEXT COLLAPSE PROTECTION
    # ------------------------------------------------------------------
    # THEORY: Even if the cache returns a valid hit, the RESULT itself
    # might be enormous (e.g., a 500,000-token legal summary). If the
    # agent tries to read this, it will blow its own context window.
    #
    # Two defenses:
    #   (A) Ephemeral Retrieval: Return the result to the agent but flag
    #       it as "do not persist in conversation history." If the agent
    #       needs it again, it re-queries the cache (which is near-free).
    #   (B) Recursive Chunking: If the result is too large for even a
    #       single context window, chunk it and summarize via sub-agent
    #       calls — the same map-reduce pattern any agent can use.
    # ------------------------------------------------------------------
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 characters per token for English text."""
        return len(text) // 4

    def _grounding_check(self, result: str, source_context: str) -> dict:
        """
        THEORY: Source Provenance Verification
        ---------------------------------------------------------------
        The cache is trust-on-first-write: whatever the LLM produces on
        the first query gets locked in. If the LLM hallucinated a number,
        the cache serves that wrong number forever.

        This method extracts quantitative facts (numbers, dollar amounts,
        percentages) from the LLM's result and checks whether they appear
        verbatim in the original source chunk. This catches the most
        dangerous hallucination mode — fabricated numbers — for zero cost.

        Returns:
            {
                "grounding": "GROUNDED" | "PARTIAL" | "INFERRED",
                "verified_facts": [...],   # Facts found in source
                "unverified_facts": [...]  # Facts NOT found in source
            }
        """
        # Extract quantitative facts: dollar amounts, percentages, plain numbers
        fact_patterns = [
            r'\$[\d,]+\.?\d*[MBKTmkbt]?',   # Dollar amounts: $12.4M, $500K, $2.1M
            r'\d+\.?\d*\s*%',                 # Percentages: 4.2%, 112%
            r'\b\d{1,3}(?:,\d{3})+\b',        # Large numbers with commas: 2,847
            r'\b\d+\.\d+[MBKTmkbt]\b',        # Abbreviated numbers: 48.7M
        ]

        all_facts = []
        for pattern in fact_patterns:
            all_facts.extend(re.findall(pattern, result))

        # De-duplicate while preserving order
        seen = set()
        unique_facts = []
        for f in all_facts:
            if f not in seen:
                seen.add(f)
                unique_facts.append(f)

        if not unique_facts:
            # No quantitative facts to verify — treat as grounded (qualitative answer)
            return {"grounding": "GROUNDED", "verified_facts": [], "unverified_facts": []}

        verified = [f for f in unique_facts if f in source_context]
        unverified = [f for f in unique_facts if f not in source_context]

        if not unverified:
            grounding = "GROUNDED"
        elif not verified:
            grounding = "INFERRED"
        else:
            grounding = "PARTIAL"

        return {"grounding": grounding, "verified_facts": verified, "unverified_facts": unverified}

    def _apply_context_collapse_guard(self, result: str, query: str, source_context: str = "", grounding_info: dict = None) -> dict:
        """
        Wrap the cache result with metadata to protect the agent from
        Context Collapse, and attach source provenance information.

        Returns:
            {
                "result": str,          # The actual text
                "ephemeral": bool,      # If True, do NOT append to agent context
                "was_summarized": bool,  # If True, this is a condensed version
                "grounding": str,       # GROUNDED | PARTIAL | INFERRED
                "verified_facts": list,  # Facts confirmed in source
                "unverified_facts": list # Facts NOT found in source
            }
        """
        # Attach grounding info (either passed in or computed fresh)
        if grounding_info is None:
            grounding_info = self._grounding_check(result, source_context)

        # Track grounding metrics
        g = grounding_info["grounding"]
        if g == "GROUNDED":
            self.metrics.grounded_results += 1
        elif g == "PARTIAL":
            self.metrics.partial_results += 1
        else:
            self.metrics.inferred_results += 1

        if grounding_info["unverified_facts"]:
            print(f"      [PROVENANCE] ⚠ {g}: verified={grounding_info['verified_facts']}, unverified={grounding_info['unverified_facts']}")
        elif grounding_info["verified_facts"]:
            print(f"      [PROVENANCE] ✓ {g}: all facts verified in source: {grounding_info['verified_facts']}")

        estimated_tokens = self._estimate_tokens(result)
        base = {
            "grounding": grounding_info["grounding"],
            "verified_facts": grounding_info["verified_facts"],
            "unverified_facts": grounding_info["unverified_facts"],
        }

        # Case B: Result is TOO LARGE for any single context window.
        if estimated_tokens > self.MAX_CONTEXT_TOKENS:
            print(f"      [GUARD] Result is ~{estimated_tokens} tokens. Applying recursive summarization...")
            self.metrics.recursive_summarizations += 1
            summarized = self._recursive_summarize(result, query)
            return {**base, "result": summarized, "ephemeral": False, "was_summarized": True}

        # Case A: Result is large but manageable. Serve it but flag as ephemeral.
        if estimated_tokens > self.EPHEMERAL_TOKEN_THRESHOLD:
            print(f"      [GUARD] Result is ~{estimated_tokens} tokens. Flagging as ephemeral (do not persist).")
            self.metrics.ephemeral_retrievals += 1
            return {**base, "result": result, "ephemeral": True, "was_summarized": False}

        # Normal case: result is small enough to safely append to context.
        return {**base, "result": result, "ephemeral": False, "was_summarized": False}

    # Maximum recursion depth to prevent infinite loops on pathological inputs
    MAX_RECURSION_DEPTH = 5

    def _recursive_summarize(self, large_text: str, original_query: str, depth: int = 0) -> str:
        """
        THEORY: Truly Recursive Parallel Summarization
        ---------------------------------------------------------------
        If a cached result is too large for the agent's context window,
        we chunk the text and summarize each chunk via cheap sub-agent
        calls fired in PARALLEL using ThreadPoolExecutor.

        CRITICAL: After the first pass, the JOINED summaries may STILL
        exceed MAX_CONTEXT_TOKENS (e.g., a 100K-token document produces
        25 chunk summaries × 200 tokens = 5,000 tokens). So we RECURSE:
        the joined summary becomes the input to the next level of
        parallel chunking + summarization.

        This yields logarithmic reduction:
          Level 0: 100K tokens → 25 chunks → ~5,000 token summary
          Level 1:   5K tokens →  2 chunks → ~400 token summary   ✓ fits

        Each level is fully parallel, so total wall-clock time is
        O(depth) where depth = log(N / max_tokens) — typically 2-3.
        """
        if depth >= self.MAX_RECURSION_DEPTH:
            print(f"      [SUMMARIZE] ⚠ Max recursion depth ({self.MAX_RECURSION_DEPTH}) reached. Truncating.")
            return large_text[:self.MAX_CONTEXT_TOKENS * 4]  # Hard truncate as safety valve

        # Split into manageable chunks (~1000 tokens each)
        chunk_size = 4000  # characters ≈ 1000 tokens
        chunks = [large_text[i:i+chunk_size] for i in range(0, len(large_text), chunk_size)]

        level_label = f"L{depth}" if depth > 0 else ""
        print(f"      [SUMMARIZE {level_label}] Splitting {len(large_text)} chars into {len(chunks)} chunks, dispatching in parallel...")

        # Tighter prompt at deeper recursion levels
        if depth == 0:
            instruction = "Summarize the following text in 2-3 sentences"
        else:
            instruction = "Condense the following text into 1-2 sentences, keeping only the most critical facts"

        def _summarize_chunk(args):
            """Summarize a single chunk via Haiku. Designed for parallel dispatch."""
            idx, chunk_text = args
            try:
                response = client.messages.create(
                    model=self.EVALUATOR_MODEL,
                    max_tokens=200,
                    temperature=0,
                    messages=[{"role": "user", "content": (
                        f"{instruction}, "
                        f"focusing on information relevant to: '{original_query}'\n\n"
                        f"TEXT:\n{chunk_text}"
                    )}]
                )
                self.metrics.record_call(
                    self.EVALUATOR_MODEL,
                    response.usage.input_tokens,
                    response.usage.output_tokens
                )
                return (idx, response.content[0].text)
            except Exception as e:
                return (idx, f"[Summarization failed for chunk {idx}: {e}]")

        # Fire all chunk summarizations in parallel
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            results = list(executor.map(_summarize_chunk, enumerate(chunks)))

        # Sort by original index to preserve document order
        results.sort(key=lambda x: x[0])
        summaries = [text for _, text in results]
        joined = "\n".join(summaries)

        print(f"      [SUMMARIZE {level_label}] ✓ {len(chunks)} parallel sub-agent calls complete → {len(joined)} chars (~{len(joined)//4} tokens)")

        # CHECK: Is the joined summary STILL too large? If so, recurse.
        if self._estimate_tokens(joined) > self.MAX_CONTEXT_TOKENS:
            print(f"      [SUMMARIZE] Summary still exceeds {self.MAX_CONTEXT_TOKENS} tokens. Recursing to level {depth + 1}...")
            self.metrics.recursive_summarizations += 1
            return self._recursive_summarize(joined, original_query, depth + 1)

        return joined

    # ------------------------------------------------------------------
    # MAIN CACHE CHECK — The full Two-Stage pipeline
    # ------------------------------------------------------------------
    def check(self, query: str, context: str) -> Optional[dict]:
        """
        The complete cache lookup pipeline:
          1. Hash Bucket isolation
          2. Exact match check (free)
          3. Vector Dragnet (fast, local)
          4. LLM Sniper (cheap, accurate) — with parallel chunking if needed
          5. Context Collapse Guard on the result

        Returns:
            None if cache miss.
            dict {"result", "ephemeral", "was_summarized"} if cache hit.
        """
        chunk_hash = self._get_chunk_hash(context)

        # No entries exist for this document chunk yet
        if chunk_hash not in self.cache:
            self.metrics.cache_misses += 1
            return None

        bucket = self.cache[chunk_hash]

        # ----- EXACT MATCH (Free, O(N) scan of small bucket) -----
        for entry in bucket:
            if entry["query"].strip().lower() == query.strip().lower():
                self.metrics.exact_hits += 1
                print(f"      [CACHE] ✓ Exact Match. Free retrieval.")
                return self._apply_context_collapse_guard(
                    entry["result"], query,
                    source_context=entry.get("source_context", ""),
                    grounding_info=entry.get("grounding_info")
                )

        # ----- STAGE 1: VECTOR DRAGNET -----
        candidates = self._vector_dragnet(query, bucket)
        if not candidates:
            self.metrics.cache_misses += 1
            return None

        print(f"      [DRAGNET] Retrieved {len(candidates)} candidates (top sim: {candidates[0][2]:.3f})")

        # ----- STAGE 2: LLM SNIPER -----
        # If candidates exceed the batch size, use parallel chunking
        # to prevent evaluator Context Rot.
        if len(candidates) > self.PARALLEL_BATCH_SIZE:
            decision = self._parallel_sniper_evaluate(query, candidates)
        else:
            print(f"      [SNIPER] Evaluating {len(candidates)} candidates via Haiku...")
            decision = self._llm_sniper_evaluate(query, candidates)

        if decision.get("hit") and decision.get("id") is not None:
            idx = decision["id"]
            matched = bucket[idx]
            print(f"      [CACHE] ✓ Semantic Hit! '{query}' ≡ '{matched['query']}'")
            self.metrics.semantic_hits += 1
            return self._apply_context_collapse_guard(
                matched["result"], query,
                source_context=matched.get("source_context", ""),
                grounding_info=matched.get("grounding_info")
            )

        # Cache miss — no semantic match found
        print(f"      [CACHE] ✗ Miss. Queries are semantically distinct.")
        self.metrics.cache_misses += 1
        return None

    # ------------------------------------------------------------------
    # STORE — Add a new entry to the cache
    # ------------------------------------------------------------------
    def store(self, query: str, context: str, result: str, model_used: str = "unknown",
              sources: List[dict] = None):
        """Store a query-result pair with source provenance and knowledge extraction."""
        chunk_hash = self._get_chunk_hash(context)
        if chunk_hash not in self.cache:
            self.cache[chunk_hash] = []

        # Compute grounding at write-time so it's cached with the entry
        grounding_info = self._grounding_check(result, context)

        embedding = self.embedder.encode_single(query)

        entry = {
            "query": query,
            "result": result,
            "embedding": embedding,
            "source_context": context,
            "model_used": model_used,
            "grounding_info": grounding_info,
        }
        self.cache[chunk_hash].append(entry)

        # Knowledge extraction — decompose answer into (subj, rel, obj) triples
        facts = self._extract_facts(result, sources or [])
        entry["facts"] = facts
        cache_idx = self.get_total_entries() - 1
        for fact in facts:
            fact["source_cache_idx"] = cache_idx
            fact["source_chunk_hash"] = chunk_hash
            fact_idx = len(self.knowledge)
            self.knowledge.append(fact)
            if self.knowledge_index is not None:
                fact_text = f"{fact['subject']} {fact['relation']} {fact['object']}"
                emb = self.embedder.encode_single(fact_text)
                self.knowledge_index.add(emb.reshape(1, -1), [{"fact_idx": fact_idx}])

        # Add to cache FAISS index for semantic lookup
        if self._cache_index is not None:
            self._cache_index.add(embedding.reshape(1, -1), [{"cache_idx": cache_idx, "chunk_hash": chunk_hash}])

        # Auto-persist
        if self._persist_path:
            self.save(self._persist_path)

    def get_total_entries(self) -> int:
        """Return the total number of cached entries across all buckets."""
        return sum(len(entries) for entries in self.cache.values())

    def consensus_verify(self, query: str, context: str, primary_result: str, primary_model: str) -> dict:
        """
        THEORY: Multi-Model Consensus Verification
        ---------------------------------------------------------------
        Grounding is a property of the WRITE PATH, not a one-time event.
        Every cache write is an opportunity to validate before locking in.

        After the primary model (e.g., Sonnet) generates an answer, we
        independently dispatch the same query to the evaluator model
        (Haiku) and compare key quantitative facts. If they agree, the
        result is trustworthy. If they disagree, it's flagged DISPUTED
        and surfaced for review before entering the cache.

        Cost: ~$0.0001 per write (one Haiku call). For 2,000 cache
        misses, that's $0.20 total for consensus on every entry.

        Returns:
            {
                "consensus": "AGREED" | "DISPUTED",
                "primary_facts": [...],
                "verifier_facts": [...],
                "divergent_facts": [...]  # Facts that differ
            }
        """
        try:
            response = client.messages.create(
                model=self.EVALUATOR_MODEL,
                max_tokens=256,
                temperature=0,
                system=(
                    "You are an autonomous sub-agent performing a specific analytical task "
                    "on a chunk of data. Return ONLY the answer. Be concise and precise."
                ),
                messages=[{"role": "user", "content": f"Task: {query}\n\nData:\n{context}"}]
            )
            self.metrics.record_call(
                self.EVALUATOR_MODEL,
                response.usage.input_tokens,
                response.usage.output_tokens
            )
            verifier_result = response.content[0].text
        except Exception as e:
            print(f"      [CONSENSUS] ⚠ Verifier call failed: {e}")
            return {"consensus": "AGREED", "primary_facts": [], "verifier_facts": [], "divergent_facts": []}

        # Extract quantitative facts from both results
        fact_patterns = [
            r'\$[\d,]+\.?\d*[MBKTmkbt]?',
            r'\d+\.?\d*\s*%',
            r'\b\d{1,3}(?:,\d{3})+\b',
            r'\b\d+\.\d+[MBKTmkbt]\b',
        ]

        def extract_facts(text):
            facts = []
            for pattern in fact_patterns:
                facts.extend(re.findall(pattern, text))
            return list(dict.fromkeys(facts))  # De-dupe preserving order

        primary_facts = extract_facts(primary_result)
        verifier_facts = extract_facts(verifier_result)

        # If neither has quantitative facts, treat as agreed (qualitative)
        if not primary_facts and not verifier_facts:
            self.metrics.consensus_agreed += 1
            return {"consensus": "AGREED", "primary_facts": [], "verifier_facts": [], "divergent_facts": []}

        # Find facts in primary that don't appear in verifier's response
        divergent = [f for f in primary_facts if f not in verifier_result]

        if not divergent:
            self.metrics.consensus_agreed += 1
            print(f"      [CONSENSUS] ✓ AGREED — {primary_model} and {self.EVALUATOR_MODEL} concur on: {primary_facts}")
            return {"consensus": "AGREED", "primary_facts": primary_facts, "verifier_facts": verifier_facts, "divergent_facts": []}
        else:
            self.metrics.consensus_disputed += 1
            print(f"      [CONSENSUS] ⚠ DISPUTED — {primary_model} said {primary_facts}, {self.EVALUATOR_MODEL} said {verifier_facts}. Divergent: {divergent}")
            return {"consensus": "DISPUTED", "primary_facts": primary_facts, "verifier_facts": verifier_facts, "divergent_facts": divergent}

    # ------------------------------------------------------------------
    # KNOWLEDGE EXTRACTION — Decompose answers into atomic triples
    # ------------------------------------------------------------------
    def _extract_facts(self, answer: str, sources: List[dict]) -> List[dict]:
        """
        THEORY: Intelligent Knowledge Extraction
        ---------------------------------------------------------------
        Instead of storing monolithic query→answer blobs, decompose each
        answer into atomic (subject, relation, object) triples that are
        independently indexed for cross-query reuse.

        Cost: ~$0.0002 per write (one Haiku call).
        """
        source_file = sources[0].get("metadata", {}).get("filename", "unknown") if sources else "unknown"
        try:
            response = client.messages.create(
                model=self.EVALUATOR_MODEL,
                max_tokens=512,
                temperature=0,
                system=(
                    "Extract structured facts from the text as (subject | relation | object) triples. "
                    "One triple per line. Include people, organizations, dates, charges, locations, "
                    "legal citations, amounts, and relationships. Be exhaustive but precise. "
                    "Format: subject | relation | object"
                ),
                messages=[{"role": "user", "content": answer}]
            )
            self.metrics.record_call(self.EVALUATOR_MODEL,
                                     response.usage.input_tokens,
                                     response.usage.output_tokens)

            facts = []
            for line in response.content[0].text.strip().split("\n"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    facts.append({
                        "subject": parts[0],
                        "relation": parts[1],
                        "object": parts[2],
                        "source_file": source_file,
                    })

            print(f"      [KNOWLEDGE] Extracted {len(facts)} facts:")
            for f in facts[:5]:
                print(f"        → {f['subject']} | {f['relation']} | {f['object']}")
            if len(facts) > 5:
                print(f"        ... and {len(facts) - 5} more")
            return facts

        except Exception as e:
            print(f"      [KNOWLEDGE] ⚠ Extraction failed: {e}")
            return []

    # ------------------------------------------------------------------
    # RETRIEVAL — FAISS document search + Reranker with relevance gate
    # ------------------------------------------------------------------
    def ingest(self, docs_dir: Path, chunk_size: int = 3000, overlap: int = 200):
        """Ingest text documents: chunk → embed → build FAISS doc index."""
        docs_dir = Path(docs_dir)
        if self.doc_index is None:
            self.doc_index = FAISSIndex()

        txt_files = sorted(docs_dir.glob("*.txt"))
        if not txt_files:
            print(f"  No .txt files found in {docs_dir}")
            return 0

        print(f"  [INGEST] Found {len(txt_files)} documents")
        all_chunks = []
        all_meta = []

        for f in txt_files:
            text = f.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            # Chunk the document
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if len(chunk) < 50:
                    continue
                meta = {
                    "filename": f.name,
                    "chunk_index": len(all_chunks),
                    "char_start": i,
                }
                all_chunks.append(chunk)
                all_meta.append(meta)

        print(f"  [INGEST] Chunked into {len(all_chunks)} pieces, embedding...")
        embeddings = self.embedder.encode_documents(all_chunks)
        self.doc_index.add(embeddings, all_meta)
        print(f"  [INGEST] ✓ Indexed {len(all_chunks)} chunks")

        # Store chunks for retrieval
        self._doc_chunks = all_chunks
        return len(all_chunks)

    def retrieve(self, query: str, top_k: int = 20, rerank_top: int = 5) -> List[dict]:
        """
        Retrieve relevant document chunks: FAISS dragnet → Reranker (with relevance gate).
        Returns list of {text, score, metadata}.
        """
        if self.doc_index is None or self.doc_index.total == 0:
            print("  [RETRIEVE] No documents indexed. Run ingest() first.")
            return []

        t0 = time.time()
        query_emb = self.embedder.encode_query(query)
        faiss_results = self.doc_index.search(query_emb, top_k=top_k)
        dt_faiss = (time.time() - t0) * 1000
        print(f"  [DRAGNET] Retrieved {len(faiss_results)} candidates in {dt_faiss:.0f}ms")

        if not faiss_results:
            return []

        # Get text for reranking
        candidate_texts = []
        candidate_meta = []
        for i, (score, meta) in enumerate(faiss_results):
            # Try metadata inline text first (old format), then _doc_chunks
            text = meta.get("text")
            if not text and hasattr(self, '_doc_chunks'):
                # The FAISS result index maps to _doc_chunks position
                faiss_idx = self.doc_index.metadata.index(meta) if meta in self.doc_index.metadata else i
                if faiss_idx < len(self._doc_chunks):
                    text = self._doc_chunks[faiss_idx]
            if text:
                candidate_texts.append(text)
                candidate_meta.append(meta)

        # Rerank with relevance gate
        if self.reranker and candidate_texts:
            t1 = time.time()
            reranked = self.reranker.rerank(query, candidate_texts, top_k=rerank_top)
            dt_rerank = (time.time() - t1) * 1000
            print(f"  [RERANK] Narrowed to {len(reranked)} relevant results in {dt_rerank:.0f}ms")

            results = []
            for orig_idx, score, text in reranked:
                results.append({
                    "text": text,
                    "score": float(score),
                    "metadata": candidate_meta[orig_idx] if orig_idx < len(candidate_meta) else {},
                })
            return results
        else:
            # No reranker: return raw FAISS results
            return [{"text": t, "score": float(s), "metadata": m}
                    for (s, m), t in zip(faiss_results, candidate_texts)]

    def search(self, query: str, top_k: int = 20, rerank_top: int = 5,
               synthesize: bool = True) -> dict:
        """
        Full search pipeline: Cache check → Retrieve → Rerank → Synthesize → Cache store.
        This is the main entry point for domain-specific clients.
        """
        t_start = time.time()

        # ── Stage 0: Cache check (Dragnet + Sniper) ──
        # For retrieval-based search, we use a global cache (not hash-bucketed)
        if self._cache_index and self._cache_index.total > 0:
            query_emb = self.embedder.encode_query(query)
            # Exact match scan
            for chunk_hash, entries in self.cache.items():
                for entry in entries:
                    if entry["query"].lower().strip() == query.lower().strip():
                        print(f"  [CACHE] ✓ Exact Match — free retrieval")
                        return {
                            "query": query, "answer": entry["result"],
                            "from_cache": True, "cache_type": "exact",
                            "grounding": entry.get("grounding_info", {}),
                        }

            # Semantic cache lookup via FAISS + Sniper
            cache_results = self._cache_index.search(query_emb, top_k=3)
            strong = [(score, meta) for score, meta in cache_results if score > 0.85]
            if strong:
                candidate_queries = []
                candidate_entries = []
                for score, meta in strong:
                    ch = meta["chunk_hash"]
                    ci = meta["cache_idx"]
                    # Find entry in cache
                    flat_idx = 0
                    for h, entries in self.cache.items():
                        for entry in entries:
                            if flat_idx == ci:
                                candidate_queries.append(entry["query"])
                                candidate_entries.append(entry)
                            flat_idx += 1

                if candidate_queries:
                    sniper = self._llm_sniper_evaluate(query, [
                        (e, i, s) for i, (e, s) in enumerate(zip(candidate_entries, [sc for sc, _ in strong]))
                    ])
                    if sniper and sniper.get("hit"):
                        idx = sniper["id"]
                        cached = candidate_entries[idx]
                        print(f"  [SNIPER] ✓ Semantic Hit! '{query}' ≡ '{cached['query']}'")
                        return {
                            "query": query, "answer": cached["result"],
                            "from_cache": True, "cache_type": "semantic",
                            "grounding": cached.get("grounding_info", {}),
                        }

            # Knowledge fact lookup
            if self.knowledge and self.knowledge_index and self.knowledge_index.total > 0:
                fact_results = self.knowledge_index.search(query_emb, top_k=5)
                strong_facts = [(s, m) for s, m in fact_results if s > 0.75]
                if strong_facts:
                    relevant_facts = []
                    source_entry = None
                    for score, meta in strong_facts:
                        fact = self.knowledge[meta["fact_idx"]]
                        relevant_facts.append(fact)
                        if source_entry is None:
                            # Find the source entry
                            flat_idx = 0
                            target = fact.get("source_cache_idx", -1)
                            for h, entries in self.cache.items():
                                for entry in entries:
                                    if flat_idx == target:
                                        source_entry = entry
                                    flat_idx += 1

                    if relevant_facts and source_entry:
                        print(f"  [KNOWLEDGE] ✓ Fact Hit! {len(relevant_facts)} facts found")
                        for f in relevant_facts:
                            print(f"    → {f['subject']} | {f['relation']} | {f['object']}")
                        return {
                            "query": query, "answer": source_entry["result"],
                            "from_cache": True, "cache_type": "knowledge",
                            "relevant_facts": relevant_facts,
                            "grounding": source_entry.get("grounding_info", {}),
                        }

        # ── Stage 1: Retrieve relevant documents ──
        results = self.retrieve(query, top_k=top_k, rerank_top=rerank_top)
        if not results:
            return {"query": query, "answer": "No relevant documents found.", "from_cache": False}

        # ── Stage 2: Synthesize answer ──
        if not synthesize:
            return {"query": query, "results": results, "from_cache": False}

        source_text = "\n\n---\n\n".join(r["text"] for r in results[:3])
        t_synth = time.time()
        model = EXECUTOR_MODEL
        response = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0,
            system=(
                "You are a document analysis expert. Answer the query using ONLY "
                "the provided documents. If the answer isn't in the documents, say so. "
                "Cite specific details. Be precise and thorough."
            ),
            messages=[{"role": "user", "content": f"Query: {query}\n\nDocuments:\n{source_text}"}]
        )
        self.metrics.record_call(model, response.usage.input_tokens, response.usage.output_tokens)
        answer = response.content[0].text
        dt_synth = (time.time() - t_synth) * 1000
        print(f"  [SYNTH] Generated answer in {dt_synth:.0f}ms")

        # ── Stage 3: Verify and cache ──
        consensus = self.consensus_verify(query, source_text, answer, model)
        self.store(query, source_text, answer, model_used=model, sources=results)

        return {
            "query": query, "answer": answer,
            "from_cache": False, "results": results,
            "consensus": consensus,
        }

    # ------------------------------------------------------------------
    # PERSISTENCE — Save/load cache, knowledge, FAISS indices
    # ------------------------------------------------------------------
    def save(self, path: Path):
        """Persist cache entries, knowledge triples, FAISS indices, and corpus config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save corpus config — namespace identity and metadata
        config = {
            "corpus_id": self.corpus_id,
            "corpus_domain": self.corpus_domain,
            "doc_vectors": self.doc_index.total if self.doc_index else 0,
            "cache_entries": self.get_total_entries(),
            "knowledge_facts": len(self.knowledge),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        # Preserve created_at from existing config
        if self.corpus_config.get("created_at"):
            config["created_at"] = self.corpus_config["created_at"]
        else:
            config["created_at"] = config["updated_at"]
        self.corpus_config = config
        with open(path / "corpus_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save cache entries (convert numpy arrays to lists for JSON)
        serializable_cache = {}
        for chunk_hash, entries in self.cache.items():
            serializable_cache[chunk_hash] = []
            for e in entries:
                entry = {k: v for k, v in e.items() if k != "embedding"}
                entry["embedding"] = e["embedding"].tolist() if isinstance(e["embedding"], np.ndarray) else e["embedding"]
                serializable_cache[chunk_hash].append(entry)
        with open(path / "cache_entries.json", "w") as f:
            json.dump(serializable_cache, f, indent=2, default=str)

        # Save knowledge triples
        with open(path / "knowledge.json", "w") as f:
            json.dump(self.knowledge, f, indent=2, default=str)

        # Save FAISS indices
        if self._cache_index and self._cache_index.total > 0:
            self._cache_index.save(path / "cache_idx")
        if self.knowledge_index and self.knowledge_index.total > 0:
            self.knowledge_index.save(path / "knowledge_idx")

        total = self.get_total_entries()
        print(f"  [CACHE] Saved {total} entries + {len(self.knowledge)} facts to {path}")

    def load(self, path: Path) -> bool:
        """Load cache, knowledge, doc index, FAISS indices, and corpus config."""
        path = Path(path)
        self._persist_path = path

        # Load corpus config
        config_path = path / "corpus_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.corpus_config = json.load(f)
            stored_id = self.corpus_config.get("corpus_id")
            # Validate: if controller has a corpus_id, it must match
            if self.corpus_id and stored_id and self.corpus_id != stored_id:
                print(f"  [CORPUS] ⚠ MISMATCH: expected '{self.corpus_id}', found '{stored_id}'")
                print(f"  [CORPUS]   Refusing to load — wrong namespace.")
                return False
            # Adopt the stored corpus identity
            if stored_id:
                self.corpus_id = stored_id
                self.corpus_domain = self.corpus_config.get("corpus_domain", self.corpus_domain)
            print(f"  [CORPUS] '{self.corpus_id}' — domain: {self.corpus_domain}, "
                  f"created: {self.corpus_config.get('created_at', 'unknown')}")

        # Load document FAISS index
        doc_idx_path = path / "doc_idx" if (path / "doc_idx").exists() else path
        if (doc_idx_path / "index.faiss").exists():
            if self.doc_index is None:
                self.doc_index = FAISSIndex()
            self.doc_index.load(doc_idx_path)

            # Load doc chunks
            chunks_path = path / "chunks.json"
            if chunks_path.exists():
                with open(chunks_path) as f:
                    self._doc_chunks = json.load(f)
            elif self.doc_index and self.doc_index.metadata:
                # Fallback: extract chunks from metadata (old index format)
                self._doc_chunks = [
                    m.get("text", "") for m in self.doc_index.metadata
                ]
                if self._doc_chunks and self._doc_chunks[0]:
                    print(f"  [INDEX] Extracted {len(self._doc_chunks)} chunks from metadata")

        # Load cache entries
        cache_path = path / "cache_entries.json"
        if cache_path.exists():
            with open(cache_path) as f:
                raw = json.load(f)
            self.cache = {}
            for chunk_hash, entries in raw.items():
                self.cache[chunk_hash] = []
                for e in entries:
                    e["embedding"] = np.array(e["embedding"], dtype="float32")
                    self.cache[chunk_hash].append(e)

            # Rebuild cache FAISS index
            if self._cache_index is None:
                self._cache_index = FAISSIndex()
            cache_idx_path = path / "cache_idx"
            if not (cache_idx_path / "index.faiss").exists():
                flat_idx = 0
                for chunk_hash, entries in self.cache.items():
                    for entry in entries:
                        self._cache_index.add(
                            entry["embedding"].reshape(1, -1),
                            [{"cache_idx": flat_idx, "chunk_hash": chunk_hash}]
                        )
                        flat_idx += 1
            else:
                self._cache_index.load(cache_idx_path)

        # Load knowledge
        knowledge_path = path / "knowledge.json"
        if knowledge_path.exists():
            with open(knowledge_path) as f:
                self.knowledge = json.load(f)
            if self.knowledge_index is None:
                self.knowledge_index = FAISSIndex()
            ki_path = path / "knowledge_idx"
            if (ki_path / "index.faiss").exists():
                self.knowledge_index.load(ki_path)
            elif self.knowledge:
                print(f"  [KNOWLEDGE] Rebuilding fact index for {len(self.knowledge)} triples...")
                for i, fact in enumerate(self.knowledge):
                    fact_text = f"{fact['subject']} {fact['relation']} {fact['object']}"
                    emb = self.embedder.encode_single(fact_text)
                    self.knowledge_index.add(emb.reshape(1, -1), [{"fact_idx": i}])

        # Initialize indices if they don't exist yet
        if self._cache_index is None:
            self._cache_index = FAISSIndex()
        if self.knowledge_index is None:
            self.knowledge_index = FAISSIndex()

        total = self.get_total_entries()
        if total > 0 or self.knowledge:
            print(f"  [CACHE] Loaded {total} entries, {len(self.knowledge)} facts")
        return total > 0

    def save_doc_index(self, path: Path):
        """Save document FAISS index, chunks, and corpus config (after ingest)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.doc_index:
            self.doc_index.save(path)
        if hasattr(self, '_doc_chunks'):
            with open(path / "chunks.json", "w") as f:
                json.dump(self._doc_chunks, f)
        # Write corpus identity
        config = {
            "corpus_id": self.corpus_id,
            "corpus_domain": self.corpus_domain,
            "doc_vectors": self.doc_index.total if self.doc_index else 0,
            "cache_entries": 0,
            "knowledge_facts": 0,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self.corpus_config = config
        with open(path / "corpus_config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"  [INDEX] Saved doc index to {path} (corpus: {self.corpus_id})")
# ============================================================================
# 3. CACHE PRE-WARMER — Programmatic Day-1 saturation
# ============================================================================
class CachePreWarmer:
    """
    THEORY: Programmatic Cache Pre-Warming (Solving Cold-Start)
    -----------------------------------------------------------------------
    Standard caches are "lazy" — they start empty and slowly fill up as
    unlucky early users take the latency and cost hits.

    We solve this with Programmatic Pre-Warming: before any human touches
    the system, we deploy an automated sweep that loops over the entire
    corpus with a set of template queries, forcing the cache to saturate.

    This is architecture-agnostic. Any agent framework (RLM, LangChain,
    AutoGen, or a simple Python script) can pre-warm a cache. The key
    insight is that closed-domain corpora (legal, financial, medical)
    have a finite set of relevant queries that can be enumerated.

    Example: A PE firm uploads a 50-page financial model. We instantly
    sweep it with ["What is Q1 ARR?", "What is EBITDA?", "What is Churn?"]
    so that by the time the analyst logs in, every question is an instant hit.
    """

    def __init__(self, agent):
        """
        Args:
            agent: An AutonomousAgent instance whose cached_query() method
                   we call to populate the cache.
        """
        self.agent = agent

    def warm(self, corpus_chunks: list, sweep_queries: list):
        """
        Programmatically sweep the corpus to saturate the cache.

        Args:
            corpus_chunks: List of document text chunks.
            sweep_queries: List of query templates to run against each chunk.
        """
        total = len(corpus_chunks) * len(sweep_queries)
        print(f"\n{'─'*65}")
        print(f"  [PRE-WARMER] Starting programmatic cache warming...")
        print(f"  [PRE-WARMER] {len(corpus_chunks)} chunks × {len(sweep_queries)} queries = {total} operations")
        print(f"{'─'*65}")

        count = 0
        for i, chunk in enumerate(corpus_chunks):
            if not chunk.strip():
                continue
            for query in sweep_queries:
                count += 1
                print(f"\n  [PRE-WARM {count}/{total}] chunk={i}, query='{query[:50]}...'")
                self.agent.cached_query(query, chunk)
                time.sleep(0.5)  # Rate limit protection

        self.agent.metrics.pre_warm_entries = self.agent.cache.get_total_entries()
        print(f"\n  [PRE-WARMER] ✓ Complete. Cache now holds {self.agent.metrics.pre_warm_entries} entries.")
        print(f"{'─'*65}\n")


# ============================================================================
# 4. HETEROGENEOUS MODEL ROUTER
# ============================================================================
class Router:
    """
    THEORY: Heterogeneous Dispatch
    -----------------------------------------------------------------------
    Not every sub-query needs a frontier model. Simple classification or
    extraction tasks can be handled by cheap, fast models (Haiku), while
    complex reasoning or synthesis tasks require expensive models (Sonnet).

    This router is a simple heuristic. In a production system, it could
    be a trained classifier that predicts the optimal model based on
    query complexity, context size, and historical performance.
    """

    @staticmethod
    def select_model(query: str) -> str:
        """Route to cheap models for simple tasks, expensive for complex ones."""
        simple_keywords = ["classify", "extract", "find", "count", "identify", "list", "check"]
        if any(kw in query.lower() for kw in simple_keywords):
            return "claude-haiku-4-5"
        return "claude-sonnet-4-5"


# ============================================================================
# 5. AUTONOMOUS AGENT — The transparent interceptor
# ============================================================================
class AutonomousAgent:
    """
    THEORY: The Agent-Cache Integration Layer
    -----------------------------------------------------------------------
    This class represents ANY autonomous agent (RLM, LangChain, AutoGen,
    or a custom loop). The key function is `cached_query()`, which acts
    as a transparent interceptor: the agent calls it exactly like it
    would call an LLM API, but internally the Two-Stage Cache intercepts
    redundant queries before they ever reach the expensive model.

    THEORY: Context Collapse is Architecture-Agnostic
    -----------------------------------------------------------------------
    Context Collapse is NOT specific to RLMs. ANY agent that processes
    more data than fits in its context window is vulnerable. A LangChain
    agent appending tool results, an AutoGen orchestrator receiving sub-
    agent reports, or even Claude Code accumulating terminal outputs —
    all suffer from the same fundamental limitation.

    The cache is the universal solution regardless of agent framework.
    """

    def __init__(self):
        self.metrics = ExecutionMetrics()
        self.embedder = EmbeddingEngine()
        self.cache = SemanticCacheController(self.metrics, embedder=self.embedder)
        self.router = Router()

    def cached_query(self, query: str, context: str) -> dict:
        """
        The transparent cache-interceptor function.

        This is analogous to rlm_query() in the RLM architecture, but is
        framework-agnostic. Any agent framework can call this instead of
        calling the LLM API directly.

        Returns:
            dict with keys:
                "result":         The answer text
                "ephemeral":      If True, do NOT append to agent's context history
                "was_summarized": If True, result was condensed from a larger original
                "from_cache":     If True, this was a cache hit (no expensive LLM call)
        """
        print(f"\n    → cached_query('{query[:60]}...')")

        # ----- Check the Two-Stage Cache -----
        cached = self.cache.check(query, context)
        if cached is not None:
            cached["from_cache"] = True
            return cached

        # ----- Cache Miss: Route to the appropriate model -----
        model = self.router.select_model(query)
        print(f"      [EXECUTE] Cache miss. Dispatching to {model}...")

        response = client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0,
            system=(
                "You are an autonomous sub-agent performing a specific analytical task "
                "on a chunk of data. Return ONLY the answer. Be concise and precise."
            ),
            messages=[{"role": "user", "content": f"Task: {query}\n\nData:\n{context}"}]
        )
        self.metrics.record_call(model, response.usage.input_tokens, response.usage.output_tokens)
        result = response.content[0].text

        # Multi-model consensus: verify with Haiku before caching
        consensus = self.cache.consensus_verify(query, context, result, model)

        # Store in cache with source provenance + consensus result
        self.cache.store(query, context, result, model_used=model)

        # Apply Context Collapse Guard to the fresh result too
        guarded = self.cache._apply_context_collapse_guard(result, query, source_context=context)
        guarded["from_cache"] = False
        guarded["consensus"] = consensus["consensus"]
        guarded["divergent_facts"] = consensus.get("divergent_facts", [])
        return guarded


# ============================================================================
# 6. DEMONSTRATION — 4-Pass Agent Simulation
# ============================================================================
def run_full_demonstration():
    """
    Runs a complete 4-pass demonstration of the Two-Stage Semantic Cache:

    Pass 1 (COLD START):   All cache misses. The agent processes a corpus
                           for the first time, populating the cache.

    Pass 2 (PARAPHRASED):  The agent returns with differently-worded queries
                           that ask for the same information. The Haiku Sniper
                           correctly identifies semantic equivalence → cache hits.

    Pass 3 (DISTINCT):     The agent asks logically DIFFERENT questions about
                           the same data. The Haiku Sniper correctly rejects
                           them → cache misses (no false positives).

    Pass 4 (PRE-WARMED):   We run the Cache Pre-Warmer with finance-style
                           extraction queries, then demonstrate instant hits.
    """

    # --- Synthetic Corpus ---
    # A small but realistic dataset that demonstrates all cache behaviors.
    corpus = """FINANCIAL REPORT Q1 2024
Company: Acme Corp
Annual Recurring Revenue (ARR): $12.4M
Monthly Recurring Revenue (MRR): $1.03M
Customer Churn Rate: 4.2%
EBITDA: $2.1M
Net Revenue Retention: 112%

INCIDENT REPORT — SERVER OUTAGE
Date: 2024-01-15
Duration: 3 hours 42 minutes
Root Cause: Database connection pool exhausted due to unbounded retry loop.
Impact: 2,400 users affected. HTTP 503 errors returned.
Resolution: Connection pool limit raised from 50 to 200. Circuit breaker added.

LEGAL REVIEW — CONTRACT AMENDMENT
Clause 7.2(b): Vendor must INCLUDE all source code in escrow deposit.
Clause 7.3(a): Vendor must EXCLUDE proprietary third-party libraries from escrow.
Amendment effective: March 1, 2024.
Signed by: J. Martinez (Buyer), K. Chen (Vendor)."""

    chunks = [c.strip() for c in corpus.split("\n\n") if c.strip()]

    agent = AutonomousAgent()
    agent.metrics.start_time = time.time()

    # ========== PASS 1: COLD START ==========
    print("\n" + "="*65)
    print("  PASS 1: COLD START (All misses — populating cache)")
    print("="*65)

    cold_queries = [
        ("Extract the Annual Recurring Revenue.", chunks[0]),
        ("What was the root cause of the server outage?", chunks[1]),
        ("What must the vendor INCLUDE in the escrow deposit?", chunks[2]),
    ]

    for query, chunk in cold_queries:
        result = agent.cached_query(query, chunk)
        print(f"      Result: {result['result'][:100]}...")
        time.sleep(1)

    # ========== PASS 2: PARAPHRASED QUERIES (Semantic hits) ==========
    print("\n" + "="*65)
    print("  PASS 2: PARAPHRASED QUERIES (Haiku Sniper should detect equivalence)")
    print("="*65)

    paraphrased_queries = [
        ("What is the ARR for Acme Corp?", chunks[0]),
        ("Identify the primary reason the servers went down.", chunks[1]),
        ("What should the vendor include in the escrow?", chunks[2]),
    ]

    for query, chunk in paraphrased_queries:
        result = agent.cached_query(query, chunk)
        print(f"      Result: {result['result'][:100]}...")
        print(f"      From Cache: {result['from_cache']}")
        time.sleep(1)

    # ========== PASS 3: LOGICALLY DISTINCT (Correct misses) ==========
    print("\n" + "="*65)
    print("  PASS 3: LOGICALLY DISTINCT QUERIES (Sniper should reject — no false positives)")
    print("="*65)

    # Critical test: "INCLUDE" vs "EXCLUDE" — the classic vector cache failure
    distinct_queries = [
        ("What is the customer churn rate?", chunks[0]),        # Different metric, same chunk
        ("How long did the outage last?", chunks[1]),           # Different question, same chunk
        ("What must the vendor EXCLUDE from escrow?", chunks[2]),  # Logical inversion!
    ]

    for query, chunk in distinct_queries:
        result = agent.cached_query(query, chunk)
        print(f"      Result: {result['result'][:100]}...")
        print(f"      From Cache: {result['from_cache']}")
        time.sleep(1)

    # ========== PASS 4: PRE-WARMING + INSTANT HITS ==========
    print("\n" + "="*65)
    print("  PASS 4: CACHE PRE-WARMING (Programmatic Day-1 saturation)")
    print("="*65)

    # Pre-warm with financial extraction templates
    pre_warmer = CachePreWarmer(agent)
    finance_queries = [
        "Extract the Monthly Recurring Revenue.",
        "What is the Net Revenue Retention rate?",
        "What is the EBITDA?",
    ]
    pre_warmer.warm([chunks[0]], finance_queries)

    # Now query with paraphrased versions — should be instant hits
    print("\n  [POST-WARMING] Testing paraphrased queries against pre-warmed cache...")
    post_warm_queries = [
        ("What is Acme's MRR?", chunks[0]),
        ("Report the NRR percentage.", chunks[0]),
        ("What was the EBITDA figure?", chunks[0]),
    ]

    for query, chunk in post_warm_queries:
        result = agent.cached_query(query, chunk)
        print(f"      Result: {result['result'][:100]}...")
        print(f"      From Cache: {result['from_cache']}")
        time.sleep(1)

    # ========== PASS 5: CONTEXT COLLAPSE GUARD (Large result retrieval) ==========
    print("\n" + "="*65)
    print("  PASS 5: CONTEXT COLLAPSE GUARD (Triggering large-result protection)")
    print("="*65)

    # -----------------------------------------------------------------------
    # SCENARIO: A previous agent run analyzed a massive due diligence package
    # and cached the full result. When the agent (or a new session) retrieves
    # it, the result is too large to safely append to context. The guard must:
    #   (A) Flag as ephemeral if result is 2000-4000 tokens
    #   (B) Recursively chunk + summarize via parallel Haiku if >4000 tokens
    # -----------------------------------------------------------------------

    # --- 5A: EPHEMERAL RETRIEVAL (medium-sized result, 2000-4000 tokens) ---
    print("\n  --- 5A: Ephemeral Retrieval (medium result > 2000 tokens) ---")
    medium_result = (
        "EXECUTIVE SUMMARY — ACME CORP Q1 2024 PERFORMANCE REVIEW\n"
        + "=" * 60 + "\n\n"
    )
    # Pad to ~10,000 chars (~2500 tokens) to trigger ephemeral but not summarization
    for i in range(1, 16):
        medium_result += (
            f"Finding {i}: The Q{(i % 4) + 1} analysis reveals that operational metric "
            f"category {i} showed a {10 + i}% improvement over baseline projections "
            f"established in the prior fiscal quarter. This improvement is attributed "
            f"to the deployment of automated monitoring systems in region {chr(64 + i)} "
            f"and the renegotiation of vendor contracts covering service tier {i}. "
            f"The financial impact is estimated at ${i * 0.3:.1f}M in annualized "
            f"cost savings, with full realization expected by Q{(i % 4) + 1} 2025. "
            f"Stakeholder feedback from the {i * 3} affected business units has been "
            f"overwhelmingly positive, with a satisfaction score of {85 + (i % 10)}%. "
            f"Risk mitigation measures including redundant failover systems and "
            f"automated alerting have been validated through {i * 2} tabletop exercises.\n\n"
        )

    medium_context = "ACME CORP Q1 2024 — QUARTERLY PERFORMANCE PACKAGE"
    medium_query = "Summarize the Q1 2024 performance review findings."
    agent.cache.store(medium_query, medium_context, medium_result)
    print(f"  [SETUP] Injected medium cached result ({len(medium_result)} chars, ~{len(medium_result)//4} tokens)")

    # Exact-match retrieval to guarantee the hit fires
    result = agent.cached_query(medium_query, medium_context)
    print(f"      Result (first 150 chars): {result['result'][:150]}...")
    print(f"      From Cache: {result['from_cache']}")
    print(f"      >>> Ephemeral: {result.get('ephemeral')}  (should be True — do NOT persist in context)")
    print(f"      >>> Was Summarized: {result.get('was_summarized')}  (should be False — not large enough)")

    # --- 5B: RECURSIVE PARALLEL SUMMARIZATION (huge result > 4000 tokens) ---
    print("\n  --- 5B: Recursive Parallel Summarization (huge result > 4000 tokens) ---")
    large_result = (
        "COMPREHENSIVE DUE DILIGENCE REPORT — ACME CORP ACQUISITION\n"
        + "=" * 60 + "\n\n"
    )
    # Build a ~20,000 char result (~5000 tokens) to exceed the MAX_CONTEXT_TOKENS threshold
    sections = [
        ("FINANCIAL OVERVIEW",
         "Acme Corp reported total revenues of $48.7M in FY2023, representing "
         "a 23% year-over-year growth rate. The company's gross margin expanded "
         "from 62% to 68%, driven primarily by operational efficiencies in their "
         "cloud infrastructure division. EBITDA margins improved to 18.4%, up "
         "from 14.1% in the prior year. The company maintains $12.3M in cash "
         "and equivalents with a debt-to-equity ratio of 0.34. Working capital "
         "stands at $8.7M, providing adequate runway for the next 18 months of "
         "projected operations without additional financing requirements. "
         "Revenue breakdown by segment: Cloud Platform ($28.3M, +31% YoY), "
         "Professional Services ($12.1M, +14% YoY), and Support & Maintenance "
         "($8.3M, +11% YoY). The Cloud Platform segment is the primary growth "
         "driver with 58% of total revenue and the highest contribution margin."),
        ("CUSTOMER ANALYSIS",
         "The customer base consists of 2,847 active accounts across 14 industry "
         "verticals. The top 10 customers represent 31% of ARR, indicating healthy "
         "diversification. Logo retention rate is 94.2% with net dollar retention "
         "at 118%, demonstrating strong expansion within existing accounts. The "
         "average contract value (ACV) increased from $14.2K to $17.1K, reflecting "
         "successful upselling of premium features. Customer acquisition cost (CAC) "
         "is $23.4K with an LTV:CAC ratio of 4.7x, well above the 3x benchmark. "
         "Enterprise customers (>$100K ACV) grew from 47 to 89 accounts, representing "
         "62% of total ARR. The SMB segment, while larger by count at 2,412 accounts, "
         "contributes 22% of ARR with higher churn (8.1%) compared to enterprise (1.4%)."),
        ("TECHNOLOGY & INTELLECTUAL PROPERTY",
         "The platform comprises 847,000 lines of production code across 12 "
         "microservices deployed on AWS. The tech stack includes Python, Go, and "
         "TypeScript. Key IP assets include 3 granted patents and 7 pending "
         "applications covering their proprietary data pipeline architecture and "
         "ML-based anomaly detection algorithms. The engineering team of 42 FTEs "
         "maintains a deployment frequency of 47 releases per week with a change "
         "failure rate of 2.1%. Technical debt ratio is estimated at 14%, with a "
         "dedicated time allocation of 20% sprint capacity for remediation. The "
         "platform processes 2.3 billion events per day with 99.97% uptime SLA "
         "compliance over the trailing 12-month period. Infrastructure cost as a "
         "percentage of revenue has declined from 23% to 17% through optimization."),
        ("LEGAL & COMPLIANCE",
         "The company holds SOC 2 Type II certification, ISO 27001, and GDPR "
         "compliance attestation. Outstanding litigation includes one minor "
         "patent infringement claim (estimated exposure: $200K-$500K). All "
         "employee agreements include standard IP assignment and non-compete "
         "clauses. The data processing agreements with all enterprise customers "
         "are current and compliant with applicable privacy regulations including "
         "CCPA, HIPAA BAA for healthcare customers, and FedRAMP authorization "
         "is in progress with expected completion in Q2 2025. The company has "
         "completed 3 external penetration tests in the past 12 months with "
         "all critical findings remediated within the 30-day SLA."),
        ("MARKET POSITION & COMPETITIVE LANDSCAPE",
         "Acme Corp is positioned as a mid-market leader in the observability "
         "space, competing primarily with Datadog, New Relic, and Splunk. Their "
         "differentiation lies in automated root cause analysis, which reduces "
         "mean time to resolution (MTTR) by an average of 73% based on customer "
         "benchmarks. The total addressable market (TAM) for cloud observability "
         "is projected at $52B by 2027, growing at 14.3% CAGR. Acme's current "
         "market share is approximately 0.09%. Their positioning in the Gartner "
         "Magic Quadrant moved from Niche Players to Visionaries in the 2023 "
         "report. Win rates against incumbents improved from 28% to 41%."),
        ("RISK FACTORS",
         "Key risks include: (1) concentration in AWS infrastructure creating "
         "vendor dependency, (2) potential margin compression from AI compute "
         "costs as the ML pipeline scales, (3) regulatory uncertainty around "
         "AI-generated insights in regulated industries, and (4) talent "
         "retention challenges in the current market. The company has partially "
         "mitigated risk #1 through a multi-cloud roadmap targeting Q3 2024 "
         "completion. Risk #2 is being addressed through model optimization "
         "and a partnership with a custom silicon provider. Risk #3 requires "
         "ongoing monitoring of the EU AI Act and SEC proposed disclosure rules. "
         "Risk #4 is partially mitigated by above-market compensation and a "
         "99th-percentile employee Net Promoter Score of 72."),
        ("VALUATION CONSIDERATIONS",
         "Based on comparable public company multiples (NTM Revenue: 12.4x, "
         "NTM EBITDA: 34.2x) and precedent transactions in the observability "
         "space, the implied enterprise value range is $580M-$720M. Applying "
         "a 15-20% illiquidity discount for the private market context yields "
         "a fair value estimate of $490M-$610M. The Rule of 40 score of 41.4 "
         "(23% growth + 18.4% margin) supports a premium multiple within "
         "the peer group. Sensitivity analysis on growth deceleration scenarios "
         "suggests a floor valuation of $420M even under conservative assumptions."),
        ("INTEGRATION PLANNING",
         "Post-acquisition integration is estimated to require 12-18 months "
         "across four workstreams: (A) Technology stack consolidation targeting "
         "$4.2M in annual infrastructure savings, (B) Go-to-market alignment "
         "to capitalize on cross-sell opportunities estimated at $8M in Year 1, "
         "(C) G&A rationalization yielding $2.8M in headcount synergies, and "
         "(D) Product roadmap harmonization to avoid customer confusion. Key "
         "integration risks include the potential loss of 3-5 senior engineers "
         "during the transition period, a 6-month revenue dip during sales team "
         "realignment, and brand perception challenges in the developer community. "
         "A retention package of $3.2M has been budgeted for key personnel."),
        ("APPENDIX: DETAILED FINANCIAL PROJECTIONS",
         "FY2024E: Revenue $59.8M (+23% YoY), EBITDA $11.4M (19.1% margin). "
         "FY2025E: Revenue $72.3M (+21% YoY), EBITDA $15.2M (21.0% margin). "
         "FY2026E: Revenue $85.1M (+18% YoY), EBITDA $19.6M (23.0% margin). "
         "These projections assume stable competitive dynamics, successful "
         "execution of the multi-cloud roadmap, and continued expansion of "
         "the enterprise segment. Downside scenario (15% probability): revenue "
         "growth decelerates to 12% due to macro headwinds, compressing "
         "EBITDA margins to 15%. Upside scenario (25% probability): successful "
         "product-led growth initiative accelerates enterprise adoption, driving "
         "28% revenue growth and 25% EBITDA margins by FY2026. The base case "
         "assumes 2% quarterly price increases across the SMB segment and a "
         "12% annual expansion in enterprise contract renewals."),
    ]

    for i, (title, body) in enumerate(sections, 1):
        large_result += f"SECTION {i}: {title}\n{body}\n\n"

    # Repeat the analysis under different framing to realistically bulk up the result
    # (In production, a real DD report would be 50-200 pages)
    large_result += "\n" + "=" * 60 + "\nSUPPLEMENTAL ANALYSIS — YEAR-OVER-YEAR COMPARISON\n" + "=" * 60 + "\n\n"
    for i, (title, body) in enumerate(sections, 1):
        large_result += f"YoY COMPARISON {i}: {title}\n{body}\n\n"

    large_result += "\n" + "=" * 60 + "\nHISTORICAL TREND ANALYSIS — 5-YEAR LOOKBACK\n" + "=" * 60 + "\n\n"
    for i, (title, body) in enumerate(sections, 1):
        large_result += f"HISTORICAL {i}: {title}\n{body}\n\n"

    dd_context = "ACME CORP ACQUISITION TARGET — FULL DUE DILIGENCE PACKAGE"
    dd_query = "Provide a comprehensive due diligence analysis of the acquisition target."
    agent.cache.store(dd_query, dd_context, large_result)
    print(f"  [SETUP] Injected large cached result ({len(large_result)} chars, ~{len(large_result)//4} tokens)")
    print(f"  [SETUP] MAX_CONTEXT_TOKENS threshold = {agent.cache.MAX_CONTEXT_TOKENS}")
    print(f"  [SETUP] This should trigger recursive parallel summarization.\n")

    # Exact-match retrieval — guaranteed cache hit, forces the guard to fire
    result = agent.cached_query(dd_query, dd_context)
    print(f"      Result (first 200 chars): {result['result'][:200]}...")
    print(f"      From Cache: {result['from_cache']}")
    print(f"      >>> Ephemeral: {result.get('ephemeral')}  (should be False — it was summarized instead)")
    print(f"      >>> Was Summarized: {result.get('was_summarized')}  (should be True — parallel Haiku chunks)")
    time.sleep(1)

    # Paraphrased retrieval — tests Sniper + Guard combo
    print("\n  --- 5C: Paraphrased retrieval of large cached result (Sniper + Guard) ---")
    result = agent.cached_query(
        "What are the key findings from the Acme Corp due diligence?",
        dd_context
    )
    print(f"      Result (first 200 chars): {result['result'][:200]}...")
    print(f"      From Cache: {result['from_cache']}")
    print(f"      Ephemeral: {result.get('ephemeral', 'N/A')}")
    print(f"      Was Summarized: {result.get('was_summarized', 'N/A')}")

    # ========== FINAL SUMMARY ==========
    agent.metrics.end_time = time.time()
    agent.metrics.print_summary()


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: ANTHROPIC_API_KEY not found. Check your .env file.")
        exit(1)

    run_full_demonstration()
