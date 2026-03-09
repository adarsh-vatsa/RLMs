"""
================================================================================
EPSTEIN DOCUMENT SEARCH ENGINE
================================================================================

Upgraded Two-Stage Semantic Cache applied to 1,000+ Epstein court documents.

Upgrades from base system:
  - Qwen3-Embedding-0.6B  (replaces all-MiniLM-L6-v2)
  - FAISS IndexFlatIP      (replaces numpy brute-force)
  - Qwen3-Reranker-0.6B   (new layer between Dragnet and Sniper)
  - Document ingestion pipeline for .txt court filings

Usage:
    python3 epstein_search.py --ingest      Ingest documents + build FAISS index
    python3 epstein_search.py --search "query"  Search the corpus
    python3 epstein_search.py --interactive  Interactive search mode

Requirements:
    pip install faiss-cpu transformers torch anthropic python-dotenv numpy
================================================================================
"""

import os
import sys
import json
import re
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from anthropic import Anthropic

# IMPORTANT: faiss is imported lazily to avoid segfault conflict with
# transformers on Apple Silicon. Imported in FAISSIndex.__init__.


load_dotenv()
client = Anthropic()

# ============================================================================
# CONFIGURATION
# ============================================================================

DOCS_DIR = Path(__file__).parent / "Epstein-Downloader" / "documents" / "github_ocr" / "IMAGES001"
INDEX_DIR = Path(__file__).parent / "epstein_index"

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
EVALUATOR_MODEL = "claude-haiku-4-5"
EXECUTOR_MODEL = "claude-sonnet-4-5"

EMBEDDING_DIM = 1024
MAX_CHUNK_CHARS = 2000          # ~500 tokens per chunk
OVERLAP_CHARS = 200             # Overlap between chunks
TOP_K_VECTOR = 20               # Dragnet retrieves 20
TOP_K_RERANK = 5                # Reranker narrows to 5
SNIPER_BATCH_SIZE = 5           # Max candidates per Sniper call


# ============================================================================
# 1. EMBEDDING ENGINE — Qwen3-Embedding-0.6B via Transformers
# ============================================================================

class EmbeddingEngine:
    """
    Qwen3-Embedding-0.6B with instruction-aware embedding.
    32K context, 1024-dim output, instruction-aware for legal retrieval.
    """

    def __init__(self):
        print("  [EMBED] Loading Qwen3-Embedding-0.6B...")
        from transformers import AutoModel, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            EMBEDDING_MODEL, trust_remote_code=True, torch_dtype=torch.float32
        )
        self.model.eval()
        # Use CPU — MPS has known segfault issues with some transformer architectures
        self.device = "cpu"
        self.model.to(self.device)
        self.torch = torch
        print(f"  [EMBED] ✓ Loaded on {self.device} ({sum(p.numel() for p in self.model.parameters())/1e6:.0f}M params)")

    def encode(self, texts: List[str], instruction: str = "") -> np.ndarray:
        """Encode texts with optional instruction prefix."""
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]

        with self.torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            # Use last hidden state [CLS] token or mean pooling
            embeddings = outputs.last_hidden_state[:, 0, :]
            # Normalize
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        return embeddings.cpu().numpy().astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a search query with retrieval instruction."""
        return self.encode(
            [query],
            instruction="Given a legal query, retrieve relevant court documents and filings"
        )

    def encode_documents(self, docs: List[str]) -> np.ndarray:
        """Encode document chunks (no instruction for documents)."""
        # Process in batches to avoid OOM
        batch_size = 16
        all_embeddings = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            emb = self.encode(batch)
            all_embeddings.append(emb)
            if (i + batch_size) % 100 == 0:
                print(f"    Embedded {min(i+batch_size, len(docs))}/{len(docs)} chunks...")
        return np.vstack(all_embeddings)


# ============================================================================
# 2. RERANKER — Qwen3-Reranker-0.6B
# ============================================================================

class Reranker:
    """
    Qwen3-Reranker-0.6B: generative cross-encoder that scores query-document pairs.
    Uses AutoModelForCausalLM with yes/no token probability scoring.
    Sits between Vector Dragnet (broad) and LLM Sniper (expensive).
    """

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

        # Token IDs for yes/no scoring
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        # Chat template prefix/suffix for the reranker
        self.max_length = 8192
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

        print(f"  [RERANK] ✓ Loaded on {self.device}")

    def _format_pair(self, query: str, doc: str, instruction: str = None) -> str:
        """Format a query-document pair for reranking."""
        if instruction is None:
            instruction = "Given a legal query, retrieve relevant court documents and filings"
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, pairs: List[str]):
        """Tokenize with prefix/suffix chat template tokens."""
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Re-score query-document pairs and return top-k.
        Returns: [(original_index, score, document_text), ...]
        """
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

        results = []
        for orig_idx, score in ranked[:top_k]:
            results.append((orig_idx, float(score), documents[orig_idx]))

        return results


# ============================================================================
# 3. FAISS INDEX — Replaces brute-force numpy
# ============================================================================

class FAISSIndex:
    """
    FAISS IndexFlatIP (inner product on normalized vectors = cosine similarity).
    Exact search but vectorized — much faster than raw numpy at scale.
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        import faiss as _faiss
        self._faiss = _faiss
        self.dim = dim
        self.index = self._faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []  # Parallel array of chunk metadata

    def add(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        """Add embeddings with associated metadata."""
        self.index.add(embeddings)
        self.metadata.extend(metadata_list)

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[float, Dict]]:
        """Return top-k results with scores and metadata."""
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for empty slots
                results.append((float(score), self.metadata[idx]))
        return results

    def save(self, path: Path):
        """Persist index and metadata to disk."""
        path.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "metadata.json", "w") as f:
            json.dump(self.metadata, f)
        print(f"  [FAISS] Saved {self.index.ntotal} vectors to {path}")

    def load(self, path: Path) -> bool:
        """Load index and metadata from disk."""
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
        return self.index.ntotal


# ============================================================================
# 4. DOCUMENT INGESTION — .txt court filings → chunks → FAISS
# ============================================================================

def chunk_document(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> List[str]:
    """Split a document into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        # Try to break at sentence boundary
        if end < len(text):
            last_period = text.rfind(".", start, end)
            last_newline = text.rfind("\n", start, end)
            break_at = max(last_period, last_newline)
            if break_at > start + max_chars // 2:
                end = break_at + 1
        chunk = text[start:end].strip()
        if len(chunk) > 50:  # Skip tiny fragments
            chunks.append(chunk)
        start = end - overlap
    return chunks


def ingest_documents(docs_dir: Path, embedding_engine: EmbeddingEngine, faiss_index: FAISSIndex) -> int:
    """Read all .txt files, chunk them, embed, and add to FAISS."""
    txt_files = sorted(docs_dir.glob("*.txt"))
    if not txt_files:
        print(f"  No .txt files found in {docs_dir}")
        return 0

    print(f"\n  [INGEST] Found {len(txt_files)} documents in {docs_dir.name}")

    all_chunks = []
    all_metadata = []

    for i, txt_file in enumerate(txt_files):
        try:
            content = txt_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"    Error reading {txt_file.name}: {e}")
            continue

        # Extract metadata from header
        doc_meta = {"filename": txt_file.name}
        for line in content.split("\n")[:20]:
            if ":" in line and not line.startswith("="):
                key, val = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                if key in ["people", "organizations", "locations", "dates", "document_type", "date"]:
                    doc_meta[key] = val.strip()

        # Extract full text section
        full_text_marker = "FULL TEXT"
        ft_idx = content.find(full_text_marker)
        if ft_idx >= 0:
            # Skip past the marker and the "=====" line
            text = content[ft_idx + len(full_text_marker):]
            text = text.lstrip("=\n ")
        else:
            text = content

        chunks = chunk_document(text)
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                **doc_meta,
                "chunk_index": j,
                "total_chunks": len(chunks),
                "text": chunk
            })

        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{len(txt_files)} files ({len(all_chunks)} chunks total)")

    print(f"  [INGEST] Total: {len(all_chunks)} chunks from {len(txt_files)} documents")

    # Embed all chunks
    print(f"  [EMBED] Encoding {len(all_chunks)} chunks with Qwen3-Embedding-0.6B...")
    embeddings = embedding_engine.encode_documents(all_chunks)
    print(f"  [EMBED] ✓ Generated {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

    # Add to FAISS
    faiss_index.add(embeddings, all_metadata)
    print(f"  [FAISS] Index now contains {faiss_index.total} vectors")

    return len(all_chunks)


# ============================================================================
# 5. SEMANTIC CACHE — Sniper + Grounding + Consensus
# ============================================================================

class SemanticCache:
    """
    Two-Stage Semantic Cache (Dragnet & Sniper) adapted for document search.

    Stores query→answer pairs. On new queries:
      1. Exact match check (free)
      2. Vector similarity in cache index (Qwen3 embeddings, free)
      3. LLM Sniper validates semantic equivalence (Haiku, ~$0.0001)

    On cache miss + new answer:
      1. Grounding check (regex, free)
      2. Consensus verification (Haiku, ~$0.0001)
      3. Store with full provenance
    """

    def __init__(self, embedder: EmbeddingEngine):
        self.embedder = embedder
        self.entries: List[Dict] = []        # All cached query→answer pairs
        self.cache_index = FAISSIndex()       # Separate FAISS index for cache queries
        self.stats = {
            "exact_hits": 0, "semantic_hits": 0, "misses": 0,
            "grounded": 0, "partial": 0, "inferred": 0,
            "consensus_agreed": 0, "consensus_disputed": 0,
            "sniper_calls": 0, "consensus_calls": 0,
            "api_cost": 0.0,
        }

    def check(self, query: str) -> Optional[dict]:
        """
        Full cache lookup: Exact → Vector Dragnet → LLM Sniper.
        Returns cached result dict or None.
        """
        if not self.entries:
            self.stats["misses"] += 1
            return None

        # ── Exact match (free) ──
        q_lower = query.lower().strip()
        for entry in self.entries:
            if entry["query"].lower().strip() == q_lower:
                self.stats["exact_hits"] += 1
                print(f"  [CACHE] ✓ Exact Match — free retrieval")
                return entry

        # ── Vector Dragnet in cache index ──
        query_emb = self.embedder.encode_query(query)
        candidates = self.cache_index.search(query_emb, top_k=5)

        if not candidates:
            self.stats["misses"] += 1
            return None

        # Filter by similarity threshold
        strong_candidates = [(score, meta) for score, meta in candidates if score > 0.85]
        if not strong_candidates:
            self.stats["misses"] += 1
            print(f"  [CACHE] ✗ No strong matches (top: {candidates[0][0]:.3f})")
            return None

        print(f"  [CACHE DRAGNET] {len(strong_candidates)} candidates (top: {strong_candidates[0][0]:.3f})")

        # ── LLM Sniper — semantic equivalence check ──
        candidate_queries = []
        for _, meta in strong_candidates[:3]:
            idx = meta["cache_idx"]
            candidate_queries.append(self.entries[idx]["query"])

        try:
            self.stats["sniper_calls"] += 1
            sniper_prompt = f"""You are evaluating whether a NEW query is semantically equivalent to cached queries.
The user wants the SAME information — just phrased differently.

NEW QUERY: {query}

CACHED QUERIES:
{chr(10).join(f'  [{i}] {q}' for i, q in enumerate(candidate_queries))}

If any cached query asks for the SAME information as the new query, respond with ONLY the number in brackets. Otherwise respond with NONE."""

            response = client.messages.create(
                model=EVALUATOR_MODEL,
                max_tokens=32,
                temperature=0,
                messages=[{"role": "user", "content": sniper_prompt}]
            )
            cost = (response.usage.input_tokens * 0.25 / 1_000_000) + \
                   (response.usage.output_tokens * 1.25 / 1_000_000)
            self.stats["api_cost"] += cost
            verdict = response.content[0].text.strip()

            # Parse Sniper response
            match = re.search(r'\[(\d+)\]', verdict)
            if match is None and verdict.isdigit():
                match_idx = int(verdict)
            elif match:
                match_idx = int(match.group(1))
            else:
                self.stats["misses"] += 1
                print(f"  [SNIPER] ✗ No semantic match (verdict: {verdict})")
                return None

            # Valid cache hit via Sniper
            if 0 <= match_idx < len(strong_candidates):
                cache_idx = strong_candidates[match_idx][1]["cache_idx"]
                cached = self.entries[cache_idx]
                self.stats["semantic_hits"] += 1
                print(f"  [SNIPER] ✓ Semantic Hit! '{query}' ≡ '{cached['query']}'")
                return cached

        except Exception as e:
            print(f"  [SNIPER] ⚠ Error: {e}")

        self.stats["misses"] += 1
        return None

    def store(self, query: str, answer: str, sources: List[dict],
              source_text: str, timing: dict) -> dict:
        """
        Store a new cache entry with grounding + consensus verification.
        """
        # ── Grounding check (free) ──
        grounding = self._grounding_check(answer, source_text)
        status = grounding["grounding"]
        if status == "GROUNDED":
            self.stats["grounded"] += 1
            print(f"  [PROVENANCE] ✓ GROUNDED: all facts verified in source")
        elif status == "PARTIAL":
            self.stats["partial"] += 1
            print(f"  [PROVENANCE] ⚠ PARTIAL: verified={grounding['verified_facts']}, unverified={grounding['unverified_facts']}")
        else:
            self.stats["inferred"] += 1
            print(f"  [PROVENANCE] ⚠ INFERRED: unverified={grounding['unverified_facts']}")

        # ── Consensus verification (Haiku call) ──
        consensus = self._consensus_verify(query, source_text, answer)

        # ── Store entry ──
        entry = {
            "query": query,
            "answer": answer,
            "sources": sources,
            "grounding": grounding,
            "consensus": consensus,
            "timing": timing,
            "timestamp": time.time(),
        }
        cache_idx = len(self.entries)
        self.entries.append(entry)

        # Add query embedding to cache FAISS index
        query_emb = self.embedder.encode_query(query)
        self.cache_index.add(query_emb, [{"cache_idx": cache_idx}])

        return entry

    def _grounding_check(self, result: str, source_context: str) -> dict:
        """Extract quantitative facts and verify against source text."""
        fact_patterns = [
            r'\$[\d,]+\.?\d*[MBKTmkbt]?',
            r'\d+\.?\d*\s*%',
            r'\b\d{1,3}(?:,\d{3})+\b',
            r'\b\d+\.\d+[MBKTmkbt]\b',
        ]
        all_facts = []
        for pattern in fact_patterns:
            all_facts.extend(re.findall(pattern, result))

        unique_facts = list(dict.fromkeys(all_facts))
        if not unique_facts:
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

    def _consensus_verify(self, query: str, context: str, primary_result: str) -> dict:
        """Independent Haiku verification for consensus on write."""
        try:
            self.stats["consensus_calls"] += 1
            response = client.messages.create(
                model=EVALUATOR_MODEL,
                max_tokens=256,
                temperature=0,
                system="You are a legal research assistant. Answer the query using ONLY the provided documents. Be concise and precise.",
                messages=[{"role": "user", "content": f"Query: {query}\n\nDocuments:\n{context[:6000]}"}]
            )
            cost = (response.usage.input_tokens * 0.25 / 1_000_000) + \
                   (response.usage.output_tokens * 1.25 / 1_000_000)
            self.stats["api_cost"] += cost
            verifier_result = response.content[0].text
        except Exception as e:
            print(f"  [CONSENSUS] ⚠ Verifier failed: {e}")
            return {"consensus": "AGREED", "divergent_facts": []}

        def extract_facts(text):
            facts = []
            for p in [r'\$[\d,]+\.?\d*[MBKTmkbt]?', r'\d+\.?\d*\s*%', r'\b\d{1,3}(?:,\d{3})+\b']:
                facts.extend(re.findall(p, text))
            return list(dict.fromkeys(facts))

        primary_facts = extract_facts(primary_result)
        verifier_facts = extract_facts(verifier_result)

        if not primary_facts and not verifier_facts:
            self.stats["consensus_agreed"] += 1
            print(f"  [CONSENSUS] ✓ AGREED (qualitative)")
            return {"consensus": "AGREED", "divergent_facts": []}

        divergent = [f for f in primary_facts if f not in verifier_result]
        if not divergent:
            self.stats["consensus_agreed"] += 1
            print(f"  [CONSENSUS] ✓ AGREED — Sonnet and Haiku concur on: {primary_facts}")
        else:
            self.stats["consensus_disputed"] += 1
            print(f"  [CONSENSUS] ⚠ DISPUTED — divergent: {divergent}")

        return {"consensus": "AGREED" if not divergent else "DISPUTED", "divergent_facts": divergent}

    def print_stats(self):
        """Print cache statistics."""
        total = self.stats["exact_hits"] + self.stats["semantic_hits"] + self.stats["misses"]
        hit_rate = (self.stats["exact_hits"] + self.stats["semantic_hits"]) / total * 100 if total else 0
        grounded_total = self.stats["grounded"] + self.stats["partial"] + self.stats["inferred"]
        grounding_rate = self.stats["grounded"] / grounded_total * 100 if grounded_total else 0
        consensus_total = self.stats["consensus_agreed"] + self.stats["consensus_disputed"]
        consensus_rate = self.stats["consensus_agreed"] / consensus_total * 100 if consensus_total else 0

        print(f"  CACHE PERFORMANCE")
        print(f"    Exact Hits       : {self.stats['exact_hits']}")
        print(f"    Semantic Hits    : {self.stats['semantic_hits']}")
        print(f"    Misses           : {self.stats['misses']}")
        print(f"    Hit Rate         : {hit_rate:.1f}%")
        print(f"  SOURCE PROVENANCE")
        print(f"    Grounded         : {self.stats['grounded']}")
        print(f"    Partial          : {self.stats['partial']}")
        print(f"    Inferred         : {self.stats['inferred']}")
        print(f"    Grounding Rate   : {grounding_rate:.1f}%")
        print(f"    Consensus Agreed : {self.stats['consensus_agreed']}")
        print(f"    Consensus Disputed: {self.stats['consensus_disputed']}")
        print(f"    Consensus Rate   : {consensus_rate:.1f}%")
        print(f"  CACHE COST         : ${self.stats['api_cost']:.4f}")


# ============================================================================
# 6. SEARCH ENGINE — Full Pipeline with Semantic Cache
# ============================================================================

class EpsteinSearchEngine:
    """
    Full search pipeline with Two-Stage Semantic Cache:
      Query → Cache Check (free/cheap)
        ├─ HIT  → Serve cached answer (with provenance)
        └─ MISS → FAISS Dragnet → Reranker → Sonnet synthesis
                    → Grounding check → Consensus verify → Cache store
    """

    def __init__(self):
        print("\n" + "=" * 70)
        print("INITIALIZING EPSTEIN SEARCH ENGINE")
        print("=" * 70)

        self.embedder = EmbeddingEngine()
        self.reranker = Reranker()
        self.faiss_index = FAISSIndex()
        self.cache = SemanticCache(self.embedder)

        # Try to load existing index
        if not self.faiss_index.load(INDEX_DIR):
            print("  [FAISS] No existing index found. Run --ingest first.")

        self.stats = {
            "queries": 0,
            "api_calls": 0,
            "api_cost": 0.0,
        }

        print("=" * 70)
        print(f"  Ready. {self.faiss_index.total} vectors indexed.")
        print("=" * 70)

    def ingest(self):
        """Ingest documents and build the FAISS index."""
        count = ingest_documents(DOCS_DIR, self.embedder, self.faiss_index)
        if count > 0:
            self.faiss_index.save(INDEX_DIR)

    def search(self, query: str, top_k: int = 5, synthesize: bool = True) -> dict:
        """
        Full search pipeline with semantic cache.
        """
        self.stats["queries"] += 1
        print(f"\n{'─'*70}")
        print(f"  QUERY: {query}")
        print(f"{'─'*70}")

        timing = {}

        # ══ SEMANTIC CACHE CHECK ══
        t0 = time.time()
        cached = self.cache.check(query)
        timing["cache_ms"] = (time.time() - t0) * 1000

        if cached:
            return {
                "query": query,
                "results": cached.get("sources", []),
                "answer": cached["answer"],
                "timing": timing,
                "from_cache": True,
                "grounding": cached.get("grounding", {}),
                "consensus": cached.get("consensus", {}),
            }

        # ══ CACHE MISS — Run full pipeline ══

        # ── Stage 1: FAISS Vector Dragnet ──
        t0 = time.time()
        query_emb = self.embedder.encode_query(query)
        dragnet_results = self.faiss_index.search(query_emb, top_k=TOP_K_VECTOR)
        timing["dragnet_ms"] = (time.time() - t0) * 1000
        print(f"  [DRAGNET] Retrieved {len(dragnet_results)} candidates in {timing['dragnet_ms']:.0f}ms")
        if dragnet_results:
            print(f"            Top score: {dragnet_results[0][0]:.4f}, Bottom: {dragnet_results[-1][0]:.4f}")

        if not dragnet_results:
            return {"query": query, "results": [], "answer": "No results found.", "timing": timing, "from_cache": False}

        # ── Stage 2: Reranker ──
        t0 = time.time()
        candidate_texts = [r[1]["text"] for r in dragnet_results]
        reranked = self.reranker.rerank(query, candidate_texts, top_k=TOP_K_RERANK)
        timing["rerank_ms"] = (time.time() - t0) * 1000
        print(f"  [RERANK] Narrowed to {len(reranked)} candidates in {timing['rerank_ms']:.0f}ms")
        for i, (orig_idx, score, _) in enumerate(reranked):
            source_meta = dragnet_results[orig_idx][1]
            print(f"           #{i+1}: score={score:.4f} | {source_meta.get('filename', '?')} chunk {source_meta.get('chunk_index', '?')}")

        # Build results
        results = []
        for orig_idx, score, text in reranked:
            meta = dragnet_results[orig_idx][1].copy()
            meta.pop("text", None)
            results.append({"score": score, "text": text, "metadata": meta})

        # ── Stage 3: LLM Synthesis ──
        answer = ""
        source_text = ""
        if synthesize and results:
            t0 = time.time()
            source_text = "\n\n---\n\n".join([
                f"[Source: {r['metadata'].get('filename', 'unknown')}, "
                f"Chunk {r['metadata'].get('chunk_index', '?')}]\n{r['text']}"
                for r in results[:3]
            ])

            try:
                response = client.messages.create(
                    model=EXECUTOR_MODEL,
                    max_tokens=512,
                    temperature=0,
                    system=(
                        "You are a legal research assistant analyzing Epstein case documents. "
                        "Answer the query using ONLY the provided source documents. "
                        "Cite specific document sources. If the documents don't contain "
                        "the answer, say so explicitly. Never fabricate information."
                    ),
                    messages=[{"role": "user", "content": f"Query: {query}\n\nSource Documents:\n{source_text}"}]
                )
                answer = response.content[0].text
                cost = (response.usage.input_tokens * 3 / 1_000_000) + \
                       (response.usage.output_tokens * 15 / 1_000_000)
                self.stats["api_calls"] += 1
                self.stats["api_cost"] += cost
                timing["synthesize_ms"] = (time.time() - t0) * 1000
                print(f"  [SYNTH] Generated answer in {timing['synthesize_ms']:.0f}ms (${cost:.4f})")
            except Exception as e:
                answer = f"Synthesis error: {e}"

        # ── Stage 4: Cache with verification ──
        if answer and synthesize:
            entry = self.cache.store(query, answer, results, source_text, timing)

        return {
            "query": query,
            "results": results,
            "answer": answer,
            "timing": timing,
            "from_cache": False,
            "grounding": entry.get("grounding", {}) if answer else {},
            "consensus": entry.get("consensus", {}) if answer else {},
        }

    def print_stats(self):
        """Print session statistics."""
        print(f"\n{'='*70}")
        print(f"  SESSION STATISTICS")
        print(f"{'='*70}")
        print(f"  Queries         : {self.stats['queries']}")
        print(f"  API Calls       : {self.stats['api_calls']} (synthesis)")
        print(f"  Synthesis Cost  : ${self.stats['api_cost']:.4f}")
        print(f"  Index Size      : {self.faiss_index.total} vectors")
        print(f"  Cache Entries   : {len(self.cache.entries)}")
        print(f"{'─'*70}")
        self.cache.print_stats()
        print(f"{'='*70}")


# ============================================================================
# 6. CLI INTERFACE
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Search Epstein court documents with Two-Stage Semantic Cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 epstein_search.py --ingest
  python3 epstein_search.py --search "Who traveled to the island?"
  python3 epstein_search.py --search "financial transactions" --no-synth
  python3 epstein_search.py --interactive
        """
    )

    parser.add_argument("--ingest", action="store_true", help="Ingest documents and build FAISS index")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--interactive", action="store_true", help="Interactive search mode")
    parser.add_argument("--no-synth", action="store_true", help="Skip LLM synthesis, show raw results only")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")

    args = parser.parse_args()

    if args.ingest:
        # Only need embedding engine for ingestion
        engine = EpsteinSearchEngine()
        engine.ingest()
        engine.print_stats()

    elif args.search:
        engine = EpsteinSearchEngine()
        result = engine.search(args.search, top_k=args.top_k, synthesize=not args.no_synth)
        print(f"\n{'='*70}")
        print(f"  ANSWER")
        print(f"{'='*70}")
        print(f"\n{result['answer']}\n")
        print(f"\n  Sources:")
        for i, r in enumerate(result["results"]):
            print(f"    {i+1}. {r['metadata'].get('filename', '?')} (chunk {r['metadata'].get('chunk_index', '?')}, score: {r['score']:.4f})")
        engine.print_stats()

    elif args.interactive:
        engine = EpsteinSearchEngine()
        print("\n  Type your query (or 'quit' to exit, 'stats' for statistics):\n")

        while True:
            try:
                query = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not query:
                continue
            if query.lower() == "quit":
                break
            if query.lower() == "stats":
                engine.print_stats()
                continue

            result = engine.search(query, synthesize=True)
            print(f"\n  ANSWER:\n  {result['answer']}\n")
            print(f"  Sources:")
            for i, r in enumerate(result["results"]):
                print(f"    {i+1}. {r['metadata'].get('filename', '?')} "
                      f"(chunk {r['metadata'].get('chunk_index', '?')}, score: {r['score']:.4f})")
                print(f"       Preview: {r['text'][:120]}...\n")

        engine.print_stats()

    else:
        parser.print_help()


if __name__ == "__main__":
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: ANTHROPIC_API_KEY not found. Check your .env file.")
        sys.exit(1)

    main()
