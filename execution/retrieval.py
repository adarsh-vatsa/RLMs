from collections import Counter
import math
import random
import re
import time


WORD = re.compile(r"\w+")
FUSION_K = 60


def index_documents(scs, controller, task, config, *, embed=True):
    """Build the active index from exact source text; keep controller internals here."""
    started = time.perf_counter()
    if config.child_tokens >= scs.EMBEDDING_MAX_LENGTH:
        raise ValueError("Child tokens must be below the embedding input limit")
    texts, metadata = [], []
    for document in task.documents:
        chunks = scs._chunk_text_with_tokenizer(
            document.text, tokenizer=controller.embedder.tokenizer,
            chunk_tokens=config.child_tokens, overlap_tokens=config.child_overlap_tokens,
        )
        if not chunks:
            raise ValueError("Exact embedding-tokenizer offsets are required")
        for text, meta in chunks:
            start, end = meta["char_start"], meta["char_end"]
            if not 0 <= start < end <= len(document.text) or text != document.text[start:end]:
                raise ValueError("Embedding tokenizer returned invalid source offsets")
            metadata.append({**meta, "chunk_index": len(texts), "child_index": len(texts),
                             "document_id": document.id, "filename": document.id,
                             "source_id": task.source_id, "tokenizer_model": scs.EMBEDDING_MODEL})
            texts.append(text)
    controller._doc_chunks = texts
    controller._doc_chunk_metadata = metadata
    if not embed:
        return {"ingested_chunks": len(texts), "document_embedding_ms": 0.0,
                "ingest_ms": (time.perf_counter() - started) * 1000, "child_encoded_length_max": 0}
    embedded_at = time.perf_counter()
    embeddings = controller.embedder.encode_documents(texts)
    embedding_ms = (time.perf_counter() - embedded_at) * 1000
    maximum = controller.embedder._last_encode_info["max_sequence_tokens"]
    if maximum >= scs.EMBEDDING_MAX_LENGTH:
        raise ValueError("A child reached the embedding input limit; reduce --child-tokens")
    controller.doc_index = scs.FAISSIndex()
    controller.doc_index.add(embeddings, metadata)
    return {"ingested_chunks": len(texts), "document_embedding_ms": embedding_ms,
            "ingest_ms": (time.perf_counter() - started) * 1000,
            "child_encoded_length_max": maximum}


def lexical_index(texts):
    documents = [Counter(WORD.findall(text.lower())) for text in texts]
    return documents, [sum(document.values()) for document in documents], Counter(
        term for document in documents for term in document)


def bm25_scores(query, index, k1=1.2, b=0.75):
    documents, lengths, frequency = index
    average = sum(lengths) / len(lengths)
    scores = []
    for document, length in zip(documents, lengths):
        score = 0.0
        for term in set(WORD.findall(query.lower())):
            count = document.get(term, 0)
            if count:
                idf = math.log(1 + (len(documents) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * count * (k1 + 1) / (count + k1 * (1 - b + b * length / average))
        scores.append(score)
    return scores


def order_candidates(selection, query, chunks, dense, lexical, *, seed, case_id):
    """Order index-order chunks for packing; `dense` is the retriever's ranking when one exists."""
    if selection == "dense":
        return dense
    if selection == "random":
        ordered = list(chunks)
        random.Random(f"{seed}:{case_id}").shuffle(ordered)
        return ordered
    if selection == "position":
        counts = Counter(chunk["metadata"]["document_id"] for chunk in chunks)
        documents = list(counts)
        seen = Counter()
        keys = []
        for chunk in chunks:
            document = chunk["metadata"]["document_id"]
            position = seen[document]
            seen[document] += 1
            keys.append((min(position, counts[document] - 1 - position), documents.index(document), position))
        return [chunk for _, chunk in sorted(zip(keys, chunks), key=lambda pair: pair[0])]
    lexical_ranked = sorted(zip(bm25_scores(query, lexical), range(len(chunks))), key=lambda pair: -pair[0])
    if selection == "bm25":
        return [dict(chunks[index], score=score) for score, index in lexical_ranked]
    rank = {}
    for ranking in ([item["metadata"]["child_index"] for item in dense],
                    [chunks[index]["metadata"]["child_index"] for _, index in lexical_ranked]):
        for position, child in enumerate(ranking, 1):
            rank[child] = rank.get(child, 0.0) + 1 / (FUSION_K + position)
    return sorted((dict(chunk, score=rank[chunk["metadata"]["child_index"]]) for chunk in chunks),
                  key=lambda chunk: -chunk["score"])


def rerank_head(reranker, query, candidates, top):
    head = candidates[:top]
    scored = reranker.rerank(query, [item["text"] for item in head], top_k=len(head),
                             relevance_threshold=float("-inf"))
    return [dict(head[index], score=score) for index, score, _ in scored] + candidates[top:]


def document_floor(candidates):
    seen, leaders, rest = set(), [], []
    for item in candidates:
        document = item["metadata"]["document_id"]
        (rest if document in seen else leaders).append(item)
        seen.add(document)
    return leaders + rest


def expand_neighbors(candidates, chunks, documents, width):
    texts = {document.id: document.text for document in documents}
    by_child = {chunk["metadata"]["child_index"]: chunk["metadata"] for chunk in chunks}
    expanded = []
    for item in candidates:
        meta = item["metadata"]
        span = [by_child[child] for child in range(meta["child_index"] - width, meta["child_index"] + width + 1)
                if child in by_child and by_child[child]["document_id"] == meta["document_id"]]
        start, end = min(m["char_start"] for m in span), max(m["char_end"] for m in span)
        expanded.append({**item, "text": texts[meta["document_id"]][start:end], "metadata": {
            **meta, "char_start": start, "char_end": end,
            "expanded_child_indices": [m["child_index"] for m in span]}})
    return expanded
