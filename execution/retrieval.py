import time


def index_documents(scs, controller, task, config):
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
    embedded_at = time.perf_counter()
    embeddings = controller.embedder.encode_documents(texts)
    embedding_ms = (time.perf_counter() - embedded_at) * 1000
    maximum = controller.embedder._last_encode_info["max_sequence_tokens"]
    if maximum >= scs.EMBEDDING_MAX_LENGTH:
        raise ValueError("A child reached the embedding input limit; reduce --child-tokens")
    controller.doc_index = scs.FAISSIndex()
    controller.doc_index.add(embeddings, metadata)
    controller._doc_chunks = texts
    controller._doc_chunk_metadata = metadata
    return {"ingested_chunks": len(texts), "document_embedding_ms": embedding_ms,
            "ingest_ms": (time.perf_counter() - started) * 1000,
            "child_encoded_length_max": maximum}
