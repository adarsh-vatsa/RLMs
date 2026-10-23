import json
from pathlib import Path
import time

from execution.client import call_with_retries
from execution.contracts import Config, Task, fingerprint
from execution.packing import pack
from execution.retrieval import (document_floor, expand_neighbors, index_documents, lexical_index,
                                 order_candidates, rerank_head)
from execution.tokens import TRUNCATION_ROUTES, chat_token_count, route, truncate_prompt


class Pipeline:
    def __init__(self, config: Config, tokenizer, complete, *, backend_factory=None, output_dir=None):
        self.config = config
        self.tokenizer = tokenizer
        self.complete = complete
        self.backend_factory = backend_factory
        self.scs = self.controller = None
        self.embedding_identity = None
        self.active_index = None
        self.lexical = None
        self.tokenizer_identity = {
            "class": type(tokenizer).__name__,
            "revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
            "chat_template_sha256": fingerprint(getattr(tokenizer, "chat_template", None)),
            "vocab_sha256": fingerprint(tokenizer.get_vocab()) if hasattr(tokenizer, "get_vocab") else None,
        }
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            metadata = {**config.metadata(), "executor_tokenizer": self.tokenizer_identity,
                        "temperature": 0, "enable_thinking": False}
            (self.output_dir / "execution_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def backend(self):
        if self.controller is None:
            if self.backend_factory is None:
                raise ValueError("Retrieval or answer caching requires a backend")
            self.scs, self.controller = self.backend_factory()
            self.embedding_identity = {key: getattr(self.scs, key, None) for key in (
                "EMBEDDING_MODEL", "EMBEDDING_MAX_LENGTH", "EMBEDDING_QUERY_INSTRUCTION", "EMBEDDING_CONTRACT_VERSION")}
            embedder = getattr(self.controller, "embedder", None)
            self.embedding_identity.update(
                tokenizer_revision=getattr(getattr(embedder, "tokenizer", None), "init_kwargs", {}).get("_commit_hash"),
                dtype_effective=getattr(embedder, "torch_dtype_name", None))
            if self.output_dir:
                embedding = {key: getattr(self.scs, key, None) for key in (
                    "EMBEDDING_MODEL", "EMBEDDING_MAX_LENGTH", "EMBEDDING_QUERY_INSTRUCTION",
                    "EMBEDDING_CONTRACT_VERSION", "EMBEDDING_BATCH_SIZE", "EMBEDDING_DTYPE", "EMBEDDING_DEVICE",
                    "RERANKER_MODEL", "RERANKER_RELEVANCE_THRESHOLD", "RERANKER_MAX_LENGTH", "MIN_RERANKED_RESULTS")}
                embedding.update(device_effective=str(getattr(embedder, "device", "")),
                    dtype_effective=getattr(embedder, "torch_dtype_name", None),
                    tokenizer_revision=getattr(getattr(embedder, "tokenizer", None), "init_kwargs", {}).get("_commit_hash"),
                    cache_verifier=getattr(self.controller, "verifier_metadata", None))
                (self.output_dir / "embedding_manifest.json").write_text(json.dumps(embedding, indent=2) + "\n")
        return self.controller

    def preflight(self, task: Task):
        full_tokens = chat_token_count(self.tokenizer, task.render(None))
        return {"case_id": task.case_id, "full_rendered_input_tokens": full_tokens,
                "route": route(self.config.mode, full_tokens, self.config.max_input_tokens, self.config.direct_overflow),
                "cache_assumption": "miss"}

    def execute(self, task: Task):
        started = time.perf_counter()
        c = self.config
        result = {
            "case_id": task.case_id, "source_id": task.source_id,
            "source_content_sha256": "", "prompt_version": task.prompt_version,
            "status": "ok", "error": "", "route": "", "prediction": "",
            "full_rendered_input_tokens": 0, "final_rendered_input_tokens": 0,
            "selected_evidence_ranges": [], "selected_child_indices": [], "candidate_count": 0,
            "packing_stop_reason": "", "packing_renders": 0, "packing_trimmed_chars": 0,
            "evidence_selection": "", "ingested_chunks": 0, "child_encoded_length_max": 0,
            "faiss_candidates": [],
            "ingest_ms": 0.0, "document_embedding_ms": 0.0, "retrieval_ms": 0.0,
            "packing_ms": 0.0, "generation_ms": 0.0, "cache_verification_ms": 0.0,
            "input_tokens": 0, "output_tokens": 0, "raw_usage": {}, "finish_reason": None,
            "attempts": 0, "from_cache": False, "cache_type": "miss", "cache_written": False,
            "cache_provenance": {}, "semantic_verifier_calls": 0,
            "verifier_attempts": 0,
            "verifier_input_tokens": 0, "verifier_output_tokens": 0,
        }
        try:
            request = task.render(None)
            full_tokens = chat_token_count(self.tokenizer, request)
            result["full_rendered_input_tokens"] = full_tokens
            source_key = fingerprint([(doc.id, doc.text) for doc in task.documents])
            result["source_content_sha256"] = source_key
            result["prompt_version"] = task.prompt_version
            if c.cache_read or c.cache_write:
                self.backend()
            scope = fingerprint([source_key, c.cache_identity(), self.tokenizer_identity, task.prompt_version, task.choices,
                                 task.required_prefix, task.fixed_instructions, self.embedding_identity])
            if c.cache_read or c.cache_write:
                self.backend().activate_data_scope(scope)
            if c.cache_read:
                controller = self.backend()
                before = controller.metrics.get_totals()
                verifier_attempts_before = getattr(controller, "verifier_attempts", 0)
                cached_at = time.perf_counter()
                if c.cache_matching == "exact":
                    cached = controller.lookup_cached_result(task.query, semantic=False, strict_exact=True)
                elif c.profile == "common":
                    cached = controller.lookup_cached_result(task.query, strict_exact=True)
                else:
                    cached = controller.lookup_cached_result(task.query)
                after = controller.metrics.get_totals()
                result.update(cache_verification_ms=(time.perf_counter() - cached_at) * 1000,
                              verifier_attempts=getattr(controller, "verifier_attempts", 0) - verifier_attempts_before,
                              semantic_verifier_calls=int(after["calls"] - before["calls"]),
                              verifier_input_tokens=after["input_tokens"] - before["input_tokens"],
                              verifier_output_tokens=after["output_tokens"] - before["output_tokens"])
                if cached is not None and self._eligible(task, cached["answer"]):
                    result.update(prediction=cached["answer"], from_cache=True,
                                  route=f"{cached['cache_type']}_cache", cache_type=cached["cache_type"],
                                  cache_provenance=cached.get("cache_provenance", {}))
            if not result["from_cache"]:
                result["route"] = route(c.mode, full_tokens, c.max_input_tokens, c.direct_overflow)
                if result["route"] == "unsupported_context":
                    result["status"] = "unsupported_context"
                else:
                    if result["route"] in TRUNCATION_ROUTES.values():
                        ending = self._ending(task, request) if c.direct_overflow == "head" else ""
                        request = truncate_prompt(self.tokenizer, request, c.max_input_tokens, c.direct_overflow, ending)
                    elif result["route"] == "dense_child_packed":
                        controller = self.backend()
                        if self.active_index != source_key:
                            if c.exact_offsets:
                                result.update(index_documents(self.scs, controller, task, c,
                                    embed=c.evidence_selection in {"dense", "fusion"}))
                                self.lexical = (lexical_index(controller._doc_chunks)
                                                if c.evidence_selection in {"bm25", "fusion"} else None)
                            else:
                                if task.legacy_docs_dir is None:
                                    raise ValueError("Legacy ingestion requires a document directory")
                                ingest_at = time.perf_counter()
                                result["ingested_chunks"] = controller.ingest(
                                    task.legacy_docs_dir, data_scope_hash=scope, source_id=task.source_id,
                                    ordered_filenames=[doc.id for doc in task.documents],
                                )
                                info = getattr(controller, "_last_ingest_info", {})
                                result.update(ingest_ms=(time.perf_counter() - ingest_at) * 1000,
                                              document_embedding_ms=info.get("embedding_ms", 0),
                                              child_encoded_length_max=info.get("child_encoded_length_max", 0))
                                if result["child_encoded_length_max"] >= self.scs.EMBEDDING_MAX_LENGTH:
                                    raise ValueError("A child reached the embedding input limit; reduce --child-tokens")
                            self.active_index = source_key
                        retrieve_at = time.perf_counter()
                        candidates = self._candidates(task, controller)
                        result["faiss_candidates"] = [{"child_index": item["metadata"].get("child_index", item["metadata"].get("chunk_index")),
                                                       "score": float(item.get("score", 0))} for item in candidates]
                        if c.neighbor_children:
                            chunks = [{"metadata": meta} for meta in controller._doc_chunk_metadata]
                            candidates = expand_neighbors(candidates, chunks, task.documents, c.neighbor_children)
                        result["evidence_selection"] = c.evidence_selection
                        result["retrieval_ms"] = (time.perf_counter() - retrieve_at) * 1000
                        packed_at = time.perf_counter()
                        request, packing = pack(self.tokenizer, task.render, task.documents, candidates,
                                                c.max_input_tokens, order=c.evidence_order, merge=c.merge_overlaps,
                                                policy=c.packing)
                        result.update(packing)
                        result["packing_ms"] = (time.perf_counter() - packed_at) * 1000
                    result["final_rendered_input_tokens"] = chat_token_count(self.tokenizer, request)
                    if result["final_rendered_input_tokens"] > c.max_input_tokens:
                        raise ValueError("Rendered executor request exceeds input budget")
                    generation_at = time.perf_counter()
                    try:
                        completion, attempts = call_with_retries(lambda: self.complete(request), max_retries=c.max_retries)
                    finally:
                        result["generation_ms"] = (time.perf_counter() - generation_at) * 1000
                    result.update(prediction=completion.text, input_tokens=completion.input_tokens,
                                  output_tokens=completion.output_tokens, raw_usage=completion.raw_usage,
                                  finish_reason=completion.finish_reason, attempts=attempts)
                    if c.cache_write and self._eligible(task, completion.text) and completion.finish_reason not in {"length", "content_filter"}:
                        controller = self.backend()
                        store = controller.store_compact_mcq if task.choices else controller.store_compact_answer
                        store(task.query, completion.text.strip(), model_used=c.executor_model,
                              source_id=task.source_id, route=result["route"], provenance={
                                  "source_scope_hash": scope, "full_rendered_input_tokens": full_tokens,
                                  "final_rendered_input_tokens": result["final_rendered_input_tokens"],
                                  "selected_evidence_ranges": result["selected_evidence_ranges"],
                              })
                        result["cache_written"] = True
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}",
                          attempts=int(getattr(exc, "attempts", result["attempts"])))
        result["total_ms"] = (time.perf_counter() - started) * 1000
        if self.output_dir:
            with (self.output_dir / "execution.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    def _candidates(self, task, controller):
        c = self.config
        dense = None
        if c.evidence_selection in {"dense", "fusion"} or not c.exact_offsets:
            kwargs = {"top_k": controller.doc_index.total, "rerank_top": 0 if c.exact_offsets else c.rerank_top,
                      "use_reranker": c.rerank_top > 0 and not c.exact_offsets}
            if c.cache_read:
                kwargs["query_embedding"] = getattr(controller, "_last_cache_query_embedding", None)
            dense = controller.retrieve(task.query, **kwargs)
        if not c.exact_offsets:
            return dense
        chunks = [{"text": text, "score": 0.0, "metadata": meta}
                  for text, meta in zip(controller._doc_chunks, controller._doc_chunk_metadata)]
        candidates = order_candidates(c.evidence_selection, task.query, chunks, dense, self.lexical,
                                      seed=c.selection_seed, case_id=task.case_id)
        # The controller's own reranking keeps only its top results, which starves budget filling.
        if c.rerank_top:
            candidates = rerank_head(controller.reranker, task.query, candidates, c.rerank_top)
        return document_floor(candidates) if c.document_floor else candidates

    @staticmethod
    def _ending(task, request):
        """The text after the documents, such as the question: what the full and evidence-free prompts end with."""
        full, empty = request[-1]["content"], task.render([])[-1]["content"]
        length = 0
        while length < min(len(full), len(empty)) and full[-1 - length] == empty[-1 - length]:
            length += 1
        return full[len(full) - length:]

    @staticmethod
    def _eligible(task, prediction):
        text = prediction.strip()
        return bool(text) and (not task.choices or text in task.choices) and (
            not task.required_prefix or text.startswith(task.required_prefix))
