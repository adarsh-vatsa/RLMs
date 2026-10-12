from dataclasses import replace
import argparse
import json
import random
import re
import types
import unittest

from execution.client import Completion
from execution.contracts import Config, Document, add_arguments, resolve_config
from execution.packing import pack, source_ranges
from execution.pipeline import Pipeline
from execution.retrieval import document_floor, expand_neighbors, order_candidates, rerank_head
from execution.tokens import TRUNCATION_MARKER, chat_token_count, token_ids, truncate_prompt
from aa_lcr.adapter import solver_task as aa_task
from mrcr_v2.adapter import solver_task as mr_task
from semantic_cache_system import _chunk_text_with_tokenizer
from test_aa_lcr import FakeTokenizer as ReversibleTokenizer
from test_execution_pipeline import backend, config, task
from test_mrcr_v2 import FakeEmbedder, FakeTokenizer, PREFIX, example, fake_scs


class WordTokenizer:
    """Counts words and punctuation, so fragments merge or split tokens where they join."""

    def apply_chat_template(self, messages, **kwargs):
        text = "".join(message["content"] for message in messages)
        return list(range(len(re.findall(r"\w+|[^\w\s]", text)) + 2))


class CoarseTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        characters = sum(len(message["content"]) for message in messages) + 20
        return list(range((characters + 9) // 10))


class ReversingReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, documents, top_k, relevance_threshold):
        self.calls.append(len(documents))
        return [(index, float(index), documents[index]) for index in reversed(range(len(documents)))][:top_k]


def counting_every_step(tokenizer, render, documents, results, budget):
    """The packing loop before bounds were added: render and count the prompt for every result."""
    selected, evidence = [], []
    for result in results:
        trial = source_ranges(documents, [*selected, result])
        if chat_token_count(tokenizer, render(trial)) > budget:
            break
        selected, evidence = [*selected, result], trial
    if not selected:
        raise ValueError("No retrieved child fits the configured input budget")
    return [item["metadata"]["child_index"] for item in selected], [
        {key: value for key, value in item.items() if key != "text"} for item in evidence]


def chunks_for(documents, size=40, overlap=8):
    chunks = []
    for document in documents:
        for text, meta in _chunk_text_with_tokenizer(document.text, tokenizer=FakeTokenizer(),
                                                      chunk_tokens=size, overlap_tokens=overlap):
            chunks.append({"text": text, "score": 0.0, "metadata": {
                **meta, "document_id": document.id, "child_index": len(chunks)}})
    return chunks


def words(rng, count):
    return " ".join(rng.choice(["alpha", "beta", "gamma", "delta", "x", "y,", "z.", "omega"]) for _ in range(count))


class PackingTests(unittest.TestCase):
    def test_stop_packing_matches_counting_every_step(self):
        for seed in range(30):
            rng = random.Random(seed)
            documents = [(f"d{index}", words(rng, rng.randint(20, 120))) for index in range(rng.randint(1, 4))]
            sources = tuple(Document(name, text) for name, text in documents)
            renders = [aa_task("aa", "s", documents, "Which?").render]
            if len(sources) == 1:
                renders.append(mr_task("mr", sources[0].id, PREFIX, sources[0].text, example()["view_ops"]).render)
            for tokenizer in (FakeTokenizer(), WordTokenizer(), CoarseTokenizer()):
                for render in renders:
                    candidates = chunks_for(sources)
                    rng.shuffle(candidates)
                    full = chat_token_count(tokenizer, render(source_ranges(sources, candidates)))
                    budget = rng.randint(full // 4, full)
                    with self.subTest(seed=seed, tokenizer=type(tokenizer).__name__):
                        try:
                            expected = counting_every_step(tokenizer, render, sources, candidates, budget)
                        except ValueError:
                            with self.assertRaises(ValueError):
                                pack(tokenizer, render, sources, candidates, budget)
                            continue
                        _, info = pack(tokenizer, render, sources, candidates, budget)
                        self.assertEqual((info["selected_child_indices"], info["selected_evidence_ranges"]), expected)

    def test_bounds_skip_most_renders(self):
        sources = (Document("d", "word " * 4000),)
        render = mr_task("mr", "d", PREFIX, sources[0].text, example()["view_ops"]).render
        candidates = chunks_for(sources, size=200, overlap=20)
        _, info = pack(FakeTokenizer(), render, sources, candidates, 12000)
        self.assertGreater(len(info["selected_child_indices"]), 40)
        self.assertLess(info["packing_renders"], 10)

    def test_skip_and_fill_use_more_of_the_budget(self):
        sources = (Document("d", "word " * 400),)
        render = mr_task("mr", "d", PREFIX, sources[0].text, example()["view_ops"]).render
        chunks = chunks_for(sources, size=100, overlap=0)
        candidates = [chunks[index] for index in (2, 6, 10, 14, 18)]
        base = chat_token_count(FakeTokenizer(), render([]))
        separator = len("\n\n[... omitted conversation ...]\n\n")
        budget = base + 3 * (100 + separator) + 90
        used = {}
        for policy in ("stop", "skip", "fill"):
            request, info = pack(FakeTokenizer(), render, sources, candidates, budget, policy=policy)
            self.assertLessEqual(chat_token_count(FakeTokenizer(), request), budget)
            used[policy] = (info["final_rendered_input_tokens"], info["packing_trimmed_chars"])
        self.assertEqual(used["stop"], (base + 3 * (100 + separator), 0))
        self.assertEqual(used["skip"], used["stop"])
        self.assertGreater(used["fill"][1], 0)
        self.assertGreater(used["fill"][0], budget - 40)


class SelectionTests(unittest.TestCase):
    def chunks(self, texts, documents=None):
        documents = documents or ["d"] * len(texts)
        return [{"text": text, "score": 0.0, "metadata": {"document_id": document, "child_index": index}}
                for index, (text, document) in enumerate(zip(texts, documents))]

    def test_bm25_ranks_term_matches_first(self):
        from execution.retrieval import lexical_index
        chunks = self.chunks(["apple banana", "carrot", "banana banana"])
        ordered = order_candidates("bm25", "banana?", chunks, None, lexical_index([c["text"] for c in chunks]),
                                   seed=0, case_id="q")
        self.assertEqual([item["metadata"]["child_index"] for item in ordered], [2, 0, 1])

    def test_fusion_combines_dense_and_lexical_ranks(self):
        from execution.retrieval import lexical_index
        chunks = self.chunks(["banana", "carrot", "banana banana"])
        dense = [chunks[1], chunks[2], chunks[0]]
        ordered = order_candidates("fusion", "banana", chunks, dense, lexical_index([c["text"] for c in chunks]),
                                   seed=0, case_id="q")
        self.assertEqual([item["metadata"]["child_index"] for item in ordered], [2, 1, 0])

    def test_random_is_seeded_per_question(self):
        chunks = self.chunks([str(index) for index in range(20)])
        first = order_candidates("random", "", chunks, None, None, seed=3, case_id="q1")
        self.assertEqual(first, order_candidates("random", "", chunks, None, None, seed=3, case_id="q1"))
        self.assertNotEqual(first, order_candidates("random", "", chunks, None, None, seed=3, case_id="q2"))
        self.assertEqual(sorted(item["metadata"]["child_index"] for item in first), list(range(20)))

    def test_position_keeps_the_head_and_tail_of_every_document_first(self):
        chunks = self.chunks(["a0", "a1", "a2", "a3", "b0", "b1"], ["a"] * 4 + ["b"] * 2)
        ordered = order_candidates("position", "", chunks, None, None, seed=0, case_id="q")
        self.assertEqual([item["text"] for item in ordered], ["a0", "a3", "b0", "b1", "a1", "a2"])

    def test_document_floor_and_neighbors(self):
        chunks = self.chunks(["a0", "a1", "b0", "a2", "c0"], ["a", "a", "b", "a", "c"])
        self.assertEqual([item["text"] for item in document_floor(chunks)], ["a0", "b0", "c0", "a1", "a2"])
        documents = (Document("d", "0123456789abcdef"), Document("e", "XYZ"))
        chunks = [{"text": text, "score": 0.0, "metadata": {"document_id": doc, "child_index": index,
                                                            "char_start": start, "char_end": end}}
                  for index, (doc, start, end, text) in enumerate(
                      [("d", 0, 6, "012345"), ("d", 5, 11, "56789a"), ("d", 10, 16, "abcdef"), ("e", 0, 3, "XYZ")])]
        middle, last = expand_neighbors([chunks[1], chunks[2]], chunks, documents, 1)
        self.assertEqual((middle["text"], middle["metadata"]["expanded_child_indices"]), ("0123456789abcdef", [0, 1, 2]))
        self.assertEqual((last["metadata"]["char_start"], last["metadata"]["char_end"]), (5, 16))

    def test_rerank_reorders_the_head_and_keeps_every_candidate(self):
        chunks = self.chunks([str(index) for index in range(6)])
        reranker = ReversingReranker()
        reranked = rerank_head(reranker, "q", chunks, 3)
        self.assertEqual([item["text"] for item in reranked], ["2", "1", "0", "3", "4", "5"])
        self.assertEqual(reranker.calls, [3])


class PipelineTests(unittest.TestCase):
    def run_task(self, settings, scs_controller=None, solver_input=None):
        calls = []
        engine = Pipeline(config(**settings), FakeTokenizer(),
                          lambda request: calls.append(request) or Completion("answer", 1, 1, {}),
                          backend_factory=scs_controller or backend)
        result = engine.execute(solver_input or task("q", "abcdefghij" * 200))
        self.assertEqual(result["status"], "ok", result["error"])
        self.assertLessEqual(result["final_rendered_input_tokens"], 500)
        return result, calls

    def test_selections_without_dense_scores_skip_embedding(self):
        for selection in ("bm25", "random", "position"):
            with self.subTest(selection=selection):
                scs = fake_scs()
                controller = scs.SemanticCacheController(embedder=scs.EmbeddingEngine())
                result, _ = self.run_task({"evidence_selection": selection}, lambda: (scs, controller))
                self.assertEqual(FakeEmbedder.calls, 0)
                self.assertEqual(controller.queries, [])
                self.assertEqual((result["route"], result["evidence_selection"]), ("dense_child_packed", selection))

    def test_reranking_keeps_the_remaining_candidates_for_packing(self):
        scs = fake_scs()
        controller = scs.SemanticCacheController(embedder=scs.EmbeddingEngine())
        controller.reranker = ReversingReranker()
        result, _ = self.run_task({"rerank_top": 2}, lambda: (scs, controller))
        self.assertEqual(controller.reranker.calls, [2])
        self.assertGreater(len(result["selected_child_indices"]), 2)

    def test_packing_policies_and_neighbors_stay_in_budget(self):
        used = {}
        for settings in ({}, {"packing": "skip"}, {"packing": "fill"}, {"neighbor_children": 1, "packing": "fill"},
                         {"document_floor": True}):
            with self.subTest(**settings):
                result, _ = self.run_task(settings)
                used[json.dumps(settings)] = result["final_rendered_input_tokens"]
        self.assertGreaterEqual(used['{"packing": "fill"}'], used["{}"])

    def test_direct_truncation_policies(self):
        solver_input = task("q", "".join(chr(97 + index % 26) for index in range(2000)), "?")
        for policy, route in (("middle", "middle_truncated"), ("middle-marked", "middle_marked_truncated"),
                              ("head", "head_truncated")):
            with self.subTest(policy=policy):
                calls = []
                engine = Pipeline(config(mode="direct", direct_overflow=policy), ReversibleTokenizer(),
                                  lambda request: calls.append(request) or Completion("answer", 1, 1, {}))
                result = engine.execute(solver_input)
                self.assertEqual((result["status"], result["route"]), ("ok", route))
                self.assertLessEqual(result["final_rendered_input_tokens"], 500)
                content = calls[0][0]["content"]
                self.assertEqual(TRUNCATION_MARKER in content, policy == "middle-marked")
                self.assertEqual(content.endswith("?"), policy != "head")
                self.assertTrue(content.startswith("abcdef"))

    def test_middle_truncation_is_unchanged(self):
        tokenizer = ReversibleTokenizer()
        messages = [{"role": "user", "content": "".join(chr(65 + index % 50) for index in range(3000))}]
        ids = token_ids(tokenizer.encode(messages[0]["content"]))
        keep = 400 - chat_token_count(tokenizer, [{"role": "user", "content": ""}])
        expected = tokenizer.decode(ids[:keep // 2] + ids[-(keep - keep // 2):])
        self.assertEqual(truncate_prompt(tokenizer, messages, 400)[0]["content"], expected)


class ConfigTests(unittest.TestCase):
    def test_invalid_combinations_are_rejected(self):
        base = Config("hybrid", 500, 20, 1000)
        for overrides in ({"rerank_top": 2, "evidence_selection": "random"}, {"evidence_selection": "other"},
                          {"packing": "greedy"}, {"neighbor_children": -1},
                          {"exact_offsets": False, "evidence_selection": "bm25"},
                          {"merge_overlaps": False, "evidence_order": "score", "packing": "fill"}):
            with self.subTest(**overrides), self.assertRaises(ValueError):
                replace(base, **overrides)

    def test_flags_reach_the_config(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        args = parser.parse_args(["--execution-profile", "common", "--evidence-selection", "fusion",
                                  "--selection-seed", "4", "--neighbor-children", "1", "--document-floor",
                                  "--packing", "fill", "--direct-overflow", "middle-marked"])
        resolved = resolve_config(args, mode="hybrid", max_input_tokens=500, max_output_tokens=20,
                                  context_window_tokens=1000)
        self.assertEqual((resolved.evidence_selection, resolved.selection_seed, resolved.neighbor_children,
                          resolved.document_floor, resolved.packing, resolved.direct_overflow),
                         ("fusion", 4, 1, True, "fill", "middle-marked"))
        defaults = resolve_config(types.SimpleNamespace(execution_profile="common"), mode="hybrid",
                                  max_input_tokens=500, max_output_tokens=20, context_window_tokens=1000)
        self.assertEqual((defaults.evidence_selection, defaults.packing), ("dense", "stop"))


if __name__ == "__main__":
    unittest.main()
