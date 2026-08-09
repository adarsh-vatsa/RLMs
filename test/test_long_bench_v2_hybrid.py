import csv
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from long_bench_v2.run_benchmark import (
    _validate_hybrid_args,
    build_arg_parser,
    pack_hybrid_children,
    resolve_cache_namespace,
    run_hybrid_route_audit,
    run_longbench_benchmark,
)
from long_bench_v2.qwen_prompt import STRICT_MCQ_SYSTEM_PROMPT


class FakeTokenizer:
    name_or_path = "Qwen/fake"
    chat_template = "fake-chat-template"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        character_count = sum(len(message["content"]) for message in messages) + 20
        token_count = max(1, (character_count + 9) // 10)
        return {"input_ids": list(range(token_count))}


class FakeMetrics:
    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self.cache_misses = 0
        self.exact_hits = 0
        self.semantic_hits = 0

    def record_call(self, model, input_tokens, output_tokens):
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def get_totals(self):
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
        }


class FakeHybridController:
    ingest_calls = []
    retrieve_calls = []
    compact_store_calls = []

    def __init__(self, metrics, embedder=None, reranker=None, corpus_id=None, corpus_domain="general"):
        self.metrics = metrics
        self.data_scope_hash = None
        self.entries = {}
        self.doc_index = types.SimpleNamespace(total=0)
        self._last_cache_lookup_info = {"semantic_verifier_calls": 0}
        self._persist_path = None

    def activate_data_scope(self, data_scope_hash):
        self.data_scope_hash = data_scope_hash

    def lookup_cached_result(self, query):
        self._last_cache_lookup_info = {"semantic_verifier_calls": 0}
        answer = self.entries.get((self.data_scope_hash, query))
        if answer is None:
            return None
        self.metrics.exact_hits += 1
        return {
            "query": query,
            "answer": answer,
            "from_cache": True,
            "cache_type": "exact",
        }

    def ingest(self, docs_dir, **kwargs):
        FakeHybridController.ingest_calls.append({"docs_dir": Path(docs_dir), **kwargs})
        self.doc_index = types.SimpleNamespace(total=3)
        self._last_ingest_info = {
            "embedding_ms": 12.5,
            "child_encoded_length_max": 101,
        }
        return 3

    def retrieve(self, query, top_k, rerank_top, use_reranker, query_embedding=None):
        FakeHybridController.retrieve_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "rerank_top": rerank_top,
                "use_reranker": use_reranker,
                "query_embedding": query_embedding,
            }
        )
        self._last_retrieval_info = {
            "faiss_candidate_count": 3,
            "candidate_text_count": 3,
            "reranker_enabled": False,
        }
        return [
            {
                "text": letter * 1500,
                "score": 0.9 - index * 0.1,
                "metadata": {
                    "child_index": index,
                    "token_start": index * 100,
                    "token_end": (index + 1) * 100,
                    "char_start": index * 1500,
                    "char_end": (index + 1) * 1500,
                },
            }
            for index, letter in enumerate(("a", "b", "c"))
        ]

    def store_compact_mcq(self, query, result, **kwargs):
        FakeHybridController.compact_store_calls.append(
            {"query": query, "result": result, **kwargs}
        )
        self.entries[(self.data_scope_hash, query)] = result
        return {"query": query, "result": result}

    def get_total_entries(self):
        return len(self.entries)

    def load(self, path):
        return False

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "fake_state.json").write_text(json.dumps({"entries": len(self.entries)}))


class FakeHybridScs:
    SEARCH_MODE = "hybrid"
    SYNTHESIS_MAX_CHUNKS = 5
    DOCUMENT_CHUNK_SIZE = 10000
    DOCUMENT_CHUNK_OVERLAP = 1000
    DOCUMENT_CHUNK_TOKENS = 0
    DOCUMENT_CHUNK_OVERLAP_TOKENS = 0
    DOCUMENT_CHUNK_TOKENIZER_MODEL = ""
    EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_CONTRACT_VERSION = "qwen3_left_last_l2_v1"
    EMBEDDING_QUERY_INSTRUCTION = "retrieve benchmark evidence"
    EMBEDDING_BATCH_SIZE = 2
    EMBEDDING_MAX_LENGTH = 8192
    EMBEDDING_DEVICE = "cuda"
    EMBEDDING_DTYPE = "auto"
    RERANKER_RELEVANCE_THRESHOLD = 0.2
    RERANKER_BATCH_SIZE = 2
    RERANKER_MAX_LENGTH = 8192
    MIN_RERANKED_RESULTS = 5
    SYNTHESIS_INPUT_TOKEN_BUDGET = 0
    MCQ_SYNTHESIS_MAX_TOKENS = 8
    MCQ_PROMPT_STYLE = "strict"
    ITERATIVE_READER_VERSION = 13
    SCAN_MIN_CHUNK_RATIO = 0.3
    SCAN_MAX_CHUNK_RATIO = 0.5
    SCAN_MIN_CHUNKS = 3
    SCAN_MAX_CHUNKS = 0
    SCAN_MAX_TOKENS = 768
    SCAN_EMPTY_LEDGER_FALLBACK_RATIO = 1.0
    ITERATIVE_PACKED_FALLBACK_INPUT_TOKEN_BUDGET = 60000
    ITERATIVE_MEMORY_MAX_CHARS = 16000
    ITERATIVE_BATCH_MAX_CHUNKS = 3
    ITERATIVE_BATCH_INPUT_TOKEN_BUDGET = 50000
    SCAN_ORDER = "faiss_ranked"
    SemanticCacheController = FakeHybridController
    ExecutionMetrics = FakeMetrics
    executor_calls = []

    @staticmethod
    def configure_llm_provider(**kwargs):
        return None

    @staticmethod
    def get_openai_compatible_extra_body_config(redact=False):
        return {
            "effective_executor": {"chat_template_kwargs": {"enable_thinking": False}},
            "effective_evaluator": {"chat_template_kwargs": {"enable_thinking": False}},
        }

    @staticmethod
    def EmbeddingEngine():
        return types.SimpleNamespace(device="cuda", torch_dtype_name="bfloat16")

    @staticmethod
    def Reranker():
        raise AssertionError("The hybrid v1 route must not construct a reranker")

    @staticmethod
    def create_llm_message(**kwargs):
        FakeHybridScs.executor_calls.append(kwargs)
        user_content = kwargs["messages"][0]["content"]
        answer = "Answer: A" if "invalid context" in user_content else "A"
        return types.SimpleNamespace(
            usage=types.SimpleNamespace(input_tokens=12, output_tokens=1),
            content=[types.SimpleNamespace(text=answer)],
        )


def source_row(source_id, context):
    return {
        "_id": source_id,
        "context": context,
        "question": "Which option is supported?",
        "choice_A": "Alpha",
        "choice_B": "Beta",
        "choice_C": "Gamma",
        "choice_D": "Delta",
        "answer": "A",
    }


def suite_row(source_id, row_type="original"):
    return {
        "case_id": f"{source_id}__{row_type}",
        "source_id": source_id,
        "row_type": row_type,
        "is_scored": "true",
        "setup_case_id": "",
        "context_id": "ctx",
        "token_count": "100",
        "expected_cache_type": "miss" if row_type == "original" else "exact",
        "expected_from_cache": "false" if row_type == "original" else "true",
        "depends_on_case_id": "",
        "domain": "Synthetic",
        "sub_domain": "Synthetic",
        "difficulty": "easy",
        "length": "long",
        "question": "Which option is supported?",
        "choice_A": "Alpha",
        "choice_B": "Beta",
        "choice_C": "Gamma",
        "choice_D": "Delta",
        "answer": "A",
    }


class LongBenchV2HybridTests(unittest.TestCase):
    def setUp(self):
        FakeHybridController.ingest_calls = []
        FakeHybridController.retrieve_calls = []
        FakeHybridController.compact_store_calls = []
        FakeHybridScs.executor_calls = []

    def test_exact_packing_stops_before_first_overflow_and_preserves_query(self):
        tokenizer = FakeTokenizer()
        query = "Question: choose one\nA. One\nB. Two\nC. Three\nD. Four"
        results = [
            {"text": "x" * 1000, "score": 0.9, "metadata": {"child_index": 0}},
            {"text": "y" * 1000, "score": 0.8, "metadata": {"child_index": 1}},
        ]

        first_only_tokens = len(STRICT_MCQ_SYSTEM_PROMPT + results[0]["text"] + query) // 10 + 20
        messages, info = pack_hybrid_children(
            tokenizer=tokenizer,
            query=query,
            results=results,
            max_input_tokens=first_only_tokens,
        )

        self.assertEqual(info["selected_child_indices"], [0])
        self.assertEqual(info["dropped_child_count"], 1)
        self.assertIn(query, messages[1]["content"])
        self.assertLessEqual(info["rendered_input_tokens"], first_only_tokens)
        self.assertTrue(all(call["kwargs"]["enable_thinking"] is False for call in tokenizer.calls))

    def test_hybrid_argument_validation(self):
        args = build_arg_parser().parse_args(
            ["--llm-provider", "openai_compatible", "--max-input-tokens", "500"]
        )
        _validate_hybrid_args(args, embedding_max_length=8192)

        args.max_input_tokens = args.context_window_tokens
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            _validate_hybrid_args(args, embedding_max_length=8192)

        args.max_input_tokens = 500
        args.child_overlap_tokens = args.child_tokens
        with self.assertRaisesRegex(ValueError, "less than"):
            _validate_hybrid_args(args, embedding_max_length=8192)

    def test_hybrid_namespace_uses_policy_and_ignores_legacy_top_k(self):
        row = suite_row("source")
        common = {
            "suite_csv_sha256": "suite",
            "source_json_sha256": "source",
            "selected_rows": [row],
            "executor_model": "Qwen/fake",
            "rerank_top": 3,
            "row_types": ["original"],
            "search_mode": "hybrid",
            "reranker_disabled": True,
            "embedding_contract_version": "qwen3_left_last_l2_v1",
        }

        namespace_a, _ = resolve_cache_namespace(
            top_k=10,
            hybrid_policy={"max_input_tokens": 240000},
            **common,
        )
        namespace_same, _ = resolve_cache_namespace(
            top_k=99,
            hybrid_policy={"max_input_tokens": 240000},
            **common,
        )
        namespace_changed, _ = resolve_cache_namespace(
            top_k=10,
            hybrid_policy={"max_input_tokens": 220000},
            **common,
        )

        self.assertEqual(namespace_a, namespace_same)
        self.assertNotEqual(namespace_a, namespace_changed)

    def test_route_audit_reports_fit_overlength_and_suggestions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_arg_parser().parse_args(
                [
                    "--llm-provider",
                    "openai_compatible",
                    "--executor-model",
                    "Qwen/fake",
                    "--max-input-tokens",
                    "400",
                    "--context-window-tokens",
                    "500",
                    "--child-tokens",
                    "100",
                    "--child-overlap-tokens",
                    "10",
                    "--output-dir",
                    tmpdir,
                ]
            )
            rows = [
                {**suite_row("short"), "context": "brief context"},
                {**suite_row("long"), "context": "x" * 5000},
            ]
            path = run_hybrid_route_audit(args, rows, lambda model: FakeTokenizer())
            audit = json.loads(path.read_text())

        self.assertEqual(audit["direct_fit_count"], 1)
        self.assertEqual(audit["overlength_count"], 1)
        self.assertEqual(audit["suggested_smoke_source_ids"]["largest_direct_fit_source_id"], "short")
        self.assertEqual(audit["suggested_smoke_source_ids"]["smallest_overlength_source_id"], "long")

    def test_runner_routes_cache_first_direct_and_overlength(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_path = tmp / "data.json"
            suite_path = tmp / "suite.csv"
            source_path.write_text(
                json.dumps(
                    [
                        source_row("direct", "brief context"),
                        source_row("over", "x" * 5000),
                        source_row("invalid", "invalid context"),
                    ]
                ),
                encoding="utf-8",
            )
            rows = [
                suite_row("direct", "original"),
                suite_row("over", "original"),
                suite_row("direct", "exact"),
                suite_row("invalid", "original"),
            ]
            with suite_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            args = build_arg_parser().parse_args(
                [
                    "--suite-csv",
                    str(suite_path),
                    "--source-json-path",
                    str(source_path),
                    "--llm-provider",
                    "openai_compatible",
                    "--executor-model",
                    "Qwen/fake",
                    "--evaluator-model",
                    "Qwen/evaluator",
                    "--context-window-tokens",
                    "500",
                    "--max-input-tokens",
                    "400",
                    "--max-output-tokens",
                    "8",
                    "--child-tokens",
                    "100",
                    "--child-overlap-tokens",
                    "10",
                    "--cache-state-root",
                    str(tmp / "cache"),
                    "--output-dir",
                    str(tmp / "artifacts"),
                ]
            )

            with patch.dict(os.environ, {"SEMANTIC_CACHE_SEARCH_MODE": "hybrid"}), patch(
                "long_bench_v2.run_benchmark._import_semantic_cache_system",
                return_value=FakeHybridScs,
            ):
                run_longbench_benchmark(args, tokenizer_factory=lambda model: FakeTokenizer())

            run_dir = next((tmp / "artifacts" / "longbench_v2").glob("20*"))
            manifest = json.loads((run_dir / "manifest.json").read_text())
            bridge_rows = [
                json.loads(line)
                for line in (run_dir / "bridge_rows.jsonl").read_text().splitlines()
            ]

        self.assertEqual(
            [row["hybrid_route"] for row in bridge_rows],
            ["direct_fit", "exact_cache", "dense_child_packed", "direct_fit"],
        )
        self.assertEqual(len(FakeHybridController.ingest_calls), 1)
        self.assertEqual(FakeHybridController.retrieve_calls[0]["top_k"], 3)
        self.assertFalse(FakeHybridController.retrieve_calls[0]["use_reranker"])
        self.assertEqual(len(FakeHybridScs.executor_calls), 3)
        self.assertTrue(all(call["max_tokens"] == 8 for call in FakeHybridScs.executor_calls))
        self.assertTrue(all(call["system"] == STRICT_MCQ_SYSTEM_PROMPT for call in FakeHybridScs.executor_calls))
        self.assertEqual(len(FakeHybridController.compact_store_calls), 2)
        self.assertEqual(bridge_rows[0]["ingested_chunks"], 0)
        self.assertEqual(bridge_rows[1]["ingested_chunks"], 0)
        self.assertEqual(bridge_rows[2]["ingested_chunks"], 3)
        self.assertLessEqual(bridge_rows[2]["final_rendered_input_tokens"], 400)
        self.assertEqual(bridge_rows[2]["faiss_candidate_count"], 3)
        self.assertEqual(len(bridge_rows[2]["faiss_candidates"]), 3)
        self.assertFalse(bridge_rows[3]["valid_choice"])
        self.assertFalse(bridge_rows[3]["compact_cache_write"])
        self.assertEqual(manifest["hybrid_route_counts"], {"dense_child_packed": 1, "direct_fit": 2, "exact_cache": 1})
        self.assertEqual(manifest["executor_answer_calls"], 3)
        self.assertEqual(manifest["semantic_verifier_calls"], 0)
        self.assertEqual(manifest["valid_choice_count"], 3)
        self.assertEqual(manifest["api_error_count"], 0)
        self.assertEqual(manifest["invalid_choice_count"], 1)
        self.assertIsNone(manifest["top_k_effective"])
        self.assertIsNone(manifest["synthesis_max_chunks"])


if __name__ == "__main__":
    unittest.main()
