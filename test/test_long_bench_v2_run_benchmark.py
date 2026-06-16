import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from long_bench_v2.run_benchmark import (
    answer_correct,
    build_arg_parser,
    build_cache_reuse_manifest,
    build_query,
    filter_suite_rows,
    load_context_by_source_id,
    load_suite_rows,
    normalize_llm_args,
    order_suite_rows,
    parse_choice,
    resolve_cache_namespace,
    run_longbench_benchmark,
)


def _source_row(row_id: str, context: str = "Context text") -> dict:
    return {
        "_id": row_id,
        "context": context,
        "question": "Which option is correct?",
        "choice_A": "Alpha",
        "choice_B": "Beta",
        "choice_C": "Gamma",
        "choice_D": "Delta",
        "answer": "A",
    }


def _suite_row(source_id: str, row_type: str = "original") -> dict:
    return {
        "case_id": f"{source_id}__{row_type}",
        "source_id": source_id,
        "row_type": row_type,
        "is_scored": "true",
        "setup_case_id": "",
        "context_id": "ctx",
        "token_count": "100",
        "expected_cache_type": "miss" if row_type == "original" else row_type,
        "expected_from_cache": "false" if row_type == "original" else "true",
        "depends_on_case_id": "",
        "domain": "Single-Document QA",
        "sub_domain": "Synthetic",
        "difficulty": "easy",
        "length": "short",
        "question": "Which option is correct?",
        "choice_A": "Alpha",
        "choice_B": "Beta",
        "choice_C": "Gamma",
        "choice_D": "Delta",
        "answer": "A",
    }


class FakeMetrics:
    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0

    def get_totals(self):
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
        }


class FakeController:
    instances = []
    load_calls = []
    save_calls = []
    ingest_calls = []
    search_calls = []

    def __init__(self, metrics, embedder=None, reranker=None, corpus_id=None, corpus_domain="general"):
        self.metrics = metrics
        self.entries = 0
        self._persist_path = None
        FakeController.instances.append(self)

    def load(self, path):
        FakeController.load_calls.append(Path(path))
        self.entries = 1
        return True

    def save(self, path):
        FakeController.save_calls.append(Path(path))
        Path(path).mkdir(parents=True, exist_ok=True)

    def get_total_entries(self):
        return self.entries

    def ingest(self, docs_dir):
        FakeController.ingest_calls.append(Path(docs_dir))
        return 2

    def search(self, query, top_k=20, rerank_top=5, synthesize=True, cache_read=True):
        fake_scan_budget = 5
        fake_faiss_top_n = 10
        fake_early_stop_min = 3
        self.metrics.calls += 1
        self.metrics.input_tokens += 10
        self.metrics.output_tokens += 1
        self.metrics.cost += 0.01
        self.entries += 1
        FakeController.search_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "rerank_top": rerank_top,
                "synthesize": synthesize,
                "cache_read": cache_read,
            }
        )
        return {
            "answer": "A",
            "from_cache": False,
            "retrieval": {
                "faiss_candidate_count": fake_faiss_top_n,
                "candidate_text_count": fake_faiss_top_n,
                "reranker_enabled": True,
                "reranker_returned_count": rerank_top,
                "reranker_fallback_used": False,
                "synthesis_input_token_budget": 60000,
                "synthesis_source_truncated": False,
                "synthesis_estimated_input_tokens_before": 1234,
                "synthesis_estimated_input_tokens_after": 1200,
                "synthesis_packed_chunk_count": 3,
                "synthesis_dropped_chunk_count": 1,
                "synthesis_selected_chunk_indices": [0, 1, 2],
                "iterative_reader_version": 6,
                "iterative_memory_max_chars": 16000,
                "iterative_scan_total_chunks": 10,
                "iterative_scan_early_stop_min_chunks": fake_early_stop_min,
                "iterative_scan_budget": fake_scan_budget,
                "iterative_scan_empty_ledger_fallback_budget": fake_faiss_top_n,
                "iterative_scan_visited_chunk_count": fake_early_stop_min,
                "iterative_scan_faiss_top_n": fake_faiss_top_n,
                "iterative_scan_faiss_result_count": fake_faiss_top_n,
                "iterative_scan_empty_ledger_fallback_used": False,
                "iterative_scan_packed_fallback_used": False,
                "iterative_scan_early_stop": True,
                "iterative_scan_stop_reason": "high_confidence_answer",
                "iterative_scan_selected_chunk_indices": list(range(fake_early_stop_min)),
                "iterative_scan_supporting_chunk_indices": [1],
                "iterative_scan_inspector_call_count": fake_early_stop_min,
                "iterative_scan_final_adjudication_call_count": 0,
                "iterative_scan_packed_fallback_call_count": 0,
                "iterative_scan_packed_fallback_reason": None,
                "iterative_scan_final_answer": None,
                "iterative_scan_final_confidence": None,
                "iterative_scan_useful_memory_count": 1,
                "iterative_scan_memory_char_count": 29,
                "iterative_scan_memory_update_count": 1,
                "iterative_scan_target_fact_count": 1,
                "iterative_scan_code_mapping_count": 0,
                "iterative_scan_open_question_count": 0,
                "iterative_scan_observation_count": 0,
                "iterative_scan_rule_count": 0,
                "iterative_scan_example_count": 0,
                "iterative_scan_parse_failure_count": 0,
                "iterative_scan_evidence_ledger": {
                    "memory": "[chunk 1] evidence supports A",
                    "memory_updates": [{"chunk_index": 1, "note": "evidence supports A"}],
                    "target_facts": [{"source": "chunk", "chunk_index": 1, "note": "target fact"}],
                    "code_mappings": [],
                    "best_choice": "A",
                    "best_choice_rationale": "evidence supports A",
                    "confidence": "high",
                    "open_questions": [],
                    "visited_chunks": [0, 1, 2],
                    "parse_failures": [],
                },
            },
        }


class FakeScs:
    DOCUMENT_CHUNK_SIZE = 10000
    DOCUMENT_CHUNK_OVERLAP = 1000
    DOCUMENT_CHUNK_TOKENS = 6000
    DOCUMENT_CHUNK_OVERLAP_TOKENS = 600
    DOCUMENT_CHUNK_TOKENIZER_MODEL = "fake-tokenizer"
    EMBEDDING_QUERY_INSTRUCTION = "fake embedding instruction"
    SYNTHESIS_INPUT_TOKEN_BUDGET = 60000
    SYNTHESIS_MAX_CHUNKS = 5
    MCQ_PROMPT_STYLE = "strict"
    SEARCH_MODE = "iterative"
    ITERATIVE_READER_VERSION = 6
    SCAN_MIN_CHUNK_RATIO = 0.30
    SCAN_MAX_CHUNK_RATIO = 0.50
    SCAN_MIN_CHUNKS = 3
    SCAN_MAX_CHUNKS = 0
    SCAN_MAX_TOKENS = 768
    SCAN_EMPTY_LEDGER_FALLBACK_RATIO = 1.0
    ITERATIVE_PACKED_FALLBACK_INPUT_TOKEN_BUDGET = 60000
    ITERATIVE_MEMORY_MAX_CHARS = 16000
    SCAN_ORDER = "faiss_ranked"
    SemanticCacheController = FakeController
    ExecutionMetrics = FakeMetrics
    reranker_calls = 0

    @staticmethod
    def configure_llm_provider(**kwargs):
        FakeScs.configure_kwargs = kwargs

    @staticmethod
    def get_openai_compatible_extra_body_config(redact=False):
        return {
            "common": {},
            "executor": {"chat_template_kwargs": {"enable_thinking": False}},
            "evaluator": {"api_key": "[REDACTED]"} if redact else {"api_key": "secret"},
            "effective_executor": {"chat_template_kwargs": {"enable_thinking": False}},
            "effective_evaluator": {"api_key": "[REDACTED]"} if redact else {"api_key": "secret"},
        }

    @staticmethod
    def EmbeddingEngine():
        return object()

    @staticmethod
    def Reranker():
        FakeScs.reranker_calls += 1
        return object()


def _reset_fake_controller():
    FakeController.instances = []
    FakeController.load_calls = []
    FakeController.save_calls = []
    FakeController.ingest_calls = []
    FakeController.search_calls = []
    FakeScs.SYNTHESIS_MAX_CHUNKS = 5
    FakeScs.MCQ_PROMPT_STYLE = "strict"
    FakeScs.SEARCH_MODE = "iterative"
    FakeScs.ITERATIVE_READER_VERSION = 6
    FakeScs.SCAN_MIN_CHUNK_RATIO = 0.30
    FakeScs.SCAN_MAX_CHUNK_RATIO = 0.50
    FakeScs.SCAN_MIN_CHUNKS = 3
    FakeScs.SCAN_MAX_CHUNKS = 0
    FakeScs.SCAN_MAX_TOKENS = 768
    FakeScs.SCAN_EMPTY_LEDGER_FALLBACK_RATIO = 1.0
    FakeScs.ITERATIVE_PACKED_FALLBACK_INPUT_TOKEN_BUDGET = 60000
    FakeScs.ITERATIVE_MEMORY_MAX_CHARS = 16000
    FakeScs.SCAN_ORDER = "faiss_ranked"
    FakeScs.EMBEDDING_QUERY_INSTRUCTION = "fake embedding instruction"
    FakeScs.reranker_calls = 0


class LongBenchV2RunBenchmarkTests(unittest.TestCase):
    def test_load_suite_rows_resolves_context_from_source_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_path = tmp / "data.json"
            suite_path = tmp / "suite.csv"
            source_path.write_text(json.dumps([_source_row("row_1", "Full long context")]), encoding="utf-8")
            with suite_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(_suite_row("row_1").keys()))
                writer.writeheader()
                writer.writerow(_suite_row("row_1"))

            contexts = load_context_by_source_id(source_path)
            rows = load_suite_rows(suite_path, contexts)

        self.assertEqual(contexts["row_1"], "Full long context")
        self.assertEqual(rows[0]["context"], "Full long context")

    def test_filter_suite_rows_by_type_and_max_rows(self):
        rows = [
            _suite_row("row_1", "original"),
            _suite_row("row_1", "exact"),
            _suite_row("row_1", "semantic"),
            _suite_row("row_2", "original"),
        ]

        selected = filter_suite_rows(rows, row_types=["original", "semantic"], max_rows=2)

        self.assertEqual([row["case_id"] for row in selected], ["row_1__original", "row_1__semantic"])

    def test_source_grouped_order_keeps_source_rows_adjacent(self):
        rows = [
            _suite_row("row_1", "original"),
            _suite_row("row_2", "original"),
            _suite_row("row_1", "exact"),
            _suite_row("row_2", "exact"),
        ]

        grouped = order_suite_rows(rows, "source_grouped")

        self.assertEqual(
            [row["case_id"] for row in grouped],
            ["row_1__original", "row_1__exact", "row_2__original", "row_2__exact"],
        )
        self.assertEqual(order_suite_rows(rows, "input"), rows)

    def test_runner_defaults_are_faster_without_changing_doc_chunk_defaults(self):
        parser = build_arg_parser()
        args = parser.parse_args([])

        self.assertEqual(args.top_k, 10)
        self.assertEqual(args.rerank_top, 3)
        self.assertEqual(args.synthesis_max_chunks, 3)
        self.assertEqual(args.row_order, "source_grouped")
        self.assertEqual(args.cache_save_interval, 10)

        import semantic_cache_system as scs

        self.assertEqual(scs.DOCUMENT_CHUNK_SIZE, 10000)
        self.assertEqual(scs.DOCUMENT_CHUNK_OVERLAP, 1000)

    def test_resolve_cache_namespace_is_deterministic(self):
        rows = [_suite_row("row_1", "original"), _suite_row("row_1", "exact")]

        first = resolve_cache_namespace("suite-sha", "source-sha", rows, "model-a", 20, 5, ["original", "exact"])
        second = resolve_cache_namespace("suite-sha", "source-sha", list(reversed(rows)), "model-a", 20, 5, ["exact", "original"])
        changed = resolve_cache_namespace("suite-sha", "source-sha", rows, "model-b", 20, 5, ["original", "exact"])
        changed_synthesis = resolve_cache_namespace(
            "suite-sha",
            "source-sha",
            rows,
            "model-a",
            20,
            5,
            ["original", "exact"],
            synthesis_max_chunks=3,
        )
        changed_extra_body = resolve_cache_namespace(
            "suite-sha",
            "source-sha",
            rows,
            "model-a",
            20,
            5,
            ["original", "exact"],
            openai_compatible_extra_body={"executor": {"chat_template_kwargs": {"enable_thinking": False}}},
        )
        changed_prompt_style = resolve_cache_namespace(
            "suite-sha",
            "source-sha",
            rows,
            "model-a",
            20,
            5,
            ["original", "exact"],
            mcq_prompt_style="strict",
        )
        changed_token_chunks = resolve_cache_namespace(
            "suite-sha",
            "source-sha",
            rows,
            "model-a",
            20,
            5,
            ["original", "exact"],
            doc_chunk_tokens=6000,
            doc_chunk_overlap_tokens=600,
            doc_chunk_tokenizer_model="model-a",
        )
        changed_embedding_instruction = resolve_cache_namespace(
            "suite-sha",
            "source-sha",
            rows,
            "model-a",
            20,
            5,
            ["original", "exact"],
            embedding_query_instruction="custom retrieval instruction",
        )
        changed_input_budget = resolve_cache_namespace(
            "suite-sha",
            "source-sha",
            rows,
            "model-a",
            20,
            5,
            ["original", "exact"],
            synthesis_input_token_budget=60000,
        )
        changed_scan_config = resolve_cache_namespace(
            "suite-sha",
            "source-sha",
            rows,
            "model-a",
            20,
            5,
            ["original", "exact"],
            search_mode="iterative",
            iterative_reader_version=5,
            scan_min_chunk_ratio=0.30,
            scan_max_chunk_ratio=0.50,
            scan_min_chunks=3,
            scan_max_chunks=0,
            scan_max_tokens=768,
            scan_empty_ledger_fallback_ratio=1.0,
            iterative_packed_fallback_input_token_budget=60000,
            iterative_memory_max_chars=16000,
            scan_order="faiss_ranked",
        )
        changed_iterative_reader_version = resolve_cache_namespace(
            "suite-sha",
            "source-sha",
            rows,
            "model-a",
            20,
            5,
            ["original", "exact"],
            search_mode="iterative",
            iterative_reader_version=6,
            scan_min_chunk_ratio=0.30,
            scan_max_chunk_ratio=0.50,
            scan_min_chunks=3,
            scan_max_chunks=0,
            scan_max_tokens=768,
            scan_empty_ledger_fallback_ratio=1.0,
            iterative_packed_fallback_input_token_budget=60000,
            iterative_memory_max_chars=16000,
            scan_order="faiss_ranked",
        )
        changed_iterative_memory_max = resolve_cache_namespace(
            "suite-sha",
            "source-sha",
            rows,
            "model-a",
            20,
            5,
            ["original", "exact"],
            search_mode="iterative",
            iterative_reader_version=5,
            scan_min_chunk_ratio=0.30,
            scan_max_chunk_ratio=0.50,
            scan_min_chunks=3,
            scan_max_chunks=0,
            scan_max_tokens=768,
            scan_empty_ledger_fallback_ratio=1.0,
            iterative_packed_fallback_input_token_budget=60000,
            iterative_memory_max_chars=32000,
            scan_order="faiss_ranked",
        )
        iterative_with_legacy_args = resolve_cache_namespace(
            "suite-sha",
            "source-sha",
            rows,
            "model-a",
            99,
            99,
            ["original", "exact"],
            synthesis_max_chunks=99,
            synthesis_input_token_budget=12345,
            search_mode="iterative",
            iterative_reader_version=5,
            scan_min_chunk_ratio=0.30,
            scan_max_chunk_ratio=0.50,
            scan_min_chunks=3,
            scan_max_chunks=0,
            scan_max_tokens=768,
            scan_empty_ledger_fallback_ratio=1.0,
            iterative_packed_fallback_input_token_budget=60000,
            iterative_memory_max_chars=16000,
            scan_order="faiss_ranked",
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertNotEqual(first, changed_synthesis)
        self.assertNotEqual(first, changed_extra_body)
        self.assertNotEqual(first, changed_prompt_style)
        self.assertNotEqual(first, changed_token_chunks)
        self.assertNotEqual(first, changed_embedding_instruction)
        self.assertNotEqual(first, changed_input_budget)
        self.assertNotEqual(first, changed_scan_config)
        self.assertNotEqual(changed_scan_config, changed_iterative_reader_version)
        self.assertNotEqual(changed_scan_config, changed_iterative_memory_max)
        self.assertEqual(changed_scan_config, iterative_with_legacy_args)

    def test_parse_choice_and_answer_correct(self):
        self.assertEqual(parse_choice("A"), "A")
        self.assertEqual(parse_choice("Final answer: C"), "C")
        self.assertEqual(parse_choice("Therefore, the correct choice is:\nC"), "C")
        self.assertEqual(parse_choice("(D) because the passage says so"), "D")
        self.assertTrue(answer_correct("Option B", "B"))
        self.assertFalse(answer_correct("Option B", "C"))

    def test_parse_choice_prefers_final_letter_over_prose_articles(self):
        generation = (
            "Based on the provided documents, the only event mentioned from the choices "
            "is that the user took a writing workshop.\n\nC"
        )

        self.assertEqual(parse_choice(generation), "C")

    def test_parse_choice_does_not_capture_answer_choice_prose(self):
        generation = (
            "The provided documents do not contain enough information to determine the "
            "correct answer choice based on the evidence.\n\nA"
        )

        self.assertEqual(parse_choice(generation), "A")

    def test_build_query_preserves_multiple_choice_fields(self):
        query = build_query(_suite_row("row_1"))

        self.assertIn("Question: Which option is correct?", query)
        self.assertIn("A. Alpha", query)
        self.assertIn("Return only the single best answer choice letter", query)

    def test_cache_reuse_manifest_baseline_and_cache_modes(self):
        baseline = build_cache_reuse_manifest(enabled=False)
        cold = build_cache_reuse_manifest(
            enabled=True,
            cache_namespace="ns",
            dataset_signature="sig",
            cache_state_existed_before_run=False,
            cache_hits=2,
            row_count=4,
        )
        warm = build_cache_reuse_manifest(
            enabled=True,
            cache_namespace="ns",
            dataset_signature="sig",
            cache_state_existed_before_run=True,
            cache_hits=3,
            row_count=4,
        )

        self.assertFalse(baseline["enabled"])
        self.assertEqual(cold["run_start_type"], "cold_start")
        self.assertEqual(cold["cache_hit_rate"], 0.5)
        self.assertEqual(warm["run_start_type"], "warm_start")

    def test_normalize_llm_args_keeps_anthropic_default_and_maps_openrouter_and_local(self):
        anthropic = type(
            "Args",
            (),
            {
                "llm_provider": "anthropic",
                "api_key_env": None,
                "executor_model": "claude-sonnet-4-5",
                "evaluator_model": "claude-haiku-4-5",
            },
        )()
        openrouter = type(
            "Args",
            (),
            {
                "llm_provider": "openrouter",
                "api_key_env": None,
                "executor_model": "claude-sonnet-4-5",
                "evaluator_model": "claude-haiku-4-5",
            },
        )()
        local = type(
            "Args",
            (),
            {
                "llm_provider": "openai_compatible",
                "api_key_env": None,
                "executor_model": "claude-sonnet-4-5",
                "evaluator_model": "claude-haiku-4-5",
            },
        )()

        normalize_llm_args(anthropic)
        normalize_llm_args(openrouter)
        normalize_llm_args(local)

        self.assertEqual(anthropic.api_key_env, "ANTHROPIC_API_KEY")
        self.assertEqual(anthropic.executor_model, "claude-sonnet-4-5")
        self.assertEqual(openrouter.api_key_env, "OPENROUTER_API_KEY")
        self.assertEqual(openrouter.executor_model, "anthropic/claude-sonnet-4.5")
        self.assertEqual(openrouter.evaluator_model, "anthropic/claude-haiku-4.5")
        self.assertIsNone(local.api_key_env)
        self.assertEqual(local.executor_model, "meta-llama/Llama-3.3-70B-Instruct")
        self.assertEqual(local.evaluator_model, "mistralai/Mistral-Small-24B-Instruct-2501")

    def test_run_benchmark_uses_synthesis_override_grouping_and_cache_save_interval(self):
        _reset_fake_controller()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_path = tmp / "data.json"
            suite_path = tmp / "suite.csv"
            output_dir = tmp / "artifacts"
            source_path.write_text(
                json.dumps([
                    _source_row("row_1", "Context one"),
                    _source_row("row_2", "Context two"),
                ]),
                encoding="utf-8",
            )
            rows = [
                _suite_row("row_1", "original"),
                _suite_row("row_2", "original"),
                _suite_row("row_1", "exact"),
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
                    "--mode",
                    "cache",
                    "--llm-provider",
                    "openai_compatible",
                    "--cache-state-root",
                    str(tmp / "cache_state"),
                    "--cache-save-interval",
                    "2",
                    "--synthesis-max-chunks",
                    "4",
                    "--rerank-top",
                    "2",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            with patch("long_bench_v2.run_benchmark._import_semantic_cache_system", return_value=FakeScs):
                run_longbench_benchmark(args)

            run_dirs = list((output_dir / "longbench_v2").glob("20*"))
            manifest = json.loads((run_dirs[0] / "manifest.json").read_text())
            bridge_rows = [
                json.loads(line)
                for line in (run_dirs[0] / "bridge_rows.jsonl").read_text().splitlines()
            ]

        self.assertEqual(FakeScs.SYNTHESIS_MAX_CHUNKS, 4)
        self.assertEqual(len(FakeController.instances), 1)
        self.assertEqual(FakeScs.reranker_calls, 0)
        self.assertEqual(len(FakeController.load_calls), 0)
        self.assertEqual(len(FakeController.save_calls), 2)
        self.assertEqual(len(FakeController.ingest_calls), 2)
        self.assertEqual([row["case_id"] for row in bridge_rows], ["row_1__original", "row_1__exact", "row_2__original"])
        self.assertTrue(all(call["top_k"] == 0 for call in FakeController.search_calls))
        self.assertTrue(all(call["rerank_top"] == 2 for call in FakeController.search_calls))
        self.assertEqual(manifest["synthesis_max_chunks"], 4)
        self.assertEqual(manifest["top_k"], 10)
        self.assertIsNone(manifest["top_k_effective"])
        self.assertTrue(manifest["iterative_top_k_ignored"])
        self.assertEqual(manifest["rerank_top"], 2)
        self.assertEqual(manifest["search_mode"], "iterative")
        self.assertEqual(manifest["iterative_reader_version"], 6)
        self.assertEqual(manifest["scan_min_chunk_ratio"], 0.30)
        self.assertEqual(manifest["scan_max_chunk_ratio"], 0.50)
        self.assertEqual(manifest["scan_min_chunks"], 3)
        self.assertEqual(manifest["scan_max_chunks"], 0)
        self.assertEqual(manifest["scan_max_tokens"], 768)
        self.assertEqual(manifest["scan_empty_ledger_fallback_ratio"], 1.0)
        self.assertEqual(manifest["iterative_packed_fallback_input_token_budget"], 60000)
        self.assertEqual(manifest["iterative_memory_max_chars"], 16000)
        self.assertEqual(manifest["scan_order"], "faiss_ranked")
        self.assertTrue(manifest["reranker_disabled"])
        self.assertEqual(manifest["doc_chunk_size"], 10000)
        self.assertEqual(manifest["doc_chunk_overlap"], 1000)
        self.assertEqual(manifest["doc_chunk_tokens"], 6000)
        self.assertEqual(manifest["doc_chunk_overlap_tokens"], 600)
        self.assertEqual(manifest["doc_chunk_tokenizer_model"], "fake-tokenizer")
        self.assertEqual(manifest["embedding_query_instruction"], "fake embedding instruction")
        self.assertEqual(manifest["synthesis_input_token_budget"], 60000)
        self.assertEqual(manifest["mcq_prompt_style"], "strict")
        self.assertEqual(manifest["cache_save_interval"], 2)
        self.assertEqual(manifest["timing_summary"]["cache_save_count"], 2)
        self.assertEqual(
            manifest["openai_compat_extra_body"]["effective_executor"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertEqual(
            manifest["openai_compat_extra_body"]["effective_evaluator"],
            {"api_key": "[REDACTED]"},
        )
        self.assertIn("ingest_ms", bridge_rows[0])
        self.assertIn("search_ms", bridge_rows[0])
        self.assertEqual(bridge_rows[0]["doc_chunk_tokens"], 6000)
        self.assertEqual(bridge_rows[0]["embedding_query_instruction"], "fake embedding instruction")
        self.assertEqual(bridge_rows[0]["synthesis_packed_chunk_count"], 3)
        self.assertEqual(bridge_rows[0]["synthesis_dropped_chunk_count"], 1)
        self.assertEqual(bridge_rows[0]["synthesis_selected_chunk_indices"], [0, 1, 2])
        self.assertEqual(bridge_rows[0]["search_mode"], "iterative")
        self.assertEqual(bridge_rows[0]["top_k"], 10)
        self.assertIsNone(bridge_rows[0]["top_k_effective"])
        self.assertTrue(bridge_rows[0]["iterative_top_k_ignored"])
        self.assertEqual(bridge_rows[0]["iterative_reader_version"], 6)
        self.assertEqual(bridge_rows[0]["scan_min_chunk_ratio"], 0.30)
        self.assertEqual(bridge_rows[0]["scan_max_chunk_ratio"], 0.50)
        self.assertEqual(bridge_rows[0]["scan_empty_ledger_fallback_ratio"], 1.0)
        self.assertEqual(bridge_rows[0]["iterative_scan_total_chunks"], 10)
        self.assertEqual(bridge_rows[0]["iterative_scan_early_stop_min_chunks"], 3)
        self.assertEqual(bridge_rows[0]["iterative_scan_budget"], 5)
        self.assertEqual(bridge_rows[0]["iterative_scan_empty_ledger_fallback_budget"], 10)
        self.assertEqual(bridge_rows[0]["iterative_scan_faiss_top_n"], 10)
        self.assertEqual(bridge_rows[0]["iterative_scan_faiss_result_count"], 10)
        self.assertFalse(bridge_rows[0]["iterative_scan_empty_ledger_fallback_used"])
        self.assertFalse(bridge_rows[0]["iterative_scan_packed_fallback_used"])
        self.assertIsNone(bridge_rows[0]["iterative_scan_packed_fallback_reason"])
        self.assertIsNone(bridge_rows[0]["iterative_scan_final_answer"])
        self.assertIsNone(bridge_rows[0]["iterative_scan_final_confidence"])
        self.assertEqual(bridge_rows[0]["iterative_scan_supporting_chunk_indices"], [1])
        self.assertEqual(bridge_rows[0]["iterative_scan_useful_memory_count"], 1)
        self.assertEqual(bridge_rows[0]["iterative_memory_max_chars"], 16000)
        self.assertEqual(bridge_rows[0]["iterative_scan_memory_char_count"], 29)
        self.assertEqual(bridge_rows[0]["iterative_scan_memory_update_count"], 1)
        self.assertEqual(bridge_rows[0]["iterative_scan_target_fact_count"], 1)
        self.assertEqual(bridge_rows[0]["iterative_scan_code_mapping_count"], 0)
        self.assertEqual(bridge_rows[0]["iterative_scan_open_question_count"], 0)
        self.assertEqual(bridge_rows[0]["iterative_scan_parse_failure_count"], 0)
        self.assertEqual(bridge_rows[0]["iterative_scan_evidence_ledger"]["best_choice"], "A")
        self.assertIn("memory_updates", bridge_rows[0]["iterative_scan_evidence_ledger"])


if __name__ == "__main__":
    unittest.main()
