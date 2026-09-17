from dataclasses import replace
import json
import csv
import os
from pathlib import Path
import tempfile
import subprocess
import types
import unittest
from unittest.mock import patch

from execution.artifacts import finalize, record_evaluation
from execution.client import Completion
from execution.contracts import Config, Document, Task
from execution.pipeline import Pipeline
from execution.tokens import chat_token_count
from aa_lcr.adapter import solver_task as aa_task
from long_bench_v2.adapter import solver_task as lb_task
from mrcr_v2.adapter import solver_task as mr_task
from test_mrcr_v2 import FakeTokenizer, fake_scs, PREFIX, example


def task(case_id="case", body="abcdefghij", query="question"):
    doc = Document("doc", body)
    def render(evidence):
        text = body if evidence is None else "|".join(item["text"] for item in evidence)
        return [{"role": "user", "content": text + query}]
    return Task(case_id, "source", (doc,), query, render, "fixture-v1")


def config(**overrides):
    return replace(Config("hybrid", 500, 20, 1000, child_tokens=100,
                          child_overlap_tokens=20, max_retries=1), **overrides)


def backend():
    scs = fake_scs()
    controller = scs.SemanticCacheController(embedder=scs.EmbeddingEngine())
    return scs, controller


class MemoryCache:
    def __init__(self):
        self.entries = {}
        self.scope = ""
        self.reads = self.writes = 0
        self.metrics = types.SimpleNamespace(get_totals=lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0})

    def activate_data_scope(self, scope):
        self.scope = scope

    def lookup_cached_result(self, query, **kwargs):
        self.reads += 1
        value = self.entries.get((self.scope, query))
        return {"answer": value, "cache_type": "exact"} if value is not None else None

    def store_compact_answer(self, query, answer, **kwargs):
        self.writes += 1
        self.entries[self.scope, query] = answer


class SharedExecutionTests(unittest.TestCase):
    def test_benchmark_label_does_not_change_execution(self):
        requests, results = [], []
        for label in ("LongBench", "AA-LCR", "MRCR", "new-benchmark"):
            calls = []
            pipeline = Pipeline(config(), FakeTokenizer(),
                lambda request: calls.append(request) or Completion("answer", 10, 1, {}), backend_factory=backend)
            result = pipeline.execute(task(label, "abcdefghij" * 200))
            self.assertEqual(result["status"], "ok", result["error"])
            requests.append(calls)
            results.append((result["route"], result["selected_evidence_ranges"], result["selected_child_indices"]))
        self.assertTrue(all(value == requests[0] for value in requests))
        self.assertTrue(all(value == results[0] for value in results))

    def test_all_adapters_use_the_same_engine_direct_and_hybrid(self):
        body = "User: request\nAssistant: response\n" * 300
        row = example(body=body)
        tasks = [lb_task("lb", "s", body, "Question?\nA. Yes\nB. No"),
                 aa_task("aa", "s", [("first", body), ("second", "second source")], "Question?"),
                 mr_task("mr", "s", PREFIX, body, row["view_ops"])]
        for solver_input in tasks:
            for mode in ("direct", "hybrid"):
                with self.subTest(task=solver_input.case_id, mode=mode):
                    calls = []
                    engine = Pipeline(config(mode=mode, max_input_tokens=1800, context_window_tokens=2000),
                        FakeTokenizer(), lambda request: calls.append(request) or Completion("answer", 10, 1, {}),
                        backend_factory=backend)
                    expected = engine.preflight(solver_input)["route"]
                    result = engine.execute(solver_input)
                    self.assertEqual(result["route"], expected)
                    self.assertEqual(result["status"], "unsupported_context" if mode == "direct" else "ok", result["error"])
                    self.assertEqual(len(calls), int(mode == "hybrid"))
                    self.assertLessEqual(result["final_rendered_input_tokens"], 1800)

    def test_direct_fit_preserves_adapter_prompt_without_backend(self):
        solver_input = mr_task("mr", "s", PREFIX, "conversation", example()["view_ops"])
        calls = []
        engine = Pipeline(config(max_input_tokens=900), FakeTokenizer(),
            lambda request: calls.append(request) or Completion("answer", 1, 1, {}),
            backend_factory=lambda: self.fail("fitting prompt must not embed"))
        result = engine.execute(solver_input)
        self.assertEqual(result["route"], "direct_fit")
        self.assertEqual(calls, [solver_input.render(None)])

    def test_inclusive_input_boundary_and_preflight_no_calls(self):
        solver_input = task()
        tokens = chat_token_count(FakeTokenizer(), solver_input.render(None))
        for budget, expected in ((tokens, "direct_fit"), (tokens - 1, "unsupported_context")):
            engine = Pipeline(config(mode="direct", max_input_tokens=budget), FakeTokenizer(),
                              lambda _: self.fail("preflight must not generate"))
            self.assertEqual(engine.preflight(solver_input)["route"], expected)

    def test_index_reuse_and_source_invalidation(self):
        scs, controller = backend()
        calls = []
        engine = Pipeline(config(), FakeTokenizer(), lambda request: Completion("answer", 1, 1, {}),
                          backend_factory=lambda: (scs, controller))
        for solver_input in (task("a", "abcd" * 500), task("b", "abcd" * 500, "another"), task("c", "efgh" * 500)):
            result = engine.execute(solver_input)
            self.assertEqual(result["status"], "ok", result["error"])
            calls.append(result["ingested_chunks"])
        self.assertGreater(calls[0], 0)
        self.assertEqual(calls[1], 0)
        self.assertGreater(calls[2], 0)

    def test_document_ids_and_equal_text_at_distinct_positions_survive(self):
        from execution.packing import source_ranges
        documents = (Document("first", "repeat--repeat"), Document("second", "repeat"))
        selected = [{"text": text, "metadata": {"document_id": name, "char_start": a, "char_end": b}}
                    for name, a, b, text in (("second", 0, 6, "repeat"), ("first", 8, 14, "repeat"), ("first", 0, 6, "repeat"))]
        ranges = source_ranges(documents, selected)
        self.assertEqual([item["document_id"] for item in ranges], ["first", "first", "second"])
        rendered = aa_task("aa", "s", [("first", "repeat--repeat"), ("second", "repeat")], "q").render(ranges)
        self.assertIn("BEGIN DOCUMENT 2:\nrepeat", rendered[0]["content"])
        self.assertEqual(rendered[0]["content"].count("repeat"), 3)

    def test_cache_switches_and_scope_invalidation(self):
        cache = MemoryCache()
        calls = []
        def engine(settings):
            return Pipeline(settings, FakeTokenizer(),
                lambda request: calls.append(request) or Completion("answer", 1, 1, {}),
                backend_factory=lambda: (types.SimpleNamespace(), cache))
        write = config(cache_write=True, cache_matching="exact")
        self.assertTrue(engine(write).execute(task())["cache_written"])
        self.assertEqual(cache.reads, 0)
        read = replace(write, cache_read=True, cache_write=False)
        self.assertTrue(engine(read).execute(task())["from_cache"])
        self.assertEqual(len(calls), 1)
        self.assertFalse(engine(read).execute(task(body="changed"))["from_cache"])
        self.assertFalse(engine(replace(read, child_tokens=99)).execute(task())["from_cache"])
        self.assertFalse(engine(read).execute(replace(task(), required_prefix="new-marker"))["from_cache"])
        self.assertEqual(cache.writes, 1)

    def test_core_exact_only_lookup_is_case_sensitive_and_never_verifies(self):
        from test_data_scope_cache import make_controller, FakeSearchIndex, scs
        controller = make_controller()
        controller.activate_data_scope("source")
        with patch.object(scs, "FAISSIndex", FakeSearchIndex):
            controller.store_compact_answer("Prepend AbCd", "AbCd answer", model_used="executor",
                                            source_id="source", route="direct_fit")
        with patch.object(controller, "_llm_sniper_evaluate", side_effect=AssertionError("no verifier")):
            self.assertIsNone(controller.lookup_cached_result("prepend abcd", semantic=False, strict_exact=True))
            self.assertEqual(controller.lookup_cached_result("Prepend AbCd", semantic=False, strict_exact=True)["answer"], "AbCd answer")

    def test_failure_saved_before_scoring_and_report_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = Pipeline(config(mode="direct"), FakeTokenizer(),
                lambda request: (_ for _ in ()).throw(RuntimeError("failed executor")), output_dir=temporary)
            result = engine.execute(task("failed"))
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["attempts"], 1)
            unsupported = engine.execute(task("too-long", "x" * 2000))
            self.assertEqual(unsupported["status"], "unsupported_context")
            engine.complete = lambda request: Completion("answer", 1, 1, {})
            engine.execute(task("good"))
            finalize(temporary)
            report = json.loads((Path(temporary) / "execution_report.json").read_text())
            self.assertEqual(report["grading_pending_count"], 1)
            record_evaluation(temporary, "good", "ok", {"accuracy": True})
            finalize(temporary)
            report = json.loads((Path(temporary) / "execution_report.json").read_text())
            self.assertEqual(report["supported_count"], 2)
            self.assertEqual(report["quality"]["accuracy"], 0.5)
            self.assertEqual(len((Path(temporary) / "execution.jsonl").read_text().splitlines()), 3)

    def test_task_contract_has_no_gold_fields(self):
        self.assertNotIn("answer", Task.__dataclass_fields__)
        self.assertNotIn("reference", Task.__dataclass_fields__)
        self.assertNotIn("metadata", Task.__dataclass_fields__)

    def test_longbench_transport_preserves_finish_reason_and_raw_usage(self):
        from long_bench_v2.run_benchmark import _call_hybrid_executor
        raw = {"choices": [{"finish_reason": "length"}], "usage": {"prompt_tokens": 10, "completion_tokens": 8}}
        response = types.SimpleNamespace(raw=raw, content=[types.SimpleNamespace(text="A")],
                                         usage=types.SimpleNamespace(input_tokens=10, output_tokens=8))
        scs = types.SimpleNamespace(create_llm_message=lambda **kwargs: response)
        controller = types.SimpleNamespace(metrics=types.SimpleNamespace(record_call=lambda *args: None))
        args = types.SimpleNamespace(executor_model="fixture", max_output_tokens=8)
        result = _call_hybrid_executor(scs, controller, args, [{"role": "system", "content": "choose"},
                                                            {"role": "user", "content": "question"}])
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.raw_usage, raw["usage"])

    def test_source_selection_bounds_precede_row_limit(self):
        from execution.selection import filter_sources
        rows = [task("outside", "x" * 100), task("one", "x" * 10), task("two", "x" * 11)]
        count = lambda row: chat_token_count(FakeTokenizer(), row.render(None))
        args = types.SimpleNamespace(min_source_tokens=count(rows[1]), max_source_tokens=count(rows[2]), max_rows=0)
        self.assertEqual(filter_sources(rows, args, FakeTokenizer(), lambda row: row), rows[1:])
        args.max_rows = 1
        self.assertEqual(filter_sources(rows, args, FakeTokenizer(), lambda row: row), rows[1:2])
        args.min_source_tokens = 0
        with self.assertRaises(ValueError):
            filter_sources(rows, args, FakeTokenizer(), lambda row: row)

    def test_aa_runner_common_hybrid_and_deferred_grading(self):
        from test_aa_lcr import write_fixture, FakeCompletionCaller
        from aa_lcr.run_benchmark import build_arg_parser, run_benchmark
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            questions, documents, manifest = write_fixture(root, long_document=True)
            args = build_arg_parser().parse_args([
                "--mode", "hybrid", "--context-window-tokens", "65536", "--execution-profile", "common", "--execution-only",
                "--questions-csv", str(questions), "--documents-root", str(documents),
                "--dataset-manifest", str(manifest), "--output-root", str(root / "runs"),
                "--max-input-tokens", "1000", "--max-output-tokens", "20",
                "--child-tokens", "100", "--child-overlap-tokens", "20",
            ])
            calls = FakeCompletionCaller()
            output = run_benchmark(args, tokenizer_factory=lambda _: FakeTokenizer(), completion_caller=calls, scs_module=fake_scs())
            rows = [json.loads(line) for line in (output / "execution.jsonl").read_text().splitlines()]
            self.assertEqual([row["status"] for row in rows], ["ok", "ok"])
            self.assertGreater(rows[0]["ingested_chunks"], 0)
            self.assertEqual(rows[1]["ingested_chunks"], 0)
            self.assertEqual(len(calls.calls), 2)
            self.assertTrue(all(call["model"] == args.executor_model for call in calls.calls))
            report = json.loads((output / "execution_report.json").read_text())
            self.assertEqual(report["grading_pending_count"], 2)
            self.assertEqual(len((output / "predictions.jsonl").read_text().splitlines()), 2)
            from aa_lcr.regrade import build_arg_parser as regrade_parser, run_regrade
            options = regrade_parser().parse_args([
                "--source-run", str(output), "--output-dir", str(root / "regraded"),
                "--questions-csv", str(questions), "--documents-root", str(documents),
                "--dataset-manifest", str(manifest), "--max-retries", "1",
            ])
            grader = FakeCompletionCaller()
            regraded = run_regrade(options, tokenizer_factory=lambda _: FakeTokenizer(), completion_caller=grader)
            self.assertEqual(len(grader.calls), 2)
            self.assertTrue(all(call["model"] == options.evaluator_model for call in grader.calls))
            self.assertTrue((regraded / "manifest.json").exists())

    def test_aa_rejects_zero_input_override_before_loading_data(self):
        from aa_lcr.run_benchmark import build_arg_parser, run_benchmark
        args = build_arg_parser().parse_args(["--experiment", "direct_64k", "--max-input-tokens", "0"])
        with self.assertRaisesRegex(ValueError, "positive"):
            run_benchmark(args, tokenizer_factory=lambda _: self.fail("must validate before loading models"))

    def test_longbench_runner_common_hybrid_without_cache(self):
        from test_long_bench_v2_hybrid import FakeHybridScs, source_row, suite_row
        from test_mrcr_v2 import FakeController, FakeEmbedder
        from long_bench_v2.run_benchmark import build_arg_parser, run_longbench_benchmark
        class Controller(FakeController):
            def __init__(self, metrics, **kwargs):
                super().__init__(**kwargs)
                self.metrics = metrics
                self.doc_index = None
        class Scs(FakeHybridScs):
            SemanticCacheController = Controller
            EmbeddingEngine = FakeEmbedder
        fixture = fake_scs()
        Scs.FAISSIndex = fixture.FAISSIndex
        Scs._chunk_text_with_tokenizer = staticmethod(fixture._chunk_text_with_tokenizer)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data.json"
            source.write_text(json.dumps([source_row("source", "x" * 5000)]))
            suite = root / "suite.csv"
            row = suite_row("source", "original")
            with suite.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            args = build_arg_parser().parse_args([
                "--suite-csv", str(suite), "--source-json-path", str(source),
                "--llm-provider", "openai_compatible", "--execution-profile", "common",
                "--context-window-tokens", "2000", "--max-input-tokens", "1800",
                "--child-tokens", "100", "--child-overlap-tokens", "20", "--output-dir", str(root / "runs"),
            ])
            with patch.dict(os.environ, {"SEMANTIC_CACHE_SEARCH_MODE": "hybrid"}), patch(
                "long_bench_v2.run_benchmark._import_semantic_cache_system", return_value=Scs):
                run_longbench_benchmark(args, tokenizer_factory=lambda _: FakeTokenizer())
            output = next((root / "runs" / "longbench_v2").glob("20*"))
            result = json.loads((output / "execution.jsonl").read_text())
            self.assertEqual(result["status"], "ok", result["error"])
            self.assertEqual(result["route"], "dense_child_packed")
            self.assertFalse(result["cache_written"])

    def test_jarvis_service_requirements_and_forwarding(self):
        env = {**os.environ, "OPENAI_COMPAT_EXECUTOR_BASE_URL": "http://executor:8000/v1",
               "AA_LCR_LAUNCH_DRY_RUN": "1", "MRCR_LAUNCH_DRY_RUN": "1"}
        env.pop("OPENAI_COMPAT_EVALUATOR_BASE_URL", None)
        result = subprocess.run(["bash", "jarvis/run_aa_lcr.sh", "hybrid_64k", "--execution-profile", "common",
                                 "--execution-only", "--max-input-tokens", "1000"], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("grader=none cache-verifier=none", result.stdout)
        self.assertIn("--max-input-tokens 1000", result.stdout)
        result = subprocess.run(["bash", "jarvis/run_mrcr_v2.sh", "hybrid", "--answer-cache-read"],
                                env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        result = subprocess.run(["bash", "jarvis/run_mrcr_v2.sh", "hybrid", "--answer-cache-read",
                                 "--cache-verifier-model", "verifier", "--cache-verifier-base-url", "http://verify/v1"],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cache-verifier=http://verify/v1", result.stdout)
        env["LONGBENCH_LAUNCH_DRY_RUN"] = "1"
        for mode in ("direct", "hybrid"):
            result = subprocess.run(["bash", "jarvis/run_longbench_v2.sh", mode,
                "--min-source-tokens", "100", "--max-source-tokens", "200"],
                env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--execution-profile common", result.stdout)
            self.assertIn("--min-source-tokens 100 --max-source-tokens 200", result.stdout)
            self.assertIn("grader=none cache-verifier=none", result.stdout)


if __name__ == "__main__":
    unittest.main()
