import csv
import difflib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from aa_lcr.api import Completion
from aa_lcr.prompting import chat_token_count
from mrcr_v2.dataset import band_urls, load_prepared, read_source, text_hash
from mrcr_v2.prepare_dataset import build_arg_parser as prepare_parser, prepare_dataset
from mrcr_v2.prompting import FEWSHOT_END, OMITTED, messages, pack_evidence
from mrcr_v2.run_benchmark import build_arg_parser as run_parser, run_benchmark
from mrcr_v2.scoring import mrcr_v2_metric, score_prediction


class FakeTokenizer:
    chat_template = "test-template-v1"
    init_kwargs = {"_commit_hash": "fixture-revision"}
    special_tokens_map = {"eos_token": "<end>"}

    def get_vocab(self):
        return {"fixture": 0}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"add_generation_prompt": True, "tokenize": True, "enable_thinking": False}
        return list(range(sum(len(message["content"]) for message in messages) + 2))

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text))),
                "offset_mapping": [(index, index + 1) for index in range(len(text))]}


def tokenizer_factory(model, revision=None):
    return FakeTokenizer()


PREFIX = "Here are some examples of conversations succeeded by a follow-up question answered correctly:\n" + FEWSHOT_END
MARKER = "AbCd1234EfGh"


def example(*, index="first", body=None, answer="response 0", published=8000000):
    if body is None:
        body = "".join(f"User: Write a poem about stars in formal style.\n\nAssistant: response {i}\n\n" for i in range(8))
    question = f"User: Prepend {MARKER} to the {index} poem about stars in a formal style. Do not include any other text in your response.\n\nAssistant:"
    return {
        "queries": PREFIX + body + question, "view_ops": question,
        "answer": MARKER + answer, "num_relevant": 8, "context_len": published,
        "answer_context_position": "GOLD_POSITION_ONLY", "sampling_or_scoring": "sampling",
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class FakeEmbedder:
    calls = 0

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.device = "cpu"
        self.torch_dtype_name = "float32"

    def encode_documents(self, texts):
        type(self).calls += 1
        self._last_encode_info = {"max_sequence_tokens": max(map(len, texts))}
        return [[1.0] for _ in texts]


class FakeIndex:
    def add(self, embeddings, metadata):
        self.metadata = metadata
        self.total = len(metadata)


class FakeController:
    instances = []

    def __init__(self, *, embedder, **kwargs):
        self.embedder = embedder
        self.queries = []
        self.instances.append(self)

    def retrieve(self, query, **kwargs):
        self.queries.append(query)
        assert kwargs == {"top_k": self.doc_index.total, "rerank_top": 0, "use_reranker": False}
        return [
            {"text": self._doc_chunks[index], "score": index, "metadata": self.doc_index.metadata[index]}
            for index in reversed(range(self.doc_index.total))
        ]

    def lookup_cached_result(self, *args, **kwargs):
        raise AssertionError("MRCR must never read the answer cache")

    def store_compact_answer(self, *args, **kwargs):
        raise AssertionError("MRCR must never write the answer cache")


def fake_scs():
    from semantic_cache_system import _chunk_text_with_tokenizer

    FakeEmbedder.calls = 0
    FakeController.instances = []
    return types.SimpleNamespace(
        EMBEDDING_MODEL="embedding-fixture", EMBEDDING_MAX_LENGTH=8192,
        EMBEDDING_QUERY_INSTRUCTION="fixture instruction", EMBEDDING_CONTRACT_VERSION="fixture",
        EMBEDDING_BATCH_SIZE=16,
        EmbeddingEngine=FakeEmbedder, SemanticCacheController=FakeController,
        ExecutionMetrics=lambda: None, FAISSIndex=FakeIndex,
        _chunk_text_with_tokenizer=_chunk_text_with_tokenizer,
    )


class MRCRTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = io.StringIO()
        self.redirect = redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def prepare(self, rows=None, extra=(), name="prepared"):
        path = self.root / f"{name}.csv"
        write_csv(path, rows or [example()])
        args = prepare_parser().parse_args([
            "--input-csv", str(path), "--data-dir", str(self.root / name),
            "--min-source-tokens", "1", "--max-source-tokens", "1000000", *extra,
        ])
        manifest = prepare_dataset(args, tokenizer_factory=tokenizer_factory)
        return args.data_dir, manifest

    def run_args(self, data_dir, mode="direct", extra=()):
        return run_parser().parse_args([
            "--mode", mode, "--data-dir", str(data_dir),
            "--output-root", str(self.root / "runs"), "--max-retries", "1", *extra,
        ])

    def execute(self, args, scs=None, fail=False):
        calls = []

        def caller(**kwargs):
            calls.append(kwargs)
            if fail:
                raise RuntimeError("fixture API failure")
            return Completion(MARKER + "response 0", 42, 7, {"prompt_tokens": 42}, "stop")

        result = run_benchmark(args, tokenizer_factory=tokenizer_factory,
                               completion_caller=caller, scs_module=scs)
        if result is None:
            return result, calls, [], {}
        records = [json.loads(line) for line in (result / "predictions.jsonl").read_text().splitlines()]
        report = json.loads((result / "eval_report.json").read_text())
        return result, calls, records, report

    def test_preparation_preserves_prompts_deduplicates_and_counts_actual_tokens(self):
        first = example()
        second = example(index="second", answer="response 1")
        data_dir, manifest = self.prepare([first, first, second])
        _, rows = load_prepared(data_dir)
        self.assertEqual(manifest["duplicate_rows"], 1)
        self.assertEqual(manifest["source_count"], 1)
        self.assertEqual(len(rows), 2)
        for row, original in zip(rows, [first, second]):
            prefix, body = read_source(data_dir, row["source_id"])
            self.assertEqual(prefix + body + row["question"], original["queries"])
            self.assertEqual(row["full_rendered_input_tokens"], len(original["queries"]) + 2)
            self.assertEqual(row["published_context_len"], 8000000)
        self.assertNotIn("GOLD_POSITION_ONLY", (data_dir / "sources" / rows[0]["source_id"] / "context.txt").read_text())

    def test_inclusive_single_point_bounds_and_empty_selection(self):
        row = example()
        count = len(row["queries"]) + 2
        _, manifest = self.prepare([row], ["--min-source-tokens", str(count), "--max-source-tokens", str(count)])
        self.assertEqual(len(manifest["selected_example_ids"]), 1)
        with self.assertRaisesRegex(ValueError, "No MRCR examples"):
            self.prepare([row], ["--min-source-tokens", str(count + 1)], name="empty")

    def test_invalid_bounds_and_deterministic_limits(self):
        for minimum, maximum in [(0, 10), (10, 9)]:
            with self.assertRaisesRegex(ValueError, "Source token bounds"):
                self.prepare(extra=["--min-source-tokens", str(minimum), "--max-source-tokens", str(maximum)], name=f"invalid{minimum}")
        rows = [example(index="first"), example(index="second")]
        _, one = self.prepare(rows, ["--max-rows", "1"], name="one")
        _, two = self.prepare(rows, ["--max-rows", "1"], name="two")
        self.assertEqual(one["selected_example_ids"], two["selected_example_ids"])
        self.assertEqual(one["selected_example_ids"], [text_hash(rows[0]["queries"])])

    def test_conflicting_reference_after_limit_is_rejected(self):
        first = example()
        conflict = {**first, "answer": MARKER + "conflict"}
        with self.assertRaisesRegex(ValueError, "Conflicting references"):
            self.prepare([first, conflict], ["--max-rows", "1"])

    def test_bad_suffix_and_format_and_needle_count_are_rejected(self):
        for index, row in enumerate([
            {**example(), "view_ops": "wrong suffix"},
            {**example(), "queries": example()["queries"].replace(FEWSHOT_END, "unsupported\n")},
            {**example(), "num_relevant": 4},
        ]):
            with self.assertRaises(ValueError):
                self.prepare([row], name=f"bad{index}")

    def test_large_multiline_csv_and_crlf_preserved(self):
        body = 'User: Write a poem.\r\n\r\nAssistant: "stars", 星\r\n' + "x" * 150000 + "\n\n"
        row = example(body=body)
        data_dir, _ = self.prepare([row])
        _, rows = load_prepared(data_dir)
        prefix, restored = read_source(data_dir, rows[0]["source_id"])
        self.assertEqual(prefix + restored + rows[0]["question"], row["queries"])

    def test_explicit_download_bands_only(self):
        urls = band_urls("131072:262144,65536:131072", 8)
        self.assertEqual(len(urls), 2)
        self.assertIn("8needle_in_(65536,131072)", urls[0])
        with self.assertRaises(ValueError):
            band_urls("100000:200000", 8)
        calls = []

        def download(url, path):
            calls.append(url)
            write_csv(path, [example()])

        args = prepare_parser().parse_args([
            "--download-bands", "65536:131072", "--data-dir", str(self.root / "downloaded"),
            "--min-source-tokens", "1", "--max-source-tokens", "10000",
        ])
        manifest = prepare_dataset(args, tokenizer_factory=tokenizer_factory, downloader=download)
        self.assertEqual(calls, urls[:1])
        self.assertEqual(manifest["sources"][0]["url"], calls[0])

    def test_runtime_bounds_budget_and_tokenizer_validation(self):
        data_dir, _ = self.prepare(extra=["--min-source-tokens", "100"])
        for extra in [
            ["--min-source-tokens", "99"], ["--max-source-tokens", "1000001"],
            ["--max-input-tokens", "65536"], ["--child-tokens", "10", "--child-overlap-tokens", "10"],
        ]:
            with self.assertRaises(ValueError):
                self.execute(self.run_args(data_dir, extra=extra))
        changed = FakeTokenizer()
        changed.chat_template = "changed"
        with self.assertRaisesRegex(ValueError, "tokenizer or chat template changed"):
            run_benchmark(self.run_args(data_dir), tokenizer_factory=lambda *args: changed)
        with self.assertRaisesRegex(ValueError, "No MRCR examples"):
            self.execute(self.run_args(data_dir, extra=["--min-source-tokens", "999999"]))

    def test_changed_source_or_questions_rejected(self):
        data_dir, _ = self.prepare()
        _, rows = load_prepared(data_dir)
        context = data_dir / "sources" / rows[0]["source_id"] / "context.txt"
        context.write_text("tampered")
        with self.assertRaisesRegex(ValueError, "transcript changed"):
            self.execute(self.run_args(data_dir, extra=["--preflight-only"]))
        (data_dir / "questions.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "questions changed"):
            load_prepared(data_dir)

    def test_direct_and_hybrid_fit_preserve_prompt_and_exclude_gold(self):
        row = example()
        data_dir, _ = self.prepare([row])
        for mode in ["direct", "hybrid"]:
            with patch("mrcr_v2.run_benchmark._import_semantic_cache_system", side_effect=AssertionError("embedding must not load")):
                directory, calls, records, report = self.execute(self.run_args(data_dir, mode))
            self.assertEqual(calls[0]["messages"], messages(row["queries"]))
            self.assertNotIn("GOLD_POSITION_ONLY", str(calls[0]))
            self.assertEqual(calls[0]["extra_body"], {"temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
            self.assertEqual(records[0]["route"], "direct_fit")
            self.assertEqual(report["mean_mrcr_score"], 1)
            self.assertEqual(report["exact_match_accuracy"], 1)
            self.assertEqual(report["input_tokens"], 42)
            self.assertTrue((directory / "bridge_rows.csv").exists())

    def test_direct_overflow_has_no_call_or_quality_denominator(self):
        data_dir, _ = self.prepare()
        _, calls, records, report = self.execute(self.run_args(data_dir, extra=["--max-input-tokens", "10"]))
        self.assertEqual(calls, [])
        self.assertEqual(records[0]["status"], "unsupported_context")
        self.assertIsNone(records[0]["mrcr_score"])
        self.assertEqual(report["supported_count"], 0)
        self.assertIsNone(report["mean_mrcr_score"])

    def test_mixed_success_failure_and_unsupported_denominators(self):
        short = "User: Write a poem.\n\nAssistant: response 0\n\n"
        data_dir, _ = self.prepare([
            example(body=short), example(body=short, index="second"),
            example(body=short + "distractor " * 200),
        ])
        calls = []

        def caller(**kwargs):
            calls.append(kwargs)
            if "second poem" in kwargs["messages"][0]["content"]:
                raise RuntimeError("failure")
            return Completion(MARKER + "response 0", 10, 2, {})

        directory = run_benchmark(
            self.run_args(data_dir, extra=["--max-input-tokens", "500"]),
            tokenizer_factory=tokenizer_factory, completion_caller=caller,
        )
        report = json.loads((directory / "eval_report.json").read_text())
        self.assertEqual(len(calls), 2)
        self.assertEqual(report["status_counts"], {"ok": 1, "error": 1, "unsupported_context": 1})
        self.assertEqual(report["supported_fraction"], 2 / 3)
        self.assertEqual(report["mean_mrcr_score"], 0.5)
        self.assertEqual(report["exact_match_accuracy"], 0.5)

    def test_runtime_narrowing_and_equal_input_budget(self):
        first, second = example(), example(index="second")
        data_dir, _ = self.prepare([first, second])
        length = len(first["queries"]) + 2
        _, calls, records, _ = self.execute(self.run_args(data_dir, extra=[
            "--min-source-tokens", str(length), "--max-source-tokens", str(length),
            "--max-input-tokens", str(length),
        ]))
        self.assertEqual(len(calls), 1)
        self.assertEqual(records[0]["case_id"], text_hash(first["queries"]))
        self.assertEqual(records[0]["route"], "direct_fit")

    def test_preflight_verifies_routes_without_models_or_artifacts(self):
        data_dir, _ = self.prepare()
        audit = self.root / "audit.json"
        args = self.run_args(data_dir, "hybrid", ["--preflight-only", "--preflight-output", str(audit), "--max-input-tokens", "500"])
        with patch("mrcr_v2.run_benchmark._import_semantic_cache_system", side_effect=AssertionError("no embedding")):
            _, calls, _, _ = self.execute(args)
        self.assertEqual(calls, [])
        self.assertFalse((self.root / "runs").exists())
        self.assertEqual(json.loads(audit.read_text())["route_counts"], {"dense_child_packed": 1})

    def test_hybrid_retrieves_in_order_reuses_index_and_has_no_cache(self):
        data_dir, _ = self.prepare([example(), example(index="second")])
        scs = fake_scs()
        args = self.run_args(data_dir, "hybrid", ["--max-input-tokens", "500", "--child-tokens", "100", "--child-overlap-tokens", "20"])
        _, calls, records, report = self.execute(args, scs)
        self.assertEqual(len(calls), 2)
        self.assertEqual(FakeEmbedder.calls, 1)
        self.assertEqual(len(FakeController.instances[0].queries), 2)
        self.assertEqual(report["status_counts"], {"ok": 2})
        for record, call in zip(records, calls):
            self.assertLessEqual(record["final_rendered_input_tokens"], 500)
            self.assertEqual(record["route"], "dense_child_packed")
            text = call["messages"][0]["content"]
            self.assertTrue(text.startswith(PREFIX))
            self.assertIn(OMITTED, text)
            ranges = record["selected_evidence_ranges"]
            self.assertEqual(ranges, sorted(ranges, key=lambda value: value["char_start"]))
            self.assertNotIn("GOLD_POSITION_ONLY", text)
        self.assertGreater(records[0]["ingested_chunks"], 0)
        self.assertEqual(records[1]["ingested_chunks"], 0)

    def test_missing_offsets_never_fall_back(self):
        data_dir, _ = self.prepare()
        scs = fake_scs()
        scs._chunk_text_with_tokenizer = lambda *args, **kwargs: None
        _, calls, records, report = self.execute(self.run_args(data_dir, "hybrid", ["--max-input-tokens", "500"]), scs)
        self.assertEqual(calls, [])
        self.assertEqual(report["mean_mrcr_score"], 0)
        self.assertIn("Exact embedding-tokenizer offsets", records[0]["error"])

    def test_embedding_limit_and_failed_requests_are_scored_zero(self):
        data_dir, _ = self.prepare()
        scs = fake_scs()
        scs.EMBEDDING_MAX_LENGTH = 100
        _, calls, records, report = self.execute(self.run_args(data_dir, "hybrid", ["--max-input-tokens", "500"]), scs)
        self.assertEqual(calls, [])
        self.assertIn("embedding input limit", records[0]["error"])
        self.assertEqual(report["supported_count"], 1)
        _, calls, records, report = self.execute(self.run_args(data_dir), fail=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(records[0]["attempts"], 1)
        self.assertEqual(report["mean_mrcr_score"], 0)
        self.assertEqual(report["status_counts"], {"error": 1})

    def test_fail_fast_preserves_processed_artifacts(self):
        data_dir, _ = self.prepare([example(), example(index="second")])
        args = self.run_args(data_dir, extra=["--fail-fast", "--run-id", "failed"])
        with self.assertRaisesRegex(RuntimeError, "results saved"):
            self.execute(args, fail=True)
        run_dir = self.root / "runs" / "direct" / "failed"
        manifest = json.loads((run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "stopped_early")
        self.assertEqual(len((run_dir / "predictions.jsonl").read_text().splitlines()), 1)

    def test_pack_merges_overlap_orders_ranges_and_respects_exact_budget(self):
        body = "User: " + "0123456789" * 5
        spans = [(30, 45), (25, 35), (5, 10)]
        results = [{"text": body[start:end], "metadata": {"char_start": start, "char_end": end, "child_index": i}}
                   for i, (start, end) in enumerate(spans)]
        request, info = pack_evidence(FakeTokenizer(), "PREFIX", body, "QUESTION", results, 1000)
        self.assertEqual(info["selected_evidence_ranges"], [{"char_start": 5, "char_end": 10}, {"char_start": 25, "char_end": 45}])
        self.assertEqual(request[0]["content"], "PREFIX" + OMITTED + body[5:10] + OMITTED + body[25:45] + OMITTED + "QUESTION")
        one, _ = pack_evidence(FakeTokenizer(), "PREFIX", body, "QUESTION", results[:1], 1000)
        budget = chat_token_count(FakeTokenizer(), one)
        request, info = pack_evidence(FakeTokenizer(), "PREFIX", body, "QUESTION", results, budget)
        self.assertEqual(info["selected_child_indices"], [0])
        self.assertEqual(info["final_rendered_input_tokens"], budget)
        with self.assertRaisesRegex(ValueError, "No retrieved child fits"):
            pack_evidence(FakeTokenizer(), "PREFIX", body, "QUESTION", results, budget - 1)
        results[0]["text"] = "wrong text"
        with self.assertRaisesRegex(ValueError, "does not match"):
            pack_evidence(FakeTokenizer(), "PREFIX", body, "QUESTION", results, 1000)

    def test_adjoining_evidence_reconstructs_body_without_separators(self):
        body = "User: question\n\nAssistant: response\n\n"
        results = [{"text": body[start:end], "metadata": {"char_start": start, "char_end": end, "child_index": i}}
                   for i, (start, end) in enumerate([(10, len(body)), (0, 10)])]
        request, info = pack_evidence(FakeTokenizer(), "PREFIX", body, "QUESTION", results, 1000)
        self.assertEqual(request, messages("PREFIX" + body + "QUESTION"))
        self.assertEqual(info["selected_evidence_ranges"], [{"char_start": 0, "char_end": len(body)}])

    def test_official_score_and_separate_strict_metrics(self):
        target = MARKER + "abcdef"
        cases = [
            (target, 1), ("  " + target + "\n", 1),
            ("abcdef", 0), ("", 0),
            ("explanation " + target, 1),
            (MARKER + "wrong " + target, 1),
            (MARKER + "abc", difflib.SequenceMatcher(a="abcdef", b="abc").ratio()),
        ]
        for prediction, expected in cases:
            self.assertEqual(mrcr_v2_metric(prediction, target), expected)
        self.assertEqual(score_prediction("explanation " + target, target),
                         {"mrcr_score": 1, "exact_match": False, "prefix_compliant": False})
        self.assertFalse(score_prediction(MARKER + "wrong " + target, target)["exact_match"])

    def test_jarvis_dry_run_forwards_bounds_without_evaluator(self):
        environment = {**os.environ, "MRCR_LAUNCH_DRY_RUN": "1", "OPENAI_COMPAT_EXECUTOR_BASE_URL": "http://fixture:8000/v1"}
        environment.pop("OPENAI_COMPAT_EVALUATOR_BASE_URL", None)
        for mode, allocation in [("direct", "client"), ("hybrid", "client-gpu")]:
            result = subprocess.run([
                "bash", "jarvis/run_mrcr_v2.sh", mode,
                "--min-source-tokens", "100000", "--max-source-tokens", "200000",
                "--max-input-tokens", "60000", "--max-output-tokens", "4096", "--context-window-tokens", "65536",
            ], env=environment, capture_output=True, text=True, check=True)
            self.assertIn(f"allocation={allocation} ", result.stdout)
            self.assertIn("--min-source-tokens 100000 --max-source-tokens 200000", result.stdout)
            self.assertIn("--max-input-tokens 60000 --max-output-tokens 4096 --context-window-tokens 65536", result.stdout)
            self.assertNotIn("--evaluator", result.stdout)


if __name__ == "__main__":
    unittest.main()
