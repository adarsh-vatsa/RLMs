import csv
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aa_lcr.api import Completion
from aa_lcr.compare_runs import compare_runs, validate_runs
from aa_lcr.dataset import (
    build_scope_hash,
    load_documents,
    load_questions,
    sha256_file,
    validate_dataset,
)
from aa_lcr.prompting import (
    build_grader_prompt,
    build_prompt,
    pack_retrieved_children,
    parse_grade,
    prepare_direct_messages,
)
from aa_lcr.prepare_dataset import _decoded_member_name
from aa_lcr.run_benchmark import build_arg_parser, run_benchmark


class FakeTokenizer:
    name_or_path = "Qwen/fake"
    chat_template = "fake"

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=True,
    ):
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return {"input_ids": self.encode(rendered)}


class FakeCompletionCaller:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["model"] == "Qwen/Qwen3.5-35B-A3B":
            text = "CORRECT"
        else:
            text = "Equinix, $901 M"
        return Completion(
            text=text,
            input_tokens=10,
            output_tokens=2,
            raw_usage={"prompt_tokens": 10, "completion_tokens": 2},
        )


class MalformedGraderCaller(FakeCompletionCaller):
    def __call__(self, **kwargs):
        completion = super().__call__(**kwargs)
        if kwargs["model"] == "Qwen/Qwen3.5-35B-A3B":
            return Completion("Probably correct", 10, 2, completion.raw_usage)
        return completion


class FailingExecutorCaller(FakeCompletionCaller):
    def __call__(self, **kwargs):
        if kwargs["model"] == "Qwen/Qwen3.6-35B-A3B":
            raise RuntimeError("executor unavailable")
        return super().__call__(**kwargs)


def write_fixture(root: Path, *, long_document=False, duplicate_questions=False):
    data_dir = root / "data"
    documents_root = data_dir / "extracted_text" / "lcr"
    set_root = documents_root / "Company_Documents" / "co_test"
    set_root.mkdir(parents=True)
    document_a = "A" * (70000 if long_document else 40)
    (set_root / "second.txt").write_text(document_a, encoding="utf-8")
    (set_root / "first.txt").write_text("First source", encoding="utf-8")
    questions_path = data_dir / "AA-LCR_Dataset.csv"
    fieldnames = [
        "",
        "document_category",
        "document_set_id",
        "question_id",
        "question",
        "answer",
        "data_source_filenames",
        "data_source_urls",
        "input_tokens",
    ]
    with questions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in (1, 2):
            writer.writerow(
                {
                    "": index - 1,
                    "document_category": "Company_Documents",
                    "document_set_id": "co_test",
                    "question_id": str(index),
                    "question": (
                        "What was adjusted EBITDA?"
                        if duplicate_questions or index == 1
                        else "Name the company and adjusted EBITDA."
                    ),
                    "answer": "Equinix, $901 million",
                    "data_source_filenames": "second.txt;first.txt",
                    "data_source_urls": "https://example.test/second;https://example.test/first",
                    "input_tokens": "100",
                }
            )
    archive_path = data_dir / "AA-LCR_extracted-text.zip"
    archive_path.write_bytes(b"fixture archive")
    manifest_path = data_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_revision": "fixture-revision",
                "questions_sha256": sha256_file(questions_path),
                "archive_path": str(archive_path),
                "archive_sha256": sha256_file(archive_path),
                "question_count": 2,
            }
        ),
        encoding="utf-8",
    )
    return questions_path, documents_root, manifest_path


class FakeMetrics:
    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_misses = 0
        self.exact_hits = 0
        self.semantic_hits = 0

    def get_totals(self):
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class FakeHybridController:
    instances = []

    def __init__(self, metrics, embedder=None, reranker=None, **kwargs):
        self.metrics = metrics
        self.entries = {}
        self.data_scope_hash = ""
        self._last_cache_lookup_info = {"semantic_verifier_calls": 0}
        self._last_cache_query_embedding = None
        self.doc_index = types.SimpleNamespace(total=0)
        self.ingest_calls = 0
        self.save_calls = 0
        FakeHybridController.instances.append(self)

    def activate_data_scope(self, scope):
        self.data_scope_hash = scope

    def lookup_cached_result(self, query):
        answer = self.entries.get((self.data_scope_hash, query))
        if answer is None:
            return None
        self.metrics.exact_hits += 1
        return {
            "answer": answer,
            "cache_type": "exact",
            "cache_provenance": {"stored": True},
        }

    def ingest(self, docs_dir, **kwargs):
        self.ingest_calls += 1
        self.doc_index = types.SimpleNamespace(total=2)
        self._last_ingest_info = {
            "embedding_ms": 3.0,
            "child_encoded_length_max": 7000,
        }
        return 2

    def retrieve(self, query, **kwargs):
        return [
            {
                "text": "evidence " * 2000,
                "score": 0.9 - index * 0.1,
                "metadata": {
                    "filename": f"doc-{index}.txt",
                    "child_index": index,
                    "token_start": index * 100,
                    "token_end": (index + 1) * 100,
                },
            }
            for index in range(2)
        ]

    def store_compact_answer(self, query, answer, **kwargs):
        self.entries[(self.data_scope_hash, query)] = answer
        return {"query": query, "result": answer}

    def save(self, path):
        self.save_calls += 1
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "state.json").write_text("{}", encoding="utf-8")

    def get_total_entries(self):
        return len(self.entries)


class FakeScs:
    EMBEDDING_MODEL = "Qwen/fake-embedder"
    EMBEDDING_MAX_LENGTH = 8192
    DOCUMENT_CHUNK_TOKENS = 0
    DOCUMENT_CHUNK_OVERLAP_TOKENS = 0
    DOCUMENT_CHUNK_TOKENIZER_MODEL = ""
    SemanticCacheController = FakeHybridController
    ExecutionMetrics = FakeMetrics

    @staticmethod
    def configure_llm_provider(**kwargs):
        return None

    @staticmethod
    def EmbeddingEngine():
        return types.SimpleNamespace(device="cuda", torch_dtype_name="bfloat16")


class AALCRDatasetAndPromptTests(unittest.TestCase):
    def test_archive_member_names_recover_upstream_utf8_punctuation(self):
        self.assertEqual(
            str(_decoded_member_name("EUΓÇÖs AI ActΓÇöoverview.txt")),
            "EU’s AI Act—overview.txt",
        )

    def test_dataset_preserves_official_document_order_and_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            questions_path, documents_root, _ = write_fixture(Path(temporary))
            questions = load_questions(questions_path)
            validation = validate_dataset(
                questions, documents_root, expected_question_count=2
            )
            documents = load_documents(questions[0], documents_root)
            self.assertEqual(
                [name for name, _ in documents], ["second.txt", "first.txt"]
            )
            self.assertEqual(validation["document_set_count"], 1)
            self.assertEqual(build_scope_hash(documents), build_scope_hash(documents))
            self.assertNotEqual(
                build_scope_hash(documents), build_scope_hash(list(reversed(documents)))
            )

    def test_missing_document_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            questions_path, documents_root, _ = write_fixture(Path(temporary))
            (documents_root / "Company_Documents" / "co_test" / "first.txt").unlink()
            with self.assertRaises(FileNotFoundError):
                validate_dataset(load_questions(questions_path), documents_root)

    def test_official_prompt_and_semantic_grader_contract(self):
        prompt = build_prompt(["Document A", "Document B"], "What happened?")
        self.assertIn("BEGIN DOCUMENT 1:\nDocument A\nEND DOCUMENT 1", prompt)
        self.assertLess(prompt.index("Document A"), prompt.index("Document B"))
        grader = build_grader_prompt(
            "What was adjusted EBITDA?",
            "Equinix, $901 million",
            "Equinix, $901 M",
        )
        self.assertIn("Equinix, $901 million", grader)
        self.assertIn("Equinix, $901 M", grader)
        self.assertEqual(parse_grade(" CORRECT\n"), "CORRECT")
        self.assertEqual(parse_grade("Correct because equivalent"), "")

    def test_direct_middle_truncation_preserves_question_tail(self):
        tokenizer = FakeTokenizer()
        messages, info = prepare_direct_messages(
            ["x" * 500],
            "UNIQUE QUESTION TAIL",
            tokenizer,
            220,
            allow_truncation=True,
        )
        self.assertTrue(info["prompt_truncated"])
        self.assertLessEqual(info["prompt_tokens_after_truncation"], 220)
        self.assertIn("UNIQUE QUESTION TAIL", messages[0]["content"])
        with self.assertRaisesRegex(ValueError, "exceeding"):
            prepare_direct_messages(
                ["x" * 500],
                "UNIQUE QUESTION TAIL",
                tokenizer,
                220,
                allow_truncation=False,
            )

    def test_retrieval_packing_stays_in_budget(self):
        tokenizer = FakeTokenizer()
        results = [
            {
                "text": letter * 80,
                "score": 0.9,
                "metadata": {"child_index": index, "filename": f"{index}.txt"},
            }
            for index, letter in enumerate("abc")
        ]
        messages, info = pack_retrieved_children(tokenizer, "Question?", results, 360)
        self.assertLessEqual(info["rendered_input_tokens"], 360)
        self.assertGreater(info["selected_child_count"], 0)
        self.assertIn("START QUESTION", messages[0]["content"])


class AALCRRunnerTests(unittest.TestCase):
    def _args(self, root, experiment, questions, documents, manifest, run_id):
        return build_arg_parser().parse_args(
            [
                "--experiment",
                experiment,
                "--questions-csv",
                str(questions),
                "--documents-root",
                str(documents),
                "--dataset-manifest",
                str(manifest),
                "--output-root",
                str(root / "artifacts"),
                "--run-id",
                run_id,
                "--max-retries",
                "1",
            ]
        )

    def test_direct_run_writes_open_answer_artifacts_without_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            questions, documents, manifest = write_fixture(root)
            caller = FakeCompletionCaller()
            args = self._args(
                root, "direct_262k", questions, documents, manifest, "direct"
            )
            run_dir = run_benchmark(
                args,
                tokenizer_factory=lambda _: FakeTokenizer(),
                completion_caller=caller,
            )
            payload = json.loads((run_dir / "manifest.json").read_text())
            rows = [
                json.loads(line)
                for line in (run_dir / "bridge_rows.jsonl").read_text().splitlines()
            ]
            self.assertEqual(payload["answer_accuracy"], 1.0)
            self.assertEqual(payload["cache_hit_count"], 0)
            self.assertEqual(payload["executor_calls"], 2)
            self.assertTrue(all(row["route"] == "direct_fit" for row in rows))
            self.assertFalse((run_dir / "cache_state").exists())
            self.assertEqual(len(caller.calls), 4)

    def test_hybrid_64k_ingests_once_and_reuses_exact_answer(self):
        FakeHybridController.instances = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            questions, documents, manifest = write_fixture(
                root,
                long_document=True,
                duplicate_questions=True,
            )
            caller = FakeCompletionCaller()
            args = self._args(
                root, "hybrid_64k", questions, documents, manifest, "hybrid"
            )
            run_dir = run_benchmark(
                args,
                tokenizer_factory=lambda _: FakeTokenizer(),
                completion_caller=caller,
                scs_module=FakeScs,
            )
            payload = json.loads((run_dir / "manifest.json").read_text())
            controller = FakeHybridController.instances[-1]
            self.assertEqual(controller.ingest_calls, 1)
            self.assertEqual(payload["cache_hit_count"], 1)
            self.assertEqual(payload["executor_calls"], 1)
            self.assertEqual(payload["route_counts"]["dense_child_packed"], 1)
            self.assertEqual(payload["route_counts"]["exact_cache"], 1)
            self.assertTrue((run_dir / "cache_state" / "state.json").is_file())

    def test_direct_and_hybrid_direct_fit_send_identical_executor_messages(self):
        FakeHybridController.instances = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            questions, documents, manifest = write_fixture(root)
            direct_caller = FakeCompletionCaller()
            direct_args = self._args(
                root, "direct_262k", questions, documents, manifest, "direct-fit"
            )
            direct_args.max_rows = 1
            run_benchmark(
                direct_args,
                tokenizer_factory=lambda _: FakeTokenizer(),
                completion_caller=direct_caller,
            )

            hybrid_caller = FakeCompletionCaller()
            hybrid_args = self._args(
                root, "hybrid_262k", questions, documents, manifest, "hybrid-fit"
            )
            hybrid_args.max_rows = 1
            run_benchmark(
                hybrid_args,
                tokenizer_factory=lambda _: FakeTokenizer(),
                completion_caller=hybrid_caller,
                scs_module=FakeScs,
            )

            self.assertEqual(
                direct_caller.calls[0]["messages"], hybrid_caller.calls[0]["messages"]
            )

    def test_hybrid_runs_start_with_fresh_state(self):
        FakeHybridController.instances = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            questions, documents, manifest = write_fixture(
                root, duplicate_questions=True
            )
            for run_id in ("one", "two"):
                args = self._args(
                    root, "hybrid_262k", questions, documents, manifest, run_id
                )
                run_benchmark(
                    args,
                    tokenizer_factory=lambda _: FakeTokenizer(),
                    completion_caller=FakeCompletionCaller(),
                    scs_module=FakeScs,
                )
            self.assertEqual(len(FakeHybridController.instances), 2)
            self.assertIsNot(
                FakeHybridController.instances[0], FakeHybridController.instances[1]
            )
            self.assertEqual(FakeHybridController.instances[0].get_total_entries(), 1)
            self.assertEqual(FakeHybridController.instances[1].get_total_entries(), 1)

    def test_malformed_grade_is_recorded_without_reporting_full_accuracy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            questions, documents, manifest = write_fixture(root)
            args = self._args(
                root, "direct_262k", questions, documents, manifest, "malformed"
            )
            run_dir = run_benchmark(
                args,
                tokenizer_factory=lambda _: FakeTokenizer(),
                completion_caller=MalformedGraderCaller(),
            )
            payload = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(payload["invalid_grade_count"], 2)
            self.assertIsNone(payload["answer_accuracy"])

    def test_executor_failure_is_separate_from_incorrect_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            questions, documents, manifest = write_fixture(root)
            args = self._args(
                root, "direct_262k", questions, documents, manifest, "failure"
            )
            args.max_rows = 1
            run_dir = run_benchmark(
                args,
                tokenizer_factory=lambda _: FakeTokenizer(),
                completion_caller=FailingExecutorCaller(),
            )
            payload = json.loads((run_dir / "manifest.json").read_text())
            row = json.loads((run_dir / "bridge_rows.jsonl").read_text().strip())
            self.assertEqual(payload["api_error_count"], 1)
            self.assertEqual(payload["grader_calls"], 0)
            self.assertEqual(payload["total_request_attempts"], 1)
            self.assertIsNone(row["answer_correct"])


def write_comparison_run(root: Path, experiment: str, answers: list[bool]):
    mode, window, input_budget = {
        "direct_262k": ("direct", 262144, 240000),
        "hybrid_262k": ("hybrid", 262144, 240000),
        "direct_64k": ("direct", 65536, 60000),
        "hybrid_64k": ("hybrid", 65536, 60000),
    }[experiment]
    run_dir = root / experiment
    run_dir.mkdir()
    rows = []
    for index, correct in enumerate(answers, start=1):
        rows.append(
            {
                "case_id": f"aa_lcr_{index:03d}",
                "question_id": str(index),
                "document_set_id": f"set-{index}",
                "document_category": "Company_Documents",
                "question": f"Question {index}",
                "candidate_answer": "answer",
                "answer_correct": correct,
                "grade_valid": True,
                "route": "direct_fit",
                "cache_type": "miss",
                "cache_provenance": {},
            }
        )
    (run_dir / "bridge_rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = {
        "experiment": experiment,
        "mode": mode,
        "context_window_tokens": window,
        "max_input_tokens": input_budget,
        "dataset_revision": "revision",
        "dataset_signature": "signature",
        "questions_sha256": "questions",
        "archive_sha256": "archive",
        "question_ids": ["1", "2"],
        "executor_model": "executor",
        "evaluator_model": "evaluator",
        "max_output_tokens": 512,
        "temperature": 0,
        "thinking_enabled": False,
        "prompt_version": "prompt",
        "prompt_template_sha256": "prompt-hash",
        "grader_prompt_version": "grader",
        "grader_prompt_template_sha256": "grader-hash",
        "executor_tokenizer_model": "executor",
        "executor_chat_template_sha256": "chat-hash",
        "rows_selected": 2,
        "api_error_count": 0,
        "invalid_grade_count": 0,
        "child_tokens": 7500 if mode == "hybrid" else None,
        "child_overlap_tokens": 750 if mode == "hybrid" else None,
        "elapsed_seconds": 1,
        "total_api_calls": 4,
        "total_request_attempts": 4,
        "total_input_tokens": 100,
        "total_output_tokens": 10,
        "truncated_row_count": 0,
        "cache_hit_count": 0,
        "cache_hit_rate": 0,
        "cache_hit_accuracy": None,
        "semantic_hit_count": 0,
        "semantic_verifier_calls": 0,
        "executor_calls": 2,
        "grader_calls": 2,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


class AALCRComparisonAndLauncherTests(unittest.TestCase):
    def test_comparison_validates_and_computes_paired_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dirs = {
                "direct_262k": write_comparison_run(root, "direct_262k", [True, True]),
                "hybrid_262k": write_comparison_run(root, "hybrid_262k", [True, False]),
                "direct_64k": write_comparison_run(root, "direct_64k", [False, False]),
                "hybrid_64k": write_comparison_run(root, "hybrid_64k", [True, False]),
            }
            report = compare_runs(
                run_dirs,
                bootstrap_replicates=100,
                bootstrap_seed=7,
                require_full_runs=False,
            )
            self.assertEqual(report["accuracy"]["direct_262k"], 1.0)
            self.assertEqual(report["paired_deltas"]["hybrid_minus_direct_262k"], -0.5)
            self.assertIn(
                "hybrid_reasoning_retention", report["confidence_intervals_95"]
            )

            manifest = json.loads(
                (run_dirs["direct_64k"] / "manifest.json").read_text()
            )
            manifest["executor_model"] = "mismatch"
            (run_dirs["direct_64k"] / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                validate_runs(
                    {
                        experiment: (
                            json.loads((path / "manifest.json").read_text()),
                            [
                                json.loads(line)
                                for line in (path / "bridge_rows.jsonl")
                                .read_text()
                                .splitlines()
                            ],
                        )
                        for experiment, path in run_dirs.items()
                    },
                    require_full_runs=False,
                )

    def test_jarvis_launcher_dry_run_selects_cpu_and_gpu(self):
        repo_root = Path(__file__).resolve().parents[1]
        environment = {
            **os.environ,
            "OPENAI_COMPAT_EXECUTOR_BASE_URL": "http://executor:8000/v1",
            "OPENAI_COMPAT_EVALUATOR_BASE_URL": "http://evaluator:8001/v1",
            "AA_LCR_LAUNCH_DRY_RUN": "1",
        }
        direct = subprocess.run(
            ["bash", "jarvis/run_aa_lcr.sh", "direct_262k"],
            cwd=repo_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        hybrid = subprocess.run(
            ["bash", "jarvis/run_aa_lcr.sh", "hybrid_64k"],
            cwd=repo_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("allocation=client mem=32G", direct.stdout)
        self.assertIn("allocation=client-gpu mem=96G", hybrid.stdout)


if __name__ == "__main__":
    unittest.main()
