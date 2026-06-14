import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from long_bench_v2.export_csv import CSV_COLUMNS, _context_id, _estimated_token_count, export_csv, load_rows
from long_bench_v2.combine_csv import combine_csv
from long_bench_v2.sample_csv import parse_token_buckets, sample_rows


def _row(row_id: str, context: str = "Context text") -> dict:
    return {
        "_id": row_id,
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
        "context": context,
    }


class LongBenchV2CsvExportTests(unittest.TestCase):
    def test_load_rows_maps_id_to_source_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "data.json"
            input_path.write_text(json.dumps([_row("row_1")]), encoding="utf-8")

            rows = load_rows(input_path, include_exact=False)

        self.assertEqual(rows[0]["case_id"], "row_1__original")
        self.assertEqual(rows[0]["source_id"], "row_1")
        self.assertEqual(rows[0]["row_type"], "original")
        self.assertEqual(rows[0]["is_scored"], "true")
        self.assertEqual(rows[0]["setup_case_id"], "")
        self.assertEqual(rows[0]["expected_cache_type"], "miss")
        self.assertEqual(rows[0]["expected_from_cache"], "false")
        self.assertEqual(rows[0]["depends_on_case_id"], "")
        self.assertEqual(rows[0]["context_id"], _context_id("Context text"))
        self.assertEqual(rows[0]["token_count"], _estimated_token_count("Context text"))
        self.assertNotIn("_id", rows[0])
        self.assertNotIn("context", rows[0])

    def test_export_csv_writes_original_and_exact_rows_without_context(self):
        long_context = "Line 1\n" + ("long text, with comma and \"quote\" " * 20)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "data.json"
            output_path = Path(tmpdir) / "data.csv"
            input_path.write_text(json.dumps([_row("row_1", context=long_context)]), encoding="utf-8")

            count = export_csv(input_path, output_path)

            with output_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

        self.assertEqual(count, 2)
        self.assertEqual(reader.fieldnames, CSV_COLUMNS)
        self.assertEqual(rows[0]["source_id"], "row_1")
        self.assertEqual(rows[0]["row_type"], "original")
        self.assertEqual(rows[0]["case_id"], "row_1__original")
        self.assertEqual(rows[0]["expected_cache_type"], "miss")
        self.assertEqual(rows[0]["expected_from_cache"], "false")
        self.assertEqual(rows[0]["depends_on_case_id"], "")
        self.assertEqual(rows[0]["context_id"], _context_id(long_context))
        self.assertEqual(rows[0]["token_count"], _estimated_token_count(long_context))
        self.assertNotIn("context", rows[0])
        self.assertEqual(rows[1]["source_id"], "row_1")
        self.assertEqual(rows[1]["row_type"], "exact")
        self.assertEqual(rows[1]["is_scored"], "true")
        self.assertEqual(rows[1]["setup_case_id"], "")
        self.assertEqual(rows[1]["case_id"], "row_1__exact")
        self.assertEqual(rows[1]["expected_cache_type"], "exact")
        self.assertEqual(rows[1]["expected_from_cache"], "true")
        self.assertEqual(rows[1]["depends_on_case_id"], "row_1__original")
        self.assertEqual(rows[1]["token_count"], _estimated_token_count(long_context))
        self.assertEqual(rows[1]["question"], rows[0]["question"])

    def test_combine_csv_backfills_token_count_from_source_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "data.csv"
            source_json_path = Path(tmpdir) / "data.json"
            output_path = Path(tmpdir) / "data_cache_suite.csv"
            context = "one two three four"
            source_json_path.write_text(json.dumps([_row("row_1", context=context)]), encoding="utf-8")
            legacy_columns = [column for column in CSV_COLUMNS if column != "token_count"]
            base_rows = [
                {
                    "case_id": "row_1__original",
                    "source_id": "row_1",
                    "row_type": "original",
                    "is_scored": "true",
                    "setup_case_id": "",
                    "context_id": _context_id(context),
                    "expected_cache_type": "miss",
                    "expected_from_cache": "false",
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
            ]
            with base_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=legacy_columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(base_rows)

            count = combine_csv(base_path, output_path, source_json_path=source_json_path)

            with output_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

        self.assertEqual(count, 1)
        self.assertEqual(reader.fieldnames, CSV_COLUMNS)
        self.assertEqual(rows[0]["token_count"], _estimated_token_count(context))

    def test_combine_csv_writes_base_and_semantic_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "data.csv"
            semantic_path = Path(tmpdir) / "data_semantic_codex.csv"
            output_path = Path(tmpdir) / "data_cache_suite.csv"
            base_rows = [
                {
                    "case_id": "row_1__original",
                    "source_id": "row_1",
                    "row_type": "original",
                    "is_scored": "true",
                    "setup_case_id": "",
                    "context_id": "ctx",
                    "expected_cache_type": "miss",
                    "expected_from_cache": "false",
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
                },
                {
                    "case_id": "row_1__exact",
                    "source_id": "row_1",
                    "row_type": "exact",
                    "is_scored": "true",
                    "setup_case_id": "",
                    "context_id": "ctx",
                    "expected_cache_type": "exact",
                    "expected_from_cache": "true",
                    "depends_on_case_id": "row_1__original",
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
                },
            ]
            semantic_rows = [
                {
                    **base_rows[0],
                    "case_id": "row_1__semantic",
                    "row_type": "semantic",
                    "is_scored": "true",
                    "setup_case_id": "",
                    "expected_cache_type": "semantic",
                    "expected_from_cache": "true",
                    "depends_on_case_id": "row_1__original",
                    "question": "Which choice is right?",
                }
            ]

            for path, rows in [(base_path, base_rows), (semantic_path, semantic_rows)]:
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)

            count = combine_csv(base_path, output_path, semantic_csv_path=semantic_path, source_json_path=None)

            with output_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(count, 3)
        self.assertEqual([row["case_id"] for row in rows], ["row_1__original", "row_1__exact", "row_1__semantic"])
        self.assertEqual(rows[2]["question"], "Which choice is right?")

    def test_combine_csv_includes_knowledge_rows_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "data.csv"
            semantic_path = Path(tmpdir) / "data_semantic_codex.csv"
            knowledge_path = Path(tmpdir) / "data_knowledge_codex.csv"
            output_path = Path(tmpdir) / "data_cache_suite.csv"
            base_rows = [
                {
                    "case_id": "row_1__original",
                    "source_id": "row_1",
                    "row_type": "original",
                    "is_scored": "true",
                    "setup_case_id": "",
                    "context_id": "ctx",
                    "expected_cache_type": "miss",
                    "expected_from_cache": "false",
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
            ]
            semantic_rows = [
                {
                    **base_rows[0],
                    "case_id": "row_1__semantic",
                    "row_type": "semantic",
                    "question": "Which choice is right?",
                }
            ]
            knowledge_rows = [
                {
                    **base_rows[0],
                    "case_id": "ctx__setup",
                    "source_id": "ctx__setup",
                    "row_type": "setup",
                    "is_scored": "false",
                    "setup_case_id": "",
                    "expected_cache_type": "",
                    "expected_from_cache": "",
                    "question": "Summarize the key facts.",
                    "choice_A": "",
                    "choice_B": "",
                    "choice_C": "",
                    "choice_D": "",
                    "answer": "",
                },
                {
                    **base_rows[0],
                    "case_id": "row_1__knowledge",
                    "row_type": "knowledge",
                    "setup_case_id": "ctx__setup",
                    "expected_cache_type": "",
                    "expected_from_cache": "",
                },
            ]

            for path, rows_to_write in [
                (base_path, base_rows),
                (semantic_path, semantic_rows),
                (knowledge_path, knowledge_rows),
            ]:
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows_to_write)

            count = combine_csv(
                base_path,
                output_path,
                semantic_csv_path=semantic_path,
                knowledge_csv_path=knowledge_path,
                source_json_path=None,
            )

            with output_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(count, 4)
        self.assertEqual(
            [row["case_id"] for row in rows],
            ["row_1__original", "row_1__semantic", "ctx__setup", "row_1__knowledge"],
        )
        self.assertEqual(rows[2]["is_scored"], "false")
        self.assertEqual(rows[3]["setup_case_id"], "ctx__setup")

    def test_combine_csv_allows_generated_inputs_to_be_omitted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "data.csv"
            output_path = Path(tmpdir) / "data_cache_suite.csv"
            base_rows = [
                {
                    "case_id": "row_1__original",
                    "source_id": "row_1",
                    "row_type": "original",
                    "is_scored": "true",
                    "setup_case_id": "",
                    "context_id": "ctx",
                    "expected_cache_type": "miss",
                    "expected_from_cache": "false",
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
            ]
            with base_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(base_rows)

            count = combine_csv(base_path, output_path, source_json_path=None)

            with output_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(count, 1)
        self.assertEqual([row["case_id"] for row in rows], ["row_1__original"])

    def test_sample_rows_keeps_balanced_source_linked_row_types(self):
        rows = []
        for source_id in ["row_1", "row_2", "row_3"]:
            for row_type in ["original", "exact", "semantic"]:
                rows.append({
                    "case_id": f"{source_id}__{row_type}",
                    "source_id": source_id,
                    "row_type": row_type,
                    "is_scored": "true",
                    "setup_case_id": "",
                    "context_id": "ctx",
                    "expected_cache_type": row_type,
                    "expected_from_cache": "false",
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
                })

        sampled = sample_rows(rows, sample_size=2, row_types=("original", "exact", "semantic"), seed=7)

        self.assertEqual(len(sampled), 6)
        sampled_by_type = {}
        for row in sampled:
            sampled_by_type.setdefault(row["row_type"], set()).add(row["source_id"])
        self.assertEqual(set(sampled_by_type), {"original", "exact", "semantic"})
        self.assertEqual(sampled_by_type["original"], sampled_by_type["exact"])
        self.assertEqual(sampled_by_type["original"], sampled_by_type["semantic"])
        self.assertEqual(len(sampled_by_type["original"]), 2)

    def test_sample_rows_filters_by_token_count_and_can_select_shortest(self):
        rows = []
        token_counts = {"row_1": "300", "row_2": "100", "row_3": "200", "row_4": "900"}
        for source_id, token_count in token_counts.items():
            for row_type in ["original", "exact", "semantic"]:
                rows.append({
                    "case_id": f"{source_id}__{row_type}",
                    "source_id": source_id,
                    "row_type": row_type,
                    "is_scored": "true",
                    "setup_case_id": "",
                    "context_id": "ctx",
                    "token_count": token_count,
                    "expected_cache_type": row_type,
                    "expected_from_cache": "false",
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
                })

        sampled = sample_rows(
            rows,
            sample_size=2,
            row_types=("original", "exact", "semantic"),
            seed=0,
            max_token_count=300,
            selection_strategy="shortest",
        )

        self.assertEqual({row["source_id"] for row in sampled}, {"row_2", "row_3"})
        self.assertEqual(len(sampled), 6)

    def test_sample_rows_can_select_token_stratified_source_groups(self):
        rows = []
        token_counts = {
            "short_1": "50",
            "short_2": "80",
            "medium_1": "150",
            "medium_2": "180",
            "long_1": "250",
            "long_2": "280",
            "xlong_1": "500",
        }
        for source_id, token_count in token_counts.items():
            for row_type in ["original", "exact", "semantic"]:
                rows.append({
                    "case_id": f"{source_id}__{row_type}",
                    "source_id": source_id,
                    "row_type": row_type,
                    "is_scored": "true",
                    "setup_case_id": "",
                    "context_id": "ctx",
                    "token_count": token_count,
                    "expected_cache_type": row_type,
                    "expected_from_cache": "false",
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
                })
        buckets = parse_token_buckets("short:0:100,medium:101:200,long:201:300")

        first = sample_rows(
            rows,
            sample_size=6,
            row_types=("original", "exact", "semantic"),
            seed=11,
            selection_strategy="token_stratified",
            token_buckets=buckets,
        )
        second = sample_rows(
            rows,
            sample_size=6,
            row_types=("original", "exact", "semantic"),
            seed=11,
            selection_strategy="token_stratified",
            token_buckets=buckets,
        )

        self.assertEqual([row["case_id"] for row in first], [row["case_id"] for row in second])
        selected_sources = {row["source_id"] for row in first}
        self.assertEqual(selected_sources, {"short_1", "short_2", "medium_1", "medium_2", "long_1", "long_2"})
        self.assertNotIn("xlong_1", selected_sources)
        self.assertEqual(len(first), 18)

    def test_sample_rows_token_stratified_fails_when_bucket_is_underfilled(self):
        rows = []
        token_counts = {
            "short_1": "50",
            "medium_1": "150",
            "medium_2": "175",
            "medium_3": "190",
            "long_1": "250",
            "long_2": "275",
        }
        for source_id, token_count in token_counts.items():
            for row_type in ["original", "exact", "semantic"]:
                rows.append({
                    "case_id": f"{source_id}__{row_type}",
                    "source_id": source_id,
                    "row_type": row_type,
                    "is_scored": "true",
                    "setup_case_id": "",
                    "context_id": "ctx",
                    "token_count": token_count,
                    "expected_cache_type": row_type,
                    "expected_from_cache": "false",
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
                })

        with self.assertRaisesRegex(ValueError, "Token bucket 'short'"):
            sample_rows(
                rows,
                sample_size=6,
                row_types=("original", "exact", "semantic"),
                seed=0,
                selection_strategy="token_stratified",
                token_buckets=parse_token_buckets("short:0:100,medium:101:200,long:201:300"),
            )


if __name__ == "__main__":
    unittest.main()
