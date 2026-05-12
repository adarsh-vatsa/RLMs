import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from long_bench_v2.export_csv import CSV_COLUMNS, _context_id, export_csv, load_rows
from long_bench_v2.combine_csv import combine_csv


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
        self.assertNotIn("context", rows[0])
        self.assertEqual(rows[1]["source_id"], "row_1")
        self.assertEqual(rows[1]["row_type"], "exact")
        self.assertEqual(rows[1]["is_scored"], "true")
        self.assertEqual(rows[1]["setup_case_id"], "")
        self.assertEqual(rows[1]["case_id"], "row_1__exact")
        self.assertEqual(rows[1]["expected_cache_type"], "exact")
        self.assertEqual(rows[1]["expected_from_cache"], "true")
        self.assertEqual(rows[1]["depends_on_case_id"], "row_1__original")
        self.assertEqual(rows[1]["question"], rows[0]["question"])

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

            count = combine_csv(base_path, semantic_path, output_path)

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

            count = combine_csv(base_path, semantic_path, output_path, knowledge_path)

            with output_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(count, 4)
        self.assertEqual(
            [row["case_id"] for row in rows],
            ["row_1__original", "row_1__semantic", "ctx__setup", "row_1__knowledge"],
        )
        self.assertEqual(rows[2]["is_scored"], "false")
        self.assertEqual(rows[3]["setup_case_id"], "ctx__setup")


if __name__ == "__main__":
    unittest.main()
