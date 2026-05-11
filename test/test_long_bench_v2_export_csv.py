import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from long_bench_v2.export_csv import CSV_COLUMNS, _context_id, export_csv, load_rows


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
        self.assertEqual(rows[1]["case_id"], "row_1__exact")
        self.assertEqual(rows[1]["expected_cache_type"], "exact")
        self.assertEqual(rows[1]["expected_from_cache"], "true")
        self.assertEqual(rows[1]["depends_on_case_id"], "row_1__original")
        self.assertEqual(rows[1]["question"], rows[0]["question"])


if __name__ == "__main__":
    unittest.main()
