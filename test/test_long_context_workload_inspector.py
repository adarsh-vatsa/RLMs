import unittest

from scripts.run_dp_memo_nolima import (
    content_hash_for_sample,
    data_kind,
    document_id_for_sample,
    stable_corpus_id,
)
from scripts.inspect_long_context_workload import estimate_length_row, parse_lengths


class LongContextWorkloadInspectorTests(unittest.TestCase):
    def test_parse_lengths_accepts_k_suffixes(self):
        self.assertEqual(parse_lengths("32K,64k,128000"), [32000, 64000, 128000])

    def test_estimate_length_row_reports_context_window_multiple_when_requested(self):
        row = estimate_length_row(
            length=128000,
            case_count=2,
            haystack_count=3,
            depth_intervals=4,
            chunk_words=1000,
            chunk_size=2,
            agent_window_tokens=32000,
        )

        self.assertEqual(row["samples"], 24)
        self.assertEqual(row["estimated_chunks_per_sample"], 128)
        self.assertEqual(row["estimated_solver_windows_per_sample"], 64)
        self.assertEqual(row["estimated_model_calls_per_cold_sample"], 65)
        self.assertEqual(row["estimated_model_calls_cold_total"], 1560)
        self.assertEqual(row["context_window_multiple"], 4.0)
        self.assertTrue(row["exceeds_agent_window"])

    def test_dp_memo_nolima_data_kind_distinguishes_fixture_and_downloaded_data(self):
        self.assertEqual(
            data_kind(
                "benchmark_fixtures/nolima/needlesets/needle_set.json",
                "benchmark_fixtures/nolima/haystack/rand_shuffle",
            ),
            "fixture",
        )
        self.assertEqual(
            data_kind(
                "benchmark_data/nolima/needlesets/needle_set.json",
                "benchmark_data/nolima/haystack/rand_shuffle_long",
            ),
            "official_or_downloaded",
        )

    def test_stable_document_and_source_hash_modes_ignore_length(self):
        sample_8k = {
            "id": "case__book__L8000__D00",
            "test_name": "case",
            "haystack_name": "book",
            "haystack_hash": "abc",
            "needle": "needle",
            "depth_index": 0,
        }
        sample_16k = {**sample_8k, "id": "case__book__L16000__D00"}

        self.assertEqual(
            document_id_for_sample(sample_8k, "stable"),
            document_id_for_sample(sample_16k, "stable"),
        )
        self.assertEqual(
            content_hash_for_sample("short context", sample_8k, "source"),
            content_hash_for_sample("longer overlapping context", sample_16k, "source"),
        )
        self.assertNotEqual(
            content_hash_for_sample("short context", sample_8k, "sample"),
            content_hash_for_sample("longer overlapping context", sample_16k, "sample"),
        )

    def test_stable_corpus_id_ignores_lengths(self):
        class Haystack:
            name = "book"
            sha256 = "abc"

        self.assertEqual(
            stable_corpus_id("needle-hash", [Haystack()], 42),
            stable_corpus_id("needle-hash", [Haystack()], 42),
        )


if __name__ == "__main__":
    unittest.main()
