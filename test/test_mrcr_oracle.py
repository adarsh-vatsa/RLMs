import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

from aa_lcr.api import Completion
from mrcr_v2.dataset import load_prepared, read_source
from mrcr_v2.oracle import needle_spans
from mrcr_v2.prepare_dataset import build_arg_parser as prepare_parser, prepare_dataset
from mrcr_v2.results_table import main as results_table
from mrcr_v2.run_benchmark import build_arg_parser as run_parser, run_benchmark
from mrcr_v2.visibility import summarize
from test_mrcr_v2 import MARKER, FakeTokenizer, example, fake_scs, tokenizer_factory, write_csv


NEEDLES = (2, 5, 8, 11)


def conversation():
    return "".join(
        f"User: Write a poem about stars in formal style.\n\nAssistant: response {turn}\n\n" if turn in NEEDLES
        else f"User: Write a riddle about ducks in informal style.\n\nAssistant: distractor {turn} {'filler ' * 20}\n\n"
        for turn in range(12))


def question(index, answer):
    return {**example(index=index, body=conversation(), answer=f"response {answer}"), "num_relevant": 4}


class OracleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.suite = self.root / "S"
        redirect = redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        write_csv(self.root / "rows.csv", [question("first", 2), question("third", 8)])
        self.data_dir = self.root / "prepared"
        prepare_dataset(prepare_parser().parse_args([
            "--input-csv", str(self.root / "rows.csv"), "--data-dir", str(self.data_dir), "--needles", "4",
            "--min-source-tokens", "1", "--max-source-tokens", "1000000"]), tokenizer_factory=tokenizer_factory)

    def run_condition(self, name, mode, *flags):
        args = run_parser().parse_args([
            "--mode", mode, "--data-dir", str(self.data_dir), "--output-root", str(self.suite),
            "--run-id", f"S_{name}", "--max-retries", "1", "--max-input-tokens", "1400",
            "--child-tokens", "100", "--child-overlap-tokens", "20", *flags])
        return run_benchmark(args, tokenizer_factory=tokenizer_factory, scs_module=fake_scs(),
                             completion_caller=lambda **kwargs: Completion(MARKER + "response 2", 1, 1, {}, "stop"))

    def test_needle_spans_follow_the_question_and_check_the_gold_answer(self):
        _, rows = load_prepared(self.data_dir)
        _, body = read_source(self.data_dir, rows[0]["source_id"])
        spans, target = needle_spans(body, rows[1]["question"], rows[1]["answer"], 4)
        self.assertEqual(target, 2)
        self.assertEqual([body[start:end].split("response ")[1].strip() for start, end in spans], ["2", "5", "8", "11"])
        with self.assertRaisesRegex(ValueError, "needle turns"):
            needle_spans(body, rows[1]["question"], MARKER + "response 5", 4)
        with self.assertRaisesRegex(ValueError, "needle turns"):
            needle_spans(body, rows[1]["question"], rows[1]["answer"], 8)

    def test_oracle_offers_the_needle_turns_first_without_changing_the_prompt_format(self):
        position = self.run_condition("position", "hybrid", "--evidence-selection", "position")
        oracle = self.run_condition("oracle", "hybrid", "--evidence-selection", "position", "--oracle")
        self.assertTrue(json.loads((oracle / "manifest.json").read_text())["oracle"])
        self.assertFalse(json.loads((position / "manifest.json").read_text())["oracle"])
        with self.assertRaisesRegex(ValueError, "requires --mode hybrid"):
            self.run_condition("direct_oracle", "direct", "--oracle")

        results_table([str(self.suite)])
        policies = {row["condition"]: row["policy"] for row in csv.DictReader((self.suite / "results.csv").open())}
        self.assertEqual(policies["oracle"], "oracle: position, 100-token chunks")

        columns, rows = summarize(self.suite, self.data_dir, tokenizer_factory=tokenizer_factory)
        rows = {row["condition"]: row for row in rows}
        self.assertEqual((rows["oracle"]["target visible"], rows["oracle"]["all needles visible"]), (1, 1))
        self.assertLess(rows["position"]["all needles visible"], 1)

    def test_visibility_rebuilds_truncated_prompts_with_the_runs_tokenizer(self):
        self.run_condition("middle", "direct", "--direct-overflow", "middle")
        columns, rows = summarize(self.suite, self.data_dir, tokenizer_factory=tokenizer_factory)
        middle = rows[0]
        # Middle truncation keeps the first and last needle turns: the "first" target, not the "third".
        self.assertEqual((middle["target visible"], middle["all needles visible"]), (0.5, 0))
        self.assertEqual((middle["exact match given target visible"], middle["exact match given target hidden"]), (1, 0))
        self.assertEqual(middle["exact match given all needles visible"], "")
        self.assertIn("target visible at 2K", columns)

        changed = FakeTokenizer()
        changed.chat_template = "changed"
        with self.assertRaisesRegex(ValueError, "differs from the one the runs used"):
            summarize(self.suite, self.data_dir, tokenizer_factory=lambda *args: changed)


if __name__ == "__main__":
    unittest.main()
