import unittest

from scripts.run_dp_memo_shared_context import planner_response


class SharedContextBenchmarkTests(unittest.TestCase):
    def test_planner_response_recovers_answer_from_partial_json(self):
        raw = """
        {
          "action": "answer",
          "answer": "Lisbon, Marta",
          "used_entry_ids": [
            "533122c3e97507e1e187c9ba",
            "077077229dcd10736310ae2f"
          ],
          "confidence": 0.95,
          "reason": "truncated
        """

        parsed = planner_response(raw, ["fallback"])

        self.assertEqual(parsed["action"], "answer")
        self.assertEqual(parsed["answer"], "Lisbon, Marta")
        self.assertEqual(
            parsed["used_entry_ids"],
            ["533122c3e97507e1e187c9ba", "077077229dcd10736310ae2f"],
        )
        self.assertEqual(parsed["confidence"], 0.95)


if __name__ == "__main__":
    unittest.main()
