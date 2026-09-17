from collections import Counter
import json
from pathlib import Path


def record_evaluation(directory, case_id, status, scores):
    with (Path(directory) / "evaluation.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps({"case_id": case_id, "status": status, "scores": scores}) + "\n")


def finalize(directory):
    directory = Path(directory)
    path = directory / "execution.jsonl"
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    evaluation_path = directory / "evaluation.jsonl"
    evaluations = {row["case_id"]: row for row in (
        json.loads(line) for line in evaluation_path.read_text().splitlines())} if evaluation_path.exists() else {}
    score_names = sorted({key for row in evaluations.values() for key in row["scores"]})

    def summary(group):
        supported = [row for row in group if row["status"] != "unsupported_context"]
        pending = [row for row in supported if row["status"] == "ok"
                   and evaluations.get(row["case_id"], {}).get("status") != "ok"]
        scores = {}
        for key in score_names:
            missing = any(key not in evaluations.get(row["case_id"], {}).get("scores", {})
                          for row in supported if row["status"] == "ok")
            scores[key] = (sum(float(evaluations[row["case_id"]]["scores"][key])
                               for row in supported if row["status"] == "ok") / len(supported)
                           if supported and not pending and not missing else None)
        return {"selected_count": len(group), "supported_count": len(supported),
                "supported_fraction": len(supported) / len(group) if group else None,
                "status_counts": dict(Counter(row["status"] for row in group)),
                "grading_pending_count": len(pending), "quality": scores,
                **{key: sum(row[key] for row in group) for key in (
                    "input_tokens", "output_tokens", "verifier_input_tokens", "verifier_output_tokens",
                    "attempts", "verifier_attempts", "ingest_ms", "retrieval_ms", "packing_ms", "generation_ms", "cache_verification_ms")}}

    def band(row):
        tokens = row["full_rendered_input_tokens"]
        lower = 1 << (max(1, tokens).bit_length() - 1)
        return f"[{lower},{lower * 2})"

    report = {**summary(rows), "denominator": "supported selected examples; execution failures score zero; unresolved grading is incomplete",
              "by_route": {route: summary([row for row in rows if row["route"] == route]) for route in sorted({row["route"] for row in rows})},
              "by_source_band": {key: summary([row for row in rows if band(row) == key]) for key in sorted({band(row) for row in rows})}}
    (directory / "execution_report.json").write_text(json.dumps(report, indent=2) + "\n")
