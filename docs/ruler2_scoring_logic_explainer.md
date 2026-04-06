# RULER2 Fractional Accuracy Explainer (Single-Sample Case)

## Why This Note Exists

In the run report below, some task-length buckets show fractional accuracy values (for example 0.5 and 0.6) even though each bucket has only 1 sample.

- Run report: `benchmark_artifacts/official_ruler_v2/20260405T194658Z/official_ruler2_eval_report.json`
- Example bucket: `mv_niah_basic|8192` with `samples=1` and `accuracy=0.6`

At first glance, this looks wrong if you assume strict binary scoring (0 or 1 only).

## Short Answer

Yes, this behavior comes from the real installed NeMo-Skills evaluator logic used by this repository.

The scorer used here computes `accuracy` as the mean of per-row `is_correct`, and `is_correct` in NeMo RULER2 can be a continuous value in `[0, 1]` (not necessarily binary).

## Exact Path Used in This Repo

1. `score_ruler2_predictions.py` loads `predictions.jsonl`.
2. It calls `nemo_skills.evaluation.evaluator.ruler.eval_ruler2(...)`.
3. It computes bucket accuracy as:

   `accuracy = sum(is_correct values in bucket) / number of rows in bucket`

So when `samples=1`, the reported accuracy is exactly that row's `is_correct` value.

## Why `mv_niah_basic|8192` Became `0.6`

From this run's `predictions.jsonl`, sample `sample_000005` has:

- Task: `mv_niah_basic`
- Length: `8192`
- Expected answer: `9653250072`
- Generated answer text: a wrong sentence containing another number (`8955404771`)

In this run shape, the expected answer is passed as a plain string. Inside the evaluator, the matching loop iterates over the reference sequence; for a plain string this effectively becomes per-character matching behavior (`'9'`, `'6'`, `'5'`, ...).

For `9653250072`, 6 out of 10 characters are found somewhere in the long generated text, so the resulting score is:

`6 / 10 = 0.6`

That is why a clearly wrong answer can still get `0.6`.

## Is This Really NeMo Library Logic?

Yes. This repository uses the installed NeMo package function directly:

- Import site in scorer: `from nemo_skills.evaluation.evaluator.ruler import eval_ruler2`
- Installed evaluator implementation file in this environment:
  - `.venv/lib/python3.13/site-packages/nemo_skills/evaluation/evaluator/ruler.py`

No local fork of the evaluator logic is implemented in this repository for this step.

## Practical Interpretation

- Fractional values in this report are expected with the current scoring path.
- Do not interpret these values as strict per-sample correctness.
- If strict binary behavior is desired for numeric-answer tasks, add a normalization layer before evaluation (for example: reference list normalization and explicit exact-match thresholding).

## Optional Verification Command

You can reproduce the report with:

```bash
./.venv/bin/python score_ruler2_predictions.py \
  --run-dir benchmark_artifacts/official_ruler_v2/20260405T194658Z
```

And inspect source logic used at runtime in:

- `score_ruler2_predictions.py`
- `.venv/lib/python3.13/site-packages/nemo_skills/evaluation/evaluator/ruler.py`
