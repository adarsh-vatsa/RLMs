# NoLiMa Benchmark Integration

This folder contains the high-parity NoLiMa benchmark bridge used to evaluate
this repository's SemanticCache system.

## Files

- `run_benchmark.py`: Expands NoLiMa needle sets and executes benchmark runs.
- `score_nolima_predictions.py`: Scores existing run predictions.
- `parity_bridge.py`: Needle-set expansion and haystack placement logic.

## Data and Artifacts Layout

Use existing top-level benchmark folders:

- Input data root: `benchmark_data/nolima/`
- Output artifacts root: `benchmark_artifacts/official_nolima/`

Suggested input structure:

- `benchmark_data/nolima/needlesets/needle_set.json`
- `benchmark_data/nolima/haystack/rand_shuffle/*.txt`

Synthetic smoke fixture included in this repository:

- `benchmark_fixtures/nolima/needlesets/needle_set.json`
- `benchmark_fixtures/nolima/haystack/rand_shuffle/*.txt`

## To-Do's
- What is depth intervals?
- Get the dataset and run the code
- What is the question asked?
- Check cost and token usage for both benchmarks (does the bridge store totals?)