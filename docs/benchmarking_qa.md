# Benchmarking Q&A Log

This file tracks newcomer-style benchmarking questions and answers.

## Q1
Question: Which benchmark is used and what kind of tasks are executed to test this project?

Answer:
- We are using the official RULER v2 benchmark path.
- The current milestone task set is:
  - mk_niah_basic
  - mv_niah_basic
  - qa_basic
- We run these tasks at two context lengths:
  - 8192
  - 32768
- Purpose of this scope: establish strict official-score comparability first, then expand to the full benchmark matrix.

## Q2
Question: So you have 3 different tasks. Can you elaborate on them? What are they trying to test?

Answer:
- mk_niah_basic:
  - Tests targeted retrieval in long context when key facts are sparse and easy to miss.
  - Focus: whether the system can reliably locate the right needle-like fact and return it accurately.
- mv_niah_basic:
  - Tests retrieval when multiple relevant values/facts are present and can be confused with each other.
  - Focus: whether the system can disambiguate and return the correct value(s) without mixing nearby distractors.
- qa_basic:
  - Tests standard question answering quality over longer context windows.
  - Focus: end-to-end ability to retrieve relevant evidence and produce a correct final answer.

Combined intent of the 3-task milestone:
- Stress retrieval precision under long-context noise.
- Check answer correctness after retrieval.
- Validate behavior consistency at medium and larger context sizes (8192 and 32768).

## Q3
Question: Which datasets are used to test these tasks?

Answer:
- For official benchmarking, we use official RULER v2 prepared datasets for the selected tasks and context lengths.
- In this project, those datasets are provided to the runner through `--official-prepared-data` (JSON/JSONL file or directory produced by the official prep flow).
- For development-only smoke checks, we also used a local fixture:
  - `benchmark_fixtures/suites/ruler_adapter_pilot.json`
  - This fixture validates pipeline wiring, but it is not a source for strict official benchmark scores.

Practical rule:
- Official score runs: use official prepared RULER v2 data.
- Local dry runs: optional fixture data is acceptable for sanity checks only.

## Q4
Question: Where do we get the data from? I see that in the README there is parameter that takes the path. Do we have the dataset locally ready?

Answer:
- The official benchmark data should come from the official RULER v2 preparation flow (for example, NeMo-Skills data prep), and the generated output path is then passed into `--official-prepared-data`.
- In this repository, we currently have local smoke fixtures and smoke-run artifacts, but no confirmed full official prepared RULER v2 dataset directory for the milestone matrix.
- What exists locally now:
  - Fixture for dry runs: `benchmark_fixtures/suites/ruler_adapter_pilot.json`
  - Prior smoke artifacts: `benchmark_artifacts/official_ruler_v2/<run_id>/...`
- So for strict official scoring runs, we still need to generate or place the official prepared dataset locally and point `--official-prepared-data` to that location.

## Q5
Question: How can I run the script? Could you summarize the commands for me?

Answer:
- 1) Activate environment and inspect CLI:
  source .venv/bin/activate
  python run_benchmark.py --help

- 2) Local smoke run (pipeline sanity check only, not official scoring):
  python run_benchmark.py \
    --official-prepared-data benchmark_fixtures/suites/ruler_adapter_pilot.json \
    --official-tasks niah,tracing \
    --official-lengths 8192 \
    --official-max-samples-per-task 1 \
    --mode cache \
    --output-dir benchmark_artifacts

- 3) Official milestone run (when official prepared data is ready):
  python run_benchmark.py \
    --official-prepared-data /path/to/ruler2_prepared_data \
    --official-tasks mk_niah_basic,mv_niah_basic,qa_basic \
    --official-lengths 8192,32768 \
    --mode cache \
    --output-dir benchmark_artifacts

- 4) Official scoring run with evaluator command (strict scoring path):
  python run_benchmark.py \
    --official-prepared-data /path/to/ruler2_prepared_data \
    --official-tasks mk_niah_basic,mv_niah_basic,qa_basic \
    --official-lengths 8192,32768 \
    --mode cache \
    --official-eval-command "python -m nemo_skills.evaluation.evaluate --predictions {predictions} --output-dir {results_dir}" \
    --output-dir benchmark_artifacts

- 5) Optional: include official prep command in same invocation:
  python run_benchmark.py \
    --official-prepared-data /path/to/ruler2_prepared_data \
    --official-tasks mk_niah_basic,mv_niah_basic,qa_basic \
    --official-lengths 8192,32768 \
    --mode cache \
    --official-prep-command "python -m nemo_skills.inference.generate_data --tasks {tasks_csv} --lengths {lengths_csv} --output {prepared_data}" \
    --official-eval-command "python -m nemo_skills.evaluation.evaluate --predictions {predictions} --output-dir {results_dir}" \
    --output-dir benchmark_artifacts

Artifacts are written under benchmark_artifacts/official_ruler_v2/<run_id>/ with predictions.jsonl, bridge_rows.jsonl, and manifest.json.

Quick rule:
- Use command 3 when checking that the pipeline can generate outputs.
- Use command 4 when you need official scores to report.
- If the pipeline is already stable and you want final results, command 4 alone is usually enough.

## Q6
Question: What is the difference between the third and fourth command you mentioned?

Answer:
- Command 3:
  - Runs the official benchmark bridge only.
  - Produces prediction and telemetry artifacts.
  - Does not execute the official evaluator.
  - Result: useful for generation validation, but not final official scores.

- Command 4:
  - Runs the same bridge generation step.
  - Also executes the official evaluator command via --official-eval-command.
  - Result: strict official scoring path, suitable for authoritative benchmark reporting.

## Q7
Question: What is the bridge generation step? Also, what's the difference between command 2 and 3 you mentioned?

Answer:
- Bridge generation step means:
  - Read prepared benchmark samples from --official-prepared-data.
  - For each sample, run this project's architecture (ingest context + answer question).
  - Write evaluator-ready predictions to `predictions.jsonl`.
  - Write run telemetry to `bridge_rows.jsonl`.
  - Save run metadata to `manifest.json`.

- Difference between command 2 and command 3:
  - Command 2 is a local smoke run using fixture data in this repo (`benchmark_fixtures/suites/ruler_adapter_pilot.json`).
    - Goal: sanity-check wiring and execution.
    - Not suitable for official score reporting.
  - Command 3 is the same bridge flow but with official prepared RULER v2 data path.
    - Goal: generate real benchmark predictions for the official task/length matrix.
    - This is the required pre-step before official scoring.

In short:
- Command 2 = pipeline test on local mock-like data.
- Command 3 = real benchmark generation on official prepared data.

## Q8
Question: How much would it cost to run the benchmark?

Answer:
- Current known signal (from local smoke telemetry):
  - 2 samples cost about $0.00331 total.
  - Approximate per-sample cost observed: about $0.0015 to $0.0018.
- Cost estimate formula for a run:
  - estimated_total_cost ~= sample_count x per_sample_cost
- Example projections using current smoke-rate range:
  - 100 samples: about $0.15 to $0.18
  - 500 samples: about $0.75 to $0.90
  - 1000 samples: about $1.50 to $1.80

Important caveats:
- These numbers come from short local fixture samples and can be lower than real official long-context runs.
- 32768-length official tasks can increase token usage and cost significantly.
- Evaluator command itself is usually cheap relative to generation; most cost comes from prediction generation calls.

Best practice:
- Run a small official subset first (for example, cap samples per task), read cost from bridge_rows.jsonl, then extrapolate before full-matrix execution.

## Q9
Question: How many samples are there in the official benchmark dataset?

Answer:
- There is no single fixed sample count defined in this repository.
- In our workflow, the count depends on the official prepared data that you generate or provide at `--official-prepared-data`.
- So the exact number is determined by the prep configuration and the selected tasks/lengths.

How to check once data is ready:
- If your prepared data is a JSONL file:
  wc -l /path/to/ruler2_prepared_data.jsonl
- If it is a directory with multiple JSONL files:
  find /path/to/ruler2_prepared_data -name "*.jsonl" -print0 | xargs -0 cat | wc -l

For the current milestone matrix (`mk_niah_basic`, `mv_niah_basic`, `qa_basic` at `8192` and `32768`), we will compute and report exact counts from the prepared files before running the full benchmark.

## Q10
Question: In this repository, where are benchmark inputs and outputs saved? Are testing and full-run outputs stored in the same location?

Answer:
- Inputs:
  - Local smoke fixture inputs are under `benchmark_fixtures/suites/`.
  - Official prepared benchmark inputs are provided externally and passed via `--official-prepared-data` (path can be outside or inside this repo).

- Outputs:
  - Official benchmark runner outputs are written under `benchmark_artifacts/official_ruler_v2/<run_id>/`.
  - Each run writes separate artifacts in its own timestamped run folder (`predictions.jsonl`, `bridge_rows.jsonl`, `manifest.json`).

- Testing vs full-run location:
  - They use the same root output location (`benchmark_artifacts/official_ruler_v2/`).
  - They are separated by run_id folders, so smoke/test and full runs do not overwrite each other.

- Additional legacy smoke output folders from earlier phases may still exist under `benchmark_artifacts/` (for example, adapter smoke folders), but the current official workflow writes to `benchmark_artifacts/official_ruler_v2/`.

## Q11
Question: How are evaluating "correct" answers?

Answer:
- In the current official-only workflow, correctness is evaluated by the official evaluator command passed with `--official-eval-command`.
- The runner itself generates predictions (`predictions.jsonl`) and telemetry (`bridge_rows.jsonl`) but does not apply custom correctness scoring.
- This is intentional so reported metrics come from the official benchmark evaluator, not project-specific scoring logic.

Practical implication:
- If `--official-eval-command` is omitted, you get generation artifacts only.
- If `--official-eval-command` is provided, that run is the strict official correctness/scoring path.

## Q12
Question: When we pass the CLI parameter, what code does it execute? How does evaluation actually happen?

Answer:
- This repository does not contain evaluator implementation code.
- The runner treats `--official-eval-command` as a shell command string and executes it via Python subprocess.
- So the executed code is whatever external command you provide (for example, a NeMo-Skills module command).

Execution chain:
- CLI value from `--official-eval-command`
- Placeholder substitution (`{predictions}`, `{results_dir}`, etc.)
- subprocess execution with `shell=True` in current working directory

Where evaluation "actually" happens:
- In the external tool/package invoked by that command (outside this repo's Python functions).

So this is the right question, and the key point is:
- runner = orchestrator and artifact producer
- external evaluator command = source of official scoring logic

## Q13
Question: What is the `nemo_skills` part?

Answer:
- `nemo_skills` refers to the NeMo-Skills Python package (external to this repository).
- It provides benchmark tooling, including command-line modules for data preparation and evaluation.
- In our commands, examples like:
  - `python -m nemo_skills.inference.generate_data ...`
  - `python -m nemo_skills.evaluation.evaluate ...`
  mean: run NeMo-Skills module entry points through Python.

Why we use it here:
- We want official-compatible benchmark processing and scoring.
- This repository generates predictions, then hands scoring to NeMo-Skills via `--official-eval-command`.

Important operational note:
- `nemo_skills` is not implemented in this repo; it must be installed and available in the active Python environment for those commands to work.

## Q14
Question: Explain the --mode parameter.

Answer:
- `--mode` controls how this repository generates predictions before official evaluation.
- Allowed values are:
  - `baseline`
  - `cache`

What each mode does:
- `baseline`:
  - Calls search with cache reads disabled.
  - Intended to represent no-reuse behavior during generation.
- `cache`:
  - Calls search with cache reads enabled.
  - Uses the architecture's normal cache-enabled behavior during generation.

Important scope note:
- `--mode` affects only prediction generation in this runner.
- The external official evaluator still scores the produced predictions in the same way; mode changes the predictions, not evaluator logic.

Typical usage:
- Use `cache` as default for normal architecture benchmarking.
- Use `baseline` when you want an ablation-style comparison against cache-enabled behavior.
