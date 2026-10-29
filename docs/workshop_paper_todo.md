# Workshop paper to-do

Tracks the work for the
[workshop paper plan](workshop_paper_plan_20261012.md). In the stage sections a
tick means the code for the item exists; items marked *(no code)* are
decisions or manual steps. The [experiment list](#experiment-list) at the end
tracks execution: every run is executed fresh on the current code, references
included. Commands are in the
[benchmark runbook](linux/BENCHMARK_RUNBOOK.md#overflow-policy-conditions).

## Stage 1: structure or selection

Earlier references, superseded by the fresh runs: AA-LCR 30 September runs
(middle truncation 37%, dense 54%; `direct/20260930T134555Z`,
`hybrid/20260930T025706Z`); MRCR 30 September runs (truncation 38%, dense 51%
over 120 questions, 30 per conversation; 3,500-token chunks, 350 overlap;
`direct/20260930T192314239638Z`, `hybrid/20260930T172140824150Z`). All on
Neselab.

- [x] AA-LCR 64K, `hybrid --evidence-selection position`
- [x] AA-LCR 64K, `hybrid --evidence-selection random --selection-seed 0`
- [x] AA-LCR 64K, `hybrid --evidence-selection random --selection-seed 1`
- [x] AA-LCR 64K, `hybrid --evidence-selection random --selection-seed 2`
- [x] AA-LCR 64K, `direct --direct-overflow middle-marked`
- [x] AA-LCR 64K, `direct --direct-overflow head` (keeps the start of the
      documents, then the question)
- [x] MRCR 262K, `hybrid --evidence-selection position`
- [x] MRCR 262K, `hybrid --evidence-selection random --selection-seed 0`
- [x] Grade the AA-LCR runs with the executor (Qwen3.6) as well. They use
      `--execution-only`, so they have no grades until then (`aa_lcr.regrade`
      with the executor as grader). These grades are the comparison point for
      the Luna agreement check (see Before submitting)

## Stage 2: lexical retrieval and the upgraded recipe

- [x] AA-LCR, `--evidence-selection bm25`
- [x] AA-LCR, `--evidence-selection fusion`
- [x] MRCR, `--evidence-selection bm25`
- [x] MRCR, `--evidence-selection fusion`
- [x] MRCR chunk size: 7,500-token chunks, 750 overlap, request query, all
      120 questions (so far run on only 40)
- [x] MRCR full-question query, `--retrieval-query question`
- [x] LongBench-v2 shared-pipeline baselines (truncation and dense), needed
      before tuning on LongBench-v2
- [x] Upgrade ladder on MRCR, starting from `fusion`, one change per run:
  - [x] `--child-tokens 1024 --child-overlap-tokens 128`
  - [x] `+ --neighbor-children 1`
  - [x] `+ --packing fill`
  - [x] `+ --document-floor`
- [x] Upgrade ladder on LongBench-v2, same steps
- [x] Fix one recipe for all three benchmarks; record it here *(no code)*
- [x] AA-LCR held-out check with that recipe, run once

Tune only on MRCR and LongBench-v2. Report every ladder step, including those
that did not help.

## Before submitting

- [x] Independent AA-LCR grader, run once after all AA-LCR runs are done:
      regrade every run with GPT-5.6 Luna at medium reasoning, the official
      AA-LCR equality checker, through OpenRouter. `aa_lcr.regrade` flags:
      `--evaluator-model openai/gpt-5.6-luna`
      `--evaluator-base-url https://openrouter.ai/api/v1`
      `--grader-api-style openai --grader-reasoning-effort medium`
      `--evaluator-api-key-env OPENROUTER_API_KEY`. The key is in the
      git-ignored `.env`, which `aa_lcr.regrade` does not read; export it
      first with `set -a; source .env; set +a`
  - [x] Send one grading call first and confirm the usage reports reasoning
        tokens, since OpenRouter may drop parameters a provider does not
        support. `aa_lcr.regrade` has no row limit, so make the call with
        `aa_lcr.grading.Grader`; regrades also save each row's `grader_usage`
  - [x] Report agreement with the Qwen3.6 grades: the Luna results table's
        `agreement with Qwen3.6-35B-A3B` column
  - [x] Hand-label about 50 grades, mostly where the two graders disagree, and
        report the grader's accuracy *(no code)*
- [x] AA-LCR results table per suite (`aa_lcr.results_table`), regenerated as
      runs finish. For the paper, use the Luna grades and add paired intervals
      against middle truncation from `aa_lcr.compare_conditions`
- [x] MRCR results table per suite (`mrcr_v2.results_table`), with paired
      gained and lost counts and an exact McNemar p-value against middle
      truncation, and exact match per conversation length
- [x] LongBench-v2 results table per suite (`long_bench_v2.results_table`),
      with the same paired comparison against middle truncation
- [ ] **Decide the lead**, after all runs and with the Luna grades for AA-LCR:
      where `position` and `random` land between truncation and dense on each
      benchmark. AA-LCR has a paired comparison (`aa_lcr.compare_conditions`,
      random seeds as repeats). Both results tables give paired gained and
      lost counts with an exact McNemar p-value, which treats questions as
      independent; questions share documents or conversations, so prefer
      intervals that resample them for the paper
- [x] LongBench-v2 on the shared pipeline, with a full-context reference
- [x] Budget sweep: the window flags cover it; see the experiment list for the
      AA-LCR budgets
- [x] GPU embedding timing for Table 3: records keep `document_embedding_ms`
- [ ] MRCR oracle and visibility for Table 2: not implemented. The
      visibility decomposition (target visible × accuracy given visible) in
      the 30 September report has no committed code either; the oracle needs
      a separate, labelled MRCR script outside the shared pipeline
- [ ] Second model family: the runners take any served model, but none has
      been chosen or tested outside Qwen. MRCR data must be re-prepared with
      its tokenizer (`mrcr_v2.prepare_dataset --executor-model`)
- [x] Check arXiv for concurrent overflow-policy papers *(no code)*

## Experiment list

Tick a run when it completes and record its run ID. All three benchmarks run on
Jarvis, one suite at a time, with `bash adarsh-rlms/jarvis/run_suite.sh
<aa_lcr|mrcr|longbench> <suite-id>`; see the
[Jarvis suite pages](jarvis_suites/README.md). Executor:
Qwen3.6-35B-A3B on vLLM (4 L40S, `--max-num-seqs 1`), temperature 0,
non-thinking. Hybrid runs embed on the client job's GPU. Table 3's GPU embedding
times come from these runs and its CPU times from the 30 September Neselab runs;
the embedding code has not changed since. Order: Stage 1, Stage 2 and
the ladders, then the recipe-dependent runs (AA-LCR held-out check, budget
sweeps).

### AA-LCR (Jarvis)

Suite: `run_suite.sh aa_lcr`; details in the
[AA-LCR suite runbook](linux/AA_LCR_SUITE_RUNBOOK.md). The name in
brackets is the suite's condition name. All 100 questions; every prompt is
88K–122K tokens. Base flags:
`--execution-only --max-output-tokens 4096 --fail-fast --serving-metadata .cache/serving_metadata.json`.
Main budget 64K: `--context-window-tokens 65536 --max-input-tokens 61440`.
Hybrid runs use the default 7,500-token chunks with 750 overlap.

- [ ] Full context, 262K (`full`): `direct --context-window-tokens 262144 --max-input-tokens 258048`
- [ ] Middle truncation (`middle`): `direct --direct-overflow middle`
- [ ] Marked middle truncation (`middle_marked`): `direct --direct-overflow middle-marked`
- [ ] Head plus question (`head`): `direct --direct-overflow head`
- [ ] Dense (`dense`): `hybrid`
- [ ] Position (`position`): `hybrid --evidence-selection position`
- [ ] Random, seed 0 (`random0`): `hybrid --evidence-selection random --selection-seed 0`
- [ ] Random, seed 1 (`random1`): `hybrid --evidence-selection random --selection-seed 1`
- [ ] Random, seed 2 (`random2`): `hybrid --evidence-selection random --selection-seed 2`
- [ ] BM25 (`bm25`): `hybrid --evidence-selection bm25`
- [ ] Fusion (`fusion`): `hybrid --evidence-selection fusion`
- [ ] Budget sweep at 32K (`--context-window-tokens 32768 --max-input-tokens 28672`):
      middle truncation (`middle_32k`), position (`position_32k`)
- [ ] Budget sweep at 80K (`--context-window-tokens 81920 --max-input-tokens 77824`):
      middle truncation (`middle_80k`), position (`position_80k`). 128K would
      fit every prompt; 80K keeps 74% on average with none fitting
- [ ] Upgraded recipe at 64K, 32K and 80K (`upgraded`, `upgraded_32k`,
      `upgraded_80k`), once the recipe is fixed: set `AA_LCR_RECIPE`
- [ ] Grade every run above with Qwen3.6 (for the agreement check); the suite
      does this after each run
- [ ] Regrade every run above with GPT-5.6 Luna (final): `AA_LCR_GRADER=luna`

### MRCR v2 (Jarvis)

Suite: `run_suite.sh mrcr`; details in the
[MRCR v2 suite runbook](linux/MRCR_SUITE_RUNBOOK.md). The name in brackets is the
suite's condition name, listed in the suite's order. The first suite takes 30
questions per conversation; a full run (`MRCR_ROWS_PER_SOURCE=0`, all 953
questions) may follow as a separate suite.
`MRCR_DATA_DIR=benchmark_data/mrcr_v2_4needle_100k_1200k`, 30 questions per
conversation (120) over conversations of 133K, 267K, 533K and 1.06M tokens.
Base flags: `--max-output-tokens 4096 --max-rows-per-source 30 --fail-fast --serving-metadata .cache/serving_metadata.json`.
Main budget 262K: `--context-window-tokens 262144 --max-input-tokens 258048`.
Hybrid runs use `--child-tokens 3500 --child-overlap-tokens 350` unless
stated. The 133K conversation fits 262K, so its rows are the full-context
reference. Scoring is deterministic, so there is no grading step.

- [ ] Middle truncation (`middle`): `direct --direct-overflow middle`
- [ ] Dense (`dense`): `hybrid`
- [ ] Position (`position`): `hybrid --evidence-selection position`
- [ ] Random, seed 0 (`random0`): `hybrid --evidence-selection random --selection-seed 0`
- [ ] Marked middle truncation (`middle_marked`): `direct --direct-overflow middle-marked`
- [ ] Head plus question (`head`): `direct --direct-overflow head`
- [ ] BM25 (`bm25`): `hybrid --evidence-selection bm25`
- [ ] Fusion (`fusion`): `hybrid --evidence-selection fusion`
- [ ] Dense with 7,500-token chunks (`dense_7500`): `hybrid --child-tokens 7500 --child-overlap-tokens 750`
- [ ] Dense ranked by the full question (`dense_question`): `hybrid --retrieval-query question`
- [ ] Ladder 1 (`ladder1`): `hybrid --evidence-selection fusion --child-tokens 1024 --child-overlap-tokens 128`
- [ ] Ladder 2 (`ladder2`): ladder 1 `+ --neighbor-children 1`
- [ ] Ladder 3 (`ladder3`): ladder 2 `+ --packing fill`
- [ ] Ladder 4 (`ladder4`): ladder 3 `+ --document-floor`
- [ ] Budget sweep at 32K (`--context-window-tokens 32768 --max-input-tokens 28672`):
      middle truncation (`middle_32k`), position (`position_32k`)
- [ ] Budget sweep at 64K (`--context-window-tokens 65536 --max-input-tokens 61440`):
      middle truncation (`middle_64k`), position (`position_64k`)
- [ ] Budget sweep at 128K (`--context-window-tokens 131072 --max-input-tokens 126976`):
      middle truncation (`middle_128k`), position (`position_128k`)
- [ ] Upgraded recipe at 262K, 32K, 64K and 128K (`upgraded`, `upgraded_32k`,
      `upgraded_64k`, `upgraded_128k`), once the recipe is fixed: set
      `MRCR_RECIPE`
- [ ] Oracle run, once the oracle script exists

### LongBench-v2 (Jarvis)

Suite: `run_suite.sh longbench`; details in the
[Jarvis LongBench-v2 suite page](jarvis_suites/LONGBENCH_V2.md). The name in
brackets is the suite's condition name, listed in the suite's order. Every run
uses the same 221 original rows whose full prompt is over the 64K budget and
within 262K: `--min-source-tokens 65529 --max-source-tokens 262136`. Main budget
64K: `--context-window-tokens 65536 --max-input-tokens 65528` (8 output tokens,
the launcher default). Hybrid runs use the default 7,500-token chunks with 750
overlap. Scoring is by answer letter, so there is no grading step.

- [ ] Full context, 262K (`full`): `direct --context-window-tokens 262144 --max-input-tokens 262136`
- [ ] Middle truncation (`middle`): `direct --direct-overflow middle`
- [ ] Dense (`dense`): `hybrid`
- [ ] Position (`position`): `hybrid --evidence-selection position`
- [ ] Random, seed 0 (`random0`): `hybrid --evidence-selection random --selection-seed 0`
- [ ] Marked middle truncation (`middle_marked`): `direct --direct-overflow middle-marked`
- [ ] Head plus question (`head`): `direct --direct-overflow head`
- [ ] BM25 (`bm25`): `hybrid --evidence-selection bm25`
- [ ] Fusion (`fusion`): `hybrid --evidence-selection fusion`
- [ ] Ladder 1–4 (`ladder1`–`ladder4`): the same four steps as MRCR
- [ ] Budget sweep at 32K (`--context-window-tokens 32768 --max-input-tokens 32760`):
      middle truncation (`middle_32k`), position (`position_32k`)
- [ ] Budget sweep at 128K (`--context-window-tokens 131072 --max-input-tokens 131064`):
      middle truncation (`middle_128k`), position (`position_128k`)
- [ ] Upgraded recipe at 64K, 32K and 128K (`upgraded`, `upgraded_32k`,
      `upgraded_128k`), once the recipe is fixed: set `LONGBENCH_RECIPE`

### Second model (if time)

- [ ] Choose the model and serve it (`EXECUTOR_MODEL=... bash linux/serve_vllm.sh executor`)
- [ ] Re-prepare MRCR with its tokenizer
- [ ] Run full context, middle truncation, position, dense and the upgraded
      recipe on all three benchmarks
