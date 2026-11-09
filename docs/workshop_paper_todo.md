# Workshop paper to-do

Tracks the work for the [workshop paper plan](workshop_paper_plan_20261012.md). In the stage sections a tick means the code for the item exists; items marked *(no code)* are decisions or manual steps. The [experiment list](#experiment-list) at the end tracks execution: every run is executed fresh on the current code, references included. Commands are in the [benchmark runbook](linux/BENCHMARK_RUNBOOK.md#overflow-policy-conditions).

## Remaining work

In order; tick an item when it is done. Writing starts while the oracle is built and run, and the grader check is manual and can happen at any point.

1. [ ] **MRCR oracle and visibility (Tables 1 and 2), and the AA-LCR document-floor check.** Code done; runs pending, see [ORACLE.md](jarvis_suites/ORACLE.md). MRCR has a full-context reference only for the 133K conversation, so the oracle is also its Table 1 reference.
   - Visibility (`mrcr_v2.visibility`, the last step of the MRCR suite): for each question, whether the prompt held the whole target turn and all needle turns, and exact match given each. Exact match given all needles visible is the analytic oracle. Hybrid runs alone, computed 2026-11-08: the recipe shows the target for every question at 262K and for 48–89% at 32K–128K; exact match with the target hidden is 0–3%.
   - Oracle runs (`mrcr_v2/oracle.py`, `--oracle`): the recipe with the chunks holding the needle turns offered first, at 262K, 32K, 64K and 128K. Only the chunk order uses gold data; the prompt is built as in any hybrid run. Same 120 questions, because the suite selects them with the same flags.
   - AA-LCR recipe without `--document-floor` (`ladder3`, `ladder3_32k`, `ladder3_80k`), in the same executor job. The recipe includes every document for every question, but it also changes chunk size, neighbours and packing, so without these runs the paper cannot attribute the AA-LCR gain to the document floor. A second Luna regrade grades only these three.
   - Qwen pushed commit `345d467` to `Qwen/Qwen3.6-35B-A3B` on 2026-11-07. It changed only `chat_template.jinja` and `tokenizer_config.json`, adding generation tags around assistant turns, and renders the same text for every prompt shape the benchmarks send, so prompts and token counts are unchanged. The AA-LCR `ladder3` runs loaded it and record a different `executor_chat_template_sha256` from the earlier runs. MRCR checks the hash against its prepared data, so its runs and visibility table now load the earlier commit `995ad96` (`mrcr_v2.dataset.DEFAULT_REVISION`).
2. [ ] **Paired intervals** for every comparison the paper reports, resampling the unit questions share:
   - AA-LCR: `aa_lcr.compare_conditions` already resamples the 30 document sets (random seeds as repeats); check it on the suite runs and Luna grades.
   - LongBench-v2: new; resample the 194 documents behind the 221 questions.
   - MRCR: new; four conversations are too few to resample, so resample questions within each conversation.
3. [ ] **Numbers for the plan's claims**, with intervals. No runs; build it with item 2, so ratios and differences come from the same resamples:
   - X%, the share of what middle truncation loses that a selector recovers: (selector − truncation) / (full context − truncation). AA-LCR: full 65, middle 40. LongBench-v2: full 103, middle 90, so the denominator is 13 questions and BM25 (108) exceeds 100%; report points there rather than a ratio. MRCR: at 262K the 133K conversation is sent whole by every condition, so X exists only at 32K–128K, on those 30 questions.
   - Y%, the share of the oracle's recovery on MRCR: (selector − truncation) / (oracle − truncation).
   - "Where it breaks": budget share comes from the sweeps. Evidence spread has no analysis yet; for example, AA-LCR accuracy split by whether the selector kept every document. Build it or drop the claim.
4. [ ] **Decide the lead** *(no code, no runs)*: where `position` and `random` land between truncation and dense on each benchmark, and how BM25 and the recipe compare across budgets, judged on the intervals.
5. [ ] **Grader check** *(no code)*: hand-label the grades where Luna and Qwen3.6 disagree (about 21 of 2,000) and about 30 where they agree; report Luna's accuracy.
6. [ ] **Tables, figures and writing.**
   - Choose the AAAI-27 workshop and check its page limit, template and anonymity rules before drafting.
   - Tables 1–3 and Figure 1 from the plan. Table 3 takes GPU embedding times from these runs; CPU times exist only from the 30 September Neselab dense runs (AA-LCR 7,500-token and MRCR 3,500-token chunks), not for LongBench-v2 or the recipe's 1,024-token chunks.
   - Report `ladder4` and the matching recipe runs as two runs of one configuration, with the run-to-run noise in the [experiment list](#experiment-list).
   - State that selectors packed with `stop` used fewer input tokens than truncation at the same budget, because whole chunks did not fit: about 1% at MRCR 262K, 5% at 64K, and 9–15% at 32K.
   - Verify related work in full text before citing, and check arXiv once more for concurrent overflow-policy papers just before submitting.

## Stage 1: structure or selection

Earlier references, superseded by the fresh runs: AA-LCR 30 September runs (middle truncation 37%, dense 54%; `direct/20260930T134555Z`, `hybrid/20260930T025706Z`); MRCR 30 September runs (truncation 38%, dense 51% over 120 questions, 30 per conversation; 3,500-token chunks, 350 overlap; `direct/20260930T192314239638Z`, `hybrid/20260930T172140824150Z`). All on Neselab.

- [x] AA-LCR 64K, `hybrid --evidence-selection position`
- [x] AA-LCR 64K, `hybrid --evidence-selection random --selection-seed 0`
- [x] AA-LCR 64K, `hybrid --evidence-selection random --selection-seed 1`
- [x] AA-LCR 64K, `hybrid --evidence-selection random --selection-seed 2`
- [x] AA-LCR 64K, `direct --direct-overflow middle-marked`
- [x] AA-LCR 64K, `direct --direct-overflow head` (keeps the start of the documents, then the question)
- [x] MRCR 262K, `hybrid --evidence-selection position`
- [x] MRCR 262K, `hybrid --evidence-selection random --selection-seed 0`
- [x] Grade the AA-LCR runs with the executor (Qwen3.6) as well. They use `--execution-only`, so they have no grades until then (`aa_lcr.regrade` with the executor as grader). These grades are the comparison point for the Luna agreement check (see Before submitting)

## Stage 2: lexical retrieval and the upgraded recipe

- [x] AA-LCR, `--evidence-selection bm25`
- [x] AA-LCR, `--evidence-selection fusion`
- [x] MRCR, `--evidence-selection bm25`
- [x] MRCR, `--evidence-selection fusion`
- [x] MRCR chunk size: 7,500-token chunks, 750 overlap, request query, all 120 questions (so far run on only 40)
- [x] MRCR full-question query, `--retrieval-query question`
- [x] LongBench-v2 shared-pipeline baselines (truncation and dense), needed before tuning on LongBench-v2
- [x] Upgrade ladder on MRCR, starting from `fusion`, one change per run:
  - [x] `--child-tokens 1024 --child-overlap-tokens 128`
  - [x] `+ --neighbor-children 1`
  - [x] `+ --packing fill`
  - [x] `+ --document-floor`
- [x] Upgrade ladder on LongBench-v2, same steps
- [x] Fix one recipe for all three benchmarks; record it here *(no code)*. Recipe, fixed 2026-11-02 before the BM25 sweep and the AA-LCR recipe runs: ladder 4, `--evidence-selection fusion --child-tokens 1024 --child-overlap-tokens 128 --neighbor-children 1 --packing fill --document-floor`. This is the final ladder step, as the plan specified. No step was significant on MRCR or LongBench-v2, ladders 3 and 4 tied (`--document-floor` has no effect on one document), and `bm25` tied them
- [x] AA-LCR held-out check with that recipe, run once

Tune only on MRCR and LongBench-v2. Report every ladder step, including those that did not help.

## Before submitting

- [x] Independent AA-LCR grader, run once after all AA-LCR runs are done: regrade every run with GPT-5.6 Luna at medium reasoning, the official AA-LCR equality checker, through OpenRouter. `aa_lcr.regrade` flags: `--evaluator-model openai/gpt-5.6-luna` `--evaluator-base-url https://openrouter.ai/api/v1` `--grader-api-style openai --grader-reasoning-effort medium` `--evaluator-api-key-env OPENROUTER_API_KEY`. The key is in the git-ignored `.env`, which `aa_lcr.regrade` does not read; export it first with `set -a; source .env; set +a`
  - [x] Confirm OpenRouter passes the reasoning effort through. Checked 2026-11-08 on a counting puzzle: `reasoning_effort` `none` gave 0 reasoning tokens and a wrong answer, `low`/`medium`/`high` gave 85-127 and the right one. Luna reasons adaptively: AA-LCR question 1's grading prompt used 0 reasoning tokens even at `high`, so a grade with no reasoning tokens does not mean the effort was dropped. Regrades save each row's `grader_usage`
  - [x] Report agreement with the Qwen3.6 grades: the Luna results table's `agreement with Qwen3.6-35B-A3B` column
- [x] AA-LCR results table per suite (`aa_lcr.results_table`), regenerated as runs finish. For the paper, use the Luna grades and add paired intervals against middle truncation from `aa_lcr.compare_conditions`
- [x] MRCR results table per suite (`mrcr_v2.results_table`), with paired gained and lost counts and an exact McNemar p-value against middle truncation, and exact match per conversation length
- [x] LongBench-v2 results table per suite (`long_bench_v2.results_table`), with the same paired comparison against middle truncation
- [x] LongBench-v2 on the shared pipeline, with a full-context reference
- [x] Budget sweep: the window flags cover it; see the experiment list for the AA-LCR budgets
- [x] GPU embedding timing for Table 3: records keep `document_embedding_ms`
- [x] Check arXiv for concurrent overflow-policy papers *(no code)*

## Experiment list

Tick a run when it completes and record its run ID. All three benchmarks run on Jarvis, one suite at a time, with `bash adarsh-rlms/jarvis/run_suite.sh <aa_lcr|mrcr|longbench> <suite-id>`; see the [Jarvis suite pages](jarvis_suites/README.md). Executor: Qwen3.6-35B-A3B on vLLM (4 L40S, `--max-num-seqs 1`), temperature 0, non-thinking. Hybrid runs embed on the client job's GPU. Table 3's GPU embedding times come from these runs and its CPU times from the 30 September Neselab runs; the embedding code has not changed since. Order: Stage 1, Stage 2 and the ladders, then the recipe-dependent runs (AA-LCR held-out check, budget sweeps).

Every run below is in suite `20261029`, with run ID `20261029_<condition>` inside `benchmark_artifacts/<benchmark>/suites/20261029/`. The first suites ran 2026-10-30 to 2026-11-02 on commit `02ef820`; the BM25 sweep and the recipe runs ran 2026-11-03 on `0f3257e`, which only adds those conditions to the launchers. Luna regraded AA-LCR on 2026-11-08. Temperature 0 does not make generation repeatable: the recipe runs at the main budget repeat ladder 4 exactly (the same chunks for every question), yet changed 6 of 221 LongBench-v2 answers (+4/−2) and 4 of 120 MRCR answers (+2/−2).

### AA-LCR (Jarvis)

Suite: `run_suite.sh aa_lcr`; details in the [AA-LCR suite runbook](linux/AA_LCR_SUITE_RUNBOOK.md). The name in brackets is the suite's condition name. All 100 questions; every prompt is 88K–122K tokens. Base flags: `--execution-only --max-output-tokens 4096 --fail-fast --serving-metadata .cache/serving_metadata.json`. Main budget 64K: `--context-window-tokens 65536 --max-input-tokens 61440`. Hybrid runs use the default 7,500-token chunks with 750 overlap.

- [x] Full context, 262K (`full`): `direct --context-window-tokens 262144 --max-input-tokens 258048`
- [x] Middle truncation (`middle`): `direct --direct-overflow middle`
- [x] Marked middle truncation (`middle_marked`): `direct --direct-overflow middle-marked`
- [x] Head plus question (`head`): `direct --direct-overflow head`
- [x] Dense (`dense`): `hybrid`
- [x] Position (`position`): `hybrid --evidence-selection position`
- [x] Random, seed 0 (`random0`): `hybrid --evidence-selection random --selection-seed 0`
- [x] Random, seed 1 (`random1`): `hybrid --evidence-selection random --selection-seed 1`
- [x] Random, seed 2 (`random2`): `hybrid --evidence-selection random --selection-seed 2`
- [x] BM25 (`bm25`): `hybrid --evidence-selection bm25`
- [x] Fusion (`fusion`): `hybrid --evidence-selection fusion`
- [x] Budget sweep at 32K (`--context-window-tokens 32768 --max-input-tokens 28672`): middle truncation (`middle_32k`), position (`position_32k`)
- [x] Budget sweep at 80K (`--context-window-tokens 81920 --max-input-tokens 77824`): middle truncation (`middle_80k`), position (`position_80k`). 128K would fit every prompt; 80K keeps 74% on average with none fitting
- [x] BM25 across budgets (`bm25_32k`, `bm25_80k`): BM25 ties the upgraded ladder at the main budget, so it needs the same sweep
- [x] Upgraded recipe at 64K, 32K and 80K (`upgraded`, `upgraded_32k`, `upgraded_80k`), once the recipe is fixed: set `AA_LCR_RECIPE`
- [x] Grade every run above with Qwen3.6 (for the agreement check); the suite does this after each run
- [x] Regrade every run above with GPT-5.6 Luna (final): `AA_LCR_GRADER=luna`
- [ ] Recipe without the document floor at 64K, 32K and 80K (`ladder3`, `ladder3_32k`, `ladder3_80k`), graded by Qwen3.6 and then Luna

### MRCR v2 (Jarvis)

Suite: `run_suite.sh mrcr`; details in the [MRCR v2 suite runbook](linux/MRCR_SUITE_RUNBOOK.md). The name in brackets is the suite's condition name, listed in the suite's order. The first suite takes 30 questions per conversation; a full run (`MRCR_ROWS_PER_SOURCE=0`, all 953 questions) may follow as a separate suite. `MRCR_DATA_DIR=benchmark_data/mrcr_v2_4needle_100k_1200k`, 30 questions per conversation (120) over conversations of 133K, 267K, 533K and 1.06M tokens. Base flags: `--max-output-tokens 4096 --max-rows-per-source 30 --fail-fast --serving-metadata .cache/serving_metadata.json`. Main budget 262K: `--context-window-tokens 262144 --max-input-tokens 258048`. Hybrid runs use `--child-tokens 3500 --child-overlap-tokens 350` unless stated. The 133K conversation fits 262K, so its rows are the full-context reference. Scoring is deterministic, so there is no grading step.

- [x] Middle truncation (`middle`): `direct --direct-overflow middle`
- [x] Dense (`dense`): `hybrid`
- [x] Position (`position`): `hybrid --evidence-selection position`
- [x] Random, seed 0 (`random0`): `hybrid --evidence-selection random --selection-seed 0`
- [x] Marked middle truncation (`middle_marked`): `direct --direct-overflow middle-marked`
- [x] Head plus question (`head`): `direct --direct-overflow head`
- [x] BM25 (`bm25`): `hybrid --evidence-selection bm25`
- [x] Fusion (`fusion`): `hybrid --evidence-selection fusion`
- [x] Dense with 7,500-token chunks (`dense_7500`): `hybrid --child-tokens 7500 --child-overlap-tokens 750`
- [x] Dense ranked by the full question (`dense_question`): `hybrid --retrieval-query question`
- [x] Ladder 1 (`ladder1`): `hybrid --evidence-selection fusion --child-tokens 1024 --child-overlap-tokens 128`
- [x] Ladder 2 (`ladder2`): ladder 1 `+ --neighbor-children 1`
- [x] Ladder 3 (`ladder3`): ladder 2 `+ --packing fill`
- [x] Ladder 4 (`ladder4`): ladder 3 `+ --document-floor`
- [x] Budget sweep at 32K (`--context-window-tokens 32768 --max-input-tokens 28672`): middle truncation (`middle_32k`), position (`position_32k`)
- [x] Budget sweep at 64K (`--context-window-tokens 65536 --max-input-tokens 61440`): middle truncation (`middle_64k`), position (`position_64k`)
- [x] Budget sweep at 128K (`--context-window-tokens 131072 --max-input-tokens 126976`): middle truncation (`middle_128k`), position (`position_128k`)
- [x] BM25 across budgets (`bm25_32k`, `bm25_64k`, `bm25_128k`)
- [x] Upgraded recipe at 262K, 32K, 64K and 128K (`upgraded`, `upgraded_32k`, `upgraded_64k`, `upgraded_128k`), once the recipe is fixed: set `MRCR_RECIPE`
- [ ] Oracle at 262K, 32K, 64K and 128K (`oracle`, `oracle_32k`, `oracle_64k`, `oracle_128k`): the recipe with `--oracle`
- [ ] Visibility table (`visibility.md`), written at the end of the suite

### LongBench-v2 (Jarvis)

Suite: `run_suite.sh longbench`; details in the [Jarvis LongBench-v2 suite page](jarvis_suites/LONGBENCH_V2.md). The name in brackets is the suite's condition name, listed in the suite's order. Every run uses the same 221 original rows whose full prompt is over the 64K budget and within 262K: `--min-source-tokens 65529 --max-source-tokens 262136`. Main budget 64K: `--context-window-tokens 65536 --max-input-tokens 65528` (8 output tokens, the launcher default). Hybrid runs use the default 7,500-token chunks with 750 overlap. Scoring is by answer letter, so there is no grading step.

- [x] Full context, 262K (`full`): `direct --context-window-tokens 262144 --max-input-tokens 262136`
- [x] Middle truncation (`middle`): `direct --direct-overflow middle`
- [x] Dense (`dense`): `hybrid`
- [x] Position (`position`): `hybrid --evidence-selection position`
- [x] Random, seed 0 (`random0`): `hybrid --evidence-selection random --selection-seed 0`
- [x] Marked middle truncation (`middle_marked`): `direct --direct-overflow middle-marked`
- [x] Head plus question (`head`): `direct --direct-overflow head`
- [x] BM25 (`bm25`): `hybrid --evidence-selection bm25`
- [x] Fusion (`fusion`): `hybrid --evidence-selection fusion`
- [x] Ladder 1–4 (`ladder1`–`ladder4`): the same four steps as MRCR
- [x] Budget sweep at 32K (`--context-window-tokens 32768 --max-input-tokens 32760`): middle truncation (`middle_32k`), position (`position_32k`)
- [x] Budget sweep at 128K (`--context-window-tokens 131072 --max-input-tokens 131064`): middle truncation (`middle_128k`), position (`position_128k`)
- [x] BM25 across budgets (`bm25_32k`, `bm25_128k`)
- [x] Upgraded recipe at 64K, 32K and 128K (`upgraded`, `upgraded_32k`, `upgraded_128k`), once the recipe is fixed: set `LONGBENCH_RECIPE`

### Second model (if time)

The runners take any served model, but none has been chosen or tested outside Qwen.

- [ ] Choose the model and serve it (`EXECUTOR_MODEL=... bash linux/serve_vllm.sh executor`)
- [ ] Re-prepare MRCR with its tokenizer (`mrcr_v2.prepare_dataset --executor-model`)
- [ ] Run full context, middle truncation, position, dense and the upgraded recipe on all three benchmarks
