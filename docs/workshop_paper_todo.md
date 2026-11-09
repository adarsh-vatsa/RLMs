# Workshop paper to-do

Tracks the work for the [workshop paper plan](workshop_paper_plan_20261012.md). In the stage sections a tick means the code for the item exists; items marked *(no code)* are decisions or manual steps. The [experiment list](#experiment-list) at the end tracks execution: every run is executed fresh on the current code, references included. Commands are in the [benchmark runbook](linux/BENCHMARK_RUNBOOK.md#overflow-policy-conditions).

## Remaining work

In order; tick an item when it is done. Writing starts while the oracle is built and run, and the grader check is manual and can happen at any point.

1. [x] **MRCR oracle and visibility (Tables 1 and 2), and the AA-LCR document-floor check.** Ran 2026-11-09, see [ORACLE.md](jarvis_suites/ORACLE.md). MRCR has a full-context reference only for the 133K conversation, so the oracle at each budget is its Table 1 reference.
   - Visibility (`mrcr_v2.visibility`, the last step of the MRCR suite): for each question, whether the prompt held the whole target turn and all needle turns, and exact match given each. Exact match given all needles visible is the analytic oracle. Exact match with the target hidden is 0% in every run except `dense_7500` (3.2%); with the target visible it is 56–86%, highest at the smallest budgets. So accuracy is the share of prompts showing the target times the model's rate on those. The recipe shows the target for every question at 262K and for 48%, 71% and 89% at 32K, 64K and 128K. Some cells rest on few questions (`middle_64k`, exact match given all needles visible: 1 question); show counts or leave such cells blank in the paper.
   - Oracle runs (`mrcr_v2/oracle.py`, `--oracle`): the recipe with the chunks holding the needle turns offered first, at 262K, 32K, 64K and 128K. Only the chunk order uses gold data; the prompt is built as in any hybrid run. Same 120 questions, because the suite selects them with the same flags. Every oracle prompt held the target and all needle turns. Recipe against oracle, exact match: 32K 34.2% vs 75.0% (57 vs 8 discordant, p < 0.001), 64K 40.8% vs 63.3% (31 vs 4, p < 0.001), 128K 56.7% vs 59.2% (10 vs 7, p = 0.63), 262K 61.7% vs 61.7% (3 vs 3). The recipe reaches the oracle from 128K; below that it loses by not finding the target. With the needles guaranteed, less context is better: the oracle at 32K beats the oracle at 262K, 90 vs 74 correct (24 vs 8, p = 0.007). On the 133K conversation alone, the oracle at 32K against full context is 21 vs 18 of 30 (p = 0.51), so the paper cannot claim that a focused prompt beats full context.
   - AA-LCR recipe without `--document-floor` (`ladder3`, `ladder3_32k`, `ladder3_80k`), graded by Qwen3.6 and Luna. Luna, without against with the floor: 32K 56 vs 56, 64K 63 vs 58 (12 vs 7, p = 0.36), 80K 62 vs 64 (3 vs 5, p = 0.73), within AA-LCR's run-to-run noise (see the [experiment list](#experiment-list)). The floor raises documents covered (78% to 98% at 32K, 95% to 100% at 64K, 97% to 100% at 80K) but not accuracy: at 32K, on the 66 questions where `ladder3` missed a document, both score 37. The AA-LCR gain over `fusion` therefore comes from the chunk size, neighbours and fill packing together; AA-LCR has no ladder to separate them. Decision (2026-11-09): keep the recipe fixed on 2026-11-02 and report the floor as a null result. AA-LCR is the held-out check, so switching to `ladder3` would choose the recipe on test data.
   - Qwen pushed commit `345d467` to `Qwen/Qwen3.6-35B-A3B` on 2026-11-07. It changed only `chat_template.jinja` and `tokenizer_config.json`, adding generation tags around assistant turns, and renders the same text for every prompt shape the benchmarks send, so prompts and token counts are unchanged. The AA-LCR `ladder3` runs loaded it and record a different `executor_chat_template_sha256` from the earlier runs. MRCR checks the hash against its prepared data, so its runs and visibility table now load the earlier commit `995ad96` (`mrcr_v2.dataset.DEFAULT_REVISION`).
2. [x] **Paired intervals** for every comparison the paper reports, resampling the unit questions share. Done 2026-11-09: `python -m <aa_lcr|mrcr_v2|long_bench_v2>.intervals <suite>` writes `intervals.md` and `.csv` (AA-LCR: `intervals-gpt-5.6-luna`) in each suite folder; commands on the [Jarvis suite pages](jarvis_suites/README.md). `execution.intervals` draws 10,000 resamples (seed 20261109) and uses the same ones for every comparison in a suite. It compares each run with `middle` at its window, then the pairs in `execution.intervals.PAIRS`: dense against position, the retrieval variants against dense, each ladder step, the recipe against fusion, BM25, `ladder3`, `ladder4` and full context, the oracle against the recipe at each budget, and the oracle at 32K against 262K. Add pairs there if item 4 needs more.
   - AA-LCR: resamples the 30 document sets, from the Luna grades. It agrees with `aa_lcr.compare_conditions`: recipe against middle +18.0 points, [+7.3, +29.9] here and [+7.2, +29.9] there.
   - LongBench-v2: resamples the 194 documents behind the 221 questions.
   - MRCR: exact match only; the MRCR score has no intervals yet. Four conversations are too few to resample, so it resamples questions within each conversation, and the intervals hold for these four conversations only.
3. [x] **Numbers for the plan's claims**, with intervals. No runs; the `gap recovered` columns of the item 2 tables come from the same resamples as the differences. A ratio's interval is left blank when the gap reaches zero in any resample; the difference and its interval then carry the result.
   - X%, the share of what middle truncation loses that a selector recovers: (selector − truncation) / (full context − truncation). AA-LCR (`intervals-gpt-5.6-luna`, against `full`): the recipe recovers 80% [57, 105] at 32K, 72% [39, 104] at 64K and 95% [71, 130] at 80K. LongBench-v2: full 103, middle 90, so the gap is 13 questions and reaches zero in 2–10% of resamples; report points (recipe +7.2 [+1.9, +12.8] against middle at 64K, full +5.9 [+0.9, +11.4]). MRCR (`intervals-full-context`): the 30 questions of the 133K conversation, which `middle` sends whole at 262K; the gap reaches zero in some resamples at every budget, so points only (recipe 36%, 89% and 75% at 32K, 64K and 128K).
   - Y%, the share of the oracle's recovery on MRCR: (selector − truncation) / (oracle − truncation) at the same window (`intervals`). The recipe recovers 100% [83, 119] at 262K, 92% [72, 115] at 128K, 53% [37, 69] at 64K and 40% [28, 52] at 32K; BM25 93%, 59%, 41% and 23%. At 533K and 1.06M tokens the recipe at 262K matches the oracle (70% and 63% exact match against middle's 27% and 10%; per-conversation points from `results.md`).
   - "Where it breaks": budget share, from the sweeps. MRCR: the recipe closes the whole oracle gap at 262K and 92% of it at 128K, but 53% at 64K and 40% at 32K, where it shows the target turn for 71% and 48% of questions. AA-LCR: 72–95% of the full-context gap across 32K–80K, so no break in the swept range. LongBench-v2: the recipe's gain over middle is +5.4 [−1.3, +12.4] at 32K, +7.2 [+1.9, +12.8] at 64K and +3.2 [+0.4, +6.2] at 128K, where middle already keeps 89% of each prompt on average.
   - Evidence spread: dropped 2026-11-09. A preview split found no break: on AA-LCR at 64K the recipe gained +30 points on questions with more than 6 documents and +10 on those with 6 or fewer, with overlapping intervals, and on LongBench-v2 it tracked full context in every domain. A post-hoc subgroup split on 100 and 221 questions would not support a claim either way.
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

Every run below is in suite `20261029`, with run ID `20261029_<condition>` inside `benchmark_artifacts/<benchmark>/suites/20261029/`. The first suites ran 2026-10-30 to 2026-11-02 on commit `02ef820`; the BM25 sweep and the recipe runs ran 2026-11-03 on `0f3257e`, which only adds those conditions to the launchers. Luna regraded AA-LCR on 2026-11-08. On 2026-11-09 the AA-LCR `ladder3` runs ran on `a3eaba3` and the MRCR oracle runs on `4f55d32`, which only pins the MRCR tokenizer revision. Temperature 0 does not make generation repeatable: the recipe runs at the main budget repeat ladder 4 exactly (the same chunks for every question), yet changed 6 of 221 LongBench-v2 answers (+4/−2) and 4 of 120 MRCR answers (+2/−2). AA-LCR, with long free-text answers, is far noisier: on the questions where `ladder3` and `upgraded` built the same prompt (32, 78 and 81 at 32K, 64K and 80K), the answer text matched only 5, 13 and 13 times, and Luna's verdict differed on 6, 15 and 7 (9–19%), never when the text matched. Single-run AA-LCR differences of about 5 points are noise.

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
- [x] Recipe without the document floor at 64K, 32K and 80K (`ladder3`, `ladder3_32k`, `ladder3_80k`), graded by Qwen3.6 and then Luna

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
- [x] Oracle at 262K, 32K, 64K and 128K (`oracle`, `oracle_32k`, `oracle_64k`, `oracle_128k`): the recipe with `--oracle`
- [x] Visibility table (`visibility.md`), written at the end of the suite

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
