# Workshop paper to-do

Tracks the work for the [workshop paper plan](workshop_paper_plan_20261012.md). In the stage sections a tick means the code for the item exists; items marked *(no code)* are decisions or manual steps. The [experiment list](#experiment-list) at the end tracks execution: every run is executed fresh on the current code, references included. Commands are in the [benchmark runbook](linux/BENCHMARK_RUNBOOK.md#overflow-policy-conditions).

## Remaining work

In order; tick an item when it is done. Open: item 5, reopened by the 12 November rerun (19 new grader disagreements to label), and item 6.

1. [x] **MRCR oracle and visibility (Tables 1 and 2), and the AA-LCR document-floor check.** Ran 2026-11-09, see [ORACLE.md](jarvis_suites/ORACLE.md). MRCR has a full-context reference only for the 133K conversation, so the oracle at each budget is its Table 1 reference.
   - Visibility (`mrcr_v2.visibility`, the last step of the MRCR suite): for each question, whether the prompt held the whole target turn and all needle turns, and exact match given each. Exact match given all needles visible is the analytic oracle. Exact match with the target hidden is 0% in every run except `dense_7500` (3.2%); with the target visible it is 56–86%, highest at the smallest budgets. So accuracy is the share of prompts showing the target times the model's rate on those. The recipe shows the target for every question at 262K and for 48%, 71% and 89% at 32K, 64K and 128K. Some cells rest on few questions (`middle_64k`, exact match given all needles visible: 1 question); show counts or leave such cells blank in the paper.
   - Oracle runs (`mrcr_v2/oracle.py`, `--oracle`): the recipe with the chunks holding the needle turns offered first, at 262K, 32K, 64K and 128K. Only the chunk order uses gold data; the prompt is built as in any hybrid run. Same 120 questions, because the suite selects them with the same flags. Every oracle prompt held the target and all needle turns. Recipe against oracle, exact match: 32K 34.2% vs 75.0% (57 vs 8 discordant, p < 0.001), 64K 40.8% vs 63.3% (31 vs 4, p < 0.001), 128K 56.7% vs 59.2% (10 vs 7, p = 0.63), 262K 61.7% vs 61.7% (3 vs 3). The recipe reaches the oracle from 128K; below that it loses by not finding the target. With the needles guaranteed, less context is better: the oracle at 32K beats the oracle at 262K, 90 vs 74 correct (24 vs 8, p = 0.007). On the 133K conversation alone, the oracle at 32K against full context is 21 vs 18 of 30 (p = 0.51), so the paper cannot claim that a focused prompt beats full context.
   - AA-LCR recipe without `--document-floor` (`ladder3`, `ladder3_32k`, `ladder3_80k`), graded by Qwen3.6 and Luna. Luna, without against with the floor: 32K 56 vs 56, 64K 63 vs 58 (12 vs 7, p = 0.36), 80K 62 vs 64 (3 vs 5, p = 0.73), within AA-LCR's noise between serving nodes (see the [experiment list](#experiment-list)). The floor raises documents covered (78% to 98% at 32K, 95% to 100% at 64K, 97% to 100% at 80K) but not accuracy: at 32K, on the 66 questions where `ladder3` missed a document, both score 37. The AA-LCR gain over `fusion` therefore comes from the chunk size, neighbours and fill packing together; AA-LCR has no ladder to separate them. Decision (2026-11-09): keep the recipe fixed on 2026-11-02 and report the floor as a null result. AA-LCR is the held-out check, so switching to `ladder3` would choose the recipe on test data.
   - Qwen pushed commit `345d467` to `Qwen/Qwen3.6-35B-A3B` on 2026-11-07. It changed only `chat_template.jinja` and `tokenizer_config.json`, adding generation tags around assistant turns, and renders the same text for every prompt shape the benchmarks send, so prompts and token counts are unchanged. The AA-LCR `ladder3` runs loaded it and record a different `executor_chat_template_sha256` from the earlier runs. MRCR checks the hash against its prepared data, so its runs and visibility table now load the earlier commit `995ad96` (`mrcr_v2.dataset.DEFAULT_REVISION`).
2. [x] **Paired intervals** for every comparison the paper reports, resampling the unit questions share. Done 2026-11-09: `python -m <aa_lcr|mrcr_v2|long_bench_v2>.intervals <suite>` writes `intervals.md` and `.csv` (AA-LCR: `intervals-gpt-5.6-luna`) in each suite folder; commands on the [Jarvis suite pages](jarvis_suites/README.md). `execution.intervals` draws 10,000 resamples (seed 20261109) and uses the same ones for every comparison in a suite. It compares each run with `middle` at its window, then the pairs in `execution.intervals.PAIRS`: dense against position, BM25 and the recipe against position at each budget, the retrieval variants against dense, each ladder step, the recipe against fusion, BM25, `ladder3`, `ladder4` and full context, the oracle against the recipe at each budget, and the oracle at 32K against 262K. Add pairs there if item 4 needs more.
   - AA-LCR: resamples the 30 document sets, from the Luna grades. It agrees with `aa_lcr.compare_conditions`: recipe against middle +18.0 points, [+7.3, +29.9] here and [+7.2, +29.9] there.
   - LongBench-v2: resamples the 194 documents behind the 221 questions.
   - MRCR: exact match only; the MRCR score has no intervals yet. Four conversations are too few to resample, so it resamples questions within each conversation, and the intervals hold for these four conversations only.
3. [x] **Numbers for the plan's claims**, with intervals. No runs; the `gap recovered` columns of the item 2 tables come from the same resamples as the differences. A ratio's interval is left blank when the gap reaches zero in any resample; the difference and its interval then carry the result.
   - X%, the share of what middle truncation loses that a selector recovers: (selector − truncation) / (full context − truncation). AA-LCR (`intervals-gpt-5.6-luna`, against `full`): the recipe recovers 80% [57, 105] at 32K, 72% [39, 104] at 64K and 95% [71, 130] at 80K. LongBench-v2: full 103, middle 90, so the gap is 13 questions and reaches zero in 2–10% of resamples; report points (recipe +7.2 [+1.9, +12.8] against middle at 64K, full +5.9 [+0.9, +11.4]). MRCR (`intervals-full-context`): the 30 questions of the 133K conversation, which `middle` sends whole at 262K; the gap reaches zero in some resamples at every budget, so points only (recipe 36%, 89% and 75% at 32K, 64K and 128K).
   - Y%, the share of the oracle's recovery on MRCR: (selector − truncation) / (oracle − truncation) at the same window (`intervals`). The recipe recovers 100% [83, 119] at 262K, 92% [72, 115] at 128K, 53% [37, 69] at 64K and 40% [28, 52] at 32K; BM25 93%, 59%, 41% and 23%. At 533K and 1.06M tokens the recipe at 262K matches the oracle (70% and 63% exact match against middle's 27% and 10%; per-conversation points from `results.md`).
   - "Where it breaks": budget share, from the sweeps. MRCR: the recipe closes the whole oracle gap at 262K and 92% of it at 128K, but 53% at 64K and 40% at 32K, where it shows the target turn for 71% and 48% of questions. AA-LCR: 72–95% of the full-context gap across 32K–80K, so no break in the swept range. LongBench-v2: the recipe's gain over middle is +5.4 [−1.3, +12.4] at 32K, +7.2 [+1.9, +12.8] at 64K and +3.2 [+0.4, +6.2] at 128K, where middle already keeps 89% of each prompt on average.
   - Evidence spread: dropped 2026-11-09. A preview split found no break: on AA-LCR at 64K the recipe gained +30 points on questions with more than 6 documents and +10 on those with 6 or fewer, with overlapping intervals, and on LongBench-v2 it tracked full context in every domain. A post-hoc subgroup split on 100 and 221 questions would not support a claim either way.
4. [x] **Decide the lead** *(no code)*: where `position` and `random` land between truncation and dense on each benchmark, and how BM25 and the recipe compare across budgets, judged on the intervals. First decided 2026-11-09, reopened 2026-11-10 and decided 2026-11-12 on the g101 rerun and the controls. The first bullet is the decision and what Results now says; the three after it are the 9–10 November position, with g202 numbers, kept for the record.
   - **Decision, 2026-11-12.** Lead: ranking by the question adds to structure, not "selection, not structure". Main method: the fusion recipe, because it was fixed on 2026-11-02 with AA-LCR held out. `bm25_recipe` is reported beside it as the variant with no embedding model. The evidence (all on g101 or g102, AA-LCR by Luna):
     - AA-LCR accuracy at 32K / 64K / 80K: `middle` 21 / 40 / 45, `position` 27 / 46 / 51, `position_recipe` 42 / 52 / 53, `bm25` 36 / 54 / 52, `bm25_recipe` 49 / 53 / 64, `upgraded` 58 / 64 / 62, `full` 65. The recipe recovers 84% [65, 105], 96% [68, 141] and 85% [53, 125] of the full-context gap.
     - Structure matters on AA-LCR once the steps are added: `position_recipe` against `middle` is +21.0 [+8.4, +32.7] at 32K, +12.0 [0.0, +22.7] at 64K and +8.0 [−5.2, +19.2] at 80K. It leaves out a document for 6, 0 and 0 questions, where `position` does for 86, 62 and 34.
     - The question still adds: `upgraded` against `position_recipe` is +16.0 [+7.2, +24.2], +12.0 [+2.9, +23.0] and +9.0 [−0.8, +20.8]. `bm25_recipe` against `position_recipe` is clear only at 80K (+11.0 [+3.4, +20.0]).
     - The steps help at small windows with the ranking fixed: `bm25_recipe` against `bm25` is +20.8, +18.3 and +10.0 on MRCR at 32K, 64K and 128K (all clear) and +13.0, −1.0 and +12.0 on AA-LCR (clear at 32K and 80K).
     - Fusion against BM25 with equal steps is split. AA-LCR, `upgraded` against `bm25_recipe`: +9.0 [−3.2, +21.2], +11.0 [+5.0, +17.8], −2.0 [−6.7, +3.5]. MRCR: −9.2 [−20.0, +1.7], −11.7 [−20.8, −2.5], −1.7 [−9.2, +5.8]; `bm25_recipe` shows the target for 65%, 86% and 98% of questions against the recipe's 48%, 71% and 89%. MRCR's query favours lexical ranking.
     - What the rerun changed: AA-LCR `upgraded` against `fusion` is now clear (+11.0 [+1.8, +21.1]). On LongBench-v2 the recipe against `middle` is clear at 32K and 64K but not at 128K (+1.8 [−0.5, +4.5]), and the recipe is not clearly better than `position` at any window.
   - Lead, holding: selection, not structure. `position` builds the hybrid prompt without the question, from the same chunks and with the same packing as `bm25` and `dense`. It stays within 6 points of `middle` at every window, never with a clear interval (AA-LCR 64K +6.0 [−4.6, +16.5], MRCR 262K +0.0 [−2.5, +2.5], LongBench-v2 64K +1.8 [−1.8, +5.6]). `random` is no better than `middle`, and `middle_marked` and `head` change nothing. The evidence is the direct contrasts with `position` at the main window: on MRCR, BM25 +21.7 [+13.3, +30.0] and dense +16.7 [+9.2, +24.2]; on LongBench-v2, BM25 +6.3 [+1.3, +11.6] but dense +3.2 [−3.0, +9.5]; on AA-LCR, dense +10.0 [+2.0, +19.3] but BM25 +8.0 [−2.4, +18.4]. BM25 does not separate from `position` on AA-LCR at any budget (32K +9.0 [−3.4, +19.8], 80K +1.0 [−10.5, +13.1]).
   - Main method, provisional: the recipe. Against `position` at the same budget it is clear on AA-LCR (32K +29.0 [+16.8, +41.5], 64K +12.0 [+1.0, +23.8], 80K +13.0 [+5.0, +22.6]) and on MRCR (+23.3 to +30.0 at every budget), but on LongBench-v2 only at 128K (+2.7 [+0.4, +5.3]; 64K +5.4 [0.0, +10.7], 32K +3.2 [−4.2, +10.6]). The margin mixes ranking with chunk size, fill packing and document coverage, which the `position_recipe` control separates.
   - Headline, open: the recipe alone, or shared with BM25. BM25 gets most of the gain. Dense is no better anywhere (BM25 against dense: AA-LCR −2.0 [−13.8, +7.9], MRCR +5.0 [−3.3, +13.3], LongBench-v2 +3.2 [−1.9, +8.3]), and BM25 needs no embedding. At the main window the recipe adds nothing clear to fusion (AA-LCR +5.0 [−5.4, +16.1], MRCR +5.8 [−0.8, +12.5], LongBench-v2 −0.5 [−3.7, +2.7]). Its margin over BM25 is clear only at tight budgets on AA-LCR (32K +20 [+9.9, +32.3], 80K +12 [+6.3, +18.3]) and MRCR (32K +10.8 [+2.5, +19.2], 128K +10.8 [+4.2, +18.3]), where it also packs more tokens. The `bm25_recipe` control decides this. The plan already has a rule for this case (Risks: if the upgrades do not help, report that the simplest selector is enough), and it favours BM25 unless the control shows otherwise.
5. [ ] **Grader check** *(no code)*: hand-label the grades where Luna and Qwen3.6 disagree and about 30 where they agree; report Luna's accuracy. Reopened 2026-11-12, see the last bullet. Over the first 23 runs they disagree on 27 of 2,300 answers: Luna says correct and Qwen3.6 incorrect on 21, the reverse on 6. The labeling sheet is `benchmark_artifacts/aa_lcr/suites/20261029/grader-check.csv`: all 27 disagreements and 30 agreements drawn at random (seed 20261109), shuffled together, labeled in `my_label` without seeing the last three columns (`luna`, `qwen`, `group`). Luna's accuracy is (27 × its accuracy on the disagreements + 2,273 × its accuracy on the agreements) / 2,300.
   - Labeled 2026-11-09. Luna is right on 21 of the 27 disagreements and Qwen3.6 on 6; both are right on all 30 agreements. Estimated accuracy over the 2,300 answers: Luna 99.7% [88.3, 99.7], Qwen3.6 99.1% [87.7, 99.1]. The disagreements are labeled in full, so the interval comes only from the agreements (Clopper–Pearson, 0 errors in 30); more labeled agreements would raise its low end. Luna's 6 errors, 4 lenient and 2 strict, fall in five runs and change no run's accuracy by more than one point. Two of the lenient ones are answers cut off at the 4,096-token cap.
   - Model answers include explanations: the official prompt asks for no answer format and thinking is off, so Qwen3.6 explains and sometimes works the problem in the answer. The grader judges only the value an answer finally commits to. 113 of the 2,300 answers (4.9%) hit the 4,096-token cap; Luna graded 109 of them incorrect.
   - Question 2's official answer names 2 cases for a question that asks for 3.
   - [ ] After the rerun the suite has 29 runs, and the graders disagree on 40 of 2,900 answers: 21 in the 18 earlier runs, already labelled (Luna right on 15), and 19 in the 11 runs of 12 November (5 in `upgraded`, Luna 64 and Qwen3.6 61). Nine rows of `grader-check.csv` are from g202 runs that are no longer reported (6 disagreements, 3 agreements), which leaves 27 labelled agreements. Label `grader-check-rerun.csv` in the same folder: the 19 new disagreements and 16 agreements drawn from the 11 new runs (seed 20261112), shuffled, same columns and same blind protocol. Then recompute Luna's accuracy over the 2,900 answers and fill the review note in the Grading paragraph and the bound in Limitations.
   - Luna is not repeatable. Of 263 answer texts that recur in more than one run, it graded 3 differently between runs, all between `ladder3` and `upgraded`: questions 20 and 25 at 64K and question 11 at 32K. Qwen3.6 graded all 263 the same. On question 25 in `upgraded` the answer is "No", the reference is "Yes", and Luna marked it correct. All three are in the new sheet.
6. [ ] **Tables, figures and writing.**
   - Workshop: [FrontierIR](https://frontier-ir-workshop.github.io/), the AAAI-27 Workshop on New Frontiers in Information Retrieval. Submit on OpenReview by 20 November 2026 AoE. Long papers 4–7 pages, short papers up to 4, both excluding references and appendices; AAAI-27 author kit; double-blind; non-archival. The paper is `aaai-2027/main.tex`; the kit's examples are in `aaai-2027/author-kit/`.
   - State on 2026-11-12: Benchmarks, Methodology, Experiments, Results and Limitations are drafted. Experiments, Results, Limitations, the tables and the figures carry the g101 rerun and the controls; Benchmarks and Methodology are being revised and do not yet define the position and BM25 recipes, which Experiments introduces. Open questions on the drafted text are the red `\review{}` notes in `main.tex`, and are not repeated here.
   - [ ] Draft Related Work now, and the Abstract, Introduction and Conclusion once item 4 is decided. `references.bib` is empty: every citation is still a `\review{cite …}` note. Three rows of the plan's Claims table no longer hold as written, so do not carry them over:
     - "Recovers X% … on three task types": X% exists only for AA-LCR (72–95%). LongBench-v2 is reported in points, and MRCR against the oracle.
     - "Position and random controls rule out structure and coverage": on AA-LCR only dense separates from `position` directly, and the recipe has no matched control until `position_recipe` runs.
     - "Where it breaks: budget share and evidence spread": evidence spread was dropped on 2026-11-09.
   - [ ] Build the paper with pdfTeX. Check the page count against the 7-page limit and the width of the full-width benchmarks table. On 2026-11-12 a test build with XeTeX (the style's pdfTeX check disabled in a copy) compiled without errors and ran to 7 pages with the Abstract, Introduction, Related Work and Conclusion still placeholders and no references, so the drafted sections need cutting.
   - [ ] Cut Limitations, now 364 words, to fit.
   - Tables 1–3 and Figure 1 from the plan, and Figure 2: MRCR exact match by conversation length at 262K. `uv run aaai-2027/figures/figures.py` draws the figures and `uv run aaai-2027/tables/tables.py` writes the tabulars, both from the suite results; rerun them after any regrade or rerun. Table 1's intervals come from the `intervals*.md` tables, because the CSVs round to 4 decimals and some bounds then round the wrong way. Table 3 takes GPU embedding times from these runs; CPU times exist only from the 30 September Neselab dense runs (AA-LCR 7,500-token and MRCR 3,500-token chunks), not for LongBench-v2 or the recipe's 1,024-token chunks. The benchmarks table (`tab:benchmarks`) is written by hand in `main.tex` from dataset facts checked against the suite results. It comes first, so the paper numbers the `tables.py` tables 2–4; "Table 1–3" in this file still means `table1.tex`–`table3.tex`.
   - [ ] Verify related work in full text before citing, including the plan's statement that Distractor-Aware Truncation stopped at 180K tokens, and check arXiv once more for concurrent overflow-policy papers just before submitting.
   - **Two control runs, which item 4 waits for.** The recipe differs from `position` and `bm25` in chunk size, neighbours, fill packing and the document floor as well as in ranking, so its margins over them do not isolate the ranking. Both controls take the recipe's flags (`--child-tokens 1024 --child-overlap-tokens 128 --neighbor-children 1 --packing fill --document-floor`) and change only `--evidence-selection`. `main.tex` marks where each result goes with a review note. What they are: [CONTROLS.md](jarvis_suites/CONTROLS.md); runbook: [RERUN.md](jarvis_suites/RERUN.md).
     - [x] Code, 2026-11-10: the conditions in `linux/run_aa_lcr_suite.sh` and `linux/run_mrcr_suite.sh`; their pairs in `execution.intervals.PAIRS` (each control against its unmatched version, `bm25_recipe` and `upgraded` against `position_recipe`, `upgraded` against `bm25_recipe`); and `EXECUTOR_NODE` in `jarvis/run.sh`, which pins the executor to a node. Dry runs on suite `20261029` list only the 9 new runs.
     - [x] Serving node, decided 2026-11-10 before the rerun: g101. g202 answers identical prompts differently from g101 and g102 (see the [experiment list](#experiment-list)), so the 17 g202 runs (5 AA-LCR, 7 MRCR, 5 LongBench-v2) are rerun on g101 together with the controls, and the g101 runs replace them in the paper whatever their results. The g202 runs move to the sibling suite folders `20261029_g202` and are not reported. [RERUN.md](jarvis_suites/RERUN.md) does it in one command (about 10 hours).
     - [x] Run [RERUN.md](jarvis_suites/RERUN.md). Ran 2026-11-12 on g101, in one executor session, with no API errors or retries. Every rerun built the same prompt as its g202 run for all 2,445 questions. `upgraded` against `ladder4` shows 0 gained and 0 lost on MRCR and LongBench-v2, with identical answer text. On AA-LCR, `upgraded` and `ladder3` give identical text on all 191 questions for which they built the same prompt (32, 78 and 81 at 32K, 64K and 80K), so AA-LCR's repeatability on a node is now measured.
     - [x] Afterwards, in `main.tex`, 2026-11-12: Statistics says two nodes served every reported run and gives the three repeat checks, with one sentence that runs first served by a third node were rerun; Limitations no longer mentions nodes. The document-floor sentence in Conditions gives the g101 scores (56, 63, 62 against 58, 64, 62; no difference clear).
     - [x] `position` with the recipe's steps (`position_recipe`), on AA-LCR only: the matched structural control. The current `position` uses 7,500-token chunks with `stop` packing and leaves out at least one document on 86, 62 and 34 of the 100 AA-LCR questions at 32K, 64K and 80K. If the control stays near `middle`, "selection, not structure" holds for the recipe too. If it closes much of the gap, coverage matters on AA-LCR and the lead needs weakening. There is no MRCR control: on one conversation, position keeps the head and tail, which is middle truncation with an omission marker. MRCR's `position` is already within 2 points of `middle` at every window, and `middle_marked` matches it exactly.
     - [x] `bm25` with the recipe's steps (`bm25_recipe`), on AA-LCR and MRCR: separates fusion from chunking and packing. At 32K `bm25` leaves 15% of the budget unused on AA-LCR and 9% on MRCR, and the recipe almost none. If the control matches the recipe, the headline is BM25 with small chunks and fill packing, and no embedding model. If the recipe stays ahead, fusion earns its place.
     - Windows: AA-LCR 32K, 64K and 80K for both controls; MRCR 32K, 64K and 128K for `bm25_recipe` (not 262K, where the recipe and `bm25` already tie). That is 9 runs: about 2 h on AA-LCR with its grades and 1 h on MRCR, plus the Luna regrade of the 6 AA-LCR runs. LongBench-v2 gets none: the recipe is not clearly better than `bm25` there at any window.
     - [x] Afterwards, 2026-11-12: the rerun recomputed the intervals; the tables and figures are regenerated; item 4 is decided. The controls are in the paper as the "position recipe" and the "BM25 recipe": rows of the main table (AA-LCR only at the main window), dashed lines in Figure 1, and BM25 recipe rows in the visibility table. The matched recipe-against-oracle counts in Results (44 against 39 of 57, 52 against 53 of 85, 67 against 63 of 107) still come from a one-off script.

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

Every run below is in suite `20261029`, with run ID `20261029_<condition>` inside `benchmark_artifacts/<benchmark>/suites/20261029/`. The first suites ran 2026-10-30 to 2026-11-02 on commit `02ef820`; the BM25 sweep and the recipe runs ran 2026-11-03 on `0f3257e`, which only adds those conditions to the launchers. Luna regraded AA-LCR on 2026-11-08. On 2026-11-09 the AA-LCR `ladder3` runs ran on `a3eaba3` and the MRCR oracle runs on `4f55d32`, which only pins the MRCR tokenizer revision. Temperature 0 makes generation repeatable on a node, but not across all nodes (found 2026-11-10). The manifests record the node (`executor_base_url`): g101 served the first AA-LCR and LongBench-v2 suites and the 9 November runs, g102 the first MRCR suite, and g202 every 3 November run, that is the BM25 sweep and all recipe runs (17 runs: 5 AA-LCR, 7 MRCR, 5 LongBench-v2). g101 and g102 agree byte for byte on identical prompts: the 14 MRCR runs at 262K on g102 and the oracle on g101 give the same text for the 30 questions of the 133K conversation, and ladders 3 and 4 match on all 120 MRCR and 221 LongBench-v2 answers. g202 does not: its recipe runs at the main budget repeat ladder 4 exactly (the same chunks for every question), yet changed 6 of 221 LongBench-v2 answers (+4/−2) and 4 of 120 MRCR answers (+2/−2). AA-LCR, with long free-text answers, differs far more between nodes: on the questions where `ladder3` (g101) and `upgraded` (g202) built the same prompt (32, 78 and 81 at 32K, 64K and 80K), the answer text matched only 5, 13 and 13 times, and Luna's verdict differed on 6, 15 and 7 (9–19%), 14 in each direction, never when the text matched. Differences of about 5 points between AA-LCR runs on different nodes are noise. AA-LCR had no repeat of one prompt on one node, so its repeatability there was inferred from the other two benchmarks, and why g202 differs is not known. On 2026-11-12 the 17 g202 runs were rerun on g101, together with the 9 control runs, and replace the g202 runs everywhere; the g202 runs are in the sibling folders `20261029_g202`. The g101 reruns load the 7 November chat template on AA-LCR and LongBench-v2, as the `ladder3` runs did; prompts and token counts match the g202 runs for every question. Between g202 and g101, on identical prompts, 8 to 20 of 100 AA-LCR grades changed per run (accuracy 36, 52, 58, 56, 64 became 36, 52, 64, 58, 62 for `bm25_32k`, `bm25_80k`, `upgraded`, `upgraded_32k`, `upgraded_80k`), 2 to 5 of 120 MRCR scores and 3 to 9 of 221 LongBench-v2 scores.

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
- [x] Position with the recipe's steps at 64K, 32K and 80K (`position_recipe`, `position_recipe_32k`, `position_recipe_80k`), graded by Qwen3.6 and then Luna; see [CONTROLS.md](jarvis_suites/CONTROLS.md)
- [x] BM25 with the recipe's steps at 64K, 32K and 80K (`bm25_recipe`, `bm25_recipe_32k`, `bm25_recipe_80k`), graded the same way

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
- [x] BM25 with the recipe's steps at 32K, 64K and 128K (`bm25_recipe_32k`, `bm25_recipe_64k`, `bm25_recipe_128k`); see [CONTROLS.md](jarvis_suites/CONTROLS.md)

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
