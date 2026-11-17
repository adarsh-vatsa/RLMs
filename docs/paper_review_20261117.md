# Workshop paper review: 17 November 2026

A review of the first full draft of the FrontierIR (AAAI-27) workshop submission,
[aaai-2027/main.tex](../aaai-2027/main.tex), as of commit `d7b0159`. It merges two
sources, tagged throughout:

- **Review**: a first read of the anonymous submission, written as a FrontierIR
  reviewer would write it.
- **Prof. P1–P12**: the professor's comments, numbered as received and kept verbatim at
  the end.

The submission deadline is **20 November 2026 (AoE)**. FrontierIR uses double-blind
review on OpenReview and is non-archival. Full papers run 4–7 pages, excluding references
and appendices, and must use the AAAI-27 author kit. Line links point at `main.tex` as of
`d7b0159`.

Finished items are marked **Done** or **Partly done**, with a dated status note. Table
numbers below also refer to `d7b0159`. The 17 November restructure renumbered them: the
old Tables 2 and 5–9 are now Tables 5, 6, 7, 8, 10 and 11. The new Table 2 lists the
policies, the new Table 9 holds the policies moved to the appendix, the new Table 12 is
the run log, the new Tables 13 and 14 are the MRCR checks, and the new Table 15 splits
LongBench-v2 by category. Figures moved too:
the old Figures 1–3 are now Figures 2–4, and the new Figure 1 is the pipeline diagram.

## Verdict

As submitted, the draft reads as unfinished: the abstract is a placeholder and the title
still says "Working Title". With those fixed, the review's recommendation is **weak
accept** (confidence 4/5).

- **For:**
  - The question suits the venue.
  - The experiments are run more carefully than in most workshop papers.
- **Against:**
  - Novelty over OP-RAG, DOS-RAG and Arjmandi (2026) is modest.
  - The results come from one model with thinking off.
  - The MRCR setup favors retrieval.
  - The abstract and conclusion claim parity where the intervals don't support it.
  - The paper carries a lot of self-coined vocabulary.

## Action list

The list is ordered by priority. Each item names who raised it and what fixing it takes:

- *text*: editing only.
- *re-analysis*: new numbers from existing runs.
- *new runs*: experiments not yet run.

### Before submission

1. **Write the abstract** (Prof. P1; Review, minor). *Text.* **Done.**
   - **Status, 17 Nov:** the abstract is in the past tense, scoped to Qwen3.6-35B-A3B with
     thinking disabled, and lists no "lexical retrieval" comparator.
     - It leads with the evaluation message.
     - It leaves out the token savings.
     - Item 9 added the oracle comparison on MRCR, framed as evidence that did not fit
       the budget, which brought it to 249 words.
   - Use the past tense and the actual results.
   - Scope it to one open-weight model with thinking disabled.
   - The placeholder also lists "lexical retrieval" as a comparator, although BM25 is one
     of the question-aware rankers.
2. **Fix the title** (Prof. P4; Review, minor). *Text.* **Done.**
   - **Status, 17 Nov:** the title is now "What Survives the Cut? Query-Aware Truncation
     and Evidence Recall for Long Inputs", the authors' choice. It drops "question
     answering", names the method and the IR metric, and still sets on two lines.
   - Remove "Working Title:".
   - "Question answering" doesn't describe MRCR, which reproduces a passage from a
     synthetic conversation. Scope the title and abstract to long-context tasks or inputs.
3. **Add a code and data statement** (Prof. P6; Review, minor). *Text.* **Done.**
   - **Status, 17 Nov:** a plain statement, as the authors chose. The Introduction ends
     "We will release our code, prompts and per-answer outputs." It fit without adding a
     line.
     - Neither FrontierIR's call for papers nor the AAAI-27 kit requires code or a
       repository.
     - An anonymized repository was ruled out for now: commit authors, cluster host names
       in run manifests, local paths and dataset copies would all need scrubbing first.
   - Give an anonymized repository link or a plain statement.
   - The links block is commented out ([L117–120](../aaai-2027/main.tex#L117-L120)).
4. **Confirm the template and length** (Prof. P5). *Text.* **Done.**
   - **Final check, 17 Nov**, on a pdfTeX build of the working tree:
     - The main text ends 89% of the way down the right column of page 7, about 6.94
       pages, within the 7-page limit. References follow on pages 7–8, and the appendix
       runs from page 9 to page 17.
     - There are no overfull boxes, undefined references or BibTeX warnings.
     - The PDF has no Type 3 fonts (only Type 1 and Type 0), no embedded links or
       bookmarks, no title or author metadata, and no identifying strings.
     - No draft notes remain.
     - The kit is `\usepackage[submission]{aaai2027}`, template 2027.2.
     - **Recheck after any edit.** Table 4 sits at the top-float limit, and the main text
       has less than one line to spare.
   - The source already uses the kit: `\usepackage[submission]{aaai2027}`, template version
     2027.2.
   - In a scratch build, the main text ended in the top half of page 7, about 6.3–6.5
     pages.
   - Re-check with pdfLaTeX after editing. Anything added below has to be paid for with
     cuts; item 8 frees the space.
   - **Status, 17 Nov:** the main text was about 6.5 pages before the restructure.
     - Item 8's cuts brought it to 5.85 pages.
     - The abstract and the new Introduction (items 1 and 5) brought it back to about
       6.1 pages.
     - Item 8's pipeline figure brought it to about 6.25 pages.
     - Item 12's account brought it to about 6.34 pages.
     - Item 11's qualifications brought it to about 6.44 pages.
     - Items 13 and 14 brought it to about 6.62 pages.
     - Items 6 and 10 brought it to about 6.88 pages.
     - Item 9 left it at about 6.87 pages.
     - Items 15–17 brought it to about 6.93 pages.
     - The title and item 17's caption and chunk-size clauses brought it to about 6.94
       pages. It now ends 89% of the way down the right column of page 7, leaving room
       for about two lines.

### High impact

5. **Make the evaluation the headline and demote the recipe** (Prof. P2; Review 1). *Text.*
   **Done.**
   - **Status, 17 Nov:**
     - "Our main method" is gone from the Introduction and Methodology.
     - The recipe is now "the configuration in our primary comparisons". "The best
       configuration tested" wouldn't be accurate. The BM25 recipe leads at 7 of the 10
       windows, and on AA-LCR the ladder rung before fill and floor scored 67 against the
       recipe's 64.
     - The evaluation message now opens the abstract, the Introduction and the Conclusion.
     - The first Results heading now names middle truncation's cost and query-aware
       truncation's recovery instead of the recipe.
     - The caveat went in with item 8. Results says the recipe's steps recover part of the
       gain without the question on AA-LCR, and that BM25 ranking adds only 1 to 11 points
       to them.
     - The title, which P2 also mentions, is item 2.
   - The introduction says the paper offers "a controlled evaluation, not a new method"
     ([L122](../aaai-2027/main.tex#L122)).
   - It then introduces "our main method" ([L126](../aaai-2027/main.tex#L126)), and
     Results defends keeping the recipe ([L265](../aaai-2027/main.tex#L265)).
   - Call the recipe "the best configuration tested". It can't be separated from the BM25
     recipe at 8 of the 10 windows (Table 7).
   - **Caveat on the new headline.** "Ranking by the question beats ranking that ignores
     it" holds for the default rankers and for the fusion recipe against the position
     recipe. It does not hold for the BM25 recipe on AA-LCR.
     - At 32K and 64K the BM25 recipe can't be told apart from the position recipe:
       +7 [−6.2, +21.1] and +1 [−7.3, +10.7].
     - On multi-document input, much of the gain comes from the coverage steps rather
       than from the question: the position recipe gains 21 points at 32K.
     - Say so, rather than letting the headline cover both rankers.
   - **Lead with the evaluation message.** It currently appears only in the Conclusion
     ([L286](../aaai-2027/main.tex#L286)): harnesses that truncate measure their own
     overflow policy as well as the model.
6. **State what the paper adds over prior work** (Prof. P8; Review 1). *Text.* **Done.**
   - **Status, 17 Nov:** the Introduction's last paragraph now states the delta.
     - **Against Arjmandi (2026):** an open-weight model, inputs of up to 1.06M tokens,
       and a policy that needs no labels.
     - **Against LongBench-v2, OP-RAG and DOS RAG:**
       - controls that ignore the question at the same budget;
       - an oracle;
       - windows of 32K–262K;
       - two more benchmarks, one over separate documents and one that asks for a
         passage reproduced exactly.
     - **LongBench-v2's own comparison:** the paper says LongBench-v2 already sets
       order-preserving retrieval against its middle-truncated full input. Its retrieval
       curve reaches about 128K, the same budget as that input, so the paper does not
       claim nobody compared at the same budget.
     - **Analysis tools:** provenance and the visibility split are now presented as
       analysis tools, not contributions.
     - Related Work lost the two passages the Introduction took over.
   - Arjmandi (2026) is the closest work and gets one paragraph. Say in the introduction
     what this paper adds:
     - an open-weight model;
     - inputs up to 1.06M tokens;
     - a realistic retrieval policy instead of oracle removal;
     - question-agnostic controls at the same budget.
   - Do the same for OP-RAG and DOS-RAG.
   - By the paper's own citations, LongBench-v2 already compares retrieval with a
     long-context baseline that the harness middle-truncates.
   - Present provenance and the visibility decomposition as analysis tools rather than as
     contributions ([L128](../aaai-2027/main.tex#L128)).
7. **Remove the overclaims** (Prof. P3; Review 2). *Text, plus pairwise intervals.*
   **Done.**
   - **Status, 17 Nov:** all five are fixed in the Introduction, Results and Conclusion, and
     no new intervals were needed.
     - Both token claims now use the 64K window. The recipe differs from full context by
       −1.0 [−8.7, +8.0] on AA-LCR and +0.5 [−4.0, +4.8] on LongBench-v2. It sends 58% and
       48% of full context's input tokens. LongBench-v2's 32K figures are no longer cited.
     - The sentence on BM25, dense and fusion was dropped. The appendix now says only that
       dense and fusion score within 5 points of BM25.
   - **"Within a point of full context"** ([L126](../aaai-2027/main.tex#L126), the
     [L269](../aaai-2027/main.tex#L269) heading, [L286](../aaai-2027/main.tex#L286)):
     - The intervals are −1.0 [−8.7, +8.0] on AA-LCR and −0.5 [−5.4, +4.7] on
       LongBench-v2.
     - Say "not distinguishable" and give the interval.
   - **The token savings mix windows.**
     - "58% as many on AA-LCR" is the 64K window.
     - "About a quarter on LongBench-v2" is the 32K window.
     - At LongBench-v2's 64K window, Table 9 gives 65.5K against 137.2K, about 48%.
   - **"Only one lead each way"** ([L265](../aaai-2027/main.tex#L265)) means one lead in
     each direction, so the sentence is correct. The professor read it as a
     contradiction, though, so reword it: "two leads have intervals that exclude zero, one
     in each direction".
   - **"The BM25 recipe does about as well"** ([L126](../aaai-2027/main.tex#L126),
     [L284](../aaai-2027/main.tex#L284)):
     - On AA-LCR, the held-out benchmark, it trails the fusion recipe by 9 and 11 points
       at 32K and 64K.
     - On MRCR at 32K and 64K it leads by 9–12 points.
     - Write "neither ranking dominates".
   - **"BM25, dense and fusion cannot be told apart (Table 3)"**
     ([L265](../aaai-2027/main.tex#L265)): Table 3 has no pairwise intervals among them.
     Add the intervals or drop the sentence.
8. **Cut vocabulary, conditions and repetition** (Prof. P7; Review 9). *Text, plus one
   figure.* **Done.**
   - **Status, 17 Nov:** the main text now reports 9 policies, listed in a new
     policies-at-a-glance table (Table 2). The rest moved to three appendix sections:
     - the 93-condition table;
     - marked and head-only truncation, default dense and fusion, and all three random
       seeds;
     - the ladder, floor and dense-variant findings.
   - **Effect:**
     - Conditions fell from 313 words to 114.
     - The Results section fell from 985 words and 80 numbers to 645 and 53.
     - The main text fell from 4,325 words to 3,831.
     - "Ladder", "rung" and "stop" packing no longer appear in the main text.
     - "Held out" appears 3 times, and each after-the-fact choice is stated once.
   - **Finished later on 17 Nov:**
     - A pipeline diagram, the new Figure 1, shows chunk, rank and pack, and render. It
       is drawn by `figure4()` in `figures.py`.
     - "Default window" now replaces all 16 uses of "each benchmark's context window".
       Benchmarks defines it once, and Table 1's column is now "Default Window".
     - Table 1's caption lists the exact window sizes instead of defining K two ways.
       The old definition was also wrong for 262K: that window is 262,144 tokens, not
       262 × 1,024.
   - **Kept on purpose:**
     - The names fusion, position and BM25 recipe. Each is now defined once and listed
       in Table 2.
     - The numbers left in Results. They are mostly the intervals that items 5 and 7
       require.
   - **Vocabulary.** There are 93 conditions and many coined terms: recipe, ladder, rung,
     floor, fill and stop packing, needles, target turn, position recipe and BM25 recipe.
   - **Wall of prose.**
     - The Methodology section is one long run of prose.
     - Conditions ([L194](../aaai-2027/main.tex#L194)) is a single 314-word paragraph.
     - The five Results paragraphs total about 920 words, with roughly ten numbers each.
   - **Figures and tables.**
     - Add a short pipeline figure: chunk, rank, pack, render.
     - A one-glance table of policies would also help. Columns: uses the question?, chunk
       size, neighbours, packing, floor.
     - Table 2 is hard to read.
   - **Ladders.** The ladder table is already in the appendix (Table 8). What remains in
     the main text is the ladder's description in Methodology, Conditions and Results;
     move that too.
   - **Repetition.**
     - "Held out" appears 5 times.
     - That the controls and comparisons were chosen after the results is restated 4 times.
   - **Terms.**
     - The Table 1 caption defines K as both 1,024 and 1,000
       ([L153](../aaai-2027/main.tex#L153)).
     - "Each benchmark's context window" appears 11 times. "Default window" is clearer.
9. **Strengthen the IR angle** (Prof. P10; Review 1, 8). *Text.* **Done.**
   - **Positioning, 17 Nov:**
     - **Main topic:** trustworthy retrieval and RAG, through evidence coverage and
       efficiency. Every overflow policy is a retrieval decision under a token budget,
       and middle truncation is a retriever that ignores the question.
     - **Second topic:** benchmarks. Long-context scores silently include that
       retriever.
     - **Hook:** agent memory. MRCR is a test of conversation memory.
     - **Not claimed:** embodied AI, code and multimodal retrieval.
   - **Status, 17 Nov:**
     - **Introduction:** says a truncating harness "runs a retriever that ignores the
       question and does not report its evidence recall". It also calls MRCR a test of
       conversation memory and keeps the RAG and agent-harness sentence.
     - **Abstract:** ends with the configuration's shortfall against the oracle on MRCR,
       which came almost entirely from evidence that did not fit the budget. Its last
       sentence calls the overflow policy "in effect a retriever that ignores the
       question".
     - **Evidence recall:** the "visibility decomposition" is now "evidence recall at the
       token budget" everywhere.
       - Methodology presents it as a retrieval factor and a reading factor, and says it
         exists only on MRCR because only MRCR marks its evidence.
       - Table 4's first two columns sit under a "Recall" header.
     - **Conclusion:** recommends reporting evidence recall where the evidence is marked.
     - **Layout:** Table 4's source moved up with the other Results floats.
     - **Caption:** Table 4's caption lost its "not comparable across rows" sentence,
       which Results already makes.
   - **Warning:** Table 4 fills almost the 70% of a column that LaTeX allows a top float.
     One more caption line sends it to a float page of its own, which pushes the
     references to page 9.
   - FrontierIR's topics don't include long-context. The closest are benchmarks and
     trustworthy RAG.
   - **Name the retrieval metric.** Call the Target column of Table 4 what it is: chunk-level
     evidence recall at the token budget. Say why it exists only on MRCR: AA-LCR and
     LongBench-v2 don't mark where the evidence is.
   - **Add framing.** One sentence should tie the paper to evaluation validity for IR and
     RAG. Another can tie it to agent harnesses, where overflow is routine (long
     histories, tool outputs) and where MRCR is a conversation-memory task.
10. **Add the missing references** (Prof. P9; Review). *Text.* **Done.**
    - **Status, 17 Nov:** 12 references were added, each checked against arXiv or its
      publisher's record (via Crossref). `references.bib` now holds 34 entries, all cited.
      - **Lost in the Middle:** Related Work, for "middle truncation removes the part
        models use least well".
      - **Gao, Chen and Huang (2026):** Results, twice.
        - At a fixed length, the text kept separates the policies, not the shorter
          context that they credit for filtering's benefit.
        - Together with Levy et al. (2024), at the oracle's "less context helps".

        Only their abstract was read.
      - **Harness conventions:** ∞Bench truncates the middle (also cited in the
        Introduction), HELMET cuts documents from the end, and RULER generates each input
        at its target length. All three are in Related Work.
      - **In-model selection:** InfLLM and SnapKV, in Related Work.
      - **IR antecedents:** Callan, Kaszkiel and Zobel, and Tombros and Sanderson, in
        Related Work.
      - **MemGPT:** the Introduction, at the agent-harness sentence.
    - **Lost in the Middle** (Liu et al. 2024). Middle truncation is the baseline, and it
      removes the region models use least.
    - **Gao, Chen and Huang (2026)**, "The First Drop of Ink: Nonlinear Impact of
      Distracting Information in Long-Context Reasoning", arXiv 2605.10828. Its abstract
      says filtering gains "mainly come from context-length reduction rather than
      distractor removal".
      - The paper holds length roughly fixed at each window.
      - At that fixed length, random chunks score below middle truncation on AA-LCR (35 vs
        40), while ranking by the question scores 53–64. So length reduction alone does not
        explain the gains over truncation.
      - The oracle's "less context helps" result (75.0 at 32K vs 61.7 at 262K) agrees with
        their length finding.
      - Cite it in both places. Only the abstract has been checked.
    - **From the review:**
      - how ∞Bench, RULER and HELMET truncate overflowing inputs;
      - InfLLM and SnapKV, which select context inside the model;
      - query-biased summaries (Tombros & Sanderson 1998);
      - passage retrieval (Callan 1994; Kaszkiel & Zobel 1997);
      - Levy et al. 2024, *Same Task, More Tokens*;
      - MemGPT or similar work.
11. **Qualify the MRCR results** (Review 3; Prof. P4 and P10 touch on it). *Re-analysis;
    one or two optional runs.* **Done, without the optional runs.**
    - **Note, 17 Nov:** the run log (Table 12) now records how MRCR's query and chunk size
      were chosen. They changed after an 8-needle pilot in which dense retrieval, with
      7,500-token chunks and the full question as the query, tied truncation.
    - **Status, 17 Nov:**
      - **Query:** the Introduction and Results qualify the oracle match: the query nearly
        repeats the needle requests. Results adds that querying with the full question
        costs dense ranking 9 points (Table 8). The rerun with the raw final message is
        skipped for now.
      - **Official metric:** Benchmarks now says the released score runs about 1 to 4
        points above exact match for every main-text policy and never reverses their
        order. The scores are in the new appendix Table 13, from `results.csv`. Across
        all 37 conditions, a few appendix pairs at 262K do swap, so the sentence is
        limited to the main-text policies.
      - **Two conversations:** Results now says that at 262K the gain comes from the two
        longest conversations, 60 of the 120 questions.
      - **Selection changes the task:** the new appendix Table 14 splits the questions
        whose target the recipe keeps by how many needle turns stay complete, with the
        oracle on the same questions.
        - At 32K the recipe's lead (44 answers against 39) comes from prompts with fewer
          than four needle turns.
        - With all four, the recipe and the oracle agree within two answers at every
          window.
        - With fewer, the recipe answers more at 32K and 128K and fewer at 64K.

        Results and the Table 4 caption say so. The caption also loses the "fewer
        candidates" sentence that was hard to parse.
      - **By construction:** Results now notes that a prompt missing part of the target
        can't be an exact match.
      - **Code:** `mrcr_v2.visibility` now also writes `visibility-needles.csv`. A new
        `--selection-only` flag reads only selection runs and needs no tokenizer. That
        matters because the full command can't rebuild middle-truncated prompts here: no
        local transformers version reproduces the tokenizer recorded for the dataset.
        `test/test_mrcr_oracle.py` covers both paths.
    - **Hand-built query.** The query is a hand-extracted description that nearly repeats
      the needle requests ([L174](../aaai-2027/main.tex#L174)). Qualify "matches the oracle
      on every conversation" accordingly. Optionally, rerun the recipes with the raw final
      user message as the query.
    - **Official metric.** Report the official similarity score next to exact match
      ([L159](../aaai-2027/main.tex#L159)).
    - **Two conversations carry the 262K effect.** At 262K the 133K conversation fits, and
      97% of the 267K one fits, so the effect comes from the other two.
    - **Selection changes the task.** At 32K the recipe keeps all four needles in only 9% of
      prompts. Yet on questions where it keeps the target, it scores 77.2 against the
      oracle's 68.4 on the same questions. Break exact match down by the number of visible
      needles.
12. **Account for how the recipe was fixed** (Review 4). *Text.* **Done.**
    - **Status, 17 Nov:** answered from the repo history, with two facts the authors
      confirmed.
      - **The floor** answers the 30 Sep AA-LCR dense pilot, which left at least one
        document out for 63 of 100 questions. Methodology now says so.
      - **AA-LCR results seen before the recipe was fixed:** yes, from the Aug–Sep pilots
        and the default policies run on 31 Oct. The 12 Oct plan (`3198356`) had already
        fixed the recipe's settings, whose exact flags `e991b92` lists. It also fixed the
        ladder order and the rule that the last rung becomes the recipe. The recipe was
        fixed by that rule on 2 Nov and first ran on AA-LCR on 3 Nov.
      - **Other configurations:** none. The code for the recipe's steps arrived on
        12 Oct. Before that, only truncation and default dense retrieval had run, and
        every later run is a reported condition.
      - **Neighbour step:** the plan's rule keeps every step, whatever its effect.
      - **Primary comparisons:** the plan posed both questions before any reported run,
        and named position as the decisive control. Only the test and Holm's correction
        came after the results, on 13 Nov. Statistics now says this. It also says that
        the redundant comparisons on MRCR and LongBench-v2 only make the correction
        stricter.
      - **Run log:** new appendix Table 12, dated by commit.
    - **Unclear step choices.**
      - The neighbour step was kept although it lowered LongBench-v2 by 3.6 points.
      - The floor can't change a single-text prompt, so it can't have been chosen on MRCR
        or LongBench-v2.
    - **Questions to answer:**
      - What motivated the floor?
      - Had any AA-LCR results been seen before the recipe was fixed?
      - How many other configurations were tried?
    - **Post-hoc primary comparisons.** The two primary comparisons were chosen after the
      results were known ([L257](../aaai-2027/main.tex#L257)).
    - **Redundant comparisons.** On MRCR and LongBench-v2, the comparison against position
      largely repeats the one against middle truncation.
    - Add a short appendix log giving the order of runs and decisions.
13. **Make LongBench-v2 interpretable** (Review 5). *Re-analysis; cheap runs.* **Done,
    without the runs.**
    - **Note, 17 Nov:** no new runs for now (authors' decision); text and re-analysis only.
    - **Status, 17 Nov:**
      - **Categories:** a new appendix Table 15 comes from `categories.csv`, which
        `long_bench_v2.results_table` now writes.
        - Full context gains on middle truncation only in single-document QA, long
          in-context learning and structured data.
        - Both recipes gain in the first two.
        - On multi-document QA no policy beats middle truncation, and on code all score
          the same.

        Results says so in one sentence. This is the opposite of the review's guess for
        in-context learning.
      - **Noise, no-context baseline, natural overflow:** Limitations now gives the seeds'
        39.8–46.6% spread. It also says that without a no-context run the model's priors
        can't be measured, and that prompts over 262K are left out.
    - **Noise.** Full context is only 5.9 points above middle truncation. The random-chunk
      seeds span 39.8–46.6, as wide as the recipe's gain.
    - **Add:**
      - a no-context baseline (question and options only);
      - a breakdown by category.
    - **Missing natural overflow.** The kept questions exclude every prompt over 262K, which
      are the ones that overflow naturally. Those could be added for the recipe vs middle
      truncation comparison, which needs no full-context reference.
14. **Address external validity** (Review 6). *Text; optional runs.* **Done, without the
    runs.**
    - **Note, 17 Nov:** no new runs for now (authors' decision); text and re-analysis only.
    - **Status, 17 Nov:**
      - **Thinking off:** Setup now gives the authors' reasons.
        - A reasoning trace would take output tokens from the same window, shrinking and
          varying the input budget.
        - We wanted each policy's effect on the model's answer without a separate
          reasoning phase.

        Limitations adds that such a phase might make up for missing text or compound its
        loss.
      - **Imposed windows:** Limitations now says a model built for 64K may use the window
        differently.
      - **Answer cap:**
        - The cost table (Table 11) gains a Capped column. At 64K, middle truncation caps
          5 answers, the recipe 2 and full context none.
        - `aa_lcr.intervals` now also writes `intervals-gpt-5.6-luna-uncapped`. It
          compares each run with middle truncation, leaving out the questions where either
          answer hit the cap.
        - The recipe still leads by 40, 25 and 14 points at 32K, 64K and 80K, with
          intervals that exclude zero. Grading says so.
    - **Thinking off.** AA-LCR is a reasoning benchmark, so justify disabling thinking.
    - **Imposed windows.** Note that the 64K window is imposed on a model that can read
      262K.
    - **Answer cap.** It interacts with the policy:
      - Under middle truncation, answers are 55% longer than under full context.
      - Luna marks 136 of 144 capped answers wrong.
      - Report cap rates per policy, and the main comparison with capped answers excluded.

### Lower priority

15. **Bring appendix results into the main text** (Prof. P11). *Text.* **Done.**
    - **Status, 17 Nov:** Results now names the test: "Holm-corrected randomization tests".
      The key cost numbers were already in Results (58% and 48% of full context's input
      tokens); times per question stay in the appendix.
    - [L261](../aaai-2027/main.tex#L261) already says "significant" and points to Table 6.
      Name the test and the correction in that sentence: a Holm-corrected randomization
      test over 20 comparisons.
    - Give the key cost numbers from Table 9 in the main text as well.
16. **Identify the judge and its repeatability** (Prof. P12; Review, minor). *Text.*
    **Done.**
    - **Status, 17 Nov:**
      - **Grading follows the published method.** The four rules in our grader's system
        prompt are AA-LCR v1.1's published grader prompt (dataset card at `9a77ef56`), and
        that card names GPT-5.6 Luna at medium effort as the judge. The old wording "rules
        for equivalent answers added" wrongly suggested we added them. This also answers
        the minor comment on L255.
      - **Grading paragraph:** now says AA-LCR is graded as its v1.1 release specifies, by
        OpenAI's GPT-5.6 Luna. "Its errors" became "the errors we found".
      - **Limitations:** adds that Luna graded 3 of 276 recurring answers inconsistently.
      - **Grader Check appendix** now gives:
        - the identifier, `openai/gpt-5.6-luna` through OpenRouter, called 8–13 Nov 2026;
        - that the check grader is the model under test, so hand labels settle
          disagreements;
        - the errors not found: 2 of 46 sampled agreements are wrong for both graders,
          about 130 of 3,058 answers, or about 4 per run;
        - Qwen-graded accuracies, within 4 points of Luna's for every run and 1.2 on
          average.
    - "GPT-5.6 Luna" needs its provider, exact model identifier and access date.
    - Move the repeatability result into the Limitations section: 3 of 276 recurring
      answers were graded differently.
    - **Detected errors only.** "Luna's errors change no run's accuracy by more than two
      points" ([L294](../aaai-2027/main.tex#L294)) covers only the errors found.
      - In the hand-checked sample, 2 of 46 answers on which both graders agreed were wrong.
      - Extrapolated, that is about 130 answers that both graders got wrong.
    - **Self-grading.** The check grader is the model under test.
    - **Reproducibility.** Report the Qwen-graded scores in the appendix, since the two
      graders agree on 98.6% of answers.
17. **Remaining review items.** See Review 7 and 8 and the minor comments below. **Done.**
    - **Second pass, 17 Nov:** the rest of Review 7 and 8, the minor comments and the
      questions to the authors. Everything went into the appendix, outside the page
      limit, except two words in Figure 2's caption.
      - **Question 4, the third node:** Serving Nodes now gives the noise floor.
        - Across the 17 reruns, the third node's scores differ by up to 6 points, 1.5 on
          average, and up to 20 of 100 AA-LCR answers change.
        - The largest gap is the recipe on AA-LCR at 64K: 58% on the third node against
          the 64% reported.
        - **Decided, 17 Nov:** it stays in the appendix. The main text's last two lines
          go to the code statement (item 3).
      - **Question 7, prefix caching:** Cost now says the token counts ignore it. About
        three AA-LCR questions share a document set, so full context could prefill those
        documents once per set; query-aware prompts can't share theirs.
      - **Question 3, MRCR rendering:** Other Policies now says each MRCR prompt is one
        user message, and gives the three omission markers.
      - **Question 6, the example:** Figure 2's caption now says six is the median number
        of documents.
      - **Minor comments:** Table 9's caption gives the three-seed means (33.3, 38.9,
        42.5). Table 5's caption says two of the 93 conditions repeat another.
      - **Declined, 17 Nov (authors' decision):** an AA-LCR hand audit, to test whether
        evidence recall explains failures beyond MRCR. The Conclusion lists it as future
        work.
    - **Status, 17 Nov:**
      - **Done:**
        - Results says the oracle is not an upper bound.
        - Limitations names the untested baselines: tail truncation, a cross-encoder
          reranker and prompt compression.
        - The spelling is American throughout (neighbor, color, gray), including the
          ladder table's label in `tables.py`.
        - Figure 3's window labels are now 7.5 pt in ink, and its caption points to
          Table 8's intervals. Error bars were left out, since they would clutter seven
          lines.
        - Figure 2's caption says which policies answer its question: middle truncation
          wrong, position and the recipe right. Authors agreed, 17 Nov.
        - **Default chunk size.** Chunking now says the 7,500-token default sits safely
          below the embedding model's 8,192-token input limit.
          - The reason comes from the 8 Aug retrieval plan, which proposed "approximately
            7.5K tokens with 750 tokens of overlap" for that reason.
          - The pipeline still caps embedding input at 8,192 (`EMBEDDING_MAX_LENGTH`).
          - The authors recall trial and error as well, but no runs with other chunk
            sizes survive in the repo history: every committed manifest uses 7,500, or
            3,500 on MRCR.
          - The ladder adds to the answer: 1,024-token chunks alone add 0.8–3.0 points,
            none with an interval excluding zero, while neighbors add 11 on AA-LCR
            (Table 10).
      - **Not changed:**
        - Leave-one-out ablations, which need runs. Limitations covers the ladder's limit.
        - "Position" keeps its name (authors' decision, 17 Nov).
      - **Duplicates in the 93 conditions:** addressed in the second pass below.
    - **Large default chunks.** The default 7,500-token chunks look like a strawman.
    - **Ablations.** Remove one step at a time instead of only adding steps in order.
    - **Missing baselines.** Tail (recency) truncation, a reranker and LongLLMLingua.
    - **The oracle is not an upper bound.**
    - **Naming and spelling** are inconsistent. *Partly done, 17 Nov:* marked middle
      truncation and head only now have one name each in text and tables. British and
      American spellings are still mixed.
    - **Minor comments, 17 Nov:**
      - The claim that AA-LCR is the only benchmark with several separate documents was
        removed along with the old Conditions paragraph.
      - Table 9 now lists all three random seeds, though without a mean.
    - **Figure 2** needs uncertainty shown and larger window labels.
    - **Figure 1** should show which policies answered correctly.

### What fits before 20 November

- **Text only:** items 1–10, 12, 14–16, and the minor comments.
- **Re-analysis of existing runs:**
  - the official MRCR score;
  - exact match by visible needles;
  - LongBench-v2 categories;
  - cap rates per policy;
  - the third node's score difference;
  - three-seed means for random chunks;
  - pairwise intervals.
- **Small new runs:**
  - no-context baselines, three runs;
  - MRCR with the raw question as the query.
- **Limitations or future work:**
  - thinking on;
  - a second model;
  - removing one step at a time;
  - a chunk-size sweep.

## Full review

Written as a FrontierIR reviewer seeing the anonymous submission for the first time.

**Summary.** The paper calls the handling of inputs longer than the context window an
*overflow policy*. It compares middle truncation, the usual harness default, with
order-preserving chunk retrieval ("query-aware truncation") at equal token budgets. One
model is used (Qwen3.6-35B-A3B, thinking off) on AA-LCR, MRCR v2 and LongBench-v2. The
conditions are:

- truncation variants;
- question-agnostic controls (position, random);
- BM25, dense and RRF rankers;
- a tuned configuration, the "fusion recipe", and a BM25 version of it;
- an oracle on MRCR.

Statistics are clustered bootstrap intervals and Holm-corrected randomization tests.

**Results.**

- Selection beats middle truncation by large, significant margins on AA-LCR and MRCR. On
  AA-LCR at 64K it scores 64 against 40, with full context at 65.
- It shows no significant gain on LongBench-v2.
- Against full context it shows no detectable loss while sending 25–58% of the input
  tokens.

**Strengths**

1. **The question is relevant and rarely examined.** A harness's overflow policy confounds
   long-context scores. The MRCR-by-length result shows this plainly: middle truncation
   falls from 60 to 10 while selection stays near 60.
2. **The controls are good.**
   - Every policy uses the same pipeline and budget.
   - Position and random share the rankers' chunks and packing, which isolates the value of
     using the question.
   - Marked and head-only truncation rule out trivial explanations.
   - Fill packing equalizes the token counts.
3. **The paper is unusually open about its process.** It reports:
   - a held-out benchmark;
   - which controls were designed after the results were known;
   - the ladder rung that beat the recipe;
   - the policy for rerunning the third node's runs.
4. **The statistics are well above workshop norms,** and the LongBench-v2 null result is
   reported honestly.
5. **The checks are thorough:** blind hand-labelled grader validation, determinism checks
   and a cost appendix.
6. **The Limitations section is specific and candid.**

**Major concerns**

1. **Framing and novelty** *(text only)*
   - **Contradictory framing.** The paper says it offers "a controlled evaluation, not a
     new method" ([L122](../aaai-2027/main.tex#L122)). It then introduces "our main
     method" ([L126](../aaai-2027/main.tex#L126)), and Results defends keeping it
     ([L265](../aaai-2027/main.tex#L265)). Pick one framing.
   - **The obvious objection.** A skeptical reviewer will sum the paper up as "retrieval
     beats cutting out the middle when the middle holds the answer." Say early what is
     *not* obvious. Candidates:
     - how large the harness confound is;
     - that question-agnostic structure alone recovers a lot on multi-document inputs;
     - that MRCR failures are failures to keep the target in the prompt.
   - **Novelty.** State what is new relative to OP-RAG, DOS-RAG and Arjmandi (2026). By the
     paper's own citations, LongBench-v2 already compares retrieval with a long-context
     baseline that the harness middle-truncates.
   - **The lead message.** The strongest message for this venue appears only in the
     Conclusion ([L286](../aaai-2027/main.tex#L286)): harnesses that truncate measure
     their own overflow policy. Lead with it.
   - **Thin contributions** ([L128](../aaai-2027/main.tex#L128)):
     - Per-answer provenance is logging.
     - The visibility decomposition is evidence recall at the budget times reader
       accuracy, the standard retriever/reader split.
     - Present both as analysis tools.
   - **Workshop fit.** The link to FrontierIR's focus on agents is never made. Overflow is
     the everyday condition of agent harnesses (long histories, tool outputs), and MRCR is
     a conversation-memory task. One paragraph would make the connection.

2. **Several claims go beyond the intervals** *(text, plus a little analysis)*
   - **"Within a point of full context"** ([L126](../aaai-2027/main.tex#L126), the
     [L269](../aaai-2027/main.tex#L269) heading, [L286](../aaai-2027/main.tex#L286)):
     - The intervals are [−5.4, +4.7] on LongBench-v2 at 32K and [−8.7, +8.0] on AA-LCR at
       64K.
     - That means "no detectable difference at n = 100 or 221", not parity.
     - Test equivalence against a stated margin, or rephrase.
   - **"The BM25 recipe does about as well"** ([L126](../aaai-2027/main.tex#L126),
     [L284](../aaai-2027/main.tex#L284)):
     - On AA-LCR, the only held-out benchmark, it trails the fusion recipe by 9 and 11
       points at 32K and 64K. The 64K interval excludes zero.
     - At those windows it can't be told apart from the position recipe, which ignores the
       question: +7 [−6.2, +21.1] and +1 [−7.3, +10.7].
     - On MRCR at 32K and 64K it leads the fusion recipe by 9–12 points.
     - "Neither ranking dominates" is the accurate summary.
   - **"BM25, dense and fusion cannot be told apart (Table 3)"**
     ([L265](../aaai-2027/main.tex#L265)): Table 3 has no pairwise intervals among them.
     Add the intervals or drop the sentence.
   - **"Ranking by the question recovers more"** is tested directly only for the recipe
     against position.
     - On AA-LCR, much of the gain comes from the recipe's steps rather than the question:
       the position recipe gains 21 points at 32K.
     - That is a finding worth stating, and the current headline hides it.

3. **The MRCR setup favors retrieval** *(rescoring, plus one or two runs)*
   - **Hand-built query.** The query is a hand-extracted format, topic and style
     description that "nearly repeats the needle requests"
     ([L174](../aaai-2027/main.tex#L174)).
     - A deployed system would query with the user's message. Using the full question
       costs dense about 9 points at 262K.
     - So "matches the oracle on every conversation" partly reflects query engineering.
     - Qualify the claim in the introduction and abstract, or report the recipes with the
       raw final message as the query.
   - **Non-standard metric.** Exact match replaces the official similarity scorer
     ([L159](../aaai-2027/main.tex#L159)), so the numbers can't be compared with published
     MRCR results. Report the official score alongside.
   - **The 262K headline rests on two conversations.** At 262K the 133K conversation fits
     under every policy, and 97% of the 267K one fits. The headline effect therefore comes
     from 60 questions on the other two.
   - **Selection changes the task.**
     - At 32K the recipe keeps all four needles complete in only 9% of prompts.
     - Yet on questions where it keeps the target, it scores 77.2 against the oracle's 68.4
       on the same questions (Table 4 caption).
     - Dropping competing needles may make the ordinal easier, or let the model be right by
       elimination.
     - Break exact match down by the number of visible needles.

4. **The held-out claim needs an account of the choices made** *(text only)*
   - **No stated rule for keeping steps.**
     - The neighbour step was kept although it lowered LongBench-v2 by 3.6 points
       [−7.7, +0.4].
     - The document floor has no effect on either development benchmark, so it cannot
       have been chosen there.
     - What motivated the floor? Had default policies already run on AA-LCR, for example
       showing position dropping whole documents as in Figure 1?
     - How many other configurations were tried and dropped?
   - **Post-hoc primary comparisons.** The two primary comparisons were chosen after the
     results were known ([L257](../aaai-2027/main.tex#L257)). Holm's correction then
     controls error within a family that was itself picked after the fact.
   - **One family is redundant on two benchmarks.** On MRCR and LongBench-v2, position is
     in effect marked middle truncation ([L194](../aaai-2027/main.tex#L194)). The second
     family is only informative on AA-LCR.
   - **Fix:** a short appendix run log giving the order of runs and decisions would settle
     all of this.

5. **LongBench-v2, as run, cannot separate the policies** *(analysis, plus cheap runs)*
   - **Little range.** Full context is only 5.9 points above middle truncation.
   - **Seed noise.** The three random-chunk seeds span 39.8–46.6, and seed 2 equals full
     context ([L244](../aaai-2027/main.tex#L244)). The spread across seeds is as large as
     the recipe's gain.
   - **Priors.** With single-letter answers and no reasoning, the benchmark may be
     measuring the model's priors.
   - **Suggested additions:**
     - A no-context baseline (question and options only).
     - A breakdown by the six categories. Long in-context learning and structured data are
       where selection by the question should struggle.
   - **Missing natural overflow.** The 221 kept questions exclude every prompt over 262K,
     which are exactly the ones that overflow without an imposed window. Comparing the
     recipe with middle truncation needs no full-context reference, so adding them would
     give a natural-overflow test on real documents.

6. **External validity** *(text; optional runs)*
   - **One model, imposed windows.** Everything rests on one model with thinking off. On
     AA-LCR and LongBench-v2 the window is imposed on a model that can read 262K.
     - AA-LCR is a reasoning benchmark, so why disable thinking?
     - A model trained natively for 64K may also use a 64K window differently.
     - One thinking-on run of the main AA-LCR comparison, or a second model, would help a
       lot. Otherwise, narrow the scope in the abstract.
   - **The answer cap interacts with the policy.**
     - Under middle truncation, answers are 55% longer than under full context.
     - Luna marks 136 of the 144 answers that hit the 4,096-token cap as wrong.
     - Report the cap rate per policy, and the main comparison with capped answers
       excluded.

**Other concerns**

7. **Baselines and configuration**
   - **Large default chunks.** The defaults are 7,500 tokens (3,500 on MRCR).
     - At 64K that allows about 8 chunks.
     - It is why position can't reach the later documents (Figure 1).
   - **The default rankers look like strawmen.** Much of the recipe's effect may come from
     normal chunk sizes plus neighbouring chunks, which together make windows of about 3K
     tokens. Justify the default or sweep the chunk size.
   - **Ablation design.** A ladder that adds steps in one order can't attribute effects.
     Removing one step at a time from the full recipe is the standard ablation.
   - **Missing baselines.** IR reviewers will expect these, or a sentence each on why they
     are out of scope:
     - tail (recency) truncation, the default for chat and agent histories and natural for
       MRCR;
     - a cross-encoder reranker;
     - LongLLMLingua, which is cited but not compared.

8. **Oracle and visibility**
   - **Not an upper bound.** The oracle scores below the recipe on the same questions at
     32K and 128K (see concern 3). Rename it, for example "evidence-first ranking", or
     explain why.
   - **True by construction.** "No prompt with an incomplete target turn gets an exact
     match" must hold when the task is verbatim reproduction. The informative number is
     accuracy given a complete target.
   - **IR framing.** Calling the Target column "evidence recall at the token budget" ties
     the analysis to IR evaluation.
   - **Visibility beyond MRCR.** Visibility is measured only on synthetic MRCR. A hand audit
     of about 20 AA-LCR failures, sorting evidence dropped from reasoning errors, would test
     whether the explanation carries over.

9. **Clarity.** The prose is precise but dense, and at times reads like an experiment log.
   - **Too many named variants to track.** A one-glance table of policies (uses the
     question?, chunk size, neighbours, packing, floor) and a small pipeline figure would
     help more than prose.
   - **Overloaded paragraphs.**
     - Conditions ([L194](../aaai-2027/main.tex#L194)) is a single 314-word paragraph.
     - The five Results paragraphs total about 920 words, with roughly ten numbers each.
     - State each finding with one number and leave the rest to the tables.
   - **Repetition.** "Held out" appears 5 times, and the after-the-fact status of the
     controls and comparisons is restated 4 times. Cutting this frees space for the
     additions above.
   - **Ambiguous terms.**
     - "Each benchmark's context window" (11 times) blurs the model's window with the
       experiment's default setting. "Default window" would be clearer.
     - K is defined two ways ([L153](../aaai-2027/main.tex#L153)).

**Questions for the authors**

1. What rule decided which ladder rungs to keep? What else was tried before the recipe was
   fixed, and had any AA-LCR results been seen?
2. How is the MRCR query extracted? What do the BM25 and fusion recipes score with the raw
   final user message as the query?
3. Is each MRCR conversation rendered as one message? What text does middle truncation cut
   there, and what does the omission marker say?
4. How far did the third node's scores differ from the reruns
   ([L362](../aaai-2027/main.tex#L362))? That difference is a noise floor for gaps of a
   few points.
5. On AA-LCR, how many answers hit the cap under each policy?
6. How was the Figure 1 example chosen, and which policies answered it correctly? Marking
   that on the figure would make it much stronger.
7. AA-LCR has about 3.3 questions per document set. With prefix caching, full-context
   prefill is shared across those questions, but query-specific prompts are not. Does the
   token-saving argument survive that?

**Minor comments**

- **Title and abstract**
  - "Working Title:" is still in the title ([L94](../aaai-2027/main.tex#L94)).
  - The abstract is a blue placeholder ([L112](../aaai-2027/main.tex#L112)).
  - It also lists "lexical retrieval" as a comparator, though BM25 is one of the
    question-aware rankers.
- **Naming and spelling**
  - Names differ between text and tables: "Marked middle truncation" vs "Middle, marked",
    and "Head" vs "Head only" ([L167](../aaai-2027/main.tex#L167)).
  - British and American spellings are mixed (neighbouring, colour vs labeled, favors).
  - "Position" is an opaque name; "per-document middle truncation" says what it does.
- **Tables and figures**
  - Table 3 gives random chunks for seed 0 only. Report the three-seed mean and range.
  - Figure 2 shows no uncertainty, although the text relies on intervals.
  - Its window labels, the experimental variable, are the smallest and lightest text on it
    (6.5 pt grey, [figures.py:116](../aaai-2027/figures/figures.py#L116)).
  - In the Table 4 caption, "leaves the ordinal fewer candidates" is hard to parse.
  - "Without the floor" builds the same prompts as the recipe on MRCR and LongBench-v2.
    Counting it as separate conditions inflates the total of 93
    ([L225](../aaai-2027/main.tex#L225)).
- **Accuracy of statements**
  - [L194](../aaai-2027/main.tex#L194) calls AA-LCR the only benchmark with several separate
    documents. LongBench-v2 has 46 multi-document questions; say that their documents
    arrive as one concatenated text.
  - [L255](../aaai-2027/main.tex#L255) says the benchmark's rules were "added". Added to
    what? Does this depart from AA-LCR's published grading?
- **Grading**
  - "Luna's errors change no run's accuracy by more than two points"
    ([L294](../aaai-2027/main.tex#L294)) covers only the detected errors. The 2 of 46
    sampled agreements that were wrong extrapolate to about 130 answers that both graders
    get wrong.
  - The check grader is also the model under test.
  - Because the judge is proprietary, report the Qwen-graded scores (98.6% agreement) in
    the appendix for reproducibility.
- **Availability.** There is no code or data statement; the links block is commented out
  ([L117–120](../aaai-2027/main.tex#L117-L120)).

**Missing references**

- **Position effects:** *Lost in the Middle* (Liu et al. 2024). Middle truncation removes
  the region models use least.
- **Harness conventions:** how ∞Bench, RULER and HELMET handle prompts that overflow.
- **Selection inside the model:** InfLLM and SnapKV, another family of overflow policy.
- **IR antecedents of query-aware truncation:**
  - query-biased summaries (Tombros & Sanderson 1998);
  - passage retrieval (Callan 1994; Kaszkiel & Zobel 1997).
- **"Less context helps":** Levy et al. 2024 (*Same Task, More Tokens*), for the oracle
  result.
- **Agent memory:** MemGPT or similar work.

**Recommendation.** As submitted, with a placeholder abstract and a working title, the paper
reads as unfinished, and some reviewers will score it on that alone. With those fixed, the
recommendation is **weak accept**.

- **For:** the question suits the venue, and the experiments are run more carefully than
  most workshop papers.
- **Against:**
  - modest novelty over OP-RAG and DOS-RAG;
  - one model with thinking off;
  - an MRCR setup that favors retrieval;
  - claims of parity that the intervals don't support.
- **What would move it to a clear accept:**
  1. Settle on the evaluation framing.
  2. Soften "within a point" and "about as well".
  3. Add the no-context baseline and the LongBench-v2 category breakdown.
  4. Document how the recipe was fixed.

Confidence: 4/5.

## Professor's comments, as received

1. The abstract is a placeholder. It's in blue, and it's written in future tense ("will
   report"). Replace it with the actual results and scope it explicitly to one
   open-weight model with thinking disabled.
2. The paper contradicts itself on what it contributes. The intro says "a controlled
   evaluation, not a new method." The results then call the fusion recipe "our main
   method," and the title frames a policy. Your own data supports a cleaner claim than the
   recipe: ranking by the question beats ranking that ignores it. The recipe and its BM25
   variant are not separable on most windows (Table 7). Make the evaluation the headline
   and demote the recipe to "best configuration tested."
3. The abstract and conclusion overclaim in three places:
   - "Within a point of full context." The intervals are −1.0 [−8.7, +8.0] on AA-LCR and
     −0.5 [−5.4, +4.7] on LongBench-v2. That isn't parity. Say "not distinguishable" and
     give the interval.
   - Token savings mix windows. "58% as many on AA-LCR" is the 64K window. "About a
     quarter on LongBench-v2" is the 32K window. At LongBench-v2's 64K window, Table 9
     gives 65.5K against 137.2K tokens, about 48%.
   - A sentence says "only one lead" and then names two (fusion's on AA-LCR at 64K and
     BM25's on MRCR at 64K).
4. "Question answering" doesn't describe all three benchmarks. MRCR reproduces a passage
   from a synthetic conversation, and your own limitations note that its query nearly
   repeats the needle requests. Scope the title and abstract accordingly.
5. Use the AAAI-27 author kit and re-check length. The workshop requires the AAAI-27
   author kit and 4–7 pages excluding references and appendices, under double-blind
   review. I can't tell from the PDF which template this is. The main text currently runs
   about 6.3 pages, so re-check after any reformat.
6. Add a code/data statement (an anonymized repo link or a plain statement). The paper has
   none, and reviewers will ask.

*Likely reviewer complaints (high impact)*

7. Too much self-coined vocabulary and too many conditions. There are 93 conditions, plus
   recipe, ladder, rung, floor, fill and stop packing, needles, target turn, position
   recipe and BM25 recipe. Methodology is a wall of prose, and Table 2 is hard to read.
   Add a short pipeline figure (chunk, rank, pack, render) and move the ladders to the
   appendix. Table 1's caption defines K as both 1,024 and 1,000, which is confusing.
8. The delta against the closest prior work is unstated. Arjmandi (2026) compares
   middle-removal against distractor-aware truncation on four benchmarks and four models
   across two providers. Your paper cites it in one paragraph. State in the intro what you
   add: an open-weight model, inputs to 1.06M tokens, and a realistic retrieval policy
   instead of an oracle removal.
9. Missing citation. "Lost in the Middle" (Liu et al.) isn't in the references, although
   middle truncation is your baseline. A recent arXiv paper (2605.10828) reports that
   filtering gains come mainly from context-length reduction rather than distractor
   removal. Your random-chunks control addresses the same question, so check it. I only
   saw a search snippet of that paper.
10. The IR angle is thin. FrontierIR's topics don't include long-context. The closest are
    benchmarks and trustworthy RAG. Reviewers will ask for a retrieval metric such as
    chunk-level evidence recall. Your visibility decomposition provides that only on MRCR,
    so say why it isn't available elsewhere. One framing sentence tying this to evaluation
    validity for IR/RAG would help.

*Lower priority*

11. Appendix-dependent claims. The significance test (Table 6) and the cost table (Table 9)
    are in the appendix, which reviewers may not read. Put the Holm result in one
    main-text sentence.
12. The judge is hard to identify. "GPT-5.6 Luna" should be named so a reviewer can
    identify it. The judge is also not fully repeatable: 3 of 276 recurring answers were
    graded differently.
