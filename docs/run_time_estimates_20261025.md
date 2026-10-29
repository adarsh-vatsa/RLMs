**Decided on 28 October:** all three benchmarks run on Jarvis, one suite at a
time against one executor job, as described in the
[workshop suites runbook](jarvis/WORKSHOP_SUITES_RUNBOOK.md). MRCR starts at 30
questions per conversation; a full run may follow as a separate suite.

---

- **AA-LCR:** 18 runs, all 100 questions each. About 4 hours now and 1.25 hours
  once the upgraded recipe is fixed.
- **LongBench-v2:** 20 runs, 221 questions each. About 6 hours now and 1.7 hours
  once the recipe is fixed.
- **MRCR:** 20 conditions, all run on the same questions. At 30 questions per
  conversation (120 in all) that takes 14–15 hours and resolves differences of
  about 20 points; all 953 questions take about 117 hours.
- **In all:** about 24 hours of executor time now, one suite per job, and about 5
  hours for the recipe runs later.
- **Basis:** the AA-LCR and MRCR times scale the 30 September Neselab runs, and
  the LongBench-v2 times the August Jarvis runs; the first suite of each
  benchmark measures its time on Jarvis.

## AA-LCR

AA-LCR takes about 4 hours for its 15 recipe-independent runs, plus 1.25 hours
for the 3 upgraded-recipe runs later. Every run uses all 100 questions, so there
is no sampling choice to make.

| Runs                                                                                | Count | Time each  |   Total |
| ----------------------------------------------------------------------------------- | ----: | ---------- | ------: |
| Direct: full context, middle, marked middle, head; middle at 32K and 80K            |     6 | 12–17 min  |  ~1.4 h |
| Hybrid without embedding: position, random seeds 0–2, BM25; position at 32K and 80K |     7 | ~14 min    |  ~1.6 h |
| Hybrid with GPU embedding: dense, fusion                                            |     2 | ~20–30 min |    ~1 h |
| Upgraded recipe at 64K, 32K and 80K, after the ladder                               |     3 | ~20–30 min | ~1.25 h |
| Grading with Qwen3.6, after each run                                                |    18 | ~20 s      |  ~6 min |

- **Final regrade with GPT-5.6 Luna** through OpenRouter: about 1,800 calls,
  under an hour with the runs regraded in parallel (3–8 hours one at a time).
  The cost depends on how many reasoning tokens Luna uses.
- **How it runs:** one command, `bash adarsh-rlms/jarvis/run_suite.sh aa_lcr <suite-id>`,
  resumable if resubmitted.

## LongBench-v2

LongBench-v2 takes about 6 hours on Jarvis for its 17 recipe-independent runs,
plus 1.7 hours for the 3 upgraded-recipe runs. Every run uses the same 221
questions: original rows whose full prompt is over the 64K budget but within
262K, averaging 137K tokens.

| Runs                                                           | Count | Time each |  Total |
| -------------------------------------------------------------- | ----: | --------- | -----: |
| Full context at 262K                                           |     1 | ~31 min   | ~0.5 h |
| Direct at 64K: middle, marked middle, head                     |     3 | ~13 min   | ~0.7 h |
| Hybrid without embedding at 64K: position, random seed 0, BM25 |     3 | ~14 min   | ~0.7 h |
| Hybrid with embedding at 64K: dense, fusion, ladder steps 1–4  |     6 | ~32 min   | ~3.2 h |
| Budget sweep: middle and position at 32K and 128K              |     4 | 6–26 min  | ~1.1 h |
| Upgraded recipe at 64K, 32K and 128K, after the ladder         |     3 | 25–45 min | ~1.7 h |

- **Basis:** the August Jarvis runs: about 1.7 s per question at 32K tokens,
  3.5 s at 64K, 7.5 s at 128K and 17 s at 200–270K, with one-letter answers.
  Embedding took 36 minutes there; scaled to these 30 million document tokens
  that is 10–20 minutes per run, and the table assumes about 18.
- **Sample size:** 221 questions, all over the 64K budget, so every question
  separates policies at 64K. At 128K, 115 of them fit whole.
- **How it runs:** `bash adarsh-rlms/jarvis/run_suite.sh longbench <suite-id>`
  from the login node submits one `client-gpu` job; it resumes when resubmitted.

## MRCR: why the sample matters

Only the 533K and 1.06M conversations separate policies, so their question count
sets what MRCR can detect. They hold 602 of MRCR's 953 questions; the planned
suite uses 60 of them.

| Conversation | Questions available | Used per run (planned) | Dense vs truncation disagreed (30 September) |
| ------------ | ------------------: | ---------------------: | -------------------------------------------: |
| 133K         |                 133 |                     30 |                                      0 of 30 |
| 267K         |                 218 |                     30 |                                      1 of 30 |
| 533K         |                 301 |                     30 |                                     14 of 30 |
| 1.06M        |                 301 |                     30 |                                     10 of 30 |

At 262K the 133K conversation fits whole, and truncation keeps 97% of the 267K
one, so every policy sees nearly the same prompt there.

Questions needed at 533K and 1.06M combined, for 80% power at p < 0.05 with the
exact paired test:

| Difference to detect | Conditions disagree on 15% of questions | 25% |                     40% |
| -------------------- | --------------------------------------: | --: | ----------------------: |
| 20 points            |                                       – |  50 |                      90 |
| 15 points            |                                       – | 100 |                     150 |
| 10 points            |                                     130 | 210 |                     330 |
| 5 points             |                                     500 | 820 | 1,300 (more than exist) |

- **Headline, truncation vs retrieval:** about 27 points on the long
  conversations in September; the planned 60 questions already give 92% power.
- **Structure or selection, position and random vs dense:** likely 10–15
  points, which needs 100–330 questions.
- **Ladder steps and BM25 vs fusion:** likely about 5 points, which even all 602
  questions cannot reliably resolve.

## MRCR sample size

All 20 conditions run on the same questions, so the choice is one number: the
questions taken from each conversation (`MRCR_ROWS_PER_SOURCE`). 30 is the
default and 0 takes all 953. The 133K and 267K conversations add no separating
power, so the informative questions are the 2 × N from the two long
conversations, up to 602. With one executor at a time, the time below is also
the wall clock.

| Questions per conversation | Questions in all | Informative (long) | Time, 20 conditions | Smallest difference resolved |
|---|---:|---:|---:|---|
| **30 (first suite)** | 120 | 60 | ~14 h | ~20–25 points |
| 60 | 240 | 120 | ~27 h | ~15–20 points |
| 100 | 400 | 200 | ~46 h | ~15 points |
| 200 | 733 | 400 | ~87 h | ~10 points |
| **All (0), possible follow-up** | 953 | 602 | ~117 h | ~10 points; 5 points stays out of reach |

- **Time per run:** at the 262K budget, ~0.9 h at 30, ~2.9 h at 100, ~7.5 h at all.
  The 6 sweep runs (32K, 64K and 128K) add ~1.5 h at 30, ~4.9 h at 100, ~11.6 h at
  all; the other 14 conditions run at 262K.
- **Upgraded recipe:** adds one 262K run and three sweep runs: ~1.6 h at 30,
  ~5.4 h at 100, ~13 h at all.
- **One value per suite.** A different N selects different questions, so runs
  from different values cannot be paired. A full run is its own suite:
  `MRCR_ROWS_PER_SOURCE=0 bash adarsh-rlms/jarvis/run_suite.sh mrcr <new-suite-id>`.
  It outlasts one executor job's 24 hours, so it takes several executor jobs and
  resubmissions.
- **Speed on Jarvis is estimated from Neselab:** about 31 s per question at the
  full budget. MRCR has never run on Jarvis; the first suite measures it.

## Decisions

- [x] Platform: all three benchmarks on Jarvis, one suite at a time
- [x] MRCR questions per conversation: 30 for the first suite; all 953 as a
      possible follow-up
- [ ] Order of the suites; LongBench-v2 first shakes out the setup on the
      shortest suite, since it has run on Jarvis before
- [ ] OpenRouter credits for the Luna regrade of AA-LCR, and the key in
      `adarsh-rlms/.env` on Jarvis

Choosing the recipe: keep a ladder step only if it helps on both MRCR and
LongBench-v2; otherwise keep the simpler setting, and report every step with
its interval.
