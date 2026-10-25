# Run times for AA-LCR, MRCR and LongBench-v2: 25 October 2026

AA-LCR and LongBench-v2 are settled: about 5.5 GPU hours on Neselab after
Monday's reboot, and about 8 on Jarvis. MRCR needs a decision, because its
sample size sets what it can detect and how long it runs.

- **AA-LCR:** 18 runs, all 100 questions each. About 4 hours now and 1.25 hours
  once the upgraded recipe is fixed.
- **LongBench-v2:** 20 runs on Jarvis, 221 questions each. About 6 hours now and
  1.7 hours once the recipe is fixed, in one GPU client job against one
  executor job.
- **MRCR:** the planned suite (20 conditions, 120 questions each) takes 14–15
  hours but only resolves differences of about 20 points. Larger samples of the
  two long conversations take 26–75 GPU hours, which is practical mainly on
  Jarvis with parallel jobs.
- **To decide on Monday:** the MRCR sample option ([A–D](#mrcr-options)) and
  whether MRCR runs on Neselab or Jarvis.

The run lists are in the [workshop to-do list](workshop_paper_todo.md#experiment-list);
the commands are in the [AA-LCR](linux/AA_LCR_SUITE_RUNBOOK.md),
[MRCR](linux/MRCR_SUITE_RUNBOOK.md) and
[LongBench-v2](jarvis/LONGBENCH_SUITE_RUNBOOK.md) suite runbooks.

## How the times were estimated

Every estimate scales the 30 September Neselab runs: Qwen3.6-35B-A3B on vLLM,
one 96 GB RTX PRO 6000, temperature 0.

| Measurement | Value | From |
|---|---|---|
| AA-LCR, direct at 64K | 13.5 min for 100 questions (7.7 s each) | middle truncation run |
| AA-LCR, full context at 262K | 17.3 min for 100 questions | full-prompt run |
| AA-LCR, dense hybrid at 64K | 165 min, of which 151 min CPU embedding | dense run |
| AA-LCR, grading with Qwen3.6 | about 20 s per run | three regrades |
| MRCR, conversation that fits (133K) | about 12 s per question | direct and dense runs |
| MRCR, at the full 258K-token budget | about 31 s per question | direct and dense runs |
| MRCR, run of 120 questions at 262K | 57 min direct, 117 min dense (44 min CPU embedding) | direct and dense runs |
| Repeat run of the same configuration | 0 of 40 answers differed | 29 vs 30 September runs |

Two things are estimated rather than measured:

- **GPU embedding.** The new suites embed on the GPU instead of the CPU; it has
  not been timed on Neselab. The estimates assume minutes per run, not hours.
- **Jarvis speed for MRCR.** MRCR has never run on Jarvis. The August
  LongBench-v2 runs there took about 17 s per question at 200–270K tokens, but
  those answers were one letter long; MRCR answers run to about 400 tokens.

## AA-LCR

AA-LCR takes about 4 hours for its 15 recipe-independent runs, plus 1.25 hours
for the 3 upgraded-recipe runs later. Every run uses all 100 questions, so there
is no sampling choice to make.

| Runs | Count | Time each | Total |
|---|---:|---|---:|
| Direct: full context, middle, marked middle, head; middle at 32K and 80K | 6 | 12–17 min | ~1.4 h |
| Hybrid without embedding: position, random seeds 0–2, BM25; position at 32K and 80K | 7 | ~14 min | ~1.6 h |
| Hybrid with GPU embedding: dense, fusion | 2 | ~20–30 min | ~1 h |
| Upgraded recipe at 64K, 32K and 80K, after the ladder | 3 | ~20–30 min | ~1.25 h |
| Grading with Qwen3.6, after each run | 18 | ~20 s | ~6 min |

- **Final regrade with GPT-5.6 Luna** through OpenRouter: about 1,800 calls,
  under an hour with the runs regraded in parallel (3–8 hours one at a time).
  The cost depends on how many reasoning tokens Luna uses.
- **How it runs:** one command, `bash linux/run_aa_lcr_suite.sh <suite-id>`,
  resumable if interrupted.
- **Blocked until Monday:** Neselab's NVIDIA driver was upgraded under the
  running kernel, so no new GPU job can start until the server is rebooted.

## LongBench-v2

LongBench-v2 takes about 6 hours on Jarvis for its 17 recipe-independent runs,
plus 1.7 hours for the 3 upgraded-recipe runs. Every run uses the same 221
questions: original rows whose full prompt is over the 64K budget but within
262K, averaging 137K tokens.

| Runs | Count | Time each | Total |
|---|---:|---|---:|
| Full context at 262K | 1 | ~31 min | ~0.5 h |
| Direct at 64K: middle, marked middle, head | 3 | ~13 min | ~0.7 h |
| Hybrid without embedding at 64K: position, random seed 0, BM25 | 3 | ~14 min | ~0.7 h |
| Hybrid with embedding at 64K: dense, fusion, ladder steps 1–4 | 6 | ~32 min | ~3.2 h |
| Budget sweep: middle and position at 32K and 128K | 4 | 6–26 min | ~1.1 h |
| Upgraded recipe at 64K, 32K and 128K, after the ladder | 3 | 25–45 min | ~1.7 h |

- **Basis:** the August Jarvis runs: about 1.7 s per question at 32K tokens,
  3.5 s at 64K, 7.5 s at 128K and 17 s at 200–270K, with one-letter answers.
  Embedding took 36 minutes there; scaled to these 30 million document tokens
  that is 10–20 minutes per run, and the table assumes about 18.
- **Sample size:** 221 questions, all over the 64K budget, so every question
  separates policies at 64K. At 128K, 115 of them fit whole.
- **How it runs:** `bash adarsh-rlms/jarvis/run_longbench_suite.sh <suite-id>`
  from the login node submits one `client-gpu` job; it resumes when resubmitted.

## MRCR: why the sample matters

Only the 533K and 1.06M conversations separate policies, so their question count
sets what MRCR can detect. They hold 602 of MRCR's 953 questions; the planned
suite uses 60 of them.

| Conversation | Questions available | Used per run (planned) | Dense vs truncation disagreed (30 September) |
|---|---:|---:|---:|
| 133K | 133 | 30 | 0 of 30 |
| 267K | 218 | 30 | 1 of 30 |
| 533K | 301 | 30 | 14 of 30 |
| 1.06M | 301 | 30 | 10 of 30 |

At 262K the 133K conversation fits whole, and truncation keeps 97% of the 267K
one, so every policy sees nearly the same prompt there.

Questions needed at 533K and 1.06M combined, for 80% power at p < 0.05 with the
exact paired test:

| Difference to detect | Conditions disagree on 15% of questions | 25% | 40% |
|---|---:|---:|---:|
| 20 points | – | 50 | 90 |
| 15 points | – | 100 | 150 |
| 10 points | 130 | 210 | 330 |
| 5 points | 500 | 820 | 1,300 (more than exist) |

- **Headline, truncation vs retrieval:** about 27 points on the long
  conversations in September; the planned 60 questions already give 92% power.
- **Structure or selection, position and random vs dense:** likely 10–15
  points, which needs 100–330 questions.
- **Ladder steps and BM25 vs fusion:** likely about 5 points, which even all 602
  questions cannot reliably resolve.

## MRCR options

The options range from 15 to 117 GPU hours; on Jarvis, parallel jobs divide the
wall clock by the number of jobs held at once.

| Option | What runs | GPU hours | Neselab | Jarvis, 2 jobs at once | Jarvis, 4 jobs at once | Smallest difference resolved |
|---|---|---:|---:|---:|---:|---|
| **A. Planned suite** | 20 conditions × 120 questions | ~15 | ~15 h | ~8 h | ~4 h | ~20–25 points |
| **B. A + key conditions** | A, plus 5 key conditions on all 602 long-conversation questions | ~40 | ~40 h | ~20 h | ~10 h | ~10 points for the key comparisons |
| **C. All 262K conditions** | 14 conditions on the 602 long questions; one short reference run for 133K and 267K; budget sweep at 120 questions | ~75 | ~3 days | ~38 h | ~19 h | ~10 points for every 262K condition |
| **D. Everything** | 20 conditions × all 953 questions | ~117 | ~5 days | ~59 h | ~30 h | as C; the extra short-conversation questions add no separating power |

- **Key conditions:** middle truncation, dense, position, random seed 0 and
  fusion, plus the upgraded recipe once it exists.
- **Time per condition at 262K:** ~55 min for 120 questions, ~1.7 h for 200
  long questions, ~5.2 h for all 602 long questions, ~7.5 h for all 953.
- **Upgraded recipe:** adds ~2 h to A, or ~5 h per condition run on 602
  questions.
- **Jarvis figures** assume Neselab's speed per question and exclude queue time.
- **Options B and C** need two small additions to `linux/run_mrcr_suite.sh`:
  questions per conversation, and a long-conversations-only selection.

## Moving MRCR to Jarvis

Jarvis shortens MRCR only through parallel jobs; each job serves the model on 4
L40S GPUs, probably at about Neselab's speed per question. It is worth it for
options B to D if we can hold two or more jobs at once.

- **All MRCR runs on one platform.** At temperature 0 the answers depend on the
  GPU type and the parallel split, so runs on Neselab and Jarvis cannot be
  paired. The references move too, which costs nothing because every run is
  being redone.
- **A Jarvis version of the suite, about half a day.** There the executor and
  the benchmark client are separate Slurm jobs, the MRCR data must be prepared
  on Jarvis, and the MRCR launcher there has never been used. The first job is a
  shakedown.
- **Job limits are not a problem.** The 24-hour limit is well above the ~5.2
  hours a full-sample condition takes.
- **Contention with LongBench-v2,** which needs one executor job for about 6
  hours, plus 1.7 hours later. Queue time is unknown.
- **Not affected by Neselab's driver problem,** so MRCR could start before
  Monday.

Open questions:

- How many 4-GPU L40S jobs can we hold at once?
- How long does an MRCR question take on Jarvis? One short test job would
  measure it.

## Recommendation and decisions

Recommended: MRCR option B, on Jarvis if we can hold two or more jobs at once,
otherwise on Neselab after AA-LCR (about 40 hours).

- **Why B over C:** the extra 35 hours in C mostly go to ladder steps and
  retrieval variants. Their likely 5-point differences stay unresolvable even at
  602 questions, so the data barely changes what we can claim.
- **Choosing the recipe:** keep a ladder step only if it helps on both MRCR and
  LongBench-v2; otherwise keep the simpler setting, and report every step with
  its interval.
- **AA-LCR** needs no decision: it runs on Neselab after the reboot.

Decisions for Monday:

- [ ] MRCR sample option: A, B, C or D
- [ ] MRCR platform: Neselab or Jarvis
- [ ] If Jarvis: how many 4-GPU L40S jobs we can hold at once
- [ ] Neselab reboot, and NVIDIA packages held back from automatic upgrades
- [ ] OpenRouter credits for the Luna regrade of AA-LCR

## Team notes

Add notes from before and during the Monday discussion here.
