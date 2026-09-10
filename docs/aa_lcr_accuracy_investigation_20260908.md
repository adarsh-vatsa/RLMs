Investigation date: 2026-09-08.

**The 44% result is reproducible from the saved artifacts, but it is not a
methodology-matched estimate of the official AA-LCR score.** The strongest
observed problems are a restrictive answer budget and incorrect equality
grades. The current benchmark also uses revised answer keys. These findings
identify concrete sources of lost credit, but do not establish how much of
the complete leaderboard gap each one explains.

The run under review is
[`direct_262k/20260830T235804Z`](../benchmark_artifacts/aa_lcr/direct_262k/20260830T235804Z/manifest.json):
Qwen/Qwen3.6-35B-A3B, temperature zero, thinking disabled, 100 questions,
44 correct. The corresponding hybrid 262K run has identical answers and grades;
it is not an independent replication.

**Official comparison.** The user reported 63.4%. The public
[model page](https://artificialanalysis.ai/models/qwen3-6-35b-a3b-non-reasoning)
fetched during this investigation instead contains `lcr: 0.643333333333333`
for the exact slug `qwen3-6-35b-a3b-non-reasoning`, or 64.3%. This does not
establish what the page showed when the user checked it. The current
[leaderboard](https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning)
is labeled AA-LCR v1.1.

| Setting | Saved local run | Published current methodology |
|---|---|---|
| Maximum answer tokens | 512 | 16,384 for non-reasoning models, subject to model limits |
| Dataset | v1.0.0, revision `bdae010bbce259820c0e34c1d7cce210d966fb75` | v1.1, with 16 revised keys |
| Equality checker | Qwen3.5-35B-A3B, non-thinking | GPT-5.6 Luna, medium |
| Grader instructions | User-only prompt; two constrained labels | System guidance plus delimited user prompt and JSON verdict |
| Repeats | One per question | Three, aggregated pass@1 |

Sources: [testing parameters and AA-LCR methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking),
[versioned dataset documentation](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/blob/9a77ef56b717057ade24ceab4d273712a0b4f19e/README.md).
The dataset repository identifies v1.1 as a September 2026 update and explicitly
disallows direct comparison with v1.0.0. The older run predates that update.

**Answer budget: the most substantial observed execution problem.**
[`run_benchmark.py`](../aa_lcr/run_benchmark.py) sets
`DEFAULT_MAX_OUTPUT_TOKENS = 512` and rejects any other value, even if supplied
through `--max-output-tokens`. Non-thinking mode still permits explanations
and calculations in the answer text; it does not make 512 tokens sufficient.

| Full-context answers | Count | Correct | Incorrect |
|---|---:|---:|---:|
| Used exactly 512 output tokens | 35 | 6 | 29 |
| Used fewer than 512 output tokens | 65 | 38 | 27 |
| Total | 100 | 44 | 56 |

Question 8 is a particularly concrete example: the answer locates quarterly
values 667,804, 638,969, 619,786, and 610,994, whose sum is the reference answer
2,537,553, but stops while listing the quarters. Question 44 stops just as it
starts enumerating analysts common to two calls. Question 33 begins correcting
its initial answer toward the reference currency, then stops.

The API adapter discards `finish_reason`, so exact limit hits are a cutoff
proxy rather than recorded server termination reasons. The visibly unfinished
answers corroborate the problem. These 29 incorrect answers are candidates
for recovery, not 29 guaranteed additional correct answers: some contain
substantive mistakes, and difficult questions naturally need more output.

**Grading: four clear false negatives already visible without regeneration.**
Each answer below was marked INCORRECT despite matching the reference value
in the units requested by its question.

| Question ID | Pinned reference | Saved final answer | v1.1 reference |
|---|---|---|---|
| 60 | 0.09 | 9 percentage points | 9% |
| 87 | 0.1 | 10 percentage points | 10% |
| 89 | 0.318 | 31.8 percentage points | 31.8% |
| 94 | 0.14 | 14% | 14% |

These are four points of erroneous penalties under semantic equivalence.
They are not a complete regrade or a corrected aggregate score.
[`prompting.py`](../aa_lcr/prompting.py) supplies no explicit unit-equivalence
guidance; it also omits the boundaries in the published legacy grader prompt.
Valid output labels and zero API errors do not imply accurate judgments.

**Dataset corrections have effects in both directions.** Comparing the local
CSV with upstream revision `9a77ef56b717057ade24ceab4d273712a0b4f19e` found only
16 changed answer fields; all questions, document lists, and other CSV fields
are unchanged. Changed IDs: 10, 21, 25, 26, 28, 30, 40, 60, 67, 76, 78, 86,
87, 89, 90, and 94. Local loading strips incidental surrounding whitespace.

For example, question 30's saved answer is No, previously rejected but
consistent with the revised key. Question 25's saved No answer was previously
credited, but the revised key is Yes. Question 67 was credited even though
its 512-token answer stops before finishing the second requested calculation.
A complete audit must inspect credited answers as well as rejected ones.

**Checks that passed.** All 100 requests used `direct_fit`, with no input
truncation or cache hits. Full prompts span 87,847–122,112 tokens, below the
240,000-token input budget. API prompt-token usage equals the locally rendered
count on all 100 requests. Both pinned download hashes match the preparation
manifest; all 229 referenced extracted files match the archive byte-for-byte
after applying its filename decoding; all saved source-scope hashes match
the loaded local documents. The loader follows CSV document order. There is
no evidence here of missing input documents or a context-accounting error.

The manifests record the executor name and tokenizer-template hash, but not
the served weights revision, vLLM version, or effective server sampling
configuration. Provider equivalence remains unverified; it is a secondary
investigation path if a matched rerun still underperforms.

**Recommended controlled follow-up.**

1. Regrade all 100 saved full-context answers using the current official
   checker and instructions. Run it once against the old keys and once against
   the revised keys to distinguish grader changes from answer-key changes.
   Preserve both outputs separately from the historical artifacts.
2. Allow a positive configurable output budget and persist `finish_reason`.
   Rerun `direct_262k` at 16,384 tokens, keeping generation prompts, document
   order, executor, and non-thinking settings fixed. Grade old and new
   generations with the same checker and keys for the output-budget comparison.
3. Use all 100 questions for the resulting score; a cutoff-question subset is
   useful only as a diagnostic. Use three repeats for leaderboard alignment
   and record the serving configuration.
4. Revisit the four-cell retrieval comparison after establishing the baseline.
   The 262K budget accommodates 240,000 + 16,384 tokens; the existing 64K
   budget does not accommodate 60,000 + 16,384. A 64K rerun needs an explicit
   input/output tradeoff and cannot silently inherit the historical contract.

The prior report's 41/44 retention ratio remains a description of that
512-token, locally graded experiment. It should not be generalized to a
baseline with adequate answer space and audited grading. The report's proposed
over-240K extension would not isolate the discrepancies identified here.

No benchmark code, historical artifacts, or pinned inputs were changed, and
no model inference or automated regrading was run for this investigation.
Counts and examples above come from
[`bridge_rows.jsonl`](../benchmark_artifacts/aa_lcr/direct_262k/20260830T235804Z/bridge_rows.jsonl).
