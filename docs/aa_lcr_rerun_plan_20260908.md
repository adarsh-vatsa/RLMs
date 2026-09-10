AA-LCR baseline rerun plan — 2026-09-08

Establish a full-context Qwen3.6-35B-A3B non-thinking baseline with sufficient
answer space, and measure the separate effects of output length, grading, and
dataset corrections. The [investigation](aa_lcr_accuracy_investigation_20260908.md)
found 35 answers at the 512-token ceiling, four clear percentage-equivalence
grading errors, and 16 upstream answer-key changes.

**Completed: dataset versioning and preservation.**

- Both releases are pinned by immutable Hugging Face revision and file hashes
  in `aa_lcr/dataset.py`. v1.0.0 remains at revision
  `bdae010bbce259820c0e34c1d7cce210d966fb75`; v1.1 uses
  `9a77ef56b717057ade24ceab4d273712a0b4f19e`.
- The legacy defaults and `benchmark_data/aa_lcr/` remain unchanged. Prepared
  v1.1 separately at `benchmark_data/aa_lcr/v1.1/`, with 100 questions, 30
  document sets, and 229 documents. Preparation refuses to replace another
  revision or a different existing CSV.
- New preflights, run manifests, and evaluation reports identify the dataset
  version. Known releases must match the pinned hashes; old manifests without
  a version field are identified from their revision. Unknown revisions are
  labeled custom. Reports also expose the grader model, grader prompt version,
  and output budget.
- Existing comparisons already reject differing dataset revisions, hashes,
  grader prompts, models, and output budgets. Regression tests cover dataset
  and grader mismatches; historical artifacts need no migration.
- Dataset version describes the inputs, not official evaluation compliance.
  Selecting v1.1 does not automatically select the v1.1 grader prompt.

**1. Implement the generation-budget change.**

Replace the fixed-512 rejection with positive-value and total-context checks.
Use 16,384 as the default for the 262K cells and retain 512 for the existing
64K cells until their input budget is deliberately redesigned. Explicit
`--max-output-tokens 512` must remain available for reproduction. Keep
LongBench's independent settings unchanged.

Preserve API `finish_reason` in `Completion` and per-answer artifacts. Report
length terminations separately from API errors and incorrect grades; do not
reinterpret old missing termination reasons as successful stops. Record the
requested output budget and actual output use. Add tests for 16K request
forwarding, invalid budgets, context overflow, and length-terminated responses.

Allow the Jarvis AA-LCR launcher to forward the selected dataset paths, output
budget, and run ID. Its printed command must make these settings reviewable.
Give each new run a unique ID that includes dataset version and output budget;
retain the exact revision, hashes, and effective settings in its manifest.

**2. Implement versioned grading and regrading.**

Preserve the current local grader prompt and its existing hash. Add a distinct
prompt version for the exact published v1.1 system/user messages and JSON
verdict contract. Model, reasoning settings, prompts, and answer-key release
must be recorded independently. Updating only the answer keys must not silently
change the grader.

Add a regrading command that reads saved predictions, joins by question ID,
checks question/document identity, and writes a new destination. It must record
the source prediction file hash and both source and grading dataset releases;
reject missing, duplicate, or mismatched questions and avoid modifying source
artifacts. Validate verdict parsing and long-answer handling without model
calls. Include representative percentage-equivalence cases in a live grader
sanity check; mocked tests alone cannot establish grading quality.

For official alignment, confirm access to the published checker, currently
GPT-5.6 Luna at medium reasoning, and its provider-specific request contract.
Do not assume the current Qwen-only, constrained-label request works for it.
If using the local Qwen checker, label the result as a local evaluation and
record that departure. Recheck the official methodology before launching.

**3. Validate the HPC services.**

Reuse the Qwen3.6 executor allocation: four L40S GPUs, tensor parallelism four,
262,144 total context, one concurrent sequence, and the existing chunked-prefill
settings. Check the live queue and actual startup logs: successful startup and
capacity for at least one 262,144-token request are required. Record the served
weights revision, vLLM version, dtype, effective generation defaults, template
hash, and launch arguments. Configured resources are not proof of live capacity.

The largest observed prompt plus the proposed output allowance is
122,112 + 16,384 = 138,496 tokens. Even the configured maximum input budget
fits: 240,000 + 16,384 = 256,384 < 262,144.

If retaining the local grader, start it with a 32,768-token window on the
existing two-GPU allocation, then verify startup memory capacity. Tokenize the
entire grading request with the grader's tokenizer, including question,
reference, candidate, instructions, and output allowance. The current 16K
grader window cannot safely accept a 16K answer plus overhead. The earlier
runbook instruction to keep the grader at 16K assumed short answers.

Smoke-test formerly cut-off questions 8 and 44, plus a question from the largest
document set. Confirm full input preservation, non-thinking mode, output
budget forwarding, termination metadata, and successful grading. Measure
latency before setting job walltimes; 16K is a maximum, not mandatory output.
Proceed to the full run only after these checks pass.

**4. Run controlled comparisons.**

Use the saved direct_262k answers for the first three new grading conditions.
Use all 100 questions, including previously credited answers.

| Condition | Generation | Answer keys | Checker | Grader prompt | Comparison isolates |
|---|---|---|---|---|---|
| Historical | Saved 512-token answers | v1.0.0 | Local Qwen3.5 | Existing local | Original 44% |
| A | Same saved answers | v1.0.0 | Official checker | Existing local | Checker change |
| B | Same saved answers | v1.0.0 | Official checker | v1.1 | Prompt change versus A |
| C | Same saved answers | v1.1 | Official checker | v1.1 | Key change versus B |
| D | New 16K-token answers | v1.1 | Official checker | v1.1 | Output budget versus C |

Keep generation prompt text, document order, executor weights, serving
configuration, temperature, and thinking setting fixed for C versus D. Recover
the old serving configuration where possible. If it cannot be recovered,
generate a fresh 512-token control on the same service as D and label the
historical comparison as confounded by unverified serving differences.

Report paired per-question changes, gains and regressions, output lengths,
length-termination counts, invalid grades/API failures, and time. Audit the
four percentage cases and revised keys, including question 25 where the new
key can remove credit. Use document-set-clustered intervals, as in the earlier
comparison. Do not count the identical historical hybrid_262k answers as an
independent repeat.

After the diagnostic baseline, use three independent full passes to match the
published repeat count. Keep repeat IDs and serving settings attributable;
report aggregate pass@1, not best-of-three. The existing four-cell comparator
requires matched contracts and should not be weakened to compare these
deliberately different conditions; the diagnostic comparison needs an explicit
join and a table of the settings that differ.

**5. Revisit 64K retrieval after the baseline.**

A 16K output allowance leaves at most 49,152 input tokens inside a 65,536-token
window; use any chosen safety margin consistently in both 64K cells. This
changes retrieval packing and truncation, so treat the resulting four-cell
comparison as a new experiment. Do not mix it with the old 60,000-input,
512-output runs or reuse the old retention ratio as evidence for the new setup.

**Completion criteria.** The dataset and grader identities are explicit;
historical results remain readable and unchanged; a full 100-question 16K
baseline completes with all failures and length stops accounted for; the
controlled comparison separates the measured changes and discloses remaining
serving uncertainty. There is no target accuracy to tune toward.

The software for steps 1, 2, and the diagnostic comparisons in step 4 is
implemented. See [runnable commands](aa_lcr_rerun_commands.md). The new
`regrade` command also supports selected questions for a grader sanity check;
`compare_conditions` supports multiple independent runs per condition and
averages their pass rates. The existing four-cell comparison remains strict.

HPC startup checks, live grader sanity checks, actual inference/regrading,
three-pass results, and the later 64K input-budget redesign remain for the
user's runs. No jobs or live grading requests were submitted. Local validation
uses Python 3.13 unittest discovery and mocked API responses; it does not
establish live grading quality or accuracy recovery.

References: [official testing methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking),
[pinned v1.1 documentation](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/blob/9a77ef56b717057ade24ceab4d273712a0b4f19e/README.md),
[AA-LCR runbook](jarvis/AA_LCR_RUNBOOK.md).
