AA-LCR rerun commands

Run these from the repository root. The examples do not submit jobs unless they
explicitly invoke the Jarvis launcher. Start your model services separately.

**Prepare and preflight.**

```bash
uv run python -m aa_lcr.prepare_dataset --dataset-version 1.1

uv run python -m aa_lcr.run_benchmark --experiment direct_262k \
  --questions-csv benchmark_data/aa_lcr/v1.1/AA-LCR_Dataset.csv \
  --documents-root benchmark_data/aa_lcr/v1.1/extracted_text/lcr \
  --dataset-manifest benchmark_data/aa_lcr/v1.1/dataset_manifest.json \
  --max-output-tokens 16384 --preflight-only
```

Preflight checks generation inputs without contacting model services. The
262K cells default to 16,384 output tokens; the 64K cells default to 512.
Explicit `--max-output-tokens 512` reproduces the previous allowance. Input
budgets remain 240,000 and 60,000 respectively; a 16K output allowance is
rejected for the existing 64K cells because the combined budget would overflow.

The default dataset remains v1.0.0, and the default grader remains the local
legacy prompt. Select the three v1.1 paths and `--grader-prompt-version
aa_lcr_equality_v1.1` explicitly. Dataset and grader versions are independent.

**Local grader and generation.**

Set `OPENAI_COMPAT_EXECUTOR_BASE_URL` and `OPENAI_COMPAT_EVALUATOR_BASE_URL` to
your running services. The local grader's total context must be 32,768 to match
the new default `--grader-context-window 32768`. Each grading request is counted
with its own tokenizer before sending; oversized answers are recorded as
grading errors without truncation. The legacy grader requests eight output
tokens, and the v1.1 JSON grader requests 64. Override those with
`--grader-max-output-tokens` when necessary.

This submits a small executor smoke test through Jarvis:

```bash
AA_LCR_DATA_DIR=benchmark_data/aa_lcr/v1.1 \
AA_LCR_RUN_ID=v1.1_out16384_smoke \
bash jarvis/run_aa_lcr.sh direct_262k \
  --question-ids 8,44,4 \
  --grader-prompt-version aa_lcr_equality_v1.1
```

Use `AA_LCR_LAUNCH_DRY_RUN=1` to inspect the command without submitting. IDs 8
and 44 previously ended mid-answer; ID 4 had the largest historical prompt
(122,112 tokens). Confirm that with the new preflight output. Remove
`--question-ids` for the full 100-question run. All additional launcher arguments
are passed as quoted arguments to `aa_lcr.run_benchmark`.

Run three full passes with distinct `AA_LCR_REPEAT_ID` values 1, 2, and 3. Omit
`AA_LCR_RUN_ID` for automatic IDs containing dataset version, output allowance,
repeat ID, and a timestamp. Existing run directories are never overwritten.
Record actual serving details in a JSON file and pass its path via
`AA_LCR_SERVING_METADATA` or `--serving-metadata`. Include the weights revision,
vLLM version, dtype, tensor parallelism, launch arguments, and effective
generation settings; omit credentials. Missing metadata remains explicitly
unverified rather than inferred from the client configuration.

**Hosted grading.**

The published v1.1 checker is GPT-5.6 Luna with medium reasoning. The supported
API model ID is `gpt-5.6-luna`. Supply `OPENAI_API_KEY` through your environment;
the code does not load or write keys into artifacts. The hosted profile uses
Chat Completions with `max_completion_tokens`, `reasoning_effort`, and no
temperature or Qwen-specific template settings. Its default 16,384-token
grader output ceiling includes reasoning; a length-terminated verdict is
invalid and reported separately. This ceiling is a local setting, not a claim
about the unpublished output budget used by AA's checker. API access and
grading quality still require your live check.

Use these flags on direct generation or regrading:

```text
--evaluator-model gpt-5.6-luna
--evaluator-base-url https://api.openai.com/v1
--evaluator-api-key-env OPENAI_API_KEY
--grader-api-style openai
--grader-reasoning-effort medium
--grader-prompt-version aa_lcr_equality_v1.1
```

Hosted grading does not load a local tokenizer; context enforcement is handled
by the provider. For local vLLM, grading uses temperature zero and non-thinking
mode. Hybrid generation still needs a local evaluator for cache verification;
use hosted **regrading** on its saved answers instead of pointing that verifier
at a hosted reasoning model. Executor and evaluator credential environment
variables can be different (`--api-key-env` versus `--evaluator-api-key-env`).

**Regrade saved answers first.**

```bash
uv run python -m aa_lcr.regrade \
  --source-run benchmark_artifacts/aa_lcr/direct_262k/20260830T235804Z \
  --output-dir benchmark_artifacts/aa_lcr/regrades/v1.1_luna_sanity \
  --questions-csv benchmark_data/aa_lcr/v1.1/AA-LCR_Dataset.csv \
  --documents-root benchmark_data/aa_lcr/v1.1/extracted_text/lcr \
  --dataset-manifest benchmark_data/aa_lcr/v1.1/dataset_manifest.json \
  --evaluator-model gpt-5.6-luna \
  --evaluator-base-url https://api.openai.com/v1 \
  --grader-api-style openai --grader-reasoning-effort medium \
  --grader-prompt-version aa_lcr_equality_v1.1 \
  --question-ids 60,87,89,94
```

These four answers should receive credit for percentage equivalence. Inspect
the saved raw grader responses. Add `--validate-only` to check source integrity
without loading a grader, calling an API, or creating output. Remove
`--question-ids` and choose a fresh output directory for all 100 answers.
The full source is validated even when selecting a sanity-check subset.

For the controlled conditions in the [plan](aa_lcr_rerun_plan_20260908.md):

| Condition | Dataset paths | Grader prompt | Checker |
|---|---|---|---|
| A | Legacy defaults | `aa_lcr_equality_v1` | Hosted Luna, medium |
| B | Legacy defaults | `aa_lcr_equality_v1.1` | Hosted Luna, medium |
| C | Explicit v1.1 paths | `aa_lcr_equality_v1.1` | Hosted Luna, medium |
| D | Generate new 16K answers using v1.1 paths | `aa_lcr_equality_v1.1` | Same as C |

Regrading copies no results back to the source directory. It records source
prediction, bridge, and manifest hashes, source and grading releases, selected
IDs, raw judgments, termination reasons, and grader usage. Regrading totals
cover grading only; per-row executor usage is retained as generation provenance.
Unknown historical termination reasons remain unknown. API failures or invalid
verdicts produce a null complete-set accuracy, with partial accuracy separately
labeled. Inspect the manifest before citing a score.

**Compare conditions or average repeats.**

```bash
uv run python -m aa_lcr.compare_conditions \
  --baseline benchmark_artifacts/aa_lcr/regrades/B \
  --candidate benchmark_artifacts/aa_lcr/regrades/C \
  --output benchmark_artifacts/aa_lcr/comparisons/B_vs_C.json
```

Provide multiple directories after `--baseline` or `--candidate` to aggregate
independent repeats. Repeats within a condition must share settings and
question IDs. Scores average all attempts, never choose the best answer.
Duplicate runs or multiple regrades of the same generation cannot masquerade
as independent generation repeats. Every grade must be valid for comparison.

The report includes settings that differ, paired changes and answers,
document-set-clustered confidence intervals, output lengths, known termination
reasons, and elapsed time. A regrade's elapsed time covers grading only; compare
generation timing using the source runs. Historical identical direct/hybrid
answers are not independent evidence. The four-cell `compare_runs` command
retains strict matching rules; use `compare_conditions` for intentional
methodology changes.

Validation command:

```bash
uv run python -m unittest discover -s test -p 'test_aa_lcr*.py'
```

Sources: [AA-LCR v1.1 prompt and keys](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/blob/9a77ef56b717057ade24ceab4d273712a0b4f19e/README.md),
[Luna model](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[Chat Completions request parameters](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
