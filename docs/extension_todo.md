# Extension to-do: a second model family and thinking

The workshop deadline moved to 10 December 2026. This extension tests whether the paper's findings hold for a second model family and with thinking enabled, the two gaps the [17 November review](paper_review_20261117.md) left in Limitations. Started 2026-11-25. The paper's own to-do is [workshop_paper_todo.md](workshop_paper_todo.md). Experiment details stay proposals until the plan (see [Status](#status)) is written and dated before the first reported run.

## Configurations

| | Thinking off | Thinking on |
|---|---|---|
| Qwen3.6-35B-A3B | done: suites `20261029` (93 conditions); the no-context baseline to run | to run, on the L40S nodes |
| DeepSeek-V4-Flash-0731 (proposed) | to run, on a gpu-h200 node | to run, on a gpu-h200 node |

The new configurations do not rerun all 93 conditions, and the paper need not report every run; see [Experiments](#experiments-proposal).

## Status

- [x] Feasibility study, 2026-11-25 (this file).
- [x] Hopper access, checked 2026-11-25 with a short script (not in the repo) that asks Slurm, through `sbatch --test-only`, whether a whole node of each GPU partition would be accepted. Membership of `jarvis-users-h100` opens all three Hopper partitions; the Slurm account is `llm_caching`.
  - gpu-h200: 3 nodes of 4 × H200 (141 GB each), 1.5 TB RAM, 24-hour jobs. A whole node was estimated to start the same evening.
  - gpu-h100sxm: 4 nodes of 4 × H100 (80 GB each), 515 GB RAM, 72-hour jobs. A whole node was estimated for the next morning.
  - gpu-h100: 4 nodes of 2 × H100 NVL, 24-hour jobs. Too small for DeepSeek.
- [x] L40S nodes, 2026-11-25: g201 is drained (hardware reseat). g101 and g102, the two nodes that agree byte for byte, are up.
- [x] Experiment matrix, 2026-11-25: [extension_experiments.csv](extension_experiments.csv), one row per condition and one column per model, X when planned or completed.
- [x] Probe jobs on the Hopper nodes, 2026-11-25. Both nodes have driver 580.82.07 (CUDA 13.0), so any recent vLLM build runs, and both reach Hugging Face.
  - g213 (gpu-h200): 4 × H200 NVL (141 GB each), 1.5 TB RAM, 254 GB free on `/local`.
  - g013 (gpu-h100sxm): 4 × H100 80GB HBM3, 515 GB RAM, 1.3 TB free on `/local`.
- [x] Per-user limits, 2026-11-25: the `normal` QOS sets none (no cap on wall time, GPUs, jobs or submissions per user), so two L40S executors, one H200 executor and three one-GPU client jobs can run at once, as the nodes allow.
- [x] Qwen with thinking: code and tests, 2026-11-25 (see [Code](#code-for-qwen-with-thinking-done-2026-11-25)); how to run the pilot and the runs: [THINKING.md](jarvis_suites/THINKING.md).
- [x] Qwen with thinking: pilot, 2026-11-26, on g101 (suite `20261128_think_pilot`); it set the allowance at 81,920 (see [Thinking budget](#thinking-budget-decided-2026-11-25)).
- [ ] Qwen with thinking: the runs (suite `20261130_think`), then the two no-context runs without thinking.
- [ ] DeepSeek: environment and smoke test on one gpu-h200 node, then pilot and runs. Cut-off for a working server: 29 November; after that, the fallback model. The vLLM environment is not installed yet; the commands are under [Smoke-test commands](#smoke-test-commands).
- [ ] Plan: the comparisons each configuration tests, written and dated before the first reported run, as the [12 October plan](workshop_paper_plan_20261012.md) was.
- [ ] Luna grading of the new AA-LCR runs, intervals, tables and the paper text.

## Decisions

### Thinking budget (decided 2026-11-25)

Each condition keeps the input budget it had with thinking off, and the output allowance comes on top of it instead of out of it. The allowance is separate from the input budget, not from the window: the model writes its reasoning and then its answer in the same sequence as the prompt, so the served window must hold all three, and one `max_tokens` caps the reasoning and the answer together. Each request gets the allowance or what the window leaves after its prompt, whichever is smaller, and an answer uses only the tokens it needs. The Qwen executor serves 262,144 tokens. Token counting stays at `enable_thinking=False`, so Qwen's prompts with thinking are byte for byte those of the `20261029` runs. Thinking on and off are then compared on identical inputs, and MRCR evidence recall cannot change, so any effect of thinking shows in exact match given the target. Check the identity per question from `selected_evidence_ranges` and `final_rendered_input_tokens`.

Qwen's allowance is 32,768 or 81,920 tokens, set from the pilot, which runs at 81,920. The Qwen3.6 card recommends 32,768 "for most queries" and 81,920 "for benchmarking on highly complex problems, such as those found in math and programming competitions", and its thinking-mode examples use 81,920. The pilot counts, per policy, the answers that finish between 32,768 and 81,920, which the smaller cap would cut off, and the traces that reach 81,920, probably loops that no cap saves. If the first group is more than a few percent of answers, or concentrated in one policy, the runs use 81,920; otherwise 32,768, which limits a loop to about 4 minutes. Cap hits must stay rare: reasoning cut off at the cap leaves no answer, which scores zero, and with thinking off middle-truncated prompts drew 55% longer answers than full ones, so a tight cap would likely hit middle truncation hardest. Loops are a real risk, because the card's remedy for endless repetition, the presence penalty, is 0 here (see [Decoding with thinking](#decoding-with-thinking-decided-2026-11-25)).

Decided after the pilot (2026-11-26): 81,920. No answer of the pilot's 132 reached it, so it cuts nothing off, and it costs time only on the answers that need it; the pilot had no loops. It also gives the full runs the pilot's requests, so the pilot's questions show whether a seeded run repeats itself without a separate rerun. This sets aside the rule above, which would have picked 32,768 if almost no answer passed it, to stop a loop sooner; the pilot's tables do not show how many passed 32,768, and the full runs count them.

- Windows: with thinking, a condition's window names its input budget. At 64K the prompts stay 61,440 tokens (65,528 on LongBench-v2), and a request can reach 61,440 plus the allowance, 143,360 at 81,920. Every windowed condition fits 81,920 in full (LongBench-v2 at 128K, the tightest, leaves 131,080), and so does AA-LCR full context (its longest prompt leaves 140,032).
- MRCR at 262K: the input budget, 258,048, leaves 4,096 tokens for the reasoning and the answer together on the 90 questions from the three conversations longer than it; the 133K conversation fits whole. Answers there ran up to 950 tokens with thinking off, which would leave about 3,100–3,700 to think. So Qwen's largest MRCR window with thinking is 128K. To keep the models comparable, 128K is MRCR's main window for every new model; the thinking-off runs at 128K exist.
- LongBench-v2 full context: prompts run up to 258,844 tokens, which leaves 3,300. The window leaves less than the allowance for 13 of the 221 questions at 32,768 (3 of them under 16,384) and for 46 at 81,920. Cap `max_tokens` per request at what the window leaves, and flag those questions. A larger allowance gives no question less room, but full context then matches the windowed runs on fewer questions.
- Cost: a trace that runs on to 81,920 instead of stopping at 32,768 takes about 5–7 more minutes, so each 1% of the 4,569 Qwen thinking questions that does so adds about 4–5 executor-hours.
- DeepSeek: the 0731 card recommends a maximum output length of 384K tokens (393,216) at the high and max effort levels, and vLLM's recipe agrees. Its pilot runs at 384K, which the 1M window fits (262,136 + 393,216 = 655,352). DeepSeek takes Qwen's allowance if almost no answer goes past it under any policy, so both models share one cap; otherwise it keeps the card's figure, and the paper reports cap hits per model. The paper compares policies within each model, so each model needs a fixed cap, not the same one. If reasoning that long is too slow for the runs, 0731 also has a low effort level.
- Paper: the thinking-off runs took their answer reserve out of the window, 4,096 tokens on AA-LCR and MRCR and 8 on LongBench-v2, so 64K meant 61,440 tokens of prompt on AA-LCR and MRCR. The paper says so, and gives the budget a reasoning trace would take as the reason thinking is off (`aaai-2027/main.tex`). It must add that the thinking runs keep those prompts and put the allowance on top, and revise that reason.
- Rejected: a fixed total window, in which the input budget shrinks by the allowance. At 32K it leaves almost no room for the documents, and it confounds thinking with less evidence.

### Decoding with thinking (decided 2026-11-25)

Sample with a fixed seed per question, derived from its case ID and the same in every policy, one request at a time on a fixed node, so that a rerun reproduces its answers. The pilot checks this by sending the same prompt and seed twice. Thinking-off runs stay greedy, so turning thinking on also changes the decoding; the paper must say so.

- Qwen: the card's thinking settings for precise tasks: temperature 0.6, top_p 0.95, top_k 20, min_p 0, presence penalty 0. The general settings' presence penalty of 1.5 would penalize MRCR's verbatim answers. The Qwen3.6 card has no explicit warning against greedy decoding (the Qwen3 cards had one), but it recommends sampling in thinking mode.
- DeepSeek (proposed): temperature 1.0 and top_p 1.0, from its card.
- The seed makes runs reproducible, not less noisy: each condition is still one sample. Effects of 20 points or more (AA-LCR, MRCR) survive that; LongBench-v2's few-point differences will not. Each condition runs once per model, at one reasoning level (Qwen's thinking is on or off; DeepSeek uses its default, high). Only AA-LCR's main conditions run again, with two more seeds each, to measure the run-to-run spread (P2 below).

### Second family (proposed 2026-11-25, pending the smoke test)

Run it locally, not through an API (decided). Proposed model: DeepSeek-V4-Flash-0731. It has 284B parameters with 13B active per token, a 1M-token window, thinking off or at low, high or max effort in one checkpoint, and the MIT licence; it is 167 GB on disk (FP4 experts, FP8 elsewhere). The second model has to come from another family, switch thinking in the same weights, read 262K tokens plus the allowance, score well above the floor on long inputs, and run on our nodes. DeepSeek meets all five and adds two things:

- Size and strength: 8 times Qwen3.6's total parameters and about 4 times its active ones. Artificial Analysis's Intelligence Index, with reasoning, is 50 against Qwen3.6's 32. With one small model, a reviewer's first question is whether the effect holds for a stronger model. Family and size change together, so the paper cannot attribute differences to family; its claims are about overflow policies.
- The 1M window leaves room for thinking at every window, and could give MRCR a full-context reference on 3 or 4 of its conversations, which the paper lacks (not planned; see [Experiments](#experiments-proposal)).

Costs: it needs vLLM 0.25 or newer, and one 8 × H100 report has 0.26.0 working and 0.27.0 failing at startup. It ships no Jinja chat template, so our token counting needs an adapter for its `encoding` module. Jobs on the shared gpu-h200 partition last at most 24 hours. It is not the newest DeepSeek: Artificial Analysis lists 0731 as superseded by V4.1-Flash, which has 763B parameters and is not documented on one H200 node.

| Partition | Per node | Fits DeepSeek-V4-Flash-0731? |
|---|---|---|
| gpu-h200 | 4 × H200, 564 GB | Yes: about 340 GB free after the weights at 0.9 memory utilization. NVIDIA documents 4 × H200. |
| gpu-h100sxm | 4 × H100, 320 GB | Probably: about 120 GB free. 4 × H100 is reported only for the DSpark checkpoint, on vLLM 0.25 with two workarounds. |
| gpu-h100 | 2 × H100 NVL, 188 GB | No |
| gpu-l40s | 4 × L40S, 192 GB | No: the weights leave about 25 GB, and nothing documents these GPUs. |

Fallback, if DeepSeek is not serving by 29 November: Nemotron 3 Super (NVIDIA; 120B, 12B active; 1M window; the same `enable_thinking` switch as Qwen and a standard chat template, so almost no code changes; FP8 on H100 or H200; a hybrid Mamba-Transformer, the architecture most unlike Qwen's).

Other candidates:

| Model | Parameters (active) | Window | Thinking switch | Runs on | Concern |
|---|---|---|---|---|---|
| Mistral Small 4 | 119B (6B) | 256K | `reasoning_effort` none or high | H100/H200; maybe FP8 on L40S | 256K window; long-context quality unknown |
| Gemma 4 26B-A4B or 31B (Google) | 26B (4B), or 31B dense | 256K | chat-template switch; with thinking off it still emits an empty thought block | L40S, or one H200 GPU | 256K window; reported bugs in vLLM's Gemma 4 reasoning parser on 0.19 |
| Nemotron 3 Nano (NVIDIA) | 30B (3B) | 1M | `enable_thinking` | L40S | Its card reports AA-LCR 35.9, against 59.0 for Qwen3-30B-A3B-Thinking: near the floor |

The 256K models have Qwen's limit: no room for thinking at MRCR's 262K window, and no MRCR full-context reference. Ruled out: GLM-5 (744B; about 200K window), DeepSeek V4-Pro (1.6T) and Kimi K3 (Moonshot's earlier Kimi models had 1T parameters), all too big for one node; GLM-4.7-Flash (128K window); gpt-oss (no thinking-off mode, 128K window); MiniMax (always thinks); Llama 4 (no thinking mode); Qwen3.8 (same family).

Rejected: DeepSeek's own API, where `deepseek-flash` now serves V4.1-Flash-1017. The proposed runs would cost about $140–300 for both modes, and they would finish in about a day because requests can go in parallel. But the priority is running locally, the model behind an API name can change, and thinking mode ignores temperature, so it cannot be seeded.

### Scope (decided 2026-11-25)

Not all 93 conditions per configuration. The new models run the same conditions as one another, so that whatever the paper reports compares them on equal terms; [extension_experiments.csv](extension_experiments.csv) marks them (see [Experiments](#experiments-proposal)). The authors reviewed the marked set on 2026-11-25 and agreed it, and every marked run will run. P1–P3 only set the order: the headline results come first, and anything that slips is the least important, the same for every model. Added the same day: a no-context baseline on AA-LCR and LongBench-v2 for every model, Qwen without thinking included, and the new models' MRCR oracle moved from 32K to 64K. Not added: AA-LCR position at 32K and 80K, the position recipe at 32K and 80K, and marked middle truncation.

### Hardware and traces (decided 2026-11-25)

Qwen with thinking runs on g101 and g102, the hardware of the thinking-off runs, which leaves gpu-h200 to DeepSeek. Reasoning traces are saved per run in their own file and committed with the other artifacts, up to about 80 MB per configuration.

## Feasibility numbers (2026-11-25)

- The 95 suite runs behind the 93 conditions took about 38 executor-hours: AA-LCR 7.5, MRCR 20 and LongBench-v2 10, from the run minutes in the suites' results tables.
- Qwen3.6 on 4 × L40S, one request at a time, fitted on the 13,507 requests of those suites: prefill takes 1.6 s at 32K, 3.4 s at 64K, 7.6 s at 128K and 16 s at 226K; decoding runs at 174, 161, 141 and 118 tokens/s. A 5K-token reasoning trace adds 30–40 s to questions that take 2–20 s now.
- Qwen with thinking, pilot of 2026-11-26: mean reasoning ran from 2,337 to 5,499 tokens per question and generation took 20–40 s per question. At those rates and the conditions' prompt lengths, the runs take about 17 hours for LongBench-v2 and about 20 for AA-LCR and MRCR together, with grading: one job per node.
- DeepSeek on 4 × H200: not measured. Expect the same order as Qwen on the L40S nodes: about 4 times the active parameters, on much faster GPUs.
- Job limits: 24 hours on gpu-l40s (executors and every suite's one-GPU client job), gpu-h200 and compute-short (Luna grading), and 72 hours on gpu-h100sxm. No single run comes near 24 hours; the longest, LongBench-v2 with thinking, takes about 2–3.5 hours at 5K reasoning tokens and about 5 at 10K. So the limit means resubmitting, about once a day per node. A resubmitted suite keeps its finished runs and restarts the unfinished one from the beginning. `jarvis/thinking_runs.sh` submits an executor pinned to a node with a suite job that waits for it, records its setup and cancels it at the end, as `jarvis/rerun_g202_runs.sh` did, so each day's resubmission is one command.

## Experiments (proposal)

The same questions as the `20261029` suites: AA-LCR 100, MRCR 120 (30 per conversation) and LongBench-v2 221. The recipe stays as fixed on 2 November, with no tuning per model, and conditions keep their `20261029` names so runs pair one to one. The main window is 64K on AA-LCR and LongBench-v2 and 128K on MRCR for every new model, because Qwen with thinking cannot run MRCR at 262K. Qwen without thinking has every planned condition already, except the no-context baseline.

[extension_experiments.csv](extension_experiments.csv) tracks the conditions: one row per condition, one column per model, X when planned or completed, and the priority below. The Qwen thinking-off column marks its 93 completed conditions and the 2 no-context runs it still needs. The three new models plan the same 29 conditions, and the two thinking models add 4 repeats, so 29 or 33 runs per model. With Qwen without thinking's 29, the comparable set is 124 runs: 116 model-condition results and 8 repeats, which the paper reports as a noise estimate rather than as conditions. 97 of the 124 are new: Qwen without thinking has 27 of its 29 already.

- **P0, pilot**, for each configuration before its runs: `middle` and `upgraded` on 20 AA-LCR questions at 64K, 12 MRCR questions (3 per conversation) at 128K and 20 LongBench-v2 questions at 64K; `full` on the 20 AA-LCR questions, to check the model is not near the floor. It measures:
  - reasoning length at the pilot's cap (81,920 for Qwen, 384K for DeepSeek), and how many answers go past 32,768 or reach the cap, per policy, which sets each model's allowance (see [Thinking budget](#thinking-budget-decided-2026-11-25));
  - time per question;
  - whether seeded reruns reproduce;
  - whether the A–D constraint applies after the reasoning;
  - whether speculative decoding (MTP for Qwen, DSpark for DeepSeek) speeds generation without breaking seeded reruns.

  The runners take `--question-ids` or `--max-rows` (AA-LCR), `--max-rows-per-source` (MRCR) and `--max-rows` (LongBench-v2).
- **P1, headline** (9 runs): AA-LCR `full`, `middle` and `upgraded`; MRCR `middle_128k`, `upgraded_128k` and `oracle_128k`; LongBench-v2 `full`, `middle` and `upgraded`. It asks whether middle truncation costs and the recipe recovers for this configuration, and how close the recipe gets to full context.
- **P2, controls and noise** (11 runs with thinking, 7 without): AA-LCR `position` and `position_recipe`, structure without the question; `bm25_recipe` at each main window; `no_context` on AA-LCR and LongBench-v2; and, with thinking, AA-LCR `middle` and `upgraded` with two more seeds each (`middle_repeat2`, `middle_repeat3`, `upgraded_repeat2`, `upgraded_repeat3`).
  - `no_context` sends the prompt with no source text, as a prompt that keeps nothing would: the instructions and the question, with the four options on LongBench-v2, where the omission marker stands in for the text. It does not depend on the window, so it is one run per model and benchmark. It measures what the model answers from prior knowledge, and on multiple choice by ruling options out: the floor every policy should beat. The 17 November review asked for it, and Limitations says LongBench-v2 cannot be read without it. With thinking it matters more, since reasoning can improve guessing. MRCR needs none, because its answers begin with a random string and reproduce a passage, which cannot be guessed.
- **P3, other windows** (13 runs): `middle` and `upgraded` on AA-LCR at 32K and 80K, MRCR at 32K and 64K and LongBench-v2 at 32K and 128K, plus MRCR `oracle_64k`. It completes the primary comparisons at every window, and gives MRCR's 64K and 128K windows their reference; 32K has none for the new models.
- **Not planned, model-specific**: MRCR at 262K, and MRCR full context in DeepSeek's 1M window (the 1.06M-token conversation may not fit). Qwen with thinking can run neither, so they would sit outside the comparison; decide later whether to report them apart.
- **Analyses, no extra runs**:
  - Thinking on against off on identical prompts (Qwen), per policy. Does thinking narrow the gap between middle truncation and the recipe? This tests the Limitations sentence that a reasoning phase "might make up for missing text or compound its loss".
  - MRCR: evidence recall is fixed by the prompt. Does thinking raise exact match given the target, and does any answer recover a target its prompt lacks?
  - Reasoning length and allowance hits by policy. With thinking off, answers to middle-truncated prompts were 55% longer than answers to full prompts.
  - Cost: tokens and time per question with thinking.
- **Not rerun**: random seeds, dense and fusion with default chunking, marked and head truncation, the ablation ladder, BM25 with default chunking, and the dense variants.

Qwen with thinking, in hours on one executor, for three guesses of the mean reasoning length: the tokens the model writes while thinking, before its answer, averaged over questions. The pilot replaces the guesses with measured means. Add about 3 hours of embedding and packing; g101 and g102 together halve the wall-clock.

| Group | Runs | 2K | 5K | 10K |
|---|---:|---:|---:|---:|
| P1 | 9 | 7 | 15 | 27 |
| P2 | 11 | 6 | 13 | 25 |
| P3 | 13 | 9 | 19 | 35 |
| All | 33 | 22 | 47 | 87 |

LongBench-v2's share of P3 is about 9 hours at 5K tokens.

Primary comparisons, to fix in the plan: the recipe against middle truncation at every window run, and against position on AA-LCR, Holm-corrected within each configuration. The thinking-on-against-off comparisons are exploratory.

## Code for Qwen with thinking (done 2026-11-25)

How to run it: [THINKING.md](jarvis_suites/THINKING.md). All 273 tests pass.

1. `--thinking` and `--sampling-seed` are shared execution options (`execution/contracts.py`), so the four runners take them and record them in their manifests. `execution.client.with_thinking` turns thinking on in the request (`chat_template_kwargs: {"enable_thinking": true}`) and adds the sampling settings and the question's seed; without thinking, requests are unchanged. The AA-LCR grader and the cache verifier stay non-thinking.
2. Token counting stays at `enable_thinking=False` (`execution/tokens.py`), so prompts stay identical.
3. The shared config check now requires only that the input budget leave room in the window, and the pipeline caps each request's output at the allowance or what the window leaves after its prompt, whichever is smaller (`max_tokens` in `execution.jsonl`). AA-LCR keeps its own stricter check, so its thinking full-context run uses an input budget of 262,144 minus the allowance, which still sends every prompt whole.
4. The answer stays the response's `content`. The client reads the reasoning from `reasoning` or `reasoning_content`, and the pipeline records `reasoning_tokens` per question and writes the text to each run's `reasoning.jsonl`.
5. LongBench-v2 keeps the A–D constraint in the request; the pilot checks that it applies after the reasoning (invalid answers). Both LongBench-v2 runners now wait up to 1,800 s for a request, like the AA-LCR and MRCR runners; the hybrid runner's client had a fixed 120 s.
6. Manifests record thinking, temperature and the sampling seed in place of the hard-coded `enable_thinking: False` and `thinking_enabled: False`.
7. Suites: `THINKING=1` with `THINKING_ALLOWANCE` (required) runs the planned conditions, headline runs first (`ONLY` now also sets the order), adds AA-LCR's repeats and serves every run the whole window. Everything that labels or pairs runs by window uses the window that a run's input budget belongs to (`execution.results_table.window`), which pairs runs as keying by `max_input_tokens` would: the results tables and intervals of all three benchmarks, MRCR's visibility analysis and AA-LCR's acceptance check. The tables add a reasoning-tokens column, and on LongBench-v2 output limit hits. `jarvis/thinking_runs.sh` submits the executor and a suite job in one command per node.
8. `--no-context` (shared, direct mode) sends the prompt rendered with no evidence, for every benchmark and model, and the AA-LCR and LongBench-v2 suites have a `no_context` condition. On LongBench-v2 that rendering shows the omission marker where the text was, as a prompt that keeps nothing would, rather than the empty context first planned.
9. Tests with fake tokenizers and clients cover the request body and seeds, identical prompts, reasoning capture, the per-request cap, the no-context prompt, the window labels and the thinking suites.

Not done: speculative decoding (MTP). The thinking runs use the executor of the thinking-off runs; add MTP only if the pilot's hours are too long, and then check seeded reruns again.

## DeepSeek setup (after the probe)

1. Done 2026-11-25: `jarvis/run.sh` takes `EXECUTOR_PARTITION`, `EXECUTOR_GRES` and `EXECUTOR_MEM` (defaults `gpu-l40s`, `gpu:l40s:4`, `220G`).
2. A separate vLLM environment, 0.26.0 first, installed on a GPU node so that `--torch-backend=auto` matches the driver. Pin DeepSeek to one node; in the default scratch storage mode the first executor start downloads DeepSeek-V4-Flash-0731 (167 GB) to that node's `/local/$USER/llm_caching` and later starts reuse it.
   - g213 has room for it, with about 85 GB to spare. The executor refuses to start below `EXECUTOR_MIN_LOCAL_FREE_GB` (250 by default): use 200 for the first start and about 20 once the weights are cached.
   - g013 (gpu-h100sxm) has 1.3 TB free and 72-hour jobs: the backup host, if 0731 runs on 4 × H100.
   - H200 NVL cards connect over NVLink bridges or only over PCIe; check with `nvidia-smi topo -m` inside the executor job, since tensor parallelism over PCIe slows prefill.
3. Serve it:
   - tensor parallel 4 with expert parallel;
   - `--kv-cache-dtype fp8 --block-size 256 --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 --trust-remote-code`;
   - one request at a time;
   - the full 1M window, so the allowance can change between runs without a restart. Each run sets its own input budget, as the thinking-off suites did against one 262,144-token server (the 32K AA-LCR run recorded a 32,768 window), so the served window changes no prompt. The suites pass it as `--context-window-tokens`.

   Thinking is set per request: `chat_template_kwargs: {"thinking": true, "reasoning_effort": "high"}`. 0731's effort levels are low, high and max. Its card and vLLM's recipe allow up to 384K output tokens at high and max; the preview card gave 384K as the context window for max only. Max is not proposed: each model runs at one reasoning level (see [Decoding with thinking](#decoding-with-thinking-decided-2026-11-25)).
4. Smoke test: start the server at 1M. vLLM refuses to start unless its KV cache holds one request of the full served length; if 1M does not fit, serve the largest input budget plus the allowance: 655,360 for the card's 384K, 294,912 for 32,768. Then thinking on and off, one prompt of about 250K tokens, and prefill and decode speed.
5. A token-counting adapter for DeepSeek's `encoding` module, checked against the server's `usage.prompt_tokens`.
6. MRCR re-prepared with DeepSeek's tokenizer (`mrcr_v2.prepare_dataset --executor-model`), checking that the same 120 questions come out. LongBench-v2 keeps the same 221.

### Smoke-test commands

Run on the Jarvis login node, from the directory that holds `adarsh-rlms`.

- [ ] **vLLM environment**, about 15 minutes. Installed on a GPU node, so that `--torch-backend=auto` picks the build that matches the driver. The first attempt, on 2026-11-25, was an interactive `srun --pty` job: it left no log, and a queued interactive job ends with the login session. A batch job keeps its place in the queue after logout and logs to `$HOME/adarsh-rlms-logs/rlms-vllm-install-<job id>.out`, whose last line gives the vLLM, torch and CUDA versions:
  ```bash
  sbatch --partition=gpu-h200 --gres=gpu:h200:1 --cpus-per-task=8 --mem=32G --time=01:00:00 \
    --job-name=rlms-vllm-install --output="$HOME/adarsh-rlms-logs/%x-%j.out" --wrap 'set -e
  uv venv "$HOME/.venvs/adarsh-vllm-v4" --python 3.12 --seed
  uv pip install --python "$HOME/.venvs/adarsh-vllm-v4/bin/python" "vllm==0.26.0" --torch-backend=auto
  "$HOME/.venvs/adarsh-vllm-v4/bin/python" -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda)"'
  ```
- [ ] **Server on g213**, at the full 1M window (step 4). Follow its log as in [EXECUTOR.md](jarvis_suites/EXECUTOR.md) and set `EXECUTOR_JOB` to its job ID; the first start also downloads the 167 GB of weights. If the log says the KV cache cannot hold one request of the served length, cancel the job and submit again with `EXECUTOR_MAX_MODEL_LEN=655360`. Once the weights are cached, later starts take `EXECUTOR_MIN_LOCAL_FREE_GB=20`.
  ```bash
  EXECUTOR_PARTITION=gpu-h200 EXECUTOR_GRES=gpu:h200:4 EXECUTOR_MEM=400G EXECUTOR_NODE=g213 \
  EXECUTOR_MODEL=deepseek-ai/DeepSeek-V4-Flash-0731 EXECUTOR_TP_SIZE=4 EXECUTOR_MAX_MODEL_LEN=1048576 \
  EXECUTOR_MIN_LOCAL_FREE_GB=200 VLLM_GPU_MEMORY_UTILIZATION=0.90 \
  VLLM_EXTRA_ARGS="--trust-remote-code --kv-cache-dtype fp8 --block-size 256 --enable-expert-parallel --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 --max-num-seqs 1" \
  VLLM_VENV="$HOME/.venvs/adarsh-vllm-v4" \
    bash adarsh-rlms/jarvis/run.sh submit executor
  ```
  If the log complains about `nvcc` or `CUDA_HOME`, add the newest toolkit from `module avail cuda` as `MODULES=...`.
- [ ] **Checks once it is up**: thinking off; thinking on, twice, where the same seed should give the same answer and the reasoning should come back in its own field; and how the four GPUs are linked.
  ```bash
  URL=$(cat "$PROJECT_LOG_DIR/executor-$EXECUTOR_JOB.url"); M=deepseek-ai/DeepSeek-V4-Flash-0731
  curl -s "$URL/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of Australia? One word.\"}],\"max_tokens\":64,\"temperature\":0}" | python3 -m json.tool
  time curl -s "$URL/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"A train leaves at 9:40 and arrives at 13:15. How long is the trip?\"}],\"max_tokens\":8192,\"temperature\":1.0,\"top_p\":1.0,\"seed\":1,\"chat_template_kwargs\":{\"thinking\":true,\"reasoning_effort\":\"high\"}}" | python3 -m json.tool
  srun --jobid "$EXECUTOR_JOB" --overlap nvidia-smi topo -m
  ```
- [ ] **A prompt of about 250K tokens**, for prefill speed at length. It should answer 17999:
  ```bash
  python3 - "$URL" "$M" <<'EOF'
  import json, sys, time, urllib.request
  url, model = sys.argv[1], sys.argv[2]
  text = "\n".join(f"Line {i}: the quick brown fox jumps over the lazy dog." for i in range(18000))
  body = {"model": model, "max_tokens": 64, "temperature": 0, "messages": [{"role": "user",
          "content": text + "\n\nWhat number does the last line have? Answer with the number only."}]}
  request = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
  start = time.time()
  reply = json.load(urllib.request.urlopen(request, timeout=1800))
  print(f"{time.time() - start:.1f} s", reply["usage"], reply["choices"][0]["message"]["content"])
  EOF
  ```
- [ ] Cancel the server when done: `scancel "$EXECUTOR_JOB"`.

## Timeline

| Dates | Work |
|---|---|
| 25–27 Nov | Qwen thinking code and tests; DeepSeek environment and smoke test once the probe is back; the plan |
| 28–29 Nov | Pilots; DeepSeek cut-off on 29 November |
| 30 Nov–4 Dec | Runs: Qwen with thinking on g101 and g102, DeepSeek on gpu-h200 |
| 5–6 Dec | Luna grading, intervals and tables |
| 7–10 Dec | Writing, with one day of buffer |

## Open questions

1. Whether to report the model-specific runs (MRCR at 262K, and MRCR in DeepSeek's 1M window) apart from the comparison.
2. DeepSeek's allowance, from its pilot: Qwen's 81,920 or its card's 384K (see [Thinking budget](#thinking-budget-decided-2026-11-25)).

## References

- Qwen3.6-35B-A3B model card: <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>
- DeepSeek-V4-Flash: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>; the 0731 release: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731>
- vLLM recipe for DeepSeek-V4-Flash: <https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash>
- NVIDIA Dynamo recipe for DeepSeek V4 Flash: <https://docs.nvidia.com/dynamo/latest/recipes/deepseek-v4-flash>
- DeepSeek API pricing and thinking mode: <https://api-docs.deepseek.com/quick_start/pricing>, <https://api-docs.deepseek.com/guides/thinking_mode>
- Nemotron 3 Super (FP8): <https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8>; vLLM blog: <https://vllm.ai/blog/2026-03-11-nemotron-3-super>
- Nemotron 3 Nano: <https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16>
- Mistral Small 4: <https://mistral.ai/news/mistral-small-4/>
- Gemma 4 model card: <https://ai.google.dev/gemma/docs/core/model_card_4>
- Artificial Analysis, DeepSeek V4 Flash against Qwen3.6 35B A3B: <https://artificialanalysis.ai/models/releases/comparisons/deepseek-v4-flash-vs-qwen3-6-35b-a3b>
