# AA-LCR on Jarvis

AA-LCR experiments run on the Neselab Linux server, not on Jarvis. Use the
[Linux setup runbook](../linux/SETUP_RUNBOOK.md) and the AA-LCR sections of the
[Linux benchmark runbook](../linux/BENCHMARK_RUNBOOK.md), which cover dataset
preparation, the 64K and full-context runs, grading and comparison. The current
results are in the
[AA-LCR report](../reports/aa_lcr_results_20260930/aa_lcr_results_20260930.md).

`jarvis/run_aa_lcr.sh` still works with the current code and defaults to the
v1.1 dataset, but it is not maintained for experiments. The August Jarvis runs in
`benchmark_artifacts/aa_lcr/{direct,hybrid}_{64k,262k}` predate the shared
pipeline; their report is in `docs/reports/archive/`, and the earlier version of
this page is in Git history (`git log -- docs/jarvis/AA_LCR_RUNBOOK.md`).
