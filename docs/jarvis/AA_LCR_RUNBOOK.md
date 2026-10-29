# AA-LCR on Jarvis

The workshop AA-LCR runs are on Jarvis: see the
[Jarvis AA-LCR suite page](../jarvis_suites/AA_LCR.md), which submits the
[AA-LCR suite](../linux/AA_LCR_SUITE_RUNBOOK.md) as a Jarvis job. The 30 September
runs used the Neselab Linux server; the AA-LCR sections of the
[Linux benchmark runbook](../linux/BENCHMARK_RUNBOOK.md) cover dataset
preparation, the 64K and full-context runs, grading and comparison. The current
results are in the
[AA-LCR report](../reports/aa_lcr_results_20260930/aa_lcr_results_20260930.md).

`jarvis/run_aa_lcr.sh` still works with the current code and defaults to the
v1.1 dataset, but it is not maintained for experiments. The August Jarvis runs in
`benchmark_artifacts/aa_lcr/{direct,hybrid}_{64k,262k}` predate the shared
pipeline; their report is in `docs/reports/archive/`, and the earlier version of
this page is in Git history (`git log -- docs/jarvis/AA_LCR_RUNBOOK.md`).
