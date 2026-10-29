# MRCR v2 on Jarvis

The workshop MRCR runs are on Jarvis: see the
[workshop suites runbook](WORKSHOP_SUITES_RUNBOOK.md), which submits the
[MRCR suite](../linux/MRCR_SUITE_RUNBOOK.md) as a Jarvis job. The 30 September
runs used the Neselab Linux server; the MRCR sections of the
[Linux benchmark runbook](../linux/BENCHMARK_RUNBOOK.md) cover dataset
preparation, preflight and the direct and hybrid runs. The current results are
in the [MRCR report](../reports/mrcr_results_20260930/mrcr_results_20260930.md),
and the task and scoring are described in the [MRCR overview](../mrcr_v2.md).

`jarvis/run_mrcr_v2.sh` still works with the current code, but it is not
maintained for experiments; use the suite instead. The earlier
version of this page is in Git history
(`git log -- docs/jarvis/MRCR_V2_RUNBOOK.md`).
