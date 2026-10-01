# MRCR v2

Separate direct/hybrid adapter for the released English text-style MRCR v2p1
CSVs, with configurable source-token bounds, chronological evidence packing,
and deterministic scoring. Hybrid routing matches LongBench; answer caching
is disabled. No evaluator service is required.

See the [benchmark overview with examples](../docs/mrcr_v2.md) for the task,
token bounds, execution modes, and scoring.

MRCR runs on the Neselab Linux server. Setup and executor startup are in the
[Linux setup runbook](../docs/linux/SETUP_RUNBOOK.md); preparation, preflight
and run commands are in the
[Linux benchmark runbook](../docs/linux/BENCHMARK_RUNBOOK.md).
