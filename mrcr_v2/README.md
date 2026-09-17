# MRCR v2

Separate direct/hybrid adapter for the released English text-style MRCR v2p1
CSVs, with configurable source-token bounds, chronological evidence packing,
and deterministic scoring. Hybrid routing matches LongBench; answer caching
is disabled. No evaluator service is required.

See the [benchmark overview with examples](../docs/mrcr_v2.md) for the task,
token bounds, execution modes, and scoring.

All preparation, executor startup, endpoint setup, preflight, evaluation,
Jarvis submission, and validation commands are in the
[MRCR v2 Jarvis runbook](../docs/jarvis/MRCR_V2_RUNBOOK.md).
The runbook also documents token bounds, metrics, and output artifacts.
