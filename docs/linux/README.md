# Linux runbooks

Run benchmarks on a Linux server without Slurm:

1. [Environment setup and model services](SETUP_RUNBOOK.md)
2. [Dataset preparation, preflight, and execution](BENCHMARK_RUNBOOK.md)

Optional reference: [runner options, defaults, and cache settings](SHARED_EXECUTION_RUNBOOK.md).
This is not a third execution step.

The [scripts](../../linux/) launch the existing benchmark runners directly.
For Slurm-based execution, use the [Jarvis runbooks](../jarvis/README.md).
