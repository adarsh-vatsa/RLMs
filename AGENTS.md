# Agent Working Guide

This repository is a rapidly changing research prototype. Optimize for the current task and current behavior, not for preserving an architecture that may already be obsolete.

## Evidence Order

When sources disagree, use this order:

1. The user's current request.
2. Executable code, tests, and configuration.
3. Focused documentation for the subsystem being changed.
4. `README.md` and broad architecture documents.

`README.md`, `docs/system_architecture.md`, plans, benchmark reports, and other prose may be incomplete or stale. Use them for context, not as authoritative specifications. Do not read large documents by default; inspect only the sections relevant to the task. Verify claims against the current code and tests before relying on them.

If documentation conflicts with behavior, follow the requested behavior and current executable evidence. Mention the mismatch when it matters, and update the relevant documentation only when that is in scope.

## Project Orientation

- `semantic_cache_system.py`: main semantic-cache implementation and current model, routing, persistence, and pricing behavior.
- `long_bench_v2/`: LongBench-v2 runners and supporting tools. Read the runner being changed before modifying benchmark behavior.
- `test/`: executable expectations and focused regression tests.
- `jarvis/`: local/HPC serving and launch scripts with their own focused documentation.
- `docs/`: design context, experiment reports, and plans; not guaranteed to describe the current implementation.
- `benchmark_artifacts/`, `benchmark_data/`, and `benchmark_fixtures/`: experimental inputs and outputs. Treat historical artifacts as read-only unless the user asks to regenerate or edit them.

Use `rg` to find the current implementation rather than relying on class names, model names, flows, or return shapes described in prose. In particular, confirm model and pricing constants in code before changing model-related behavior.

## Prototype Engineering Rules

- Write the smallest clear change that satisfies the request.
- Prefer direct code over new abstractions, indirection, configuration, compatibility layers, or extension points.
- Do not preserve an existing idea merely because it appears intentional in an old document. Preserve it only when the current task, callers, tests, or reproducibility requirements justify it.
- Remove code, imports, branches, and comments made obsolete by your change within the touched scope.
- Do not add speculative fallbacks or handle scenarios that the current system cannot reach.
- Avoid broad refactors and unrelated cleanup. Every changed line should trace to the request.
- Match the surrounding style unless changing it is necessary for the task.

## Comments And Documentation

- Prefer clear names and straightforward control flow over explanatory comments.
- Add a comment only when it explains a non-obvious reason, constraint, or tradeoff that the code cannot express.
- Do not narrate what the next line does, leave commented-out code, or add decorative section comments.
- Keep docstrings limited to useful contracts or surprising behavior.
- Documentation should follow the implementation. Do not distort code to match stale prose.
- Update only the narrow documentation affected by a user-facing command, contract, or workflow change; do not refresh unrelated documents.

## Validation

- Start with the narrowest relevant tests, then broaden only when risk warrants it.
- Add or change tests for behavior introduced by the task, not for incidental implementation details.
- Prefer `uv` for Python commands when practical.
- Do not regenerate benchmark outputs as a side effect of validation.
- For benchmark changes, keep runs attributable and comparable. Preserve existing artifact data and record behavior-affecting settings, but do not treat the current experiment design as immutable when the task explicitly changes it.

## Working Style

- Inspect the relevant execution path before editing.
- State assumptions when ambiguity would materially change the implementation.
- Offer a simpler interpretation when one exists instead of silently building a larger solution.
- Stop and ask when different plausible interpretations would produce meaningfully different results.
- Report stale or contradictory guidance you encounter; do not silently encode it into new code.
