# AGENTS.md — Behavioral Directives for AI Agents

> Canonical list of standing instructions that any AI agent working on this codebase **must** follow.
> Each entry is a rule, not a suggestion. Entries are cumulative — newer entries do not override older ones unless explicitly stated.

---

## Entries

### 001 — Lint with Ruff after significant code changes

After completing any non-trivial code modification (new module, refactor, bug-fix touching multiple functions, or schema-driven code generation), run `ruff check .` from the project root and resolve all reported issues before considering the change complete. Formatting should also be verified with `ruff format --check .`. Trivial edits (comments, docstrings, single-line fixes) are exempt.

### 002 — Execute test suite with Pytest after code changes

After adding or modifying functions in `src/hypostases/` or `tests/`, run `pytest` from the project root and ensure all tests pass cleanly before completing the task.

### 003 — Audit branch completeness on schema functions

When adding or modifying schema-level functions with case-by-case branch definitions (e.g. per action type in `feedback` or per goal in `action_likelihood`), verify and document whether every branch is state-dependent or a declared simplification (Part III §5.8).

### 004 — Monitor `MOOD_DECAY_RATE` calibration

`MOOD_DECAY_RATE` is configured to `0.1` (10% decay per Tier-1 tick). Pay attention to this parameter value if implementation failures or unexpected behavioral convergence arise during simulation/inference.


