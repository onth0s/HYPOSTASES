# AGENTS.md — Behavioral Directives for AI Agents

> Canonical list of standing instructions that any AI agent working on this codebase **must** follow.
> Each entry is a rule, not a suggestion. Entries are cumulative — newer entries do not override older ones unless explicitly stated.

---

## Entries

### 001 — Lint with Ruff after significant code changes

After completing any non-trivial code modification (new module, refactor, bug-fix touching multiple functions, or schema-driven code generation), run `ruff check .` from the project root and resolve all reported issues before considering the change complete. Formatting should also be verified with `ruff format --check .`. Trivial edits (comments, docstrings, single-line fixes) are exempt.
