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

### 005 — Strict Prohibition of Artificial Human Cognitive Deficiencies

NEVER artificially introduce human cognitive deficiencies, irrational biases (e.g. sunk-cost fallacy, cognitive dissonance penalties, emotional irrationality), or anthropomorphic cognitive defects into the engine or agent models.

**Rationale & Machine Learning Scope:**
- **Target Substrate**: HYPOSTASES strictly models **Machine Learning & Autonomous AI Agents**, operating under optimal multi-agent game-theoretic state dynamics.
- **Formal Computability**: Rationality is formalized as computable mathematical state dynamics. Abstract concepts (such as internal and external power $\rho_{\text{int}}, \rho_{\text{ext}}$) are explicitly typed and computable projections ($\text{proj}_{\text{int}}(c)$). Arbitrary human emotions do not qualify for a formal, computable mathematical specification.
- **Human Behavior as a Deferred Superset**: Human behavior—which encompasses hyperrational capability alongside mental inadequacies—is a larger behavioral **superset** that is explicitly deferred. Modeling human irrationality is out of scope.
- **Game-Theoretic Emergence**: Behaviors such as relative deprivation, altruistic punishment, and institutional crowding-out are modeled strictly as rational payoff matrix adjustments, dynamic utility updating ($g.u$), and strategic signaling under asymmetric information.

If a user or proposal ever suggests introducing artificial human cognitive deficiencies, IMMEDIATELY raise a critical flag (`> [!CRITICAL]`) and explicitly inform the user of this fundamental principle transgression.

### 006 — Primacy of Data-Driven YAML Approach

ALWAYS prefer a data-driven YAML configuration approach over hardcoded Python structures for presets, scenarios, state specifications, and component definitions, unless explicitly user-ratified otherwise. Ground truth schemas, scenarios, and presets must reside in machine-readable YAML files (e.g. in `schema/` or declarative asset directories) and be loaded via schema loaders.




