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

### 007 — YAML Serialization & Performance Assessment for Plan Storage Persistence Format

Ratify that YAML serialization of the Plan Storage Persistence Format is not significantly computationally taxing during simulation/inference. If profiling or benchmarking demonstrates that YAML serialization creates a computational bottleneck, prompt the User whether to write a plan for a more compressed, non-human readable serialization format (e.g., Protocol Buffers, MessagePack, or binary IPC).

### 008 — Calibration of Kinematic Motion Primitive (kMP) Basis Dimension $K$

The default Gaussian basis dimension for continuous `SkillArtifact` procedural memory is configured to $K=4$ for optimal performance during Monte Carlo counterfactual rollouts. If test precision failures occur or if higher trajectory expressiveness is required for performance/competence trade-offs, evaluate and benchmark $K=8$ as an alternative configuration.

### 009 — Default Configuration of Friston Expected Free Energy (EFE) Action Selection Mode

The active sensing action selection engine is configured by default to Friston Expected Free Energy mode (`efe_mode: true` in `schema/active_sensing_config.yaml`), while coexisting alongside traditional linear pragmatic-epistemic utility mixing ($U_{\text{total}} = (1-\beta) U_{\text{pragmatic}} + \beta U_{\text{epistemic}}$) as a configurable fallback option. If implementation failures, unexpected trajectory divergence, or numerical instability arise during active perception simulation or inference, investigate `efe_mode` calibration as a primary direct cause.

### 010 — Strict Prohibition of PDF Tracking or Pushing in Git

NEVER track, stage, or push PDF files (`*.pdf`) to the Git repository. Literature PDF files stored in `docs/WAVE_*/papers/` or anywhere else in the workspace are for local agent ingestion only and MUST remain ignored by Git. Markdown documentation (`*.md`), specifications, code, YAML presets, and test files are the only permitted tracked assets.

### 011 — Dual Persistence & Performance Monitoring for Meta-Parameters ($\theta_{\text{meta}}$)

Implement dual persistence for meta-parameters ($\theta_{\text{meta}}$): (1) in-memory tuple projection within $c.m_{\text{procedural}}$, and (2) persistent human-readable YAML serialization as the default state snapshot format. DO NOT purge or deprecate YAML human-readable serialization. Monitor and benchmark YAML serialization of $\theta_{\text{meta}}$ during high-frequency simulation/inference ticks for potential performance bottlenecks.

### 012 — Mandatory Formal Mathematical Implementation Verification

Every wave, front, and engine feature MUST be backed by explicit formal mathematical verification tests, not just unit/syntax surface assertions. Tests MUST empirically verify end-to-end mathematical theorems, limit behavior ($N \to \infty$), asymptotic convergence, game-theoretic equilibrium bounds, variational free energy bounds, and simplex projection invariants. To prevent giant monolithic test files, formal mathematical test suites MUST be modularly split into domain-specific test modules under `tests/formal_math/`.

### 013 — Color All User-Facing Terminal Output with `rich`

ALWAYS color and format all user-facing terminal outputs, experiment progress logs, CLI scripts, evaluation telemetry tables, and status reports using the `rich` library (`rich.console.Console`, `rich.table.Table`, `rich.panel.Panel`, `rich.progress.Progress`). Never output plain monochrome unformatted text for interactive CLI workflows or benchmark results.

### 014 — Strict Execution Boundary for Heavy vs. Light Scripts

The `scripts/` directory is partitioned into subdirectories based on computational cost:
1. **`scripts/heavy/`**: Contains long-running, computationally expensive training routines, Monte Carlo sweeps, and multi-game tournaments. **AI agents must NEVER execute commands or scripts inside `scripts/heavy/`.** The `scripts/heavy/` folder is reserved exclusively for manual execution by the User.
2. **`scripts/light/`**: Contains fast ($<10$ second) utility scripts, telemetry plotters, report generators, and diagnostic parsers. **AI agents ARE permitted to execute scripts inside `scripts/light/`** to display reports or verify telemetry outputs.





