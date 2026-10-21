# Changelog

All notable changes to the HYPOSTASES framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2026-10-20

### Packaging, Build & Environment Foundation
- **Declared Dependencies**: Added `scipy>=1.10.0` to `pyproject.toml` dependencies, resolving runtime `ModuleNotFoundError` during causal discovery and NLP Hungarian graph matching.
- **Linter Exclusions**: Excluded vendored Lean 4 Mathlib scripts in `formal/.lake` from Ruff configuration, eliminating 47 false-positive upstream errors.
- **Optional Dependency Guards**: Added `pytest.importorskip("chess")` across all 10 domain-specific test suites, allowing test collection and execution to succeed gracefully in base environments without the optional chess extra.

### Core Engine De-duplication, Hygiene & Invariant Reconcile
- **Power Projections Reconciled**: Reconciled `power_dynamics.py` with Schema v4 Target (§6.6b) and Lean 4 `StateSpace.lean` by aligning internal power $\text{proj}_{\text{int}}(c) = c.\text{reserve}$ and external power $\text{proj}_{\text{ext}}(\rho_{\text{ext}}) = w_t \cdot \text{time\_budget} + w_{\text{sc}} \cdot \text{social\_capital}$, eliminating broken attribute access crashes (`c.social_status`, `rho_ext.institutional_access`). Added dedicated unit test suite in `tests/test_power_dynamics.py`.
- **Dead Code Elimination**: Removed shadowed, dead `action_likelihood` implementation from `utility_dynamics.py` which had inverted parameter ordering and redundant inline imports.
- **Obsolete Stub Cleanup**: Removed unmaintained `step_env` stub from `state_transitions.py`, consolidating `dynamics.py` as the single canonical source of truth.
- **Memory Event Deduplication**: Removed byte-for-byte duplicate `_record_memory_event` in `dynamics.py`, consolidating event recording into `state_transitions.py`.
- **Multiprocessing Utilities**: Centralized multiprocessing worker initialization (`_init_worker_process`) and pool cleanup (`_terminate_pool_workers`) into `src/hypostases/plugins/domains/chess/multiprocess_utils.py`.
- **Directives Calibration**: Aligned `run_chess_emergence_check.py` to import `const.MOOD_DECAY_RATE` rather than hardcoding `0.9` (Rule 004).

### Architectural Layering & Dependency Inversion
- **Epistemic Action Dispatch Protocol**: Introduced `EpistemicHandler` protocol and `register_epistemic_handler()` in `dynamics.py`, decoupling the foundational core engine from higher-level `active_perception`.
- **Auto-Registration**: Configured `active_perception` to register its dispatcher on module import with zero manual caller configuration.
- **Zero-Dependency Reference Domain**: Extracted pure mathematical subtraction game (Nim) into `hypostases.domains.toy_nim` and registered with `@DomainRegistry.register("toy_nim")`, allowing testing and engine execution without the optional `chess` extra.

### Static Typing Hardening & Zero-Error Mypy Baseline
- **Strongly-Typed Kalman Summaries**: Defined `KalmanSummary(TypedDict)` in `hypostases.inference.summaries`, resolving downstream arithmetic operator union failures in `simulation/benchmarks.py` and `cli/infer.py`.
- **Safe Version-Branched StrEnum**: Standardized `StrEnum` definitions with `sys.version_info >= (3, 11)` across `engine/types.py` and `abduction/types.py`, eliminating `[no-redef]` errors.
- **Zero-Error Mypy Baseline**: Resolved all 92 static typing errors across 23 source files, establishing 100% clean `mypy src` verification over 129 modules.
- **Type-Safe Plan Repair & AST Mutator**: Hardened `isinstance(state, AgentState)` checks in `plan_repair.py` and typed operator maps in `alphaevolve/mutator.py`.
- **Native Dataclass Support in NLP**: Enhanced `TextBeliefUpdater` to natively accept and return `AgentState` dataclasses alongside legacy dicts.
- **Parameterized Simulation & Script Modernization**: Modernized `scripts/light/` diagnostic scripts to utilize parameterized signatures without mutating module-level constants in `const.*`.

### Domain-Agnostic Search & Test Modernization
- **True Domain-Agnostic Alpha-Beta**: Resolved hidden chess domain leak in `AlphaBetaSearch._quiescence()` (`is_capture` / `gives_check` predicate crash on non-chess objects) with graceful `default_tactical_filter`.
- **Pure Abstract Domain Suite**: Added `tests/test_alphabeta_domain_agnostic.py` testing minimax convergence on a finite mathematical subtraction game (Nim) without domain dependencies.
- **Test Optimization**: Registered `@pytest.mark.slow` in `pyproject.toml` and marked long-running Monte Carlo sweeps, reducing fast iteration test runtime from ~140s to 21s (`pytest -m "not slow"`).

### Verified
- 463 / 463 tests passing (`pytest`), 0 `mypy src` errors across 129 modules, 0 `ruff check` errors, `ruff format --check` 100% compliant, Lean 4 proofs clean (`lake build`), AST decoupling audit 100% clean.

## [0.4.0] - 2026-08-11

### Refactored & Enhanced
- **`StrEnum` Standardisation & Typed State Deltas**: Migrated `GoalCategory` and `ActionType` to `enum.StrEnum` (Ruff UP042 clean). Replaced loose `dict[str, float]` state deltas with strongly-typed `DeltaCharacteristics`, `DeltaWorldModel`, `DeltaPowerExternal` `TypedDict` structures within `FeedbackDelta`.
- **Data-Driven YAML Scenarios (Directive 006)**: Added `schema/scenarios.yaml` and refactored `scenarios.py` to parse ground-truth preset multi-agent scenarios dynamically from YAML via `hypostases.schemas.loader`. Added Directive 006 to `AGENTS.md`.
- **Concurrency Operator Strategy Pattern**: Decomposed `step_env()` in `dynamics.py` into modular strategy resolver functions (`_resolve_shares_first`, `_resolve_pro_rata`, `_resolve_priority`, `_resolve_lottery`) with dedicated unit tests in `tests/test_concurrency_operators.py`.
- **Modular Feedback Pipeline**: Refactored monolithic `feedback()` in `dynamics.py` into composable single-responsibility helpers (`_feedback_request`, `_feedback_share`, `_feedback_withdraw`, `_feedback_punish`, `_apply_cross_cutting_c`, `_apply_crowding_out`, `_compute_world_surprise`).
- **Inference Module Modularisation**: Extracted generic systematic particle resampling into `src/hypostases/inference/resampling.py` and state space prior sampling into `src/hypostases/inference/prior.py`.
- **Global Constant Immutability**: Frozen numpy array constants `ACTION_COSTS` and `DEFAULT_XI` with `writeable = False` to prevent accidental runtime state mutation.
- **Pytest Factory Fixture**: Enriched `tests/conftest.py` with `make_state` factory fixture and `default_delta_log` fixture, eliminating inline `AgentState` boilerplate across 15+ test files.

### Verified
- 169 / 169 tests passing (`pytest -q`), zero `ruff check` errors, `ruff format --check` clean.

## [0.3.0] - 2026-08-11

### Added
- **Full 4D Goal-Hierarchy Feedback Coverage**: `feedback()` now emits nonzero `delta_g` updates across all four `GoalCategory` dimensions (`SURVIVAL`, `ACQUISITION`, `RELATIONAL`, `STATUS`) based on action semantics. Previously only `RELATIONAL` and `ACQUISITION` (partially) were wired, causing `g.u` to collapse onto a 1D invariant subspace and producing `cos-sim=1.0` across distinct random priors.
- **Softmax Jacobian Attenuation**: `feedback()` scales each `delta_g[k]` by the marginal policy sensitivity $\pi_k(1-\pi_k)$ before emitting `FeedbackDelta`. As utility dimension $k$ dominates ($\pi_k \to 1$), $\Delta g[k] \to 0$ naturally, yielding a pure un-regularized fixed-point attractor without any artificial decay term (`UTILITY_DECAY_RATE = 0.0`).
- **Pre-Registered 3×3 Scarcity-Governance Grid Sweep**: `tests/test_condition1_grid_sweep.py` exercises $(\kappa, \lambda) \in \{0.0, 0.5, 1.0\} \times \{0.0, 1.0, 2.0\}$ under persistent scarcity pressure ($S_{\text{thresh}} = 15.0$), demonstrating parameter-dependent regime transitions (Fixed-Point at $\kappa=0.0$; High-Pressure Adaptive Drift at $\kappa=1.0$).
- **Gate Unit Tests**: `tests/test_full_goal_feedback.py` and `tests/test_jacobian_attenuation.py` verify the direction and attenuation of all new `delta_g` branches.
- **Diagnostic Scripts**: `scripts/diag_prior_seeding.py` and `scripts/diag_delta_g_growth.py` for tracing per-seed trajectory uniqueness and `‖Δg‖` growth characterization.

### Fixed
- **Dynamic `SCARCITY_COST_KAPPA` Binding (`_math.py`)**: `dynamic_action_costs()` previously imported `SCARCITY_COST_KAPPA` as a static module-level float via `from constants import`. Runtime overrides of `const.SCARCITY_COST_KAPPA` during parameter sweeps were silently ignored. Fixed via module-level `SCARCITY_COST_KAPPA` alias and `globals().get()` resolution that supports both `monkeypatch.setattr(m, ...)` in tests and `const.SCARCITY_COST_KAPPA = k` sweep overrides.

### Verified
- 158 / 158 tests passing (`pytest -q`), `ruff check` and `ruff format --check` clean.

## [0.2.0] - 2026-08-11

### Added
- `GoalCategory` Enum to replace stringly-typed goal identifiers.
- `DeltaLog` TypedDict for strong environment update metadata typing.
- Robust systematic resampling (Part VII §12.7) in the particle filter.
- Normalised Gaussian action likelihood calculation.
- World model surprise calculation correction excluding own action impacts.
- Mood decay baseline update toward zero at each evolution tick.
- Unification of project path resolution, builder pattern, and temperature helpers.
- Enhanced test coverage of 82+ verified tests, including maths helper tests, edge-case dynamics, invariants, and warnings.
- AGENTS.md monitoring rule for `MOOD_DECAY_RATE`.

## [0.1.0] - 2026-08-10

### Added
- Specification v4 target reference implementation.
- Core simulation engine in `hypostases.engine` (`AgentState`, `step_env`, `feedback`, `evolve`, `action_likelihood`).
- Sequential Monte Carlo (SMC) Bootstrap Particle Filter in `hypostases.inference`.
- Programmatic schema invariant validator in `hypostases.schemas`.
- Unified command line interface `hypostases` with `trace`, `infer`, `sweep`, and `spec merge` subcommands.
- Ground truth YAML schemas in `schema/` (`schema_v1.yaml`, `invariants.yaml`, `components.yaml`, `time_model.yaml`, `update_dynamics.yaml`).
- Directive 004 in `AGENTS.md` deferring `memory_decay` implementation until explicit calibration analysis.

