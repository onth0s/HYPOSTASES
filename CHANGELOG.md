# Changelog

All notable changes to the HYPOSTASES framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

