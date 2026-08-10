# HYPOSTASES

Agent-based modeling framework — formal specification, reference implementation, and inverse inference engine.

## Project Structure

```
HYPOSTASES/
├── AGENTS.md                  # AI agent behavioral directives
├── HYPOSTASES.md              # Original monolithic spec (preserved)
├── pyproject.toml             # Project config, ruff & pytest settings
│
├── spec/                      # Formal specification (spliced from HYPOSTASES.md)
│   ├── 01_foundations.md          # Part I:  Notation, time model, typed state spaces
│   ├── 02_update_dynamics.md      # Part II: Core loop, three computational modes
│   ├── 03_multi_agent_composition.md  # Part III: Multi-agent, redundancy resolution
│   ├── 04_reference_schemas.md    # Part IV: Schema v1 field layouts
│   ├── 05_reference_impl_trace.md # Part V–VI: Ref implementation & worked trace
│   └── 06_inverse_inference.md    # Part VII: Particle filter, worked examples
│
├── schema/                    # Ground-truth YAML schemas + human-readable specs
│   ├── components.yaml            # Component taxonomy (primitives, derived, params)
│   ├── _spec_components.md        # ↳ human-readable companion
│   ├── time_model.yaml            # Three-tier time model
│   ├── _spec_time_model.md        # ↳ human-readable companion
│   ├── update_dynamics.yaml       # Core loop function signatures & constraints
│   ├── _spec_update_dynamics.md   # ↳ human-readable companion
│   ├── schema_v1.yaml             # Concrete reference schema (social agents domain)
│   ├── _spec_schema_v1.md         # ↳ human-readable companion
│   ├── invariants.yaml            # Cross-cutting invariants & constraints
│   └── _spec_invariants.md        # ↳ human-readable companion
│
├── src/                       # Source code
│   ├── engine/                    # Forward simulation, state evolution, env stepping
│   ├── inference/                 # Inverse inference (particle filter, summaries)
│   └── schemas/                   # Schema loaders & validators
│
├── utils/                     # Developer utilities
│   └── merge_spec.py              # Reassemble spec parts → HYPOSTASES_<epoch>.md
│
├── tests/                     # Test suite
│
└── misc/                      # Auxiliary documents and notes
```

## Design Principles

1. **Minimalism** — every abstraction justifies its existence
2. **Domain Independence** — the framework is schema-parameterized, not domain-hardcoded
3. **Structural Correspondence** — code mirrors the formal spec 1:1
4. **Emergence over Enumeration** — no hardcoded social primitives
5. **Falsifiability** — the architecture revises when reality breaks it

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev]"

# Lint
ruff check .
ruff format --check .

# Merge spec parts into a single timestamped document
python utils/merge_spec.py

# Run tests
pytest
```

## Schema Architecture

All ground-truth definitions live in `schema/` as YAML files. Each schema file has a companion `_spec_*.md` with human-readable documentation. The YAML schemas are the single source of truth; code and documentation derive from them.

## Spec Management

The monolithic `HYPOSTASES.md` has been spliced into `spec/` parts. Use `utils/merge_spec.py` to regenerate a single document:

```bash
python utils/merge_spec.py                    # writes HYPOSTASES_<epoch>.md to project root
python utils/merge_spec.py --dry-run          # preview without writing
python utils/merge_spec.py --output-dir dist/ # custom output directory
```
