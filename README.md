# HYPOSTASES

Agent-based modeling framework — formal specification (v4 Target), reference implementation, and inverse inference engine.

## Project Structure

```
HYPOSTASES/
├── AGENTS.md                  # AI agent behavioral directives
├── pyproject.toml             # Project config, ruff & pytest settings
│
├── spec/                      # Formal specification (Parts I–VII)
│   ├── 01_foundations.md          # Part I:  Notation, time model, typed state spaces (v4)
│   ├── 02_update_dynamics.md      # Part II: Core loop, three computational modes
│   ├── 03_multi_agent_composition.md  # Part III: Multi-agent, redundancy resolution
│   ├── 04_reference_schemas.md    # Part IV: Schema v1 field layouts
│   ├── 05_reference_impl_trace.md # Part V–VI: Ref implementation & worked trace
│   └── 06_inverse_inference.md    # Part VII: Particle filter, worked examples
│
├── schema/                    # Ground-truth YAML schemas + human-readable specs
│   ├── components.yaml            # Component taxonomy (primitives, derived, params)
│   ├── schema_v1.yaml             # Concrete reference schema (v4 target)
│   ├── invariants.yaml            # Cross-cutting invariants & constraints
│   ├── time_model.yaml            # Three-tier time model
│   └── update_dynamics.yaml       # Core loop signatures & constraints
│
├── src/hypostases/            # Core package
│   ├── engine/                    # v4 Core simulation engine (types, dynamics, likelihood)
│   ├── inference/                 # Inverse inference (SMC particle filter, summaries)
│   ├── schemas/                   # Schema loaders, programmatic validators & static auditor
│   ├── cli/                       # Command-line architecture (main, trace, infer, sweep, spec, sweep-memory)
│   └── utils/                     # Package utilities (spec merging)
│
├── tests/                     # Test suite (91 tests covering types, dynamics, CLI, sweeps, math, specs)
│
└── misc/                      # Auxiliary documents and notes
```

## Specification v4 Highlights

1. **Primitive State $\sigma = (c, w, g, \rho_{\text{ext}})$**: Strictly four persistent primitives.
2. **Goal Hierarchy $g = u \in \mathbb{R}^{n_k}$**: Latent utility weight vector is primitive. Stochastic policy allocation $\pi \in \Delta(K)$ is transient, recomputed dynamically.
3. **World Model Theory of Mind**: Joint belief distribution over environment state and peer latent states.
4. **Internal Power**: Read-only derived view $\rho_{\text{int}} = \text{proj}_{\text{int}}(c)$. Depletion integrates directly as $\Delta c_{\text{internal}}$ during State Evolution.

## Quick Start

```bash
# Install package in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Code quality & formatting
ruff check .
ruff format --check .
```

## Command Line Architecture (`hypostases` CLI)

```bash
# Display CLI commands overview
hypostases --help

# 1. Run forward simulation trace (Part VI §8 worked example)
hypostases trace --steps 12 --seed 7

# 2. Run inverse inference particle filter (Part VII §10)
hypostases infer --particles 300 --steps 12 --output-format table

# 3. Run diagnostic / formal 3-condition sweep (Part VII §12.7)
hypostases sweep --steps 10 50 200 --particles 300 --seeds 1 2 3 4 5

# 4. Calibration sweep for memory decay stability
hypostases sweep-memory --steps 20 --decay 0.9 --mode variance

# 5. Merge specification parts into a single document
hypostases spec merge --dry-run
```

## Engine: Belief Dynamics & Memory Decay

`AgentState` carries a `decay_mode` field that selects between two calibrated update equations applied each tick in `evolve()`:

| Mode | Equation |
|------|----------|
| `"variance"` *(default)* | $w.\sigma^2 \leftarrow w.\sigma^2 + (\sigma^2_{\max} - w.\sigma^2)(1 - \text{memory\_decay})$ |
| `"precision"` | $\tau \leftarrow \tau + (\tau_{\min} - \tau)(1 - \text{memory\_decay})$ where $\tau = 1/w.\sigma^2$ |

> **Note (Directive 004):** Memory decay is intentionally inert unless `memory_decay < 1.0` is set on the agent's `Characteristics`. Any calibration change requires a dedicated sweep analysis via `hypostases sweep-memory`.

## Inference: Prior Types & Adaptive SMC

`infer()` and `sample_prior()` accept a `prior_type` argument:

| Value | Distribution |
|-------|-------------|
| `"uniform"` *(default)* | Uniform over `reserve_range` |
| `"truncated_normal"` | Normal distribution clipped to `reserve_range` |
| `"log_normal"` | Log-normal centered at 10.0, clipped to `reserve_range` |

Resampling applies **adaptive roughening**: jitter standard deviation scales with the empirical spread of the surviving particle reserve values, preventing degeneracy without over-dispersing tight posteriors.

## Environment: Concurrency Operators

`step_env()` supports four configurable `concurrency_operator` modes for multi-agent resource allocation:

| Operator | Semantics |
|----------|-----------|
| `"shares-first"` *(default)* | Shares replenish pool first; requests served from enlarged pool |
| `"pro-rata"` | Requests served from pre-share pool; shares added after |
| `"priority"` | Greedy allocation in descending `priorities` dict order |
| `"lottery"` | Greedy allocation in random shuffled order |

`WITHDRAW` consequences are opt-in per call:
- `enable_withdraw_fee=True` — deducts `WITHDRAW_FEE` per withdrawal from the pool
- `enable_withdraw_degrade=True` — deducts `WITHDRAW_DEGRADE` per withdrawal (degraded replenishment)

All action choices are logged publicly in `DeltaLog["actions_log"]` for use by the inference engine.

## Multi-Agent Inverse Inference

Two strategies for inferring latent states across a population of agents:

### Joint Particle Filter (`infer_joint`)
Maintains a single particle set over the **product state space** $\Sigma_1 \times \cdots \times \Sigma_N$. Captures cross-agent correlations exactly. Complexity scales as $O(N \cdot P)$ per step.

```python
from hypostases.inference import infer_joint

particles = infer_joint(
    observed_actions,   # list[dict[str, Action]]
    observed_pool_trace,
    xi,
    agent_names=["Agent_A", "Agent_B"],
    n_particles=300,
    concurrency_operator="shares-first",
)
```

### Mean-Field Particle Filter (`infer_mean_field`)
Runs **one independent particle filter per agent**, sharing only the common environment output (`delta_log`) for state propagation. A factorized approximation — scalable to large populations.

```python
from hypostases.inference import infer_mean_field

per_agent_particles = infer_mean_field(
    observed_actions,
    observed_pool_trace,
    xi,
    agent_names=["Agent_A", "Agent_B"],
    n_particles=300,
)
# per_agent_particles["Agent_A"] → list[Particle]
```

## Schema Management & Validation

All ground-truth definitions live in `schema/` as YAML files. Validate any agent state against `invariants.yaml` programmatically:

```python
from hypostases.schemas import assert_invariants, assert_schema_completeness

assert_invariants(agent_state)         # Raises InvariantViolationError if invalid
assert_schema_completeness()           # Audits schema branches for state-independence
```

The static auditor (`audit_schemas.py`) detects branches that degenerate to state-independent constants and raises warnings unless they are decorated with `@declared_simplification` (Directive 003).
