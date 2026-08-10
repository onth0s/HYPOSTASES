# HYPOSTASES

Agent-based modeling framework — formal specification (v4 Target), reference implementation, continuous substrate integration, vanguard game-theoretic mechanism design, and inverse inference engine.

## Project Structure

```
HYPOSTASES/
├── AGENTS.md                  # AI agent behavioral directives (including Rule 005)
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
│   ├── engine/                    # v4 Core simulation engine (types, dynamics, likelihood, continuous)
│   ├── inference/                 # Inverse inference (SMC particle filter, hierarchical, Rao-Blackwellized)
│   ├── simulation/                # Multi-agent simulation & preset scenario generators
│   ├── schemas/                   # Schema loaders, programmatic validators & static auditor
│   ├── cli/                       # Command-line architecture (main, trace, infer, sweep, spec, sweep-memory)
│   └── utils/                     # Package utilities (spec merging)
│
├── tests/                     # Test suite (148 tests covering engine, dynamics, inference, scenarios, CLI)
│
└── misc/                      # Auxiliary documents and notes
```

## Specification v4 Highlights

1. **Primitive State $\sigma = (c, w, g, \rho_{\text{ext}})$**: Strictly four persistent primitives.
2. **Goal Hierarchy $g = u \in \mathbb{R}^{n_k}$**: Latent utility weight vector is primitive. Stochastic policy allocation $\pi \in \Delta(K)$ is transient, recomputed dynamically.
3. **World Model Theory of Mind**: Joint belief distribution over environment state and peer latent states.
4. **Internal Power**: Read-only derived view $\rho_{\text{int}} = \text{proj}_{\text{int}}(c)$. Depletion integrates directly as $\Delta c_{\text{internal}}$ during State Evolution.
5. **Tier-0 Continuous Substrate**: Euler-Maruyama SDE integration for continuous substrate drift and continuous reserve decay.
6. **Strict Cognitive Integrity (Rule 005)**: Strictly forbids artificial human cognitive deficiencies or irrational biases; enforces rational state dynamics.

## Quick Start

```bash
# Install package in development mode
pip install -e ".[dev]"

# Run full test suite (148 tests)
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

# 2. Run inverse inference particle filter with NP-hard mitigations
hypostases infer --particles 300 --steps 12 --lag-window 5 --use-rao-blackwell --output-format table

# 3. Run inverse inference with hierarchical macro/micro filtering
hypostases infer --particles 300 --hierarchical --scenario tragedy

# 4. Run diagnostic / formal 3-condition sweep (Part VII §12.7)
hypostases sweep --steps 10 50 200 --particles 300 --seeds 1 2 3 4 5

# 5. Calibration sweep for memory decay stability
hypostases sweep-memory --steps 20 --decay 0.9 --mode variance
```

## Vanguard Game-Theoretic Mechanisms (2025–2026 Mechanism Design)

1. **Endogenous Scarcity Action Costs ($C_k(S_t)$)**:
   Action costs inflate as pool resources drop below threshold (`SCARCITY_POOL_THRESHOLD = 5.0`), suppressing high-cost actions during resource scarcity via `compute_omega`.
   $$C_k(S_t) = C^{\text{base}}_k \cdot \left(1 + \kappa \cdot \frac{\max(0,\, S_{\text{thresh}} - S_t)}{S_t + \varepsilon}\right)$$

2. **Adaptive Regime-Shift Belief Learning ($\Delta\sigma^2$)**:
   World model belief variance expands non-linearly upon detecting surprise acceleration ($|\Delta\text{surprise}|$), enabling rapid adaptation during environmental regime breaks.

3. **Dynamic Supervisory Governance Scaling ($\lambda$)**:
   Supervisory withdrawal fees scale dynamically based on recent defection prevalence ($n_{\text{withdraw}} / n_{\text{agents}}$), curbing cascading defection spirals.

## Multi-Agent Preset Scenarios (`scenarios.py`)

Generate pre-configured agent populations for stress-testing:

```python
from hypostases.simulation.scenarios import create_scenario_agents

agents = create_scenario_agents("punishment")
```

| Scenario | Description / Dynamics |
|----------|------------------------|
| `'tragedy'` | Tragedy of the Commons (greedy survival & acquisition agents) |
| `'altruism'` | Cooperative pool maintenance (relational-dominant agents) |
| `'freerider'` | Mixed population (cooperators vs. defector status-seeking withdrawers) |
| `'punishment'` | Second-Order Altruistic Punishment (Vigilante paying reserve costs to punish Defectors) |
| `'inequity'` | Inequity Aversion & Relative Deprivation (Peer reserve disparity triggering mood decay) |
| `'deceptive'` | Deceptive Signaling & Asymmetric Information (High reserve agent claiming scarcity via low REQUESTs) |
| `'crowding_out'` | Institutional Crowding-Out / Fine Dilemma (Multi-epoch fee toggling utility hysteresis) |

## Inverse Inference: NP-Hard Mitigation Options

The inference pipeline operationalizes four scalable techniques to mitigate SMC particle degeneracy:

1. **Bounded Lag Window (`lag_window`)**: Truncates observation history to the last $L$ steps, matching memory decay.
2. **Fast Feasibility Gate (`_is_infeasible`)**: Prunes physically impossible particle hypotheses before float arithmetic.
3. **Hierarchical SMC Filter (`infer_hierarchical`)**: Two-pass macro/micro filtering to prevent goal-cluster collapse.
4. **Rao-Blackwellization (`use_rao_blackwell`)**: Closed-form Kalman updates (`evolve_rb`) for continuous Gaussian beliefs.

## Environment: Concurrency Operators

`step_env()` supports four configurable `concurrency_operator` modes for multi-agent resource allocation:

| Operator | Semantics |
|----------|-----------|
| `"shares-first"` *(default)* | Shares replenish pool first; requests served from enlarged pool |
| `"pro-rata"` | Requests served from pre-share pool; shares added after |
| `"priority"` | Greedy allocation in descending `priorities` dict order |
| `"lottery"` | Greedy allocation in random shuffled order |

`WITHDRAW` consequences are opt-in per call:
- `enable_withdraw_fee=True` — deducts dynamic prevalence-scaled `WITHDRAW_FEE` per withdrawal
- `enable_withdraw_degrade=True` — deducts `WITHDRAW_DEGRADE` per withdrawal

All action choices are logged publicly in `DeltaLog["actions_log"]` for inference reweighting.

## Multi-Agent Inverse Inference

Strategies for inferring latent states across populations:

### Joint Particle Filter (`infer_joint`)
Maintains a single particle set over the **product state space** $\Sigma_1 \times \cdots \times \Sigma_N$. Captures cross-agent correlations.

### Mean-Field Particle Filter (`infer_mean_field`)
Runs **one independent particle filter per agent**, sharing only the common environment output (`delta_log`) for state propagation.

## Schema Management & Validation

All ground-truth definitions live in `schema/` as YAML files. Validate any agent state against `invariants.yaml` programmatically:

```python
from hypostases.schemas import assert_invariants, assert_schema_completeness

assert_invariants(agent_state)         # Raises InvariantViolationError if invalid
assert_schema_completeness()           # Audits schema branches for state-independence
```

The static auditor (`audit_schemas.py`) detects branches that degenerate to state-independent constants and raises warnings unless decorated with `@declared_simplification` (Directive 003).
