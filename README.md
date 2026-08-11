# HYPOSTASES

Agent-based modeling framework — formal specification (v4 Target), reference implementation, continuous substrate integration, vanguard game-theoretic mechanism design, cognitive expansion research fronts, and inverse inference engine.

**Current Version**: `v0.3.0` | **Status**: 164/164 tests passing · `ruff` clean · Fully audited and refactored

## Project Structure

```
HYPOSTASES/
├── AGENTS.md                  # AI agent behavioral directives (including Rule 005)
├── pyproject.toml             # Project config (Python >=3.11, ruff & pytest settings)
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
│   ├── engine/                    # v4 Core simulation engine (types, dynamics, likelihood, _math)
│   ├── inference/                 # Inverse inference (SMC particle filter, hierarchical, summaries)
│   ├── simulation/                # Multi-agent simulation & scenario registry
│   ├── schemas/                   # Schema loaders, programmatic validators & static auditor
│   ├── cli/                       # Command-line architecture (main, trace, infer, sweep, spec, sweep-memory)
│   └── utils/                     # Package utilities (spec merging)
│
├── tests/                     # Test suite (164 tests covering engine, dynamics, inference, regressions)
│
└── misc/                      # Cognitive Expansion Seeds & Spliced Research Fronts
    ├── fronts_index.md            # Master index of 14 Cognitive Expansion Fronts
    ├── next-steps.md              # Architectural directions & research seeds
    └── front_01_..._front_14.md   # Individual self-contained research front specs
```

## Cognitive Expansion Roadmap (SOTA 2026 Aligned)

HYPOSTASES is evolving from an agent simulation engine into a unified generative engine for **intelligent reasoning, multi-agent swarm distillation, and evolutionary algorithm discovery**.

All 14 cognitive expansion fronts maintain the core state invariant: $\sigma = (c, w, g, \rho_{\text{ext}})$.

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ WAVE 1: Single-Agent Foundations (Memory, Lookahead & Active Sensing)                     │
│ [Front 03: Memory] ──► [Front 04: Counterfactual Simulation] ──► [Front 09: Active Sensing] │
└───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ WAVE 2: Structural Abstraction & Metacognitive Planning                                  │
│ [Front 02: Explicit Planning] ──► [Front 01: Hierarchical Models] ──► [Front 08: Causal]  │
└───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ WAVE 3: Multi-Agent Epistemology & Swarm Mechanics                                       │
│ [Front 06: Bayesian Comm] ──► [Front 11: Abduction & Hypotheses] ──► [Front 05: Inst.]    │
└───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ WAVE 4: Recursive Adaptation & Discovery Loops                                           │
│ [Front 07: Meta-Learning] ──► [Front 10: Mechanism Search] ──► [Front 12: Discovery Loop] │
└───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ WAVE 5: Universal Scaling & Symbolic Generalization                                      │
│ [Front 13: AlphaEvolve Engine] ──► [Front 14: Natural Language Compression]               │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

- **[Front 01: Hierarchical World Models](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/misc/front_01_hierarchical_world_models.md)**: Multi-level semantic representations (`Env -> Objects -> Relations -> Norms`).
- **[Front 02: Explicit Planning Layer](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/misc/front_02_explicit_planning_layer.md)**: Reusable `Plan` objects with interruption, repair, and plan libraries (`Goal -> Plan -> Action`).
- **[Front 03: Memory Architecture](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/misc/front_03_memory_architecture.md)**: Structured `WorkingMemory`, `EpisodicMemory`, and `SkillArtifact` consolidation.
- **[Front 04: Counterfactual Simulation](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/misc/front_04_counterfactual_simulation.md)**: Multi-future rollouts & expected utility evaluation (aligned with ICML/ICLR 2026 EvoCF).
- **[Front 05: Institution Layer](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/misc/front_05_institution_layer.md)**: First-class institutional entities operating as agents with governance rules and authority.
- **[Front 06: Communication as Bayesian Evidence](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/misc/front_06_communication_bayesian_evidence.md)**: Messages as probabilistic likelihood evidence for belief posterior updates.
- **[Front 07: Meta-Learning](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/misc/front_07_meta_learning.md)**: Self-adaptation of internal exploration heuristics ($\xi$), search depths, and learning parameters.
- **[Front 08: Causal World Models](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/misc/front_08_causal_world_models.md)**: Structural Causal Models (SCMs) and intervention ($do$-calculus) in $w$.
- **[Front 09: Active Information Gathering](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/misc/front_09_active_information_gathering.md)**: Epistemic actions (`INSPECT`, `PROBE`, `MONITOR`) trading material utility for variance reduction ($\Delta \sigma^2$).
- **[Front 10: Mechanism Search](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/misc/front_10_mechanism_search.md)**: Optimization over institutional rule spaces using the simulation harness as an oracle.
- **[Front 11: Abductive Reasoning & Hypotheses](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/misc/front_11_abductive_reasoning_hypothesis_objects.md)**: Explanations as explicit `Hypothesis` objects ($H_1, H_2, \dots$) and abductive inference.
- **[Front 12: Scientific Discovery Loop](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/misc/front_12_scientific_discovery_loop.md)**: Iterative hypothesis generation, experimental design, and empirical model refinement.
- **[Front 13: Evolutionary Algorithm Discovery](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/misc/front_13_evolutionary_algorithm_discovery_alphaevolve.md)**: AlphaEvolve engine with game-theoretic & endogenous scarcity feedback oracles.
- **[Front 14: Natural Language as Symbolic Compression](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/misc/front_14_natural_language_symbolic_compression.md)**: Natural language as a lossy, high-density compression operator over continuous state $\sigma \in \mathbb{R}^d$.

## Specification v4 Highlights

1. **Primitive State $\sigma = (c, w, g, \rho_{\text{ext}})$**: Strictly four persistent primitives.
2. **Goal Hierarchy $g = u \in \mathbb{R}^{n_k}$**: Latent utility weight vector is primitive. Stochastic policy allocation $\pi \in \Delta(K)$ is transient, recomputed dynamically.
3. **World Model Theory of Mind**: Joint belief distribution over environment state and peer latent states.
4. **Internal Power**: Read-only derived view $\rho_{\text{int}} = \text{proj}_{\text{int}}(c)$. Depletion integrates directly as $\Delta c_{\text{internal}}$ during State Evolution.
5. **Tier-0 Continuous Substrate**: Euler-Maruyama SDE integration for continuous substrate drift and continuous reserve decay.
6. **Strict Cognitive Integrity (Rule 005)**: Designed strictly to simulate **Machine Learning & Autonomous AI Agents**. Artificially introducing human cognitive deficiencies or irrational biases is strictly prohibited. Human behavior (which includes mental inadequacies on top of rational capability) is a larger **superset** that is explicitly deferred.

## Machine Learning Agent Modeling & Cognitive Integrity (Rule 005)

HYPOSTASES is fundamentally engineered to model **Machine Learning Agents and Autonomous AI Systems**, operating under spec-compliant, rational, and optimal multi-agent state dynamics.

* **We Are Not Modeling Humans**: Human cognitive flaws, emotional irrationality, sunk-cost fallacies, and anthropomorphic cognitive defects are **strictly prohibited** in the core engine.
* **Human Psychology as Deferred Superset**: Humans *can* act hyperrationally, but also encompass mental inadequacies—a much larger behavioral **superset** than we scope here. Human behavioral defects are explicitly deferred in favor of foundational optimal AI game theory and mechanism design.
* **Strict Behavioral Enforcer**: Any proposal or code modification that attempts to inject artificial human cognitive deficiencies into the engine triggers an immediate critical flag (`> [!CRITICAL]`) per Rule 005.



```bash
# Install package in development mode
pip install -e ".[dev]"

# Run full test suite (158 tests)
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

4. **Full 4D Goal-Hierarchy Feedback ($\Delta g$)**:
   `feedback()` emits nonzero utility deltas across **all four** goal-hierarchy dimensions on every action:
   - `REQUEST` (success) → $\Delta g[\text{SURVIVAL}] > 0$, $\Delta g[\text{ACQUISITION}] > 0$
   - `REQUEST` (shortfall) → $\Delta g[\text{SURVIVAL}] < 0$
   - `SHARE` → $\Delta g[\text{RELATIONAL}] > 0$
   - `WITHDRAW` (no governance fee) → $\Delta g[\text{STATUS}] > 0$; cancelled under active fee

   This prevents latent utility $g.u$ from collapsing onto a lower-dimensional invariant subspace.

5. **Softmax Jacobian Attenuation (Pure Fixed-Point Attractor)**:
   Before emitting $\Phi$, `feedback()` scales each $\Delta g[k]$ by the marginal policy sensitivity $\pi_k(1-\pi_k)$ — the diagonal of the softmax Jacobian:
   $$\Delta g[k] \leftarrow \Delta g[k] \cdot \pi_k (1 - \pi_k)$$
   As dimension $k$ dominates ($\pi_k \to 1$), $\Delta g[k] \to 0$ naturally. This yields **pure un-regularized fixed-point attractors** (`UTILITY_DECAY_RATE = 0.0`) without any artificial restoring force. Empirically verified across a $3 \times 3$ $\{\kappa, \lambda\}$ grid sweep:

   | $\kappa$ (Scarcity) | $\lambda$ (Governance) | Stationarity Ratio | Classification |
   |---|---|---|---|
   | `0.0` | `0.0–2.0` | **0.160–0.179** | Fixed-Point Attractor |
   | `0.5` | `0.0–2.0` | **0.347–0.377** | Transition Regime |
   | `1.0` | `0.0–2.0` | **0.554–0.652** | High-Pressure Adaptive Drift |

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
