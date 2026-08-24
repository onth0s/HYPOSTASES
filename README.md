# HYPOSTASES

Agent-based modeling framework — formal specification (v4 Target), reference implementation, continuous substrate integration, vanguard game-theoretic mechanism design, cognitive expansion research fronts, and inverse inference engine.

**Current Version**: `v0.4.0` | **Status**: 360/360 tests passing · `ruff` clean · Formal math & schema verified

## Project Structure

```
HYPOSTASES/
├── AGENTS.md                  # AI agent behavioral directives (including Rules 005 & 008)
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
│   ├── engine/                    # v4 Core simulation engine (types, dynamics, likelihood, memory, _math)
│   ├── world_model/               # Hierarchical world models, conceptual spaces, TEM & CAN grid attractors
│   ├── causal/                    # Structural Causal Models, do-calculus, NOTEARS discovery, RSCMs & policy planning
│   ├── planning/                  # Explicit plan executor, plan library, plan repair
│   ├── alphaevolve/               # Wave 5 AlphaEvolve AST mutator, MAP-Elites QD archives, reservoir & engine
│   ├── active_perception/         # Active sensing & variational free energy dynamics package
│   ├── counterfactual/            # Multi-future rollouts & Dubins PDE reachability package
│   ├── epistemic_utility.py       # Shannon entropy, KL divergence & FEP utility
│   ├── inference/                 # Inverse inference (SMC particle filter, hierarchical, summaries)
│   ├── simulation/                # Multi-agent simulation & scenario registry
│   ├── schemas/                   # Schema loaders, programmatic validators & static auditor
│   ├── cli/                       # Command-line architecture (main, trace, infer, sweep, spec, sweep-memory)
│   └── utils/                     # Package utilities (spec merging)
│
├── tests/                     # Test suite (360 tests covering formal math, engine, dynamics, planning, memory, causal models, inference)
│
├── docs/                      # Cognitive Expansion Fronts & Compass Architecture
│   ├── roadmap_compass.md         # Ratified High-Level Specification and Strategic Compass
│   ├── fronts_index.md            # Master index of 14 Cognitive Expansion Fronts with Status flags
│   └── front_01_..._front_14.md   # Individual self-contained research front specs
│
├── misc/                      # Auxiliary notes & seed documents
│   └── next-steps.md              # Architectural directions & research seeds
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

### Front Implementation Status Matrix

- **[Front 01: Hierarchical World Models](docs/WAVES_AND_FRONTS/front_01_hierarchical_world_models.md)** `[IMPLEMENTED]`: Multi-level semantic representations (`Env -> Objects -> Relations -> Norms`), Gärdenfors conceptual spaces, Whittington et al. (2020) TEM factorization $G \otimes X$, and Burak & Fiete (2009) Continuous Attractor Network (CAN) grid velocity integration (`world_model/`).
- **[Front 02: Explicit Planning Layer](docs/WAVES_AND_FRONTS/front_02_explicit_planning_layer.md)** `[IMPLEMENTED]`: Reusable `Plan` objects with interruption, repair, and plan libraries (`Goal -> Plan -> Action`) (`planning/`).
- **[Front 03: Memory Architecture](docs/WAVES_AND_FRONTS/WAVE_1_FRONT_03/front_03_memory_architecture_spec.md)** `[IMPLEMENTED]`: Structured `WorkingMemory`, `EpisodicMemory`, and `SkillArtifact` consolidation with Zelman et al. (2013, *Front. Comput. Neurosci.*) Kinematic Motion Primitives (kMPs) 2D Gaussian decomposition and Gutfreund et al. (1998, *J. Neurosci.*) stiffening wave control (`engine/memory.py`).
- **[Front 04: Counterfactual Simulation](docs/WAVES_AND_FRONTS/front_04_counterfactual_simulation.md)** `[IMPLEMENTED]`: Multi-future rollouts & expected utility evaluation via Evolutionary Counterfactual Planning (EvoCF, ICLR 2026 MALGAI Workshop), curvature-constrained elastica PDE rollouts, and Dubins (1957) reachability distance pruning (`counterfactual/`).
- **[Front 05: Institution Layer](docs/WAVES_AND_FRONTS/WAVE_3_FRONT_05/front_05_institution_layer_spec.md)** `[IMPLEMENTED]`: First-class institutional entities operating as agents with governance rules, dynamic authority, and fee scaling (`institutions/`).
- **[Front 06: Communication as Bayesian Evidence](docs/WAVES_AND_FRONTS/WAVE_3_FRONT_06/front_06_communication_bayesian_evidence_spec.md)** `[IMPLEMENTED]`: Messages as probabilistic likelihood evidence for belief posterior updates and peer Theory of Mind tracking (`communication/`).
- **[Front 07: Meta-Learning](docs/WAVES_AND_FRONTS/WAVE_4_FRONT_07/front_07_meta_learning_spec.md)** `[IMPLEMENTED]`: Self-adaptation of internal exploration heuristics ($\xi$), search depths, meta-gradient stability, and meta-parameters ($\theta_{\text{meta}}$) (`meta_learning/`).
- **[Front 08: Causal World Models](docs/WAVES_AND_FRONTS/front_08_causal_world_models.md)** `[IMPLEMENTED]`: Structural Causal Models (SCMs), Pearl's 3-rung hierarchy, symbolic $do$-calculus engine (Rules 1-3), 3-step counterfactual cycle (Abduction, Action, Prediction), NOTEARS continuous DAG optimization ($h(W) = \text{tr}(e^{W \circ W}) - d = 0$), cost-optimal interventional planner ($X^* = \arg\min \text{Cost}(X)$), and Relational SCMs (RSCMs) zero-shot cross-skeleton transfer (`causal/`).
- **[Front 09: Active Information Gathering](docs/WAVES_AND_FRONTS/front_09_active_information_gathering.md)** `[IMPLEMENTED]`: Epistemic actions (`INSPECT`, `PROBE`, `MONITOR`) trading material utility for variance reduction ($\Delta \sigma^2$) and Dodig-Crnkovic (2022) Variational Free Energy active inference (`active_perception/`).
- **[Front 10: Mechanism Search](docs/WAVES_AND_FRONTS/WAVE_4_FRONT_10/front_10_mechanism_search_spec.md)** `[IMPLEMENTED]`: Optimization over institutional rule spaces using the simulation harness as a game-theoretic evaluation oracle (`mechanism_search/`).
- **[Front 11: Abductive Reasoning & Hypotheses](docs/WAVES_AND_FRONTS/WAVE_3_FRONT_11/front_11_abductive_reasoning_hypothesis_objects_spec.md)** `[IMPLEMENTED]`: Explanations as explicit `Hypothesis` objects ($H_1, H_2, \dots$) and abductive belief updating (`abduction/`).
- **[Front 12: Scientific Discovery Loop](docs/WAVES_AND_FRONTS/WAVE_4_FRONT_12/front_12_scientific_discovery_loop_spec.md)** `[IMPLEMENTED]`: Iterative hypothesis generation, active experimental design, and empirical model refinement (`scientific_discovery/`).
- **[Front 13: Evolutionary Algorithm Discovery](docs/WAVES_AND_FRONTS/front_13_evolutionary_algorithm_discovery_alphaevolve.md)** `[IMPLEMENTED]`: AlphaEvolve engine with game-theoretic oracles, Quality-Diversity (MAP-Elites) archives, regularized aging evolution, AST code mutator, and FEC functional equivalence checking (`alphaevolve/`).
- **[Front 14: Natural Language as Symbolic Compression](docs/WAVES_AND_FRONTS/WAVE_5_FRONT_14/front_14_natural_language_symbolic_compression_spec.md)** `[IMPLEMENTED]`: Natural language as a lossy, high-density compression operator over continuous state $\sigma \in \mathbb{R}^d$ paired with NLP belief decoder (`natural_language_compression/`, `nlp/`).

## Specification v4 Highlights

1. **Primitive State $\sigma = (c, w, g, \rho_{\text{ext}})$**: Strictly four persistent primitives.
2. **Goal Hierarchy $g = u \in \mathbb{R}^{n_k}$**: Latent utility weight vector is primitive. Stochastic policy allocation $\pi \in \Delta(K)$ is transient, recomputed dynamically.
3. **World Model Theory of Mind**: Joint belief distribution over environment state and peer latent states.
4. **Internal Power**: Read-only derived view $\rho_{\text{int}} = \text{proj}_{\text{int}}(c)$. Depletion integrates directly as $\Delta c_{\text{internal}}$ during State Evolution.
5. **Tier-0 Continuous Substrate**: Euler-Maruyama SDE integration for continuous substrate drift and continuous reserve decay.
6. **Strict Cognitive Integrity (Rule 005)**: Designed strictly to simulate **Machine Learning & Autonomous AI Agents**. Artificially introducing human cognitive deficiencies or irrational biases is strictly prohibited. Human behavior (which includes mental inadequacies on top of rational capability) is a larger **superset** that is explicitly deferred.

## Core State Equation & Closed-Loop Dynamics

### What is an Agent?

#### Informal Definition (Natural Language Abstraction)
In **HYPOSTASES**, an **Agent** is an autonomous, goal-directed computational entity that maintains internal traits and capabilities, models its environment and peers, prioritizes competing objectives, and expends finite resources to execute actions and update its beliefs in response to environmental feedback.

#### Formal Definition
Formally, an agent $i \in \mathcal{I}$ is a discrete-event dynamical system characterized by a 4-tuple of persistent primitive state spaces, a policy parameter space, and an action-perception interface:

$$\text{Agent}^{(i)} \triangleq \langle \mathcal{C}, \mathcal{W}, \mathcal{G}, \mathcal{R}_{\text{ext}}, \Xi, \mathcal{A}, \pi_{\text{decision}}, \text{observe}, \text{feedback}, \text{Evolve} \rangle$$

where:
- $\sigma_t^{(i)} = (c_t^{(i)}, w_t^{(i)}, g_t^{(i)}, \rho_{\text{ext}, t}^{(i)}) \in \Sigma = \mathcal{C} \times \mathcal{W} \times \mathcal{G} \times \mathcal{R}_{\text{ext}}$ is the persistent state at event tick $t$.
- $\xi_t \in \Xi$ is the exploration/temperature policy parameter.
- $\pi_{\text{decision}} : \Sigma \times \Xi \to \Delta(\mathcal{A})$ maps internal state to an action probability distribution.
- $\text{observe} : \mathcal{E} \to \mathcal{S}$ projects global environment state $\mathcal{E}$ to local observations $\mathcal{S}$.
- $\text{feedback} : \mathcal{S} \times \mathcal{S} \times \mathcal{A} \times \mathcal{W} \to \Phi$ computes transition feedback deltas $\Phi = \Delta\mathcal{C} \times \Delta\mathcal{W} \times \Delta\mathcal{G} \times \Delta\mathcal{R}_{\text{ext}}$.
- $\text{Evolve} : \Sigma \times \Phi \times \mathcal{A} \to \Sigma$ integrates feedback deltas into the next persistent state $\sigma_{t+1}^{(i)}$.

---

### 1. The Core State Invariant ($\sigma$)
An agent's persistent latent configuration at discrete Tier-1 event time $t$ is strictly defined as a 4-tuple of independent primitives:

$$\sigma^{(i)}_t = (c^{(i)}_t, w^{(i)}_t, g^{(i)}_t, \rho_{\text{ext}, t}^{(i)}) \in \mathcal{C} \times \mathcal{W} \times \mathcal{G} \times \mathcal{R}_{\text{ext}}$$

- **Characteristics ($c \in \mathcal{C} = \mathbb{R}^{n_c}$)**: Persistent agent traits and internal states (e.g. resilience, sociality, stamina/energy reserve, mood).
- **World Model ($w \in \mathcal{W}$)**: Joint belief distribution over environmental state $S$ and peer latent states $\prod_{j \neq i} \Sigma^{(j)}$ (Theory of Mind representation), parameterized via $(\mu, \sigma^2)$ or continuous attractor grids.
- **Goal Hierarchy ($g = u \in \mathcal{G} = \mathbb{R}^{n_k}$)**: Latent utilities/priorities over schema-defined goal dimensions (Survival, Acquisition, Relational, Status).
- **External Power ($\rho_{\text{ext}} \in \mathcal{R}_{\text{ext}} = \mathbb{R}_{\ge 0}^{n_r}$)**: Externally-held resource vectors (capital, authority, social capital).

**Derived Views & Transient Quantities (Non-Persistent):**
- **Internal Power (read-only)**: $\rho_{\text{int}} = \text{proj}_{\text{int}}(c) \in \mathbb{R}_{\ge 0}^{n_{r, \text{int}}}$
- **Willingness**: $\omega = \text{derive}_{\Omega}(u, \rho_{\text{ext}}, \rho_{\text{int}}, c) \in \mathbb{R}_{\ge 0}^{n_k}$
- **Dynamic Policy Allocation**: $\pi_t = \text{softmax}(u_t / \xi_t) \in \Delta(K)$
- **Potentialities**: $\mathcal{P}(c) = \{c' \in \mathcal{C} \mid c' \text{ reachable from } c \text{ under budget } R\}$

---

### 2. Closed-Loop Function Composition (Four-Stage Loop)

The operational loop maps the current agent state to the next state via total function composition:

$$\sigma_{t+1} = \text{Evolve}(\sigma_t, a_t, \delta_t) = \sigma_t \oplus \Phi_t$$

1. **Policy Stage (Forward Simulation)**:
   $$a_t \sim \pi_{\text{decision}}(\sigma_t; \, \xi_t) \quad \text{where} \quad \text{cost}_{\text{ext}}(a_t) \le \rho_{\text{ext}, t}$$

2. **Environment Stage**:
   $$e_{t+1} = \text{step\_env}(e_t, \{a^{(i)}_t\})$$

3. **Feedback Stage**:
   $$\Phi_t = \text{feedback}(\text{obs}(e_t), \text{obs}(e_{t+1}), a_t, w_t) = (\Delta c, \Delta w, \Delta g, \Delta \rho_{\text{ext}})$$

4. **State Evolution Stage**:
   $$\begin{aligned}
   c_{t+1} &= c_t + \phi_t.\Delta c \\
   w_{t+1} &= \text{update}_{\mathcal{W}}(w_t, \phi_t.\Delta w) \\
   g_{t+1} &= g_t + \phi_t.\Delta g \\
   \rho_{\text{ext}, t+1} &= \rho_{\text{ext}, t} + \phi_t.\Delta \rho_{\text{ext}} - \text{cost}_{\text{ext}}(a_t, \rho_{\text{ext}, t}, \rho_{\text{int}, t})
   \end{aligned}$$

## Machine Learning Agent Modeling & Cognitive Integrity (Rule 005)

HYPOSTASES is fundamentally engineered to model **Machine Learning Agents and Autonomous AI Systems**, operating under spec-compliant, rational, and optimal multi-agent state dynamics.

* **We Are Not Modeling Humans**: Human cognitive flaws, emotional irrationality, sunk-cost fallacies, and anthropomorphic cognitive defects are **strictly prohibited** in the core engine.
* **Human Psychology as Deferred Superset**: Humans *can* act hyperrationally, but also encompass mental inadequacies—a much larger behavioral **superset** than we scope here. Human behavioral defects are explicitly deferred in favor of foundational optimal AI game theory and mechanism design.
* **Strict Behavioral Enforcer**: Any proposal or code modification that attempts to inject artificial human cognitive deficiencies into the engine triggers an immediate critical flag (`> [!CRITICAL]`) per Rule 005.



```bash
# Install package in development mode
pip install -e ".[dev]"

# Run full test suite (360 tests)
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
   Empirically evaluated across a pre-registered $3 \times 3$ $\{\kappa, \lambda\}$ grid sweep (`tests/test_condition1_grid_sweep.py`, 10 seeds `[100, 110)`, 500 steps, $N=5$ agents):

   **Run A: Pure Un-Regularized Dynamics (`UTILITY_DECAY_RATE = 0.0`)**
   | $\kappa$ (Scarcity) | $\lambda$ (Governance) | Stationarity Ratio ($V_{500}/V_{100}$) | Cosine Sim ($S_{\text{cos}}$) | Pre-Registered Classification |
   |---|---|---|---|---|
   | `0.0` | `0.0–2.0` | **0.348–0.407** | **0.802–0.838** | Non-Attracting / Regime Drift |
   | `0.5` | `0.0–2.0` | **0.529–0.640** | **0.721–0.788** | Non-Attracting / Regime Drift |
   | `1.0` | `0.0–2.0` | **0.582–0.672** | **0.728–0.808** | Non-Attracting / Regime Drift |

   **Run B: Leaky Decay Regularized (`UTILITY_DECAY_RATE = 0.05`)**
   | $\kappa$ (Scarcity) | $\lambda$ (Governance) | Stationarity Ratio ($V_{500}/V_{100}$) | Cosine Sim ($S_{\text{cos}}$) | Pre-Registered Classification |
   |---|---|---|---|---|
   | `0.0` | `0.0–2.0` | **0.015–0.023** | **0.938–0.944** | **Attractor Confirmed** |
   | `0.5` | `0.0–2.0` | **0.014–0.023** | **0.937–0.944** | **Attractor Confirmed** |
   | `1.0` | `0.0–2.0` | **0.014–0.024** | **0.937–0.944** | **Attractor Confirmed** |

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

## Pluggable Domain Architecture & Segmentation

The HYPOSTASES engine is fully domain-agnostic. All game or environment plugins strictly adhere to the `Domain` protocol interface (`hypostases.domains.Domain`) and register with `DomainRegistry`.

### Core Installation vs Plugin Installation
- **Core Engine (Domain-Agnostic & Headless)**:
  ```bash
  pip install -e .
  ```
- **With Chess Domain Plugin**:
  ```bash
  pip install -e .[chess]
  ```

### Registering and Accessing Domains via `DomainRegistry`

Domain plugins automatically register themselves when imported:

```python
from hypostases.domains import DomainRegistry
import hypostases.plugins.domains.chess  # Registers 'chess'

# Dynamically instantiate domain by string identifier
domain = DomainRegistry.get("chess", representation_mode="full")
initial_state = domain.initial_state()
valid_moves = domain.valid_actions(initial_state)
```

