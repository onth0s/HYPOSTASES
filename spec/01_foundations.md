---
part_number: 1
title: "Part I: Foundations"
depends_on: []
sections: [0, 1, 2]
---
# HYPOSTASES — Formal Specification
## Part I: Foundations

Status: draft v2 (formalization pass over the v1 architectural abstract)
Scope of this part: notation, time model, typed state spaces for all ten components.

---

## 0. Notational Conventions

- Agents are indexed `i ∈ I`, where `I` is a finite or countable agent index set.
- Scalars are lowercase italic (`n`, `t`). Vectors are bold lowercase (**c**, **g**). Sets are uppercase calligraphic-style written as plain caps (`C`, `G`). Functions are lowercase with explicit domain/codomain.
- `ℝ≥0` denotes non-negative reals. `Δ(X)` denotes the probability simplex over finite set `X` (i.e. distributions over `X`).
- Superscript `(i)` marks agent ownership: `c^(i)_t` is agent `i`'s Characteristics vector at time `t`.
- `t` is a *tier-1 event index* for a given agent (see §1), not wall-clock time, unless stated.

---

## 1. Time Model

### 1.1 Motivation

Three candidate time models were considered:

| Model | Pros | Cons |
|---|---|---|
| Discrete synchronous | Deterministic, reproducible, simple to implement and test | Forces heterogeneous agents onto one clock; wastes resolution on slow agents, starves fast ones |
| Discrete asynchronous | Matches real heterogeneous agents (event-driven); no wasted steps | No global "state of the system at time t"; feedback attribution is ambiguous under overlapping actions; harder to prove properties about |
| Continuous | Physically honest; supports equilibrium/stability analysis via calculus | Not directly implementable; every simulation discretizes it anyway |

These are not mutually exclusive theories of time — they are different **sampling resolutions of one underlying process**. HYPOSTASES adopts a three-tier model that recovers each as a limiting case.

### 1.2 The Three Tiers

**Tier 0 — Substrate (continuous).**
The "true" process is a continuous-time stochastic process. This tier is never simulated directly; it exists so that Part III (Dynamics & Stability) can define equilibria, attractors, and Lyapunov-style stability using standard continuous-time tools. Formally, each component trajectory (e.g. **c**^(i)(t)) is a càdlàg (right-continuous, left-limit) stochastic process on `t ∈ ℝ≥0`.

**Tier 1 — Operational (discrete, asynchronous).**
This is the tier the engine actually runs. Each agent `i` has its own local event clock producing a strictly increasing sequence of event times `t^(i)_0 < t^(i)_1 < t^(i)_2 < ...`. At each `t^(i)_k`, agent `i` executes exactly one iteration of the core loop (§ Part II). Inter-event intervals `Δt^(i)_k = t^(i)_k − t^(i)_{k-1}` may be fixed, sampled, or determined by the agent's own Willingness/Power state (e.g., high Power → shorter reaction latency). This tier is the ground truth for causal ordering: Feedback at `t^(i)_k` is attributed only to Environment deltas visible to agent `i` as of `t^(i)_k`, resolved by a global total order over all `(i, k)` event pairs (broken by a tie-rule, e.g. agent-index order, for simultaneous events).

**Tier 2 — Epochal (discrete, synchronous).**
A coarser global barrier at times `T_0 < T_1 < T_2 < ...`, independent of any single agent's clock. Between consecutive epoch boundaries, arbitrarily many Tier-1 events occur, in any order, for any agents. At each `T_n`, the Environment state and every agent's Characteristics are *snapshotted*. Tier 2 exists purely for measurement, reproducibility, and cross-agent comparison — it has no causal role and does not gate Tier-1 execution.

### 1.3 Reconciliation & Game-Theoretic Divergence

The three tiers are related by limits, but with an explicit game-theoretic divergence note:

1. **Continuous (Tier 0) is the continuous limit of Tier 1** as `max_i Δt^(i)_k → 0`. Tier 1 is formally a point-process discretization of Tier 0.
2. **Synchronous discrete (Tier 2) is a non-equivalent discrete discretization operator.** While Tier 1 enforces sequential causal ordering where simultaneous moves have measure zero, collapsing Tier 1 onto a Tier 2 synchronous epoch barrier (`T_n = t^(i)_n ∀i`) introduces simultaneous-move game dynamics (e.g. strategic ambiguity matrices, payoff collisions) that do not exist in sequential Tier 1 execution. 

Therefore, Tier 2 is **not** a trivial special case of Tier 1; it is an explicit discretization scheme that alters game-theoretic equilibrium structures. An engine running in pure Tier 2 mode must define an explicit simultaneous collision-resolution operator over concurrent actions.

### 1.4 Consequence for the rest of the spec

All update equations in Part II are written at Tier 1 (per-agent, per-event), since that is the operational semantics. Part III (Dynamics & Stability) additionally uses the Tier 0 continuous limit for equilibrium analysis. Cross-agent claims (e.g. "the market cleared") are Tier-2 statements evaluated over a snapshot barrier.

---

## 2. Typed State Spaces

### 2.0 Architectural revision note (v4)

This document updates the v3 formalization to resolve five formal architectural contentions:

| Component | v3 status | v4 status | Reason |
|---|---|---|---|
| Characteristics | primitive | **primitive** | expanded to receive internal power consumption deltas |
| World Model | primitive | **primitive** (extended) | `w` expanded to model peer latent states $\Delta(S \times \prod_{j \neq i} \Sigma^{(j)})$ |
| Goal Hierarchy | primitive | **primitive** (redefined) | `g = u ∈ ℝ^{n_k}` (latent utility parameters); stochastic policy $\pi \in \Delta(K)$ moved strictly to `π_decision` |
| Action | primitive | **primitive** | unchanged |
| Environment | primitive | **primitive** | localized field interaction dynamics |
| Feedback | primitive | **primitive** | carries explicit $\Delta c_{\text{internal}}$ for internal resource depletion |
| Power (external) | primitive | **primitive** | unchanged |
| Power (internal) | derived | **derived (read-only)** | consumption maps to $\Delta c_{\text{internal}}$ in State Evolution; never mutated directly |
| Willingness | derived | **derived** | unchanged |
| Potentialities | derived | **derived** | unchanged |
| Index of Exploration | policy parameter | **policy parameter** | unchanged |

### 2.1 Design decision: fixed-dimension vectors with versioned schemas

Per spec directive, every component state is a **fixed-dimension real (or product) vector at any given schema version**. "Fixed-dimension" is reconciled with "the architecture must be revisable" (Design Philosophy V, Falsifiability) via **schema versioning**: a component's dimensionality is fixed *within* a schema version `v`, and a schema migration `v → v+1` is a declared, total function on the old space, not a silent reinterpretation.

```
Schema(X, v) : dimensionality and field layout of component X at version v
Migrate(X, v→v+1) : ℝ^{dim(X,v)} → ℝ^{dim(X,v+1)}, total, declared with the revision
```

### 2.2 Component State Spaces

#### 2.2.1 Primitives (independent, persistent state)

**Characteristics** `c ∈ C = ℝ^{n_c}`
Coordinates partitioned into named blocks by convention: knowledge, skill, cognitive/personality, physical/internal-energy, emotional-baseline. `C` is the space the whole loop reads and writes. Internal resource depletion (energy, physical stamina) is integrated directly into `c` via feedback deltas $\Delta c_{\text{internal}}$.

**World Model** `w ∈ W = Δ(S × ∏_{j ≠ i} Σ^{(j)}) × F`
A joint belief state over environment-state space `S` and peer latent primitive states `Σ^{(j)}` (Theory of Mind representation), paired with a predictive transition function family `F = { f : (S × ∏_{j ≠ i} Σ^{(j)}) × A → Δ(S × ∏_{j ≠ i} Σ^{(j)}) }`.

**Goal Hierarchy** `g = u ∈ G = ℝ^{n_k}`
`K` is a finite set of goal categories (Survival, Curiosity, Status, ...; schema-defined). `u ∈ ℝ^{n_k}` gives latent per-goal utilities/priorities. **Canonical semantics (v4): `g` stores strictly persistent utility weights $u$.** Goal allocation probabilities $\pi \in \Delta(K)$ are non-persistent and computed dynamically inside `π_decision`.

**Power (external)** `ρ_ext ∈ R_ext = ℝ≥0^{n_r}`
A non-negative resource vector over externally-held resources only: capital, time, authority, technology access, social capital.

**Action** `a ∈ A` (finite or continuous, schema-defined)
The output space. `A` supports `cost_ext : A × (R_ext, ρ_int) → R_ext` and `cost_int : A × (R_ext, ρ_int) → ΔC_internal`.

**Environment** `e ∈ E = ℝ^{n_e}`
Shared environment space representing spatial/local field states. Formally global `E`; each agent observes `obs^(i) : E → S`.

**Feedback** `φ ∈ Φ = ΔC × ΔW × ΔG × ΔR_ext`
A tuple of deltas updating primitive persistent states. `ΔC` explicitly includes internal power consumption deltas $\Delta c_{\text{internal}}$.

#### 2.2.2 Derived quantities (no independent state; recomputed from primitives)

**Power (internal)** `ρ_int = proj_int(c) ∈ ℝ≥0^{n_{r,int}}`
A declared, read-only projection of Characteristics coordinates (health, energy, physical competence) into a resource view. `ρ_int` is read by `π_decision` to constrain action affordability. It is never mutated directly; action execution emits $\Delta c_{\text{internal}} \in \Phi$, integrated into $c$ during State Evolution.

**Potentialities** `p = P(c) = { c' ∈ C : c' reachable from c under transition kernel within budget R } ⊆ C`
A reachable-set query parameterized by current Characteristics and resource budget `R`.

**Willingness** `ω = derive_Ω(u, ρ_ext, ρ_int, c) ∈ ℝ≥0^{n_k}`
A non-negative vector computed each step as a function of latent Goal Hierarchy utilities `u` scaled by current affordability.

#### 2.2.3 Policy parameter (not state; governs the Decision Policy)

**Index of Exploration** `ξ ∈ Ξ = ℝ≥0^{n_ξ}`
A temperature/entropy parameter passed into the Decision Policy function `π_decision`.

### 2.3 The Full Per-Agent State (v4, revised)

```
σ^(i)_t = ( c^(i)_t, w^(i)_t, g^(i)_t, ρ_ext^(i)_t ) ∈ C × W × G × R_ext
```
Four primitive persistent states: Characteristics $c$ (including internal energy state), World Model $w$ (including peer belief states), Goal Hierarchy $g=u$ (latent utilities), and External Power $\rho_{\text{ext}}$. Derived quantities ($\rho_{\text{int}}, P(c), \omega$) are read-only views; $\xi$ is a policy parameter; $a$ is an action event; $\phi$ is a transient delta tuple.

---

*(Continued in Part II: Update Dynamics, and §3: Critique / Redundancy Analysis)*
