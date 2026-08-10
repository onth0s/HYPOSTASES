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

### 1.3 Reconciliation

The three tiers are related by two limits, not by three separate assumptions:

1. **Synchronous discrete is a special case of Tier 1**, obtained when `Δt^(i)_k` is identical and simultaneous across all agents — i.e., Tier 1 collapses onto Tier 2 with `T_n = t^(i)_n ∀i`.
2. **Continuous (Tier 0) is the limit of Tier 1** as `max_i Δt^(i)_k → 0`. Tier 1 is formally a Poisson-clock (or arbitrary point-process) discretization of Tier 0.

This gives one time model with adjustable resolution rather than three competing ones. An implementation may legally special-case to pure Tier 2 (simplest, matches "discrete synchronous") without violating the spec, since it is a strict special case.

### 1.4 Consequence for the rest of the spec

All update equations in Part II are written at Tier 1 (per-agent, per-event), since that is the operational semantics. Part III (Dynamics & Stability) additionally uses the Tier 0 continuous limit for equilibrium analysis. Cross-agent claims (e.g. "the market cleared") are Tier-2 statements evaluated over a snapshot.

---

## 2. Typed State Spaces

### 2.0 Architectural revision note (v3)

This document supersedes the v2 formalization's treatment of the ten v1 components as flat, equal-status primitives. Per Design Philosophy V (Falsifiability — "whenever reality cannot be explained by the current architecture, the architecture must be revised") and Design Philosophy I (Minimalism — "every abstraction must justify its existence"), the v2 formalization pass (Part III §5 of that draft) surfaced four components that do not independently justify their existence as primitive state. This revision resolves each:

| Component | v1/v2 status | v3 status | Reason |
|---|---|---|---|
| Characteristics | primitive | **primitive** | unchanged |
| World Model | primitive | **primitive** | unchanged |
| Goal Hierarchy | primitive | **primitive** (redefined, §2.2) | soft `Δ(K)` is canonical, not one-hot |
| Action | primitive | **primitive** | unchanged |
| Environment | primitive | **primitive** | unchanged |
| Feedback | primitive | **primitive** | unchanged |
| Power | primitive (unified) | **split**: `Power_external` primitive, `Power_internal` derived | internal block was an undeclared duplicate of Characteristics |
| Willingness | primitive | **derived** | total, stateless function of Goals × Power × Characteristics; no independent update rule ever existed for it even in v1 |
| Potentialities | primitive | **derived** | a reachable-set query on Characteristics; never had independent state |
| Index of Exploration | primitive (pipeline stage) | **policy parameter** | formally a temperature term on the Decision Policy, not a component with its own state or update dynamics |

Net result: **six unconditional primitives**, **one split primitive** (`Power_external`), **three derived quantities** (`Power_internal`, `Willingness`, `Potentialities`), **one policy parameter** (`ξ`). This is a real architectural revision, not a relabeling — the persistent per-agent state tuple `σ` (§2.3) shrinks accordingly.

### 2.1 Design decision: fixed-dimension vectors with versioned schemas

Per spec directive, every component state is a **fixed-dimension real (or product) vector at any given schema version**. "Fixed-dimension" is reconciled with "the architecture must be revisable" (Design Philosophy V, Falsifiability) via **schema versioning**: a component's dimensionality is fixed *within* a schema version `v`, and a schema migration `v → v+1` is a declared, total function on the old space, not a silent reinterpretation.

```
Schema(X, v) : dimensionality and field layout of component X at version v
Migrate(X, v→v+1) : ℝ^{dim(X,v)} → ℝ^{dim(X,v+1)}, total, declared with the revision
```

This means: within any run, all math is fixed-dimension linear/vector algebra (clean, provable). Across architecture revisions (the kind Design Philosophy V calls for when reality breaks the current model), a explicit, auditable migration function is required — you cannot silently redefine a vector's meaning without a recorded `Migrate` function. This is the "account for major architectural revisions" requirement: revision is a first-class, typed operation, not an escape hatch from typing.

### 2.2 Component State Spaces

#### 2.2.1 Primitives (independent, persistent state)

**Characteristics** `c ∈ C = ℝ^{n_c}`
Coordinates partitioned into named blocks by convention (not enforced by the type, only by schema docs): knowledge, skill, cognitive/personality, physical, emotional-baseline. `C` is the space the whole loop reads and writes.

**World Model** `w ∈ W = Δ(S) × F`
A belief state over an environment-state space `S` (a distribution, `Δ(S)`), paired with a causal/predictive function family `F = { f : S × A → Δ(S) }` (the agent's model of environment dynamics, `A` = action space, defined below). `w = (belief, f)`.

**Goal Hierarchy** `g ∈ G = Δ(K) × ℝ^{n_k}`
`K` is a finite set of goal categories (Survival, Curiosity, Status, ...; schema-defined). `g = (π, u)` where `π ∈ Δ(K)` is a priority distribution over goals and `u ∈ ℝ^{n_k}` gives per-goal utility/target values. **Canonical semantics (v3): `π` is a soft distribution.** Strict hierarchy (lexicographic dominance of one goal over all others) is recovered only as the degenerate one-hot special case of `π`. v1's own examples ("Survival, Security, Curiosity, Status...") read as competing, weighted objectives in the surrounding prose, not a strict stack — the soft form is what the component actually describes; the name "Hierarchy" is retained for continuity but should not be read as implying strict lexicographic ordering.

**Power (external)** `ρ_ext ∈ R_ext = ℝ≥0^{n_r}`
A non-negative resource vector over externally-held resources only: capital, time, authority, technology access, social capital. This is the sole Power primitive in v3 (see §2.2.2 for why internal "Power" is not independently primitive).

**Action** `a ∈ A` (finite or continuous, schema-defined)
The output space. No internal structure is imposed beyond: `A` must support a `cost : A × (R_ext, ρ_int) → R_ext` function (an action consumes Power) and `A` must be within the support of `w.f`.

**Environment** `e ∈ E = ℝ^{n_e}`
Shared, agent-indexed only through visibility/access functions, not through separate copies. Formally one global `E`; each agent observes `obs^(i) : E → S`.

**Feedback** `φ ∈ Φ = ΔC × ΔW × ΔG × ΔR_ext`
A tuple of deltas, one per *primitive* it updates. v2 included `ΔP` (Potentiality-estimate deltas) in this tuple; v3 removes it, since Potentialities is derived (§2.2.2) and is recomputed from `c`, never updated by Feedback directly — folding a derived quantity into the delta tuple was itself a symptom of the v2 ambiguity this revision resolves.

#### 2.2.2 Derived quantities (no independent state; recomputed from primitives)

These are not part of `σ` (§2.3). Each is a pure, declared function of primitive state, evaluated on demand.

**Power (internal)** `ρ_int = proj_int(c) ∈ ℝ≥0^{n_{r,int}}`
A declared projection of a subset of Characteristics coordinates (health, energy, executive function, competence, knowledge) into a resource-shaped view. v1/v2 listed these same quantities under both Power and Characteristics as if independently duplicated; v3 resolves the overlap by typing internal Power as *read-only derived from* `c`, never separately stored or separately updated. `proj_int` is schema-declared (which `c` coordinates map to which `ρ_int` slots).

**Potentialities** `p = P(c) = { c' ∈ C : c' reachable from c under the agent's transition kernel within budget R } ⊆ C`
A reachable-set query, parameterized by current Characteristics and a resource/time budget `R` (itself typically drawn from `ρ_ext` and `ρ_int`). No update equation exists for `P` because none is needed: it is fully determined by `c` and `R` at query time.

**Willingness** `ω = derive_Ω(g, ρ_ext, ρ_int, c) ∈ ℝ≥0^{n_k}`
A non-negative vector, one scalar per goal category, computed each step as a function of current Goal Hierarchy priorities scaled by current affordability (Power). v1 named Willingness a top-level component but never specified how Feedback updates it (Feedback's own listed targets — Characteristics, World Model, Goal Hierarchy, Potentiality- and Power-*estimates* — omitted it). v3 resolves this gap by demotion rather than invented rule: Willingness has no memory of its own; it is recomputed fresh from primitives at every decision point.

#### 2.2.3 Policy parameter (not state; governs the Decision Policy)

**Index of Exploration** `ξ ∈ Ξ = ℝ≥0^{n_ξ}`
v1 diagrammed this as a pipeline stage of equal standing to Characteristics or World Model. Formally it has no update equation of its own and no role except as a temperature/entropy parameter passed into the Decision Policy function `π_decision` (Part II §3.1). v3 therefore excludes it from `σ`: it is a hyperparameter of the policy, optionally following its own schedule (e.g. annealed over epochs), not agent state that Feedback writes to.

### 2.3 The Full Per-Agent State (v3, revised)

```
σ^(i)_t = ( c^(i)_t, w^(i)_t, g^(i)_t, ρ_ext^(i)_t ) ∈ C × W × G × R_ext
```

Four primitives, down from the v2 draft's six-tuple. `ρ_int`, `P(c)`, `ω` are derived on demand from `σ_t` (§2.2.2); `ξ` is a policy parameter passed alongside `σ_t` into `π_decision` but is not part of `σ_t` itself (§2.2.3); `a` is an event/output; `φ` is a transient update signal. This distinction — primitive persistent state vs. derived-on-demand vs. policy parameter vs. transient event — replaces v1's flat "ten equal Core Components" list and is the direct architectural resolution of the redundancies that formalization surfaced.

---

*(Continued in Part II: Update Dynamics, and §3: Critique / Redundancy Analysis)*
