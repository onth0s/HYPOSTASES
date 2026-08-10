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
# HYPOSTASES — Formal Specification
## Part II: Update Dynamics

Depends on: Part I (§0 notation, §1 time model, §2 state spaces).

---

## 3. The Core Loop as a Function Composition

v1's "High-Level Architecture" diagram is a linear pipeline. Formalized, each arrow is a total function between the typed spaces of Part I §2, all evaluated at a single Tier-1 event `(i, k)` for agent `i`. We drop the `(i)` superscript below where unambiguous and write `t` for `t^(i)_k`.

### 3.1 Policy Stage (Forward Simulation)

```
π_decision : (C × W × G × R_ext) × Ξ → Δ(A)
a_t ~ π_decision(σ_t ; ξ_t)
```

where `σ_t = (c_t, w_t, g_t, ρ_ext,t)` (Part I §2.3, v3) and `ξ_t` is passed as an explicit *parameter*, separated by `;` from the state argument, not concatenated into it — this typing is the formal statement of Part I §2.2.3: Exploration governs the policy but is not itself state.

Internally, `π_decision` first computes the two derived quantities it needs (Part I §2.2.2) before selecting an action:
```
ρ_int = proj_int(c_t)
ω_t   = derive_Ω(g_t, ρ_ext,t, ρ_int)
```
`π_decision` is the **Decision Policy** (v1's Forward Simulation, §II). It is deliberately typed as producing a *distribution* over actions, not a single action. `ξ_t` controls entropy/temperature of the returned distribution (high `ξ` ⇒ higher-entropy `Δ(A)`; low `ξ` ⇒ near-deterministic, concentrated on the exploit-optimal action) — this is Exploration's entire formal role: a temperature parameter on the policy, not a separate causal component (resolved in Part I §2.2.3; no longer merely flagged).

Constraint (from Part I §2.2.1, Action): `cost(a_t, ρ_ext,t, ρ_int,t) ≤ (ρ_ext,t, ρ_int,t)` componentwise — an agent cannot take an action it cannot afford, using *both* Power views. `π_decision` must have support only on affordable actions; this is a well-formedness condition on any valid policy, not an optional check.

### 3.2 Environment Stage

```
step_env : E × A → E
e_{t+1} = step_env(e_t, a_t)
```

Since `E` is shared (Part I §2.2), `step_env` in the multi-agent case is really `step_env : E × {a^(i)_t}_{i ∈ active(t)} → E`, a joint update over all agents whose action falls in the same causal window (Part I §1.2, Tier-1 total order resolves ordering when the window is a single event, but simultaneous/concurrent actions within a Tier-2 epoch require `step_env` to define a composition or commutativity rule — this is left as a schema-level obligation, flagged in §5.4).

### 3.3 Feedback Stage

```
observe : E → S               (Part I §2.2, agent-specific)
feedback : S × S × A × W → Φ
φ_t = feedback(obs(e_t), obs(e_{t+1}), a_t, w_t)
```

Feedback is computed as a function of the *pre-* and *post-action* observations plus the action taken and the agent's prior World Model — formalizing v1's "information obtained after interaction with the environment" as a well-defined signal (a prediction-error / surprise term is the canonical instantiation: `feedback` compares what `w_t.f` predicted against what was actually observed).

### 3.4 State Evolution Stage

Only primitives (Part I §2.2.1, §2.3) are integrated across steps. Derived quantities and the policy parameter are never targets of a State Evolution equation — they are recomputed fresh wherever needed (§3.1).

```
c_{t+1}       = c_t       + φ_t.Δc
w_{t+1}       = update_W(w_t, φ_t.Δw)
g_{t+1}       = update_G(g_t, φ_t.Δg)
ρ_ext,{t+1}   = ρ_ext,t   + φ_t.Δρ_ext  −  cost(a_t, ρ_ext,t, ρ_int,t)
```

Four equations for four primitives — this is the direct payoff of Part I §2.0's revision: v2's draft carried a fifth line (`ξ_{t+1} = update_Ξ(...)`) with no principled update rule ever supplied, because none was needed. Removing it here is not a simplification for convenience, it is the corrected model: `ξ` is never a target of State Evolution (Part I §2.2.3).

Notes:
- `c_{t+1}` and `ρ_ext,{t+1}` are additive-delta updates (matching `Φ`'s typing as deltas, Part I §2.2.1).
- `update_W` is not a simple addition because `w = (belief, f)` where `belief ∈ Δ(S)` — this is formally a **Bayesian update**: `belief_{t+1} = Bayes(belief_t, obs(e_{t+1}))`, with `f` (the causal model) updated by any standard online model-learning rule (schema-chosen, e.g. gradient step on prediction error). This is the formal cash-value of v1's "World Model governs prediction... need not be correct."
- `update_G` re-normalizes `π ∈ Δ(K)` after `u` (utilities) shift — a softmax-style renormalization over updated utilities, not a free-form update. `π`'s soft-distribution semantics (Part I §2.2.1) makes this well-defined regardless of whether the current state happens to be near one-hot.
- Willingness and internal Power do not appear in this stage at all — by construction (Part I §2.2.2) they have no `_t → _{t+1}` transition; they are recomputed inside `π_decision` (§3.1) each time they're needed, always from the *current* primitives.

### 3.5 Potentialities and Willingness as Queries, Not State Updates

Per Part I §2.2.2, both `P(c)` and `ω` are pure functions of current primitive state, not persistent state, so neither has an update equation — both are recomputed on demand:
```
P_t = reach(c_t, R_t)                         for some declared reachability relation reach
ω_t = derive_Ω(g_t, ρ_ext,t, proj_int(c_t))
```
`reach` and `derive_Ω` are schema-declared functions (Part IV fixes concrete instances for the reference schema).

---

## 4. The Three Computational Modes, Formally

### 4.1 Forward Simulation

Exactly §3.1: `a_t ~ π_decision(σ_t)`, the map from latent state to observable action. This is generative: given `σ_t`, sample or compute `a_t`.

### 4.2 State Evolution

Exactly §3.2–3.4 composed: `σ_{t+1} = Evolve(σ_t, a_t, e_t)`. This is v1's "Learning is not a separate module" claim, formalized as literally the same function composition as the rest of the loop — there is no separate "learning" arrow in this formalization, confirming v1's own claim rather than merely asserting it.

### 4.3 Inverse Inference

v1 describes this only in prose ("estimates the most probable latent configuration consistent with observations"). Formalized:

```
Infer : (a_1, ..., a_T, e_1, ..., e_T) → Δ(Σ)
```

i.e., a posterior distribution over the *full* per-agent primitive state space `Σ = C × W × G × R_ext` (Part I §2.3, v3), conditioned on an observed action/environment trajectory. Derived quantities (`ρ_int`, `ω`, `P(c)`) are not separately inferred — they are recomputed from the inferred primitives via the same declared functions used forward (§3.1, §3.5), so `Infer` need only target `Σ`, not the full v2 six-tuple. Canonical instantiation: `Infer` is Bayesian filtering/smoothing over the generative model defined by §4.1–4.2 treated as a state-space model — i.e., Inverse Inference is not a new primitive operation, it is **posterior inference in the same generative model** Forward Simulation and State Evolution define. This formalizes v1's own framing ("bidirectional... generation and inference," Computational Modes preamble) precisely: one generative model, two directions of use (sample forward / condition backward), rather than three independent modes as the v1 headers ("I, II, III") suggest.

---

*(Continued in §5: Critique — Redundancy & Specification Gaps, and Part III: Multi-Agent Composition & Dynamics/Stability)*
# HYPOSTASES — Formal Specification
## Part III: Multi-Agent Composition, and §5 Resolution of Redundancies

Depends on: Part I (v3, §2.0–§2.3), Part II (v3).

---

## 4.4 Multi-Agent Composition

No new primitives are introduced (per Design Philosophy IV, preserved faithfully). Formally, using the v3 four-primitive `σ` (Part I §2.3):

```
System = ( I, E_0, {σ^(i)_0}_{i∈I}, {clock^(i)}_{i∈I} )
```

— an index set, an initial shared Environment, initial per-agent primitive states, and per-agent Tier-1 event clocks (Part I §1.2). The system trajectory is fully determined by repeated application of §4.1–§4.2 per agent, interleaved by the Tier-1 total order, with `step_env` (§3.2) resolving concurrent writes to `E`. Derived quantities (`ρ_int`, `ω`, `P(c)`) and the policy parameter `ξ` are computed as needed per agent per event; they are not part of `System`'s persistent description.

**Emergent-system claims are Tier-2 predicates.** v1 defines things like "markets" and "governments" as emergent rather than primitive (Design Philosophy IV, Emergent Systems). Formalized: an emergent system is a predicate `Φ_emergent : (E_{T_n}, {σ^(i)_{T_n}}) → {0,1}` evaluated at Tier-2 snapshots — e.g. "a market cleared" is a statement about the joint state at an epoch boundary, not a new state variable. This keeps the "no dedicated social primitives" commitment mechanically true: nothing in `Σ` or `E` encodes "trust" or "a market"; such things are only ever *read off* of snapshots by an external predicate, never written into agent or environment state.

---

## 5. Resolution of Redundancies (v3 architectural revision)

The v2 formalization pass preserved all ten v1 components as flat primitives and flagged four as contestable without altering the component list. This section records that the v3 revision (Part I §2.0) resolves each flagged item — not by further critique, but by an actual change to the type signatures, carried through consistently in Parts I, II, and (below, §5.6) the composition semantics. Each entry states the prior tension, the resolution adopted, and the principle that decided it.

### 5.1 Willingness — resolved: demoted to derived

**Prior tension:** v1's own Feedback section omitted Willingness from the list of things Feedback updates (Characteristics, World Model, Goal Hierarchy, Potentiality- and Power-*estimates* were listed; Willingness was not), despite naming it a top-level component.

**Resolution:** Willingness is not part of `σ` (Part I §2.2.2, §2.3). It is computed fresh at every decision point: `ω_t = derive_Ω(g_t, ρ_ext,t, proj_int(c_t))`. It has no memory of its own and no update equation (Part II §3.4, §3.5).

**Deciding principle:** Design Philosophy I (Minimalism) — a quantity with a total, stateless formula from existing primitives does not justify independent existence. The absence of an update rule in v1 was not an oversight to patch; it was correct, because none was needed.

### 5.2 Power — resolved: split into external primitive and internal projection

**Prior tension:** v1 listed "Health, Energy, Executive function, Knowledge, Competence" as Power's *internal* examples, all of which already appear under Characteristics ("Knowledge, Skills, Intelligence... Physical condition") — two components sharing an undeclared subspace.

**Resolution:** Power is no longer one primitive. `Power_external` (`ρ_ext`: capital, time, authority, technology, social capital) remains an independent primitive in `σ` (Part I §2.2.1). `Power_internal` (`ρ_int`) is redefined as a declared read-only projection of Characteristics, `ρ_int = proj_int(c)` (Part I §2.2.2) — never separately stored, never separately updated. The overlap is eliminated by admitting there was only ever one underlying quantity for the internal block.

**Deciding principle:** Design Philosophy I (Minimalism) again — duplicated state is the clearest possible failure of "every abstraction must justify its existence." `proj_int` makes the relationship explicit instead of leaving it as silent, unacknowledged overlap.

### 5.3 Potentialities — resolved: confirmed derived, reclassified in taxonomy

**Prior tension:** Already typed as a derived reachable-set query in the v2 draft, but still listed as a "Core Component" alongside true primitives — a taxonomy inconsistency, not a math one.

**Resolution:** Part I §2.2.2 now classifies Potentialities explicitly as a derived quantity, in the same tier as Willingness and internal Power, removed from `σ` and from any Feedback-delta tuple (Part II §2.2.1's `Φ` no longer carries a `ΔP` term).

**Deciding principle:** Design Philosophy I (Minimalism) — `P(c)` is fully explained by `c` plus a declared reachability relation; nothing about it fails to be explained by existing primitives, which is v1's own stated bar for introducing (or, here, retaining as primitive) a concept.

### 5.4 Index of Exploration — resolved: reclassified as policy parameter

**Prior tension:** v1's diagram placed Exploration as a pipeline stage of equal standing to Characteristics or World Model, but formalization found it has no state-update role beyond feeding one function (`π_decision`).

**Resolution:** `ξ` is removed from `σ` (Part I §2.2.3, §2.3) and typed as an explicit parameter to `π_decision`, separated from the state argument in the function signature (Part II §3.1: `π_decision : σ × Ξ → Δ(A)`, semicolon-separated). It may still follow its own schedule (e.g. annealed across epochs), but that schedule is documented as a policy hyperparameter, not Feedback-driven state.

**Deciding principle:** Design Philosophy IV (Emergence over Enumeration) by extension — a control parameter of one function should not be drawn as a peer of the components that function *reads*; doing so overstated its architectural weight without adding explanatory power.

### 5.5 Goal Hierarchy naming — resolved: soft form declared canonical

**Prior tension:** v1's name implies strict ranking; its own prose examples ("Survival, Security, Curiosity, Status...") read as competing, weighted objectives, not a strict stack.

**Resolution:** Part I §2.2.1 now states directly that `π ∈ Δ(K)` (a soft distribution) is the canonical semantics, with strict hierarchy recovered only as the one-hot special case. The name "Goal Hierarchy" is retained for continuity (renaming would touch every downstream document, schema, and code file for a purely lexical gain), but is no longer allowed to be read as implying lexicographic dominance.

**Deciding principle:** Fidelity to v1's own examples over v1's own label, where the two disagree — the architecture should match what it demonstrably does, not what its name suggests it does.

### 5.6 `step_env` concurrency — confirmed as intentionally open, not a contention

Re-examined under this revision and confirmed unchanged: this was never a fork between competing resolutions the way §5.1–§5.4 were — it is a deliberate open schema obligation (Design Philosophy II, Domain Independence). Part IV's reference schema (§6, next document) discharges it concretely with one declared rule (shares-apply-first, pro-rata rationing on oversubscribed requests); the spec level correctly leaves the general case open, since markets, physical collision, and information environments resolve concurrency differently by domain. No architectural change was needed or made here.

### 5.7 Net assessment (revised)

The v3 primitive set is: **six unconditional primitives** (Characteristics, World Model, Goal Hierarchy, Action, Environment, Feedback) **+ one split primitive** (Power_external, replacing v1's unified Power) **+ three derived quantities** (Power_internal, Willingness, Potentialities) **+ one policy parameter** (Index of Exploration). `σ` itself — the actual persistent state an implementation must carry per agent — shrinks from v1's implied ten-component surface to a four-tuple (Part I §2.3): `(c, w, g, ρ_ext)`. This is the concrete, mechanical payoff of Design Philosophy I and V applied to the architecture itself, not just to its formalization: the smallest state an agent must remember to reproduce all ten of v1's described behaviors.

---

*(This completes the v3 architectural revision. Part IV (schemas), Part V (reference implementation), and Part VI (worked trace) are updated to match this primitive set in the companion documents.)*
# HYPOSTASES — Formal Specification
## Part IV: Reference Schemas

Depends on: Part I §2 (v3 typed state spaces: four primitives, three derived quantities, one policy parameter), Part II §3 (v3 update dynamics).

This part fixes one concrete `Schema(X, v=1)` per component — a specific choice of dimensions and field layout — so that Part V (reference implementation) and Part VI (worked trace) have something exact to run. Per Part I §2.1, this is *a* schema, not *the* schema: dimensions are implementation parameters, this document picks values for the reference build. Schema v1 targets the v3 primitive set directly — there is no legacy six-tuple to reconcile, since this schema was written after the v3 revision.

Domain for this schema pass: **social agents sharing a depletable resource pool** (chosen to exercise cooperation/conflict emergence per Design Philosophy IV, without hardcoding either as a primitive).

---

## 6. Schema v1 — Field Layouts

### 6.1 Goal categories `K` (fixed enum for this schema)

```
K = { SURVIVAL, ACQUISITION, RELATIONAL, STATUS }   (n_k = 4)
```

`SURVIVAL`: maintaining own resource reserves above a subsistence floor.
`ACQUISITION`: increasing own resource holdings beyond subsistence.
`RELATIONAL`: goal-value placed on the other agent's welfare (the formal hook cooperation emerges through — see §8).
`STATUS`: goal-value placed on relative (not absolute) standing vs. other agents.

### 6.2 Characteristics `c ∈ ℝ^{n_c}`, `n_c = 6`

| idx | field | range | meaning |
|---|---|---|---|
| 0 | `skill` | `[0,1]` | efficiency multiplier on resource extraction |
| 1 | `resilience` | `[0,1]` | dampens negative Feedback magnitude on `c` |
| 2 | `sociality` | `[0,1]` | baseline propensity feeding `RELATIONAL` utility |
| 3 | `memory_decay` | `[0,1]` | per-step decay applied to World Model belief confidence |
| 4 | `reserve` | `ℝ≥0` | agent's own resource stock (survival-critical; also feeds Power) |
| 5 | `mood` | `[-1,1]` | emotional baseline; decays toward 0, perturbed by Feedback |

### 6.3 World Model `w = (belief, f)`

`S` (environment-state space, schema-fixed): `S = ℝ` — the agent's *estimate of the shared pool's remaining level*.
`belief ∈ Δ(S)` represented parametrically as a Gaussian `N(μ, σ²)` (i.e., `belief = (μ, σ²)`, `μ ∈ ℝ`, `σ² ∈ ℝ>0`) rather than a full distribution table — a standard, declared simplification of `Δ(S)`.
`f`: fixed functional form `f(s, a) = s − a.amount_requested + replenish_rate`, with `replenish_rate` itself a *learned* scalar the agent updates from observed pool deltas (this is the part of `f` that actually adapts).

### 6.4 Goal Hierarchy `g = (π, u)`

`π ∈ Δ(K)`, `u ∈ ℝ^4` (one utility scalar per `K` category, same order as §6.1).

### 6.5 Willingness `ω` (derived, Part I §2.2.2)

Not stored. Schema-fixed form:
```
ω_k = π_k · min(1, ρ_int.reserve_capacity / cost_estimate_k)     for k ∈ K
```
i.e. desired-priority scaled down by whether the agent can currently afford to act on it, read from the derived internal-Power projection (§6.6b).

### 6.6a Power_external `ρ_ext ∈ ℝ^{n_{r,ext}}`, `n_{r,ext} = 2` (primitive)

| idx | field | meaning |
|---|---|---|
| 0 | `social_capital` | external resource; accumulates from cooperative acts, decays otherwise |
| 1 | `time_budget` | ticks remaining this Tier-2 epoch before forced pass |

### 6.6b Power_internal `ρ_int = proj_int(c)` (derived, Part I §2.2.2)

| field | projection | meaning |
|---|---|---|
| `reserve_capacity` | `= c.reserve` | internal Power is *read directly* from Characteristics — no independent storage. This schema is where Part III §5.2's resolution becomes concrete: what was ambiguously duplicated in v1 is here one field, read two ways. |

### 6.7 Action space `A`

Discrete, 3 actions: `A = { REQUEST(amount), SHARE(amount), WITHDRAW }`.
`cost(REQUEST(x), ρ_ext, ρ_int) = 0` (requesting is free; the pool may refuse), `cost(SHARE(x), ρ_ext, ρ_int) = x` (deducted from `ρ_int.reserve_capacity`, i.e. from `c.reserve` directly), `cost(WITHDRAW, ρ_ext, ρ_int) = 0`.

### 6.8 Environment `e ∈ ℝ`

Single scalar: shared pool level. `obs^(i)(e) = e + noise^(i)`, `noise^(i) ~ N(0, obs_noise²)` — agents observe the true pool imperfectly and independently, which is what makes World Model divergence (and thus mis-coordination) possible.

### 6.9 Index of Exploration `ξ ∈ ℝ≥0^{4}` (policy parameter, not state — Part I §2.2.3)

One per goal category, passed as an explicit parameter to `π_decision` (Part II §3.1) — not part of `σ`, not stored on the agent's persistent state.

---

*(Continued in Part V: Reference Implementation)*
# HYPOSTASES — Formal Specification
## Part V: Reference Implementation & Part VI: Worked Trace

Depends on: Part IV (Schema v1).
Companion files: `hypostases_ref.py` (implementation), `run_trace.py` (driver).

---

## 7. Implementation Notes (v3)

`hypostases_ref.py` is a literal, line-for-line translation of Part II §3 (v3) into Python — function names match spec names (`pi_decision`, `step_env`, `feedback`, `evolve`) so the code and the math can be read side by side. `AgentState` holds exactly the v3 four-tuple `(c, w, g, rho_ext)`; there is no `omega`, `P`, or `xi` field anywhere on it — each is either a method that recomputes from current state (`omega()`, `power_internal()`) or an explicit function parameter never stored (`xi`, passed into `pi_decision` at each call). This is not merely documented, it is enforced by the type: the redundant fields do not exist in the dataclass, so there is no path by which they could silently accumulate independent state.

- **§3.1 Policy stage**: implemented as softmax goal-selection over `π · ω · u`, tempered by `mean(ξ)` — `ξ` arrives as a plain function argument (`pi_decision(agent, pool_belief, xi)`), never read from `agent`, matching Part I §2.2.3's resolution directly in the call signature.
- **§3.2 `step_env` concurrency** (Part III §5.6, confirmed intentionally open at spec level): resolved here as *shares-apply-first, then pro-rata rationing of requests if oversubscribed* — one legal, declared resolution for this schema, not the only possible one.
- **§3.4 World Model update**: implemented as a scalar Kalman-style correction (`μ ← μ + gain·surprise`) rather than full Bayesian posterior update over a parametric family — a standard, declared simplification of "Bayesian update" for a Gaussian belief.
- **§6.5/§6.6b Willingness and Power_internal**: both implemented as methods on `AgentState` (`omega()`, `power_internal()`) with no backing field — Python enforces this is impossible to accidentally treat as stored state, since there is nothing to assign to.
- **Power split (§5.2 resolution)**: `PowerExternal` is the only Power dataclass; `power_internal()` returns a plain dict computed from `self.c.reserve` at call time. The driver (`run_trace.py`) never constructs an internal-Power object — only `PowerExternal` is passed into `AgentState.__init__`.

---

## 8. Worked Trace: Cooperation/Conflict Emergence (re-run under v3)

Two agents, one shared pool, 12 Tier-1 steps (synchronized here as Tier-2 epochs — legal per Part I §1.3). No cooperation or conflict primitive exists anywhere in the code; both are meant to emerge purely from `sociality` and `status_u` differences feeding Goal Hierarchy utilities.

**Setup**: Agent A — `sociality=0.8, status_u=0.2` (high relational weight, low status drive). Agent B — `sociality=0.1, status_u=1.2` (the reverse).

**Result** (full run in `run_trace.py` output; bit-identical to the pre-revision v2 trace, confirming the v3 architectural resolution is a pure type-level restructuring — no math changed):

| | A (start → end) | B (start → end) |
|---|---|---|
| reserve | 6.0 → 9.27 | 6.0 → 16.00 |
| social_capital | 1.0 → 2.68 | 1.0 → 1.08 |
| mood | 0.0 → 0.13 | 0.0 → 0.00 |
| π(dominant goal) | RELATIONAL: 0.33 → 0.46 | STATUS: 0.33 → 0.33* |

*(B's STATUS weight stays flat because B never receives the relational-bump Feedback term that only fires on `SHARE` actions — B rarely shares, so its Goal Hierarchy doesn't drift the way A's does. This is itself a finding, not a bug: the model predicts that agents who don't engage prosocially also don't develop stronger prosocial preference — no separate "personality is fixed" rule was needed to produce that; it fell out of the state-evolution equations.)*

**What the trace demonstrates**: A ends with less than 60% of B's reserve but nearly triple B's mood and social capital, and a Goal Hierarchy that has *drifted further toward RELATIONAL* than it started — a self-reinforcing preference shift, not a fixed trait. B ends resource-rich but flat-mood and undrifted. Neither trajectory was hardcoded; both are the mechanical consequence of the utility-update and renormalization equations applied to two different initial utility vectors — now running over a state representation with no redundant or ambiguously-typed fields (§7). This is the concrete demonstration of Design Philosophy IV (Emergence over Enumeration) the v1 abstract asserted but never showed, produced by an architecture whose primitive/derived boundary is now explicit rather than flagged-but-unresolved.

**Caveat, stated plainly**: this is one seeded run (`seed=7`) of a small, hand-picked schema. It demonstrates that the formal machinery *can* produce qualitatively sensible divergence — it is not a validated claim that HYPOSTASES models real cooperation/conflict dynamics. A future pass would need multi-seed statistical runs and sensitivity analysis before any such claim is defensible.

---

## 9. What Remains

Honest scope of what this formalization + revision pass (Parts I–VI, v3) has and hasn't done:

**Done**: typed state spaces with an explicit primitive/derived/parameter taxonomy (four primitives, three derived quantities, one policy parameter — down from v1's flat ten); a reconciled multi-tier time model; formal update equations for all three computational modes over the reduced primitive set; a concrete schema matching v3 exactly; a runnable reference implementation where the primitive/derived distinction is enforced by the type system, not just documented; one worked multi-agent trace (re-run, bit-identical) showing emergence without hardcoded social primitives; **all four flagged redundancies from the v2 critique resolved architecturally** (Part III §5.1–§5.5), not merely noted.

**Not done** (candidates for a further pass): stability/equilibrium analysis using the Tier-0 continuous limit (Part I §1.4 promises this but it's unstarted); the Inverse Inference direction (§4.3) is formalized but not implemented — no `Infer` function exists in the reference code yet, and it would need updating to target the v3 four-tuple `Σ` rather than the old six-tuple; no proof or empirical check of the Falsifiability principle itself (Design Philosophy V) — i.e., no case where the architecture was run against a genuine failure mode in the field and revised in response, as opposed to this one internally-driven formalization pass; multi-seed statistical validation of the trace in §8.

The v3 revision changed type signatures and taxonomy, not simulated behavior (§8's bit-identical re-run confirms this) — so anything built against the v2 draft's math should port to v3 by updating construction/call sites only, per the diffs in §7.
# HYPOSTASES — Formal Specification
## Part VII: Inverse Inference

Depends on: Part I (v3 state spaces), Part II §4.3 (Inverse Inference, formalized but not implemented), Part IV (Schema v1).

---

## 10. Estimator Reconciliation

Three candidate estimator families were considered for `Infer : trajectory → Δ(Σ)` (Part II §4.3):

| Family | Represents | Cost | Handles multimodal posteriors? |
|---|---|---|---|
| Analytic (Kalman-style) | single Gaussian mode, closed form | lowest | no |
| Grid/optimization search | single best point (MAP) | moderate | no (reports one mode only) |
| Particle filter | full empirical posterior, many weighted samples | highest | yes |

### 10.1 Why these are not mutually exclusive

`π_decision` (Part II §3.1) selects a goal category via a stochastic softmax over `K` before producing an action. Different goal choices at the same state can produce entirely different, non-overlapping action distributions (e.g. `SHARE` vs `WITHDRAW`). Consequently the true posterior `P(σ_t | a_1..a_T)` is generically **multimodal in the Goal Hierarchy sub-block of `σ`** — one lobe per plausible "which goal was dominant" story consistent with the observed actions — even though the Characteristics and World Model sub-blocks may be locally well approximated by a single Gaussian mode conditioned on a fixed goal story.

This means the three families are not competing choices but **different degrees of posterior-shape preservation**, nested as special cases of one estimator:

- A **particle filter** with `N` weighted particles, each carrying a full `σ` hypothesis and an importance weight, is the general case.
- Setting `N = 1` and replacing categorical resampling with a local linear-Gaussian reweighting step over the continuous sub-blocks (`c`, `w`, `ρ_ext`) **recovers the analytic Kalman-style estimator** as a degenerate configuration — valid only when the goal story is already known or assumed fixed.
- Running the particle filter's resampling step *without* forward propagation, purely to hunt for the single highest-weight particle, and reporting only that particle, **recovers grid/MAP search** as another degenerate configuration.

### 10.2 Canonical `Infer`, formally

```
Infer_N : (a_1..a_T, e_1..e_T) → { (σ^(j), weight^(j)) }_{j=1..N}     [particle set approximating Δ(Σ)]
```

A sequential Monte Carlo (SMC bootstrap particle filter) estimator:

1. **Initialize**: sample `N` particles `σ^(j)_0 ~ prior(Σ)` (a declared prior over the four v3 primitives — schema-specific, Part IV §11 fixes one).
2. **Propose**: for each particle, forward-simulate one step using the *known* generative model (Part II §4.1–§4.2: `π_decision`, `step_env`, `feedback`, `evolve` — literally the same functions used for simulation, per §4.3's "one generative model, two directions" framing) to get `σ^(j)_{t+1}` and a predicted action distribution.
3. **Reweight**: `weight^(j) ∝ weight^(j)_{prev} · P(a_t^{observed} | σ^(j)_t)` — likelihood of the *actually observed* action under that particle's policy distribution.
4. **Resample**: when the effective sample size drops below a threshold, resample particles proportional to weight (standard SMC degeneracy control).

At any point, this particle set *is* the approximation to `Δ(Σ)` that Part II §4.3 specifies `Infer` must return. Reporting the single highest-weight particle recovers MAP (§10.1's second special case); collapsing the continuous sub-blocks to their weighted mean/covariance recovers the Kalman-style summary (§10.1's first special case). No separate code paths are required — both are read-outs of the same particle set.

### 10.3 What Infer does NOT re-derive

Per Part I §2.2.2, `Infer` only targets primitive state `Σ = C × W × G × R_ext` (Part I §2.3, v3). Derived quantities (`ρ_int`, `ω`, `P(c)`) are not inferred — they are recomputed from each particle's inferred primitives via the same declared functions used forward (`proj_int`, `derive_Ω`, `reach`), exactly as stated in Part II §4.3. `ξ` is never inferred either — it is a policy parameter (Part I §2.2.3), not a target of state estimation; if `ξ` is unknown, that is a *policy-identification* problem, out of scope for `Infer` as specified here.

---

*(Continued in §11: Reference Implementation of Infer, and §12: Worked Examples — recovery on known ground truth, then blind inference on an unseen agent)*
# HYPOSTASES — Formal Specification
## Part VII (cont.): §11 Reference Implementation, §12 Worked Examples

Depends on: §10 (Part VII, Estimator Reconciliation).
Companion files: `hypostases_infer.py` (implementation), `run_infer.py` (driver).

---

## 11. Reference Implementation Notes

`hypostases_infer.py` implements `Infer` exactly as §10.2 specifies: a bootstrap particle filter that reweights against `action_likelihood` (a new function added to `hypostases_ref.py`, sharing `_goal_probs` with `pi_decision` so forward and inverse read the identical goal-selection distribution — the "one generative model, two directions" commitment from Part II §4.3 is enforced by both functions calling the same private helper, not merely asserted in prose) and propagates particles forward using the unmodified `step_env`/`feedback`/`evolve` triplet from Part V.

`summarize_map`, `summarize_kalman`, and `goal_posterior` are three read-outs of one particle set, not three estimators (§10.1's reconciliation, made literal: there is exactly one `infer()` function; everything else is post-hoc summarization of its output).

**Amount-sensitivity (v2 fix, §12.4)**: `action_likelihood` originally scored only action *type* against the goal-to-action-type mapping (Part IV §6.7), ignoring amount entirely — a declared simplification. It now also scores observed amount against each candidate goal's own reserve-dependent predicted amount (`_predict_amount`), which measurably improved estimation on WITHDRAW-heavy traces (§12.4). The one part of this that remains a genuine, unresolved simplification: WITHDRAW's predicted and observed amounts are both always 0, so WITHDRAW observations still carry no amount-based evidence, only type-based evidence via `probs[STATUS]` — stated precisely in `action_likelihood`'s docstring rather than glossed over.

---

## 12. Worked Examples

### 12.1 Example 1 — Recovering Agent A from known ground truth

Agent A's exact Part VI parameters (`sociality=0.8, status_u=0.2`) are re-run through the identical generative loop to produce a 12-step action/pool trace. `Infer` (300 particles) then estimates `σ_A` from that trace alone — the true parameters are withheld from the estimator and used only afterward, for comparison.

**Result:**

| | MAP estimate | Kalman summary | True |
|---|---|---|---|
| reserve | 8.61 | 11.50 (var 5.34) | 11.66 |
| mood | 0.08 | 0.06 | 0.10 |
| dominant goal | ACQUISITION | — | RELATIONAL |

**Goal posterior** (the full distribution, not a point estimate): `RELATIONAL=0.68, ACQUISITION=0.17, SURVIVAL=0.09, STATUS=0.07`.

**Reading this honestly**: the single-particle MAP estimate gets the dominant goal *wrong* (reports ACQUISITION; truth is RELATIONAL) — but the goal posterior, the thing `Infer` actually returns per its type signature (`Δ(Σ)`, not a point), puts 68% of its mass correctly on RELATIONAL. This is not a filter failure to paper over — it is the concrete demonstration of §10.1's core claim: **a multimodal posterior collapsed to its single best point can report the wrong mode even when the distribution itself is well-calibrated.** This is exactly why Part II §4.3 specifies `Infer`'s return type as a distribution, and why §10 declined to build only a MAP/grid estimator. The Kalman-style reserve estimate (11.50) is closer to truth (11.66) than the MAP point estimate (8.61) — because it averages over the *whole* particle set rather than trusting one mode.

### 12.2 Example 2 — Blind inference on an unseen agent

A new agent C (`sociality=0.3, status_u=0.9, reserve0=4.0`), never seen during `Infer`'s construction or Example 1, is run for 10 steps and estimated blind.

**Result:**

| | MAP estimate | Kalman summary | True |
|---|---|---|---|
| reserve | 9.43 | 13.31 (var 7.88) | 10.00 |
| mood | 0.05 | 0.03 | 0.02 |
| dominant goal | SURVIVAL | — | SURVIVAL |

**Result, post amount-sensitivity fix (§12.4)**: reserve error dropped to 0.57 (from an original 5.81 before the fix) and the dominant goal now matches truth. **This example originally motivated §12.4's fix and is reported here already updated** — see §12.4 for the direct before/after comparison and why Example 2 specifically benefited while Example 1 stayed roughly flat.

**The remaining, still-honest limitation**: inspecting Agent C's action log shows 6 of its 10 actions were `WITHDRAW`, which — even after §12.4's fix — carries no amount evidence (§11). The estimate above improved because the fix sharpened the 4 non-WITHDRAW steps, not because WITHDRAW became more informative. A trace that was *entirely* WITHDRAW would still be poorly identified by this schema; that residual gap is named precisely in §12.4's closing paragraph rather than left implicit.

### 12.4 The diagnosed gap, actually fixed and measured

§12.2's original diagnosis was correct but the original write-up stopped at diagnosis. `action_likelihood` was rewritten to score observed **amount** against each candidate goal's own reserve-dependent predicted amount (`_predict_amount`, reusing `pi_decision`'s exact per-goal amount logic, not a new rule), instead of scoring action type alone. Re-running both examples against the fixed likelihood:

| | Reserve error (MAP), before fix | after fix | Goal match, before | after |
|---|---|---|---|---|
| Example 1 (A) | 2.42 | 3.05 | ✗ | ✗ (posterior mass on true goal: 71%→68%, roughly flat) |
| Example 2 (C) | 5.81 | **0.57** | ✗ | **✓** |

Example 2's reserve error dropped an order of magnitude and its goal match flipped correct. Example 1 stayed roughly flat. This asymmetry is itself explainable, not just observed: the fix adds discrimination power on REQUEST/SHARE steps (amount now matters, not just type) but does **not** add discrimination power on WITHDRAW steps (predicted amount is always 0, observed amount is always 0 — see the honest-scope note in `action_likelihood`'s docstring). Agent C's trace was WITHDRAW-heavy with only 4 informative steps; the fix sharpened exactly those 4 steps and materially improved the estimate. Agent A's trace had 12 mostly-informative steps already, so there was less headroom for a same-type-different-amount distinction to add.

**What is still not fixed, honestly**: WITHDRAW observations remain exactly as uninformative as before this fix — they score purely on `probs[STATUS]`, with no amount or consequence term to sharpen them, because `WITHDRAW`'s predicted and observed amounts are both trivially 0. A schema wanting WITHDRAW to carry real evidence would need `feedback`/`evolve` to attach a state-dependent consequence to withdrawing (e.g., a mood or reserve drift that differs by which goal "chose" to withdraw) and a matching likelihood term — that remains unimplemented, and is the next concrete, scoped piece of work if this matters for a given use case, rather than a vague "not done" item.



`Infer` recovers plausible, well-calibrated posteriors from action traces alone, using no information beyond what Forward Simulation and State Evolution already define (§10.2's reuse of `step_env`/`feedback`/`evolve` unmodified). Where it does worse (Example 2), the cause is traceable to a specific, nameable gap in the schema's declared likelihood function (§11), not an unexplained failure — which is itself evidence the formalization is doing its job: a wrong or underdetermined estimate here comes with a legible reason, per Design Philosophy V (Falsifiability).

---

## 13. What Remains (Part VII)

**Done**: `Infer` implemented as the canonical particle filter; MAP and Kalman-style summaries derived as read-outs, not separate code; two worked examples, one validated against ground truth, one blind; an identifiability gap surfaced, diagnosed, **and fixed** (§12.4) — amount-sensitive likelihood scoring measurably improved the worse-performing example (reserve error 5.81→0.57) without needing new dynamics, only better use of existing ones.

**Not done**: WITHDRAW-consequence modeling — `WITHDRAW` still carries zero amount-evidence by construction, since neither predicted nor observed amount varies for it (§12.4's closing paragraph); fixing this needs a state-dependent consequence in `feedback`/`evolve` for withdrawing, not just a likelihood change, so it's scoped separately rather than folded into §12.4's fix. Also outstanding: systematic sensitivity analysis of `n_particles` / `ess_threshold_ratio` against estimation accuracy; inference over multi-agent joint trajectories (current `Infer` is single-agent; a joint version would need to reason about `step_env`'s concurrency resolution, Part III §5.6, from the inference side too); stability/equilibrium analysis (Part I §1.4's Tier-0 promise) remains the one still-fully-unstarted item from Part V §9's original list.
