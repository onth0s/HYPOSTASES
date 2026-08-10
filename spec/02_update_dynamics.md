---
part_number: 2
title: "Part II: Update Dynamics"
depends_on: [1]
sections: [3, 4]
---
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
