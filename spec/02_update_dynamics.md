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

where `σ_t = (c_t, w_t, g_t, ρ_ext,t)` (Part I §2.3, v4) with persistent latent utilities `g_t = u_t ∈ ℝ^{n_k}` and `ξ_t` passed as explicit policy temperature.

Internally, `π_decision` computes derived values and dynamic goal allocations $\pi \in \Delta(K)$ on the fly before selecting an action:
```
ρ_int = proj_int(c_t)
π_t   = softmax(u_t / ξ_t)                (transient policy allocation, not state)
ω_t   = derive_Ω(u_t, ρ_ext,t, ρ_int)
```
`π_decision` is the **Decision Policy** (v1's Forward Simulation, §II). `ξ_t` controls entropy/temperature over goal allocation and action choice.

Constraint: `cost_ext(a_t, ρ_ext,t, ρ_int,t) ≤ ρ_ext,t` componentwise — an action cannot exceed external resource bounds.

### 3.2 Environment Stage

```
step_env : E × L(A) → E
e_{t+1} = step_env(e_t, {a^(i)_t})
```

`step_env` applies localized spatial field perturbations $\delta e^{(i)}(x)$ across the interaction coordinates $x \in E$ resulting from agent action vectors $a^{(i)}_t$.

### 3.3 Feedback Stage

```
observe : E → S               (Part I §2.2, agent-specific)
feedback : S × S × A × W → Φ
φ_t = feedback(obs(e_t), obs(e_{t+1}), a_t, w_t)
```

Feedback emits deltas for persistent primitive states $\Phi = \Delta C \times \Delta W \times \Delta G \times \Delta R_{\text{ext}}$, where $\Delta C$ explicitly includes internal resource consumption deltas $\Delta c_{\text{internal}} = \text{cost\_int}(a_t, \rho_{\text{ext}}, \rho_{\text{int}})$.

### 3.4 State Evolution Stage

Only primitive states (Part I §2.2.1, §2.3) are integrated across steps:

```
c_{t+1}       = c_t       + φ_t.Δc          (incorporates Δc_internal)
w_{t+1}       = update_W(w_t, φ_t.Δw)       (updates joint env & peer belief distributions)
g_{t+1}       = update_G(g_t, φ_t.Δg)       (updates latent utility weights u)
ρ_ext,{t+1}   = ρ_ext,t   + φ_t.Δρ_ext  −  cost_ext(a_t, ρ_ext,t, ρ_int,t)
```

Notes:
- `c_{t+1}` integrates physical/internal trait changes $\Delta c_{\text{internal}}$ directly during State Evolution, resolving internal resource depletion cleanly without mutating derived views $\rho_{\text{int}}$ during action selection.
- `update_W` updates joint beliefs over environment $S$ and peer latent states $\prod_{j \neq i} \Sigma^{(j)}$ (Theory of Mind filtering).
- `update_G` updates latent utility weights $u_{t+1} = u_t + \varphi_t.\Delta g$. Dynamic probabilities $\pi_{t+1}$ are computed fresh during the next `π_decision` step.

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
