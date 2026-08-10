---
part_number: 4
title: "Part IV: Reference Schemas"
depends_on: [1, 2]
sections: [6]
---
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

### 6.7b Feedback branches per action (declared; fixes Part III §5.8's named gap)

Schema v1's original `feedback` function gave `WITHDRAW` a constant, state-independent `Φ` (`Δc = {reserve: 0, mood: 0}`) — legal under Part II §3.3's general signature, but undeclared as a simplification and, as Part VII §12.2/§12.4 found, evidentially inert for Inverse Inference. This is now a declared branch instead of a silent default:

| action | `Δc.reserve` | `Δc.mood` | rationale |
|---|---|---|---|
| `REQUEST(x)` | `granted_amt` | `-0.1 · shortfall · (1 − resilience)` | unchanged from original schema |
| `SHARE(x)` | `−x` | `+0.05 · sociality` | unchanged from original schema |
| `WITHDRAW` | `0` | `−0.03 · sociality` | **new**: abstaining is mildly mood-negative, more so for agents with higher `sociality` — withdrawing is a more atypical, costlier choice for a sociable agent than for one already inclined toward self-interest. `Δρ_ext.social_capital` for `WITHDRAW` is set to a small negative drift (`−0.01`, versus SHARE's `+0.3` and REQUEST's `−0.02`) reflecting that abstaining neither builds nor actively spends social capital, but forgoes the opportunity SHARE would have created. |

`WITHDRAW`'s reserve delta remains correctly `0` — withdrawing genuinely does not touch reserve, and that branch was never the problem; only its mood/social-capital blindness was.

### 6.8 Environment `e ∈ ℝ`

Single scalar: shared pool level. `obs^(i)(e) = e + noise^(i)`, `noise^(i) ~ N(0, obs_noise²)` — agents observe the true pool imperfectly and independently, which is what makes World Model divergence (and thus mis-coordination) possible.

### 6.9 Index of Exploration `ξ ∈ ℝ≥0^{4}` (policy parameter, not state — Part I §2.2.3)

One per goal category, passed as an explicit parameter to `π_decision` (Part II §3.1) — not part of `σ`, not stored on the agent's persistent state.

---

*(Continued in Part V: Reference Implementation)*
