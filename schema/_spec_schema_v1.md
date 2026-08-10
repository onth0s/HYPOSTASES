# Schema Specification: Schema v1 (Reference)

Concrete field layouts and formulas for the reference domain: **social agents sharing a depletable resource pool**. This is *a* schema, not *the* schema — dimensions are implementation parameters (Part I §2.1).

> Companion to [`schema_v1.yaml`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/schema/schema_v1.yaml)

---

## Goal Categories `K` (§6.1)

| Category | Semantics |
|---|---|
| `SURVIVAL` | Maintaining own reserve above subsistence floor |
| `ACQUISITION` | Increasing own holdings beyond subsistence |
| `RELATIONAL` | Goal-value placed on the other agent's welfare |
| `STATUS` | Goal-value placed on *relative* (not absolute) standing |

## Characteristics (§6.2)

`c ∈ ℝ^6` — six named fields:

| idx | Field | Range | Meaning |
|---|---|---|---|
| 0 | `skill` | `[0,1]` | Efficiency multiplier on resource extraction |
| 1 | `resilience` | `[0,1]` | Dampens negative Feedback magnitude on `c` |
| 2 | `sociality` | `[0,1]` | Baseline propensity feeding `RELATIONAL` utility |
| 3 | `memory_decay` | `[0,1]` | Per-step decay on World Model belief confidence |
| 4 | `reserve` | `ℝ≥0` | Agent's own resource stock (survival-critical; feeds Power_internal) |
| 5 | `mood` | `[-1,1]` | Emotional baseline; decays toward 0, perturbed by Feedback |

## World Model (§6.3)

- **S** = `ℝ` — agent's estimate of the shared pool's remaining level
- **belief** = `N(μ, σ²)` — Gaussian parametric representation of `Δ(S)`
- **f**: `f(s, a) = s − a.amount_requested + replenish_rate` where `replenish_rate_est` is learned from observed pool deltas

## Power (§6.6)

**External (primitive):** 2 fields — `social_capital` (accumulates from cooperation), `time_budget` (epoch ticks remaining)

**Internal (derived):** `ρ_int.reserve_capacity = c.reserve` — read directly from Characteristics, no independent storage. This is Part III §5.2's resolution made concrete.

## Action Space (§6.7)

| Action | Parameters | Cost |
|---|---|---|
| `REQUEST(amount)` | `amount ∈ ℝ≥0` | 0 (requesting is free; the pool may refuse) |
| `SHARE(amount)` | `amount ∈ ℝ≥0` | `x` (deducted from `c.reserve`) |
| `WITHDRAW` | none | 0 |

## Feedback Branches (§6.7b)

| Action | `Δc.reserve` | `Δc.mood` | `Δρ_ext.social_capital` |
|---|---|---|---|
| `REQUEST(x)` | `granted_amt` | `−0.1 · shortfall · (1 − resilience)` | `−0.02` |
| `SHARE(x)` | `−x` | `+0.05 · sociality` | `+0.3` |
| `WITHDRAW` | `0` | `−0.03 · sociality` | `−0.01` |

> `WITHDRAW`'s mood/social-capital consequence is the v3 fix (Part III §5.8). Reserve delta remains correctly 0.

## Environment (§6.8)

Single scalar: shared pool level. Observation: `obs^(i)(e) = e + noise^(i)`, `noise^(i) ~ N(0, obs_noise²)`.

## Willingness (§6.5, derived)

```
ω_k = π_k · min(1, ρ_int.reserve_capacity / cost_estimate_k)    for k ∈ K
```

## Concurrency Resolution (§5.6)

Shares apply first; then pro-rata rationing of oversubscribed requests.

## STATUS Reserve Sensitivity (§12.5)

```
u_eff[STATUS] = u[STATUS] · (1 + 0.08 · max(0, reserve − 5))
```

A declared behavioral assumption: status contests matter more once subsistence (reserve > 5) is secure.
