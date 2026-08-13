# Schema Specification: Invariants

Cross-cutting constraints that hold across **all** schemas and implementations. Every invariant here is a hard rule — violations indicate bugs, not edge cases.

> Companion to [`invariants.yaml`](invariants.yaml)

---

## Affordability

**Spec ref:** Part II §3.1, Part I §2.2.1

```
cost(a_t, ρ_ext_t, ρ_int_t) ≤ (ρ_ext_t, ρ_int_t)  componentwise
```

An agent cannot take an action it cannot afford. `π_decision` must have support only on affordable actions — this is a well-formedness condition on any valid policy, not an optional runtime check.

## Distribution Normalization

**Spec ref:** Part I §2.2.1

| Constraint | Formal | Enforcement |
|---|---|---|
| Goal priorities | `π ∈ Δ(K)`: all `π_k ≥ 0`, `Σπ_k = 1` | softmax renormalization in `update_G` |
| World Model belief | `belief ∈ Δ(S)`; Gaussian: `σ² > 0` | clamped in `update_W` |

## Non-negativity

**Spec ref:** Part I §2.2.1

- `c.reserve ≥ 0` — reserve depletion below zero is physically meaningless
- `∀j: ρ_ext_j ≥ 0` — external Power fields are non-negative reals

## Architectural Taxonomy

These constraints enforce the v3 primitive/derived/parameter distinction:

| Rule | Formal | Why |
|---|---|---|
| **Derived-not-stored** | `ρ_int, ω, P(c) ∉ σ` | No persistent field, no `_t→_{t+1}` transition. Recomputed on demand from primitives. (Part I §2.2.2) |
| **Exploration-not-state** | `ξ ∉ σ` | Policy parameter passed separately to `π_decision`. Never a Feedback target. (Part I §2.2.3) |
| **Feedback targets primitives only** | `Φ = ΔC × ΔW × ΔG × ΔR_ext` | No `ΔP`, no `Δω`, no `Δξ` in the delta tuple. (Part II §3.4) |
| **σ is a four-tuple** | `σ = (c, w, g, ρ_ext)` | No more, no less. Payoff of v3 revision. (Part I §2.3) |

## Schema Versioning

**Spec ref:** Part I §2.1

Dimensionality is fixed within a schema version `v`. Cross-version changes require an explicit migration function `Migrate(X, v→v+1)` — total, declared, auditable. You cannot silently redefine a vector's meaning.

## Emergent Systems

**Spec ref:** Part III §4.4

Emergent-system claims (markets, governments, trust) are **Tier-2 predicates** evaluated at epoch snapshots — `Φ_emergent: (E_{T_n}, {σ^(i)}) → {0,1}`. They are never state variables. Nothing in `Σ` or `E` encodes social constructs.

## step_env Concurrency

**Spec ref:** Part III §5.6

Concurrent-action composition is a **schema-level obligation**, not spec-level resolved. Each schema must declare how simultaneous actions compose. Schema v1 uses shares-apply-first, pro-rata rationing.
