---
part_number: 3
title: "Part III: Multi-Agent Composition & Resolution of Redundancies"
depends_on: [1, 2]
sections: [4, 5]
---
# HYPOSTASES — Formal Specification
## Part III: Multi-Agent Composition, and §5 Resolution of Redundancies

Depends on: Part I (v3, §2.0–§2.3), Part II (v3).

---

## 4.4 Multi-Agent Composition

No new primitives are introduced (per Design Philosophy IV, preserved faithfully). Formally, using the v4 four-primitive `σ` (Part I §2.3):

```
System = ( I, E_0, {σ^(i)_0}_{i∈I}, {clock^(i)}_{i∈I} )
```

— an index set, an initial shared spatial Environment field, initial per-agent primitive states, and per-agent Tier-1 event clocks (Part I §1.2). The system trajectory is fully determined by repeated application of §4.1–§4.2 per agent, interleaved by the Tier-1 total order, with `step_env` (§3.2) applying localized field perturbations $\delta e^{(i)}(x)$ across spatial interaction coordinates. Derived quantities (`ρ_int`, `ω`, `P(c)`), transient dynamic policies $\pi_t$, and the policy parameter `ξ` are computed as needed per agent per event; they are not part of `System`'s persistent description.

**Emergent-system claims are Tier-2 predicates.** v1 defines things like "markets" and "governments" as emergent rather than primitive (Design Philosophy IV, Emergent Systems). Formalized: an emergent system is a predicate `Φ_emergent : (E_{T_n}, {σ^(i)_{T_n}}) → {0,1}` evaluated at Tier-2 snapshot barriers — e.g. "a market cleared" is a statement about the joint state at an epoch boundary, not a new state variable. This keeps the "no dedicated social primitives" commitment mechanically true.

---

## 5. Resolution of Redundancies (v4 architectural revision)

### 5.1 Willingness — resolved: demoted to derived

**Prior tension:** v1's own Feedback section omitted Willingness from the list of things Feedback updates.

**Resolution:** Willingness is not part of `σ` (Part I §2.2.2, §2.3). It is computed fresh at every decision point: `ω_t = derive_Ω(u_t, ρ_ext,t, proj_int(c_t))`. It has no memory of its own and no update equation.

### 5.2 Power — resolved: split into external primitive and internal read-only projection

**Prior tension:** Internal power (health, stamina) overlapped with Characteristics.

**Resolution:** Power_external (`ρ_ext`) is primitive. Power_internal (`ρ_int`) is a read-only projection of Characteristics `c`. Action costs on internal traits emit deltas $\Delta c_{\text{internal}}$ into Feedback $\Phi$, integrated into $c$ during State Evolution.

### 5.3 Potentialities — resolved: confirmed derived

**Resolution:** Re-confirmed derived reachable-set query $P(c)$ on demand.

### 5.4 Index of Exploration — resolved: reclassified as policy parameter

**Resolution:** $\xi$ is an explicit temperature parameter on policy allocation and decision-making, excluded from persistent state $\sigma$.

### 5.5 Goal Hierarchy — resolved: utility parameters primitive, choice policy transient

**Prior tension:** Conflating persistent utility/priority vectors with stochastic goal selection distributions.

**Resolution:** $g = u \in \mathbb{R}^{n_k}$ stores latent utility parameters in primitive state. Stochastic allocation vectors $\pi \in \Delta(K)$ are computed dynamically inside `π_decision` via softmax temperature scaling ($\pi_t = \text{softmax}(u_t / \xi_t)$).

### 5.6 Environment Concurrency — resolved: localized spatial field integration

**Resolution:** Monolithic centralized serialization is replaced by spatial field dynamics $\text{step\_env}: E \times \mathcal{L}(A) \to E$, applying localized field perturbations across spatial interaction coordinates.

### 5.7 Net Assessment (v4)

The v4 primitive set is: **six unconditional primitives** (Characteristics $c$, World Model $w$ with Theory of Mind peer states, Goal Hierarchy utilities $g=u$, Action $a$, Environment $e$, Feedback $\phi$) **+ one split primitive** ($R_{\text{ext}}$) **+ three derived quantities** ($\rho_{\text{int}}$, Willingness $\omega$, Potentialities $P(c)$) **+ one policy parameter** ($\xi$). Persistent per-agent state is strictly four-tuple $\sigma = (c, w, u, \rho_{\text{ext}})$.

### 5.8 Meta-gap: schema-level completeness was never audited (discovered post hoc, Part VII §12.4)

**What §5.1–§5.7 actually checked**: whether each *component* (Willingness, Power, Potentialities, etc.) justified its existence as primitive versus derived — a question about the architecture's type-level taxonomy.

**What §5.1–§5.7 never checked**: whether a concrete schema's instantiation of a declared function (`feedback`, `pi_decision`, `_predict_amount`, etc.) is non-degenerate on every branch it's supposed to handle. Nothing in the formal spec (Part II §3.3's `feedback : S × S × A × W → Φ` signature) restricts any action's consequence to be trivial — the general model was never at fault. But Schema v1 (Part IV §6) silently gave the `WITHDRAW` action branch a constant, all-zero `Φ` (`d_c = {reserve: 0, mood: 0}`), which is a legal instance of the general signature but a degenerate one — and nothing in this audit process would have caught it, because §5.1–§5.7's method only ever asked "is this component's *existence* justified," never "is this component's *concrete definition*, per action/branch, actually doing work."

This was discovered empirically — the WITHDRAW branch was found to carry zero inferential evidence only once Inverse Inference (Part VII) was built and tested against it (§12.2, §12.4) — rather than caught by inspection against the spec, which is the wrong order for a spec that claims falsifiability (Design Philosophy V) as a governing principle. A completeness check that could have caught this ahead of time would ask, for every declared schema function with a case-by-case definition (per action type, per goal category, etc.): does every branch produce output that depends on the particle/agent's own state, or does any branch degenerate to a state-independent constant? A constant branch is not automatically wrong (some actions may legitimately have no consequence in some schemas), but it should be a **declared** choice, flagged the way §11's amount-scoring simplification was flagged, not a silent default.

**Resolution deferred to Part VII §12.5** (schema-level fix + re-tested), since fixing this specific instance requires implementation work, not further formal argument. This entry exists to record that the *auditing method* itself has a gap, independent of whether this one instance gets patched — future schema authors should run an explicit per-branch completeness pass, not only the primitive/derived pass §5.1–§5.7 performed.

---

*(Part IV (schemas), Part V (reference implementation), Part VI (worked trace), and Part VII (inverse inference) carry this v3 revision forward. Part VII §12.5 addresses the schema-completeness instance named in §5.8.)*
