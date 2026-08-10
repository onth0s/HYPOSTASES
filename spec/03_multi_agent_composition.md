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

### 5.8 Meta-gap: schema-level completeness was never audited (discovered post hoc, Part VII §12.4)

**What §5.1–§5.7 actually checked**: whether each *component* (Willingness, Power, Potentialities, etc.) justified its existence as primitive versus derived — a question about the architecture's type-level taxonomy.

**What §5.1–§5.7 never checked**: whether a concrete schema's instantiation of a declared function (`feedback`, `pi_decision`, `_predict_amount`, etc.) is non-degenerate on every branch it's supposed to handle. Nothing in the formal spec (Part II §3.3's `feedback : S × S × A × W → Φ` signature) restricts any action's consequence to be trivial — the general model was never at fault. But Schema v1 (Part IV §6) silently gave the `WITHDRAW` action branch a constant, all-zero `Φ` (`d_c = {reserve: 0, mood: 0}`), which is a legal instance of the general signature but a degenerate one — and nothing in this audit process would have caught it, because §5.1–§5.7's method only ever asked "is this component's *existence* justified," never "is this component's *concrete definition*, per action/branch, actually doing work."

This was discovered empirically — the WITHDRAW branch was found to carry zero inferential evidence only once Inverse Inference (Part VII) was built and tested against it (§12.2, §12.4) — rather than caught by inspection against the spec, which is the wrong order for a spec that claims falsifiability (Design Philosophy V) as a governing principle. A completeness check that could have caught this ahead of time would ask, for every declared schema function with a case-by-case definition (per action type, per goal category, etc.): does every branch produce output that depends on the particle/agent's own state, or does any branch degenerate to a state-independent constant? A constant branch is not automatically wrong (some actions may legitimately have no consequence in some schemas), but it should be a **declared** choice, flagged the way §11's amount-scoring simplification was flagged, not a silent default.

**Resolution deferred to Part VII §12.5** (schema-level fix + re-tested), since fixing this specific instance requires implementation work, not further formal argument. This entry exists to record that the *auditing method* itself has a gap, independent of whether this one instance gets patched — future schema authors should run an explicit per-branch completeness pass, not only the primitive/derived pass §5.1–§5.7 performed.

---

*(Part IV (schemas), Part V (reference implementation), Part VI (worked trace), and Part VII (inverse inference) carry this v3 revision forward. Part VII §12.5 addresses the schema-completeness instance named in §5.8.)*
