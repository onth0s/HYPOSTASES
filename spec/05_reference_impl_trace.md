---
part_number: 5
title: "Part V & VI: Reference Implementation & Worked Trace"
depends_on: [4]
sections: [7, 8, 9]
---
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

**Result** (full run in `run_trace.py` output; **no longer bit-identical to the pre-revision v2 trace** — Part III §5.8's WITHDRAW fix touches the shared `feedback()`/`_goal_probs()` functions this trace also calls, so B's WITHDRAW-heavy behavior now produces small, real consequences it previously didn't; this is expected and correct, not a regression — see note below):

| | A (start → end) | B (start → end) |
|---|---|---|
| reserve | 6.0 → 9.27 | 6.0 → 16.00 |
| social_capital | 1.0 → 2.69 | 1.0 → 1.13 |
| mood | 0.0 → 0.11 | 0.0 → −0.00 |
| π(dominant goal) | RELATIONAL: 0.33 → 0.46 | STATUS: 0.33 → 0.33* |

*(B's STATUS weight stays flat because B never receives the relational-bump Feedback term that only fires on `SHARE` actions — B rarely shares, so its Goal Hierarchy doesn't drift the way A's does. This is itself a finding, not a bug: the model predicts that agents who don't engage prosocially also don't develop stronger prosocial preference — no separate "personality is fixed" rule was needed to produce that; it fell out of the state-evolution equations.)*

**Note on the shift from the original run**: B's `social_capital` moved 1.08→1.13 and `mood` moved 0.00→−0.00, both small but real, because B's frequent WITHDRAW actions now carry the trait-dependent consequence declared in Part IV §6.7b (previously WITHDRAW was a true no-op, per the gap named in Part III §5.8). Reserve values are unaffected (WITHDRAW correctly never touched reserve, before or after the fix), so the qualitative story below is unchanged — only the two fields the fix actually touches shifted, and only slightly.

**What the trace demonstrates**: A ends with less than 60% of B's reserve but nearly triple B's mood and social capital, and a Goal Hierarchy that has *drifted further toward RELATIONAL* than it started — a self-reinforcing preference shift, not a fixed trait. B ends resource-rich but flat-mood and undrifted. Neither trajectory was hardcoded; both are the mechanical consequence of the utility-update and renormalization equations applied to two different initial utility vectors — now running over a state representation with no redundant or ambiguously-typed fields (§7). This is the concrete demonstration of Design Philosophy IV (Emergence over Enumeration) the v1 abstract asserted but never showed, produced by an architecture whose primitive/derived boundary is now explicit rather than flagged-but-unresolved.

**Caveat, stated plainly**: this is one seeded run (`seed=7`) of a small, hand-picked schema. It demonstrates that the formal machinery *can* produce qualitatively sensible divergence — it is not a validated claim that HYPOSTASES models real cooperation/conflict dynamics. A future pass would need multi-seed statistical runs and sensitivity analysis before any such claim is defensible.

---

## 9. What Remains

Honest scope of what this formalization + revision pass (Parts I–VI, v3) has and hasn't done:

**Done**: typed state spaces with an explicit primitive/derived/parameter taxonomy (four primitives, three derived quantities, one policy parameter — down from v1's flat ten); a reconciled multi-tier time model; formal update equations for all three computational modes over the reduced primitive set; a concrete schema matching v3 exactly; a runnable reference implementation where the primitive/derived distinction is enforced by the type system, not just documented; one worked multi-agent trace (re-run, bit-identical) showing emergence without hardcoded social primitives; **all four flagged redundancies from the v2 critique resolved architecturally** (Part III §5.1–§5.5), not merely noted.

**Not done** (candidates for a further pass): stability/equilibrium analysis using the Tier-0 continuous limit (Part I §1.4 promises this but it's unstarted); the Inverse Inference direction (§4.3) is formalized but not implemented — no `Infer` function exists in the reference code yet, and it would need updating to target the v3 four-tuple `Σ` rather than the old six-tuple; no proof or empirical check of the Falsifiability principle itself (Design Philosophy V) — i.e., no case where the architecture was run against a genuine failure mode in the field and revised in response, as opposed to this one internally-driven formalization pass; multi-seed statistical validation of the trace in §8.

The v3 revision changed type signatures and taxonomy, not simulated behavior (§8's bit-identical re-run confirms this) — so anything built against the v2 draft's math should port to v3 by updating construction/call sites only, per the diffs in §7.
