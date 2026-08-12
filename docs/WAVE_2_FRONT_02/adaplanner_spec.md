# AdaPlanner: Adaptive Planning from Feedback — Literature Reference Spec

**Source**: Sun et al., NeurIPS / ICLR (2024–2025)  
**Relevance to HYPOSTASES Wave 2 Front 02**: Foundational architecture for closed-loop execution monitoring, out-of-plan interruption, and refine-then-resume sub-plan repair.

---

## 1. Architectural Overview

AdaPlanner introduces a dual-role planning-and-refinement framework:
- **Planner**: Decomposes top-level goals $g \in \mathcal{G}$ into a sequential DAG of sub-goals and actions, associating expected state transitions and environmental observations with each step.
- **Refiner**: Continuously monitors execution feedback against expected state transitions. When deviations occur, it modifies the plan rather than restarting execution.

---

## 2. In-Plan vs. Out-of-Plan Feedback Mechanics

### In-Plan Feedback
- **Condition**: Observed state update $w_{t+1}$ matches predicted state $\hat{w}_{t+1}$ within acceptable tolerance $\|\text{proj}_w(\sigma) - \text{proj}_w(\hat{\sigma})\| \le \epsilon$.
- **Action**: Execution continues seamlessly along the active plan sequence. Information extraction occurs without triggering re-planning.

### Out-of-Plan Feedback
- **Condition**: Precondition failure, invariant breach, or state deviation exceeding threshold:
  $$\|\text{proj}_w(\sigma) - \text{proj}_w(\hat{\sigma})\| > \epsilon$$
- **Action**: Execution is immediately halted. The Refiner receives the execution trace $(s_0, a_0, s_1, a_1, \dots, s_t, \text{error})$ and triggers **"Refine-Then-Resume"**:
  1. Identify failed node $n_k$.
  2. Preserve valid preceding trajectory $n_{1 \dots k-1}$.
  3. Generate local patch sequence $\Pi_{\text{patch}}$ targeting state recovery or goal progress from $s_t$.
  4. Resume execution at $t+1$ using $\Pi_{\text{repaired}} = n_{1 \dots k-1} \mathbin{\Vert} \Pi_{\text{patch}}$.

---

## 3. Skill Discovery & Plan Exemplar Caching

Successful plan executions (high net utility gain $\Delta g / \text{cost}$) are abstracted into reusable plan exemplars $T_\Pi$ and stored in a persistent strategy library (`PlanLibrary`).
