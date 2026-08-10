# Schema Specification: Time Model

This document outlines the three-tier time model for the HYPOSTASES project, establishing how time is represented and reconciled across continuous analysis, asynchronous simulation, and synchronous observation.

## Tiers of Time

### Tier 0 — Substrate (Continuous)
*   **Nature**: Continuous time $t \in \mathbb{R}_{\ge 0}$.
*   **Processes**: càdlàg (right-continuous with left limits).
*   **Usage**: Never directly simulated. Reserved purely for theoretical equilibrium analysis and limits.

### Tier 1 — Operational (Discrete, Async)
*   **Nature**: Discrete, asynchronous events.
*   **Clocks**: Per-agent event clocks $t^{(i)}_0 < t^{(i)}_1 < \dots$ with variable inter-event intervals $\Delta t^{(i)}_k$.
*   **Usage**: The ground truth for causal ordering. Establishes a total order over all $(i,k)$ pairs across agents. All state update equations are evaluated in Tier 1.

### Tier 2 — Epochal (Discrete, Sync)
*   **Nature**: Discrete, synchronous global barriers.
*   **Clocks**: Global points $T_0 < T_1 < \dots$
*   **Usage**: Used solely for system-wide snapshots. Has no causal role in the simulation. Cross-agent emergent claims are evaluated here as Tier 2 predicates.

## Reconciliation
*   **Tier 2 $\rightarrow$ Tier 1**: Tier 2 is simply a special case of Tier 1 where $\Delta t$ is identical and synchronized across all agents.
*   **Tier 0 $\rightarrow$ Tier 1**: Tier 0 is the mathematical limit of Tier 1 as the maximum $\Delta t \rightarrow 0$.

## Invariants
1.  Tier 1 events must be strictly increasing per agent.
2.  Tier 2 snapshots are measurement-only (read-only state).
3.  All update equations are strictly Tier 1.
4.  Cross-agent claims are Tier 2 predicates.
