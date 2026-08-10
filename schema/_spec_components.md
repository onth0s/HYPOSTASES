# Schema Specification: Components

This document specifies the complete component taxonomy for the HYPOSTASES project (v3 revised). It details primitives, derived components, and policy parameters that form the agent state and environment.

## 1. Primitives

Primitives are independent, persistent state components that form the core state $\sigma$. The per-agent persistent state $\sigma = (c, w, g, \rho_{ext})$ is a four-tuple.

| Name | Symbol | Space | Dim | Fields / Details |
|---|---|---|---|---|
| Characteristics | $c$ | $C = \mathbb{R}^{n_c}$ | 6 | `skill` $\in [0,1]$, `resilience` $\in [0,1]$, `sociality` $\in [0,1]$, `memory_decay` $\in [0,1]$, `reserve` $\ge 0$, `mood` $\in [-1,1]$. |
| World Model | $w$ | $W = \Delta(S) \times F$ | - | `belief` (Gaussian $\mu, \sigma^2$), `f` (learned replenish rate estimate scalar). |
| Goal Hierarchy | $g$ | $G = \Delta(K) \times \mathbb{R}^{n_k}$ | 4 | $K=\{SURVIVAL, ACQUISITION, RELATIONAL, STATUS\}$. $\pi$ soft distribution. |
| Power_external | $\rho_{ext}$ | $\mathbb{R}_{\ge 0}^{n_r}$ | 2 | `social_capital`, `time_budget`. |
| Action | $a$ | $A$ | - | Discrete: `REQUEST(amount)`, `SHARE(amount)`, `WITHDRAW`. |
| Environment | $e$ | $\mathbb{R}$ | 1 | Single scalar shared pool. $obs^{(i)}(e) = e + noise$. |
| Feedback | $\phi$ | $\Phi$ | - | $\Delta C \times \Delta W \times \Delta G \times \Delta R_{ext}$ (tuple of deltas). |

## 2. Derived

Derived components have no independent state; they are recomputed from primitives dynamically.

*   **Power_internal ($\rho_{int}$)**: Computed as $proj_{int}(c)$. `reserve_capacity` = $c.reserve$.
*   **Potentialities ($P(c)$)**: Reachable set query on $C$ given budget $R$.
*   **Willingness ($\omega$)**: Computed as $derive\_\Omega(g, \rho_{ext}, \rho_{int}, c)$. Formula: $\omega_k = \pi_k \cdot \min(1, \frac{\rho_{int}.reserve\_capacity}{cost\_estimate\_k})$.

## 3. Policy Parameter

Policy parameters do not constitute state but influence decision-making.

*   **Index of Exploration ($\xi$)**: $\in \mathbb{R}_{\ge 0}^{n_\xi}$ (where $n_\xi=4$, one per goal category). Acts as a temperature parameter on $\pi_{decision}$.
