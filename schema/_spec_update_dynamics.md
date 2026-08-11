# Schema Specification: Update Dynamics

This document describes the core update loop of the HYPOSTASES project, formalized as function composition, covering decision making, environmental interaction, feedback, and state evolution.

## Core Loop Functions

### 1. Decision Policy ($\pi_{decision}$)
*   **Signature**: $(C \times W \times G \times R_{ext}) ; \Xi \rightarrow \Delta(A)$
*   **Internals**: Dynamically computes $\rho_{int} = proj_{int}(c)$ and $\omega = derive\_\Omega(\dots)$.
*   **Mechanism**: Applies a softmax over $\pi \cdot \omega \cdot u$, tempered by the mean of $\xi$. Note that $\xi$ is a policy parameter, separated by a semicolon, not part of the state.
*   **Constraint**: The chosen action $a$ must satisfy $cost(a) \le (\rho_{ext}, \rho_{int})$ component-wise.

### 2. Environment Step (`step_env`)
*   **Signatures**:
    *   Single-agent: $E \times A \rightarrow E$
    *   Multi-agent: $E \times \{a^{(i)}\} \rightarrow E$
*   **Obligation**: Handling concurrency properly in multi-agent settings is a schema-level obligation.

### 3. Feedback ($\phi$)
*   **Signature**: $S \times S \times A \times W \rightarrow \Phi$
*   **Description**: Evaluates pre/post environment observations alongside the chosen action and the internal world model to generate a feedback delta tuple $\phi$.
*   **Full 4D Goal-Hierarchy Action Branches**:
    *   `REQUEST` (granted): $\Delta g[\text{SURVIVAL}] = G_{\text{surv}} \cdot (2r - 1)$, $\Delta g[\text{ACQUISITION}] = G_{\text{acq}} \cdot r$ where $r = \text{granted}/\text{amount}$.
    *   `REQUEST` (shortfall): $\Delta g[\text{SURVIVAL}] < 0$ (same formula, $r < 0.5$). Mood penalised proportional to shortfall and $(1 - \text{resilience})$.
    *   `SHARE`: $\Delta g[\text{RELATIONAL}] = G_{\text{rel}} \cdot c.\text{sociality}$. Reserve decremented, mood boosted.
    *   `WITHDRAW` (no active fee): $\Delta g[\text{STATUS}] = G_{\text{stat}} \cdot (1 - c.\text{sociality})$.
    *   `WITHDRAW` (active governance fee): STATUS positive and negative terms cancel ($\Delta g[\text{STATUS}] \approx 0$). Crowding-out shifts $\Delta g[\text{ACQUISITION}] \uparrow$, $\Delta g[\text{RELATIONAL}] \downarrow$.
*   **Softmax Jacobian Attenuation**: All $\Delta g[k]$ are scaled by $\pi_k(1 - \pi_k)$ (diagonal of the softmax Jacobian) before emission:
    $$\Delta g[k] \leftarrow \Delta g[k] \cdot \pi_k (1 - \pi_k)$$
    This yields a pure un-regularized fixed-point attractor from intrinsic dynamics alone; no external decay term is required.

### 4. State Evolution
The evolution of the persistent state components $\sigma$ from $t$ to $t+1$:
*   $c_{t+1} = c_t + \phi.\Delta c$
*   $w_{t+1}$: via Bayesian/Kalman update on the new observation.
*   $g_{t+1}$: $u_{t+1} = u_t + \phi.\Delta g$ (additive integration of Jacobian-attenuated deltas; $\pi_{t+1}$ recomputed fresh next tick).
*   $\rho_{ext, t+1}$: updated additively minus the action cost.
*   **Rule**: *Only primitives are integrated over time.* Derived quantities are recomputed fresh.


## Computational Modes
1.  **Forward Simulation (§3.1)**: Generative forward stepping.
2.  **State Evolution (§3.2-3.4)**: The composed functions forming the state transition.
3.  **Inverse Inference**: Computing the posterior over $\Sigma$ conditioned on a given trajectory. This uses the same generative model structure but evaluates it in reverse.
