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
*   **Action Branches**:
    *   `REQUEST`: Results in either granted amount or a mood penalty.
    *   `SHARE`: Results in resource loss ($-x$) and a mood boost.
    *   `WITHDRAW`: Results in $0$ change and a mood penalty (fixed in v3).

### 4. State Evolution
The evolution of the persistent state components $\sigma$ from $t$ to $t+1$:
*   $c_{t+1} = c_t + \phi.\Delta c$
*   $w_{t+1}$: via Bayesian/Kalman update on the new observation.
*   $g_{t+1}$: via softmax renormalization based on feedback.
*   $\rho_{ext, t+1}$: updated additively minus the action cost.
*   **Rule**: *Only primitives are integrated over time.* Derived quantities are recomputed fresh.

## Computational Modes
1.  **Forward Simulation (§3.1)**: Generative forward stepping.
2.  **State Evolution (§3.2-3.4)**: The composed functions forming the state transition.
3.  **Inverse Inference**: Computing the posterior over $\Sigma$ conditioned on a given trajectory. This uses the same generative model structure but evaluates it in reverse.
