# Front 09 — Active Information Gathering & Active Perception

Master Spec Ref: [`docs/WAVE_1_FRONT_09/front_09_active_information_gathering_spec.md`](../WAVE_1_FRONT_09/front_09_active_information_gathering_spec.md) | Ingested Literature: [`docs/WAVE_1_FRONT_09/papers/`](../WAVE_1_FRONT_09/papers)

## Overview
Observation becomes an intentional, goal-directed action. Agents deliberately trade immediate utility for uncertainty reduction — epistemic utility enters the decision calculus alongside material utility.

Active Perception establishes observation as a first-class execution branch in environment feedback loops (`execute_epistemic_action`), updating belief variance ($\Delta \sigma^2$) and quality space coordinates.

## Epistemic Action Types (`EPISTEMIC_ACTION_TYPES`)
- `INSPECT`: Directed visual/spatial observation of a targeted entity or canvas location.
- `PROBE`: Active stimulus injection to measure environment feedback variance.
- `QUERY`: Epistemic query emitted to peer agents or institutions.
- `EXPERIMENT`: Systematic multi-step empirical test of competing hypotheses.
- `MONITOR`: Continuous sensory tracking over a designated region.
- `SPY`: Passive covert information gathering.
- `VERIFY`: Epistemic confirmation check against ground truth.

## Mechanics & Feedback Integration
When an epistemic action is executed, it routes directly into `execute_epistemic_action()` in the dynamics loop:
$$\text{Utility}_{\text{epistemic}} = \alpha_{\text{info}} \Delta H(w) - \text{Cost}(a_{\text{epistemic}})$$

Where $\Delta H(w)$ measures the reduction in Shannon entropy / variance $\sigma^2$ over WorldModel beliefs.

## Theoretical Info-Computational Foundation & Active Inference (Dodig-Crnkovic 2022)
Following **Dodig-Crnkovic (2022)** (*Entropy*, Cognition as Morphological/Morphogenetic Embodied Computation In Vivo):
- **Info-Computational Active Inference**: Epistemic actions operate under an info-computational paradigm where physical/abstract environment structures represent information and state transitions represent computation.
- **Variational Free Energy Minimization**: Active perception minimizes expected free energy by executing sensory affordances (`INSPECT`, `PROBE`, `MONITOR`) that resolve environmental ambiguity and update Bayesian belief posteriors $P(w \mid o)$.
- **Rule 005 Compliance**: Epistemic decision dynamics operate strictly under optimal game-theoretic Bayesian state updates and computable projections without introducing artificial human biases or emotional irrationality hacks.

## Implementation Status
- **Status**: **IMPLEMENTED** ([`src/hypostases/active_perception.py`](../../src/hypostases/active_perception.py), [`epistemic_utility.py`](../../src/hypostases/epistemic_utility.py), `test_active_perception.py`).

## Objective
Trade immediate material utility for uncertainty reduction — minimizing posterior entropy over latent state $\sigma$ as an explicit sub-goal within the goal hierarchy $g$.

## Core State Constraint
Epistemic actions extend `ActionType` without altering primitive state structure $\sigma = (c, w, g, \rho_{\text{ext}})$; information gain is modeled as a computable utility signal.
