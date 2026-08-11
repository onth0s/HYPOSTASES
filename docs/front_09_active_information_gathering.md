# Front 09 — Active Information Gathering & Active Perception

Spec Ref: `misc/next-steps.md` Section IX | Epistemic Action Integration

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

## Objective
Trade immediate material utility for uncertainty reduction — minimizing posterior entropy over latent state $\sigma$ as an explicit sub-goal within the goal hierarchy $g$.

## Core State Constraint
Epistemic actions extend `ActionType` without altering primitive state structure $\sigma = (c, w, g, \rho_{\text{ext}})$; information gain is modeled as a computable utility signal.
