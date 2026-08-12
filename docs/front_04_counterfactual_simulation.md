# Front 04 — Counterfactual Simulation

Spec Ref: `misc/next-steps.md` Section IV

## Overview
Replaces direct greedy action selection with internal multi-future hypothetical simulations evaluated prior to physical execution.

## Simulation Pipeline
```
Current State σ = (c, w, g, ρ_ext)
     ↓
Branch Generation (Hypothetical Actions & Skill Primitives)
     ↓
Forward Rollouts (Internal step_env / Elastica PDE Rollouts)
     ↓
Evaluation (Expected Utility E[u] & Epistemic Information Gain ΔH)
     ↓
Execution Selection (Commitment to Physical Action / Plan)
```

## Continuous Physical Rollouts & Reachability Controls
Following **Cacace et al. (2020)** (*arXiv*, Octopus Tentacle Control & Dynamic Elastica), internal hypothetical simulations in continuous physical domains incorporate continuous PDE forward operators and optimal control bounds:
- **Curvature-Constrained Dynamic Elastica**: Simulates flexible continuum trajectories subject to physical curvature bounds $\kappa(s, t) \le \kappa_{\max}$ and inextensibility constraints.
- **Dubins Reachability Equivalence**: Evaluates stationary reachability sets by connecting current state configurations to target goal positions via minimal-length constrained curvature paths.
- **Adjoint-Based Optimal Control**: Computes exact adjoint gradients to rapidly evaluate alternative future trajectories during Monte Carlo counterfactual branch evaluation without mutating external physical state.

## Implementation Status
- **Status**: **IMPLEMENTED** ([`src/hypostases/counterfactual.py`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/src/hypostases/counterfactual.py), `test_counterfactual.py`).

## Targeted Capabilities
- Lookahead search & Monte Carlo tree evaluation
- Planning under uncertainty & PDE trajectory rollout
- Hypothetical counterfactual reasoning
- Branch evaluation & expected utility estimation
- Adjoint-guided path pruning

## Core State Constraint
Evaluates forward rollout branches conditioned on primitive state tuple $\sigma = (c, w, g, \rho_{\text{ext}})$.
