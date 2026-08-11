# Front 04 — Counterfactual Simulation

Spec Ref: `misc/next-steps.md` Section IV

## Overview
Replaces direct greedy action selection with internal multi-future hypothetical simulations evaluated prior to physical execution.

## Simulation Pipeline
```
Current State
     ↓
Future A | Future B | Future C
     ↓
Evaluation (Expected Utility)
     ↓
Execution
```

## Targeted Capabilities
- Lookahead search
- Planning under uncertainty
- Hypothetical reasoning
- Branch evaluation & expected utility estimation
- Monte Carlo search

## Core State Constraint
Evaluates forward rollout branches conditioned on primitive state tuple $\sigma = (c, w, g, \rho_{\text{ext}})$.
