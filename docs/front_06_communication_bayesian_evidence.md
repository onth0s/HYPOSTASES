# Front 06 — Communication as Bayesian Evidence

Spec Ref: `misc/next-steps.md` Section VI

## Overview
Messages become probabilistic observations rather than deterministic information transfers. Communication acts as likelihood evidence that updates receiver belief states.

## Evidence Pipeline
```
Message
    ↓
Likelihood
    ↓
Posterior Belief
```

## Targeted Capabilities
- Trust modeling
- Deception detection
- Reputation tracking
- Uncertainty propagation through communication
- Misinformation resistance
- Evidence accumulation over message sequences

## Core State Constraint
Message likelihoods update $w$ (WorldModel) fields in $\sigma = (c, w, g, \rho_{\text{ext}})$ via standard Bayesian update on peer beliefs.
