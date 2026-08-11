# Front 10 — Mechanism Search

Spec Ref: `misc/next-steps.md` Section X

## Overview
Move beyond simulating a fixed institutional design toward searching over the space of possible mechanisms to find those that best satisfy a desired objective.

## Search Pipeline
```
Desired Objective
    ↓
Generate Mechanism
    ↓
Simulate
    ↓
Evaluate
    ↓
Modify
    ↓
Repeat
```

## Potential Applications
- Governance design
- Economics & market design
- Coordination protocols
- Institutional optimization
- Public policy
- Resource allocation rules

## Core State Constraint
The mechanism search layer operates externally over the simulation harness, treating $\sigma = (c, w, g, \rho_{\text{ext}})$ dynamics as a black-box evaluation oracle.
