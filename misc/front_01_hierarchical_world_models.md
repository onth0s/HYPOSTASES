# Front 01 — Hierarchical World Models

Spec Ref: `misc/next-steps.md` Section I

## Overview
The current World Model primarily maintains beliefs over the environment and peer latent states. Future iterations should investigate **multi-level semantic world representations**, allowing beliefs to exist across several abstraction layers.

## Abstraction Hierarchy
```
Environment
    ↓
Objects
    ↓
Relations
    ↓
Institutions
    ↓
Norms
    ↓
Meta-models
```

## Targeted Capabilities
- Semantic abstraction
- Relational reasoning
- Institutional reasoning
- Nested representations
- Abstract situation understanding

## Core State Constraint
Operates over persistent primitive state $\sigma = (c, w, g, \rho_{\text{ext}})$ without introducing non-computable primitives.
