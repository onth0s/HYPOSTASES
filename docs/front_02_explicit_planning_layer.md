# Front 02 — Explicit Planning Layer

Spec Ref: `misc/next-steps.md` Section II

## Overview
Transitions decision policy from direct `Goal → Action` mapping to a layered `Goal → Plan → Action` architecture. Plans become first-class reusable computational objects.

## Architectural Transition
```
Goal → Plan → Action
```

## Targeted Capabilities
- Hierarchical planning
- Reusable strategies
- Contingency plans
- Plan interruption & repair
- Plan libraries
- Long-horizon reasoning

## Core State Constraint
Planners operate over persistent primitive state $\sigma = (c, w, g, \rho_{\text{ext}})$ as higher-order computational layers.
