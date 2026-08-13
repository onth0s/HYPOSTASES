# Front 02 — Explicit Planning Layer

Spec Ref: `misc/next-steps.md` Section II | Master Spec: [`docs/WAVE_2_FRONT_02/front_02_explicit_planning_layer_spec.md`](../WAVE_2_FRONT_02/front_02_explicit_planning_layer_spec.md)

## Overview
Transitions decision policy from direct `Goal → Action` mapping to a layered `Goal → Plan → Action` architecture. Plans become first-class reusable computational objects ($\Pi$).

## Architectural Transition
```
Goal → Plan → Action
```

## SOTA Literature References
- **AdaPlanner**: Closed-loop refine-then-resume planning & out-of-plan repair ([`adaplanner_spec.md`](../WAVE_2_FRONT_02/adaplanner_spec.md))
- **Reasoning via Planning (RAP)**: MCTS state evaluation with LLM world model ([`rap_spec.md`](../WAVE_2_FRONT_02/rap_spec.md))
- **Language Agent Tree Search (LATS)**: Contingency trees & episodic trajectory reflection ([`lats_spec.md`](../WAVE_2_FRONT_02/lats_spec.md))

## Targeted Capabilities
- Hierarchical planning & action DAGs
- Reusable strategy library (`PlanLibrary`)
- Dynamic contingency branching
- Closed-loop execution monitoring & out-of-plan interruption
- Plan repair & refine-then-resume (`PlanRepairEngine`)

## Core State Constraint
Planners operate over persistent primitive state $\sigma = (c, w, g, \rho_{\text{ext}})$ as higher-order computational layers.

