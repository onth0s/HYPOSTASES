# Front 07 — Meta-Learning

Spec Ref: `misc/next-steps.md` Section VII

## Overview
Agents adapt not only their persistent state but also their own internal reasoning mechanisms. The learning process itself becomes an optimization target.

## Adaptation Targets
- Policy allocation (π)
- Planner structure
- Utility update rules
- Learning rates and EMA coefficients
- Inference parameters (ESS thresholds, particle counts)
- Decision heuristics

## Goal
Learning how to learn — agents develop adaptive meta-strategies that improve sample efficiency and behavioral generalization across environments.

## Core State Constraint
Meta-learning operators modify the functional mapping over $\sigma = (c, w, g, \rho_{\text{ext}})$ without introducing non-computable state mutations; parameters are themselves computable projections.
