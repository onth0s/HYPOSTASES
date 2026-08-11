# Front 12 — Scientific Discovery Loop

Spec Ref: `misc/next-steps.md` Section XIII

## Overview
Extends the basic cognitive loop (`Observe → Infer → Act`) into a complete iterative scientific discovery loop where agents generate hypotheses, rank explanations, design experiments, collect evidence, and update internal world representations.

## Scientific Cognitive Pipeline
```
Observe
   ↓
Infer
   ↓
Generate Hypotheses
   ↓
Rank Explanations
   ↓
Design Experiment
   ↓
Collect Evidence
   ↓
Update Hypotheses
   ↓
Act
```

## Objective
Enable agents to perform iterative model refinement and active hypothesis testing within unknown or shifting environments.

## Core State Constraint
Integrates Front 09 (Active Sensing), Front 11 (Abductive Reasoning), and Front 08 (Causal Models) over state $\sigma = (c, w, g, \rho_{\text{ext}})$.
