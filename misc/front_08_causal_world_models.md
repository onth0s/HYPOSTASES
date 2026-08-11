# Front 08 — Causal World Models

Spec Ref: `misc/next-steps.md` Section VIII

## Overview
Move beyond predictive correlational relationships. Represent causal structure explicitly within the world model, enabling intervention reasoning and counterfactual causality.

## Epistemic Shift
```
A predicts B      →      A causes B
```

## Targeted Capabilities
- Intervention reasoning (do-calculus)
- Causal diagnosis
- Counterfactual causality
- Structural causal models (SCMs)
- Policy evaluation under intervention

## Core State Constraint
Causal graph structures are computable projections over $w$ (WorldModel) within $\sigma = (c, w, g, \rho_{\text{ext}})$; formally typed and admissible for inference.
