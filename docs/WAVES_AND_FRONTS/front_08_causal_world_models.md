# Front 08 — Causal World Models & Structural Causal Graphs

**Ratified Specification**: [`docs/WAVE_2_FRONT_08/front_08_causal_world_models_spec.md`](../WAVE_2_FRONT_08/front_08_causal_world_models_spec.md)  
**Spec Ref**: `misc/next-steps.md` Section VIII | **Compass**: [`docs/roadmap_compass.md`](../roadmap_compass.md)

## Overview
Move beyond predictive correlational relationships. Represent causal structure explicitly within world model $w$ in $\sigma = (c, w, g, \rho_{\text{ext}})$, enabling intervention reasoning ($do$-calculus), counterfactual causality, and automated continuous/constraint-based causal discovery.

## Epistemic Shift
```
A predicts B      →      A causes B (SCM & do-calculus)
```

## Targeted Capabilities
- Structural Causal Models (SCMs) & Pearl's 3-rung Causal Hierarchy
- Symbolic $do$-Calculus Engine (Rules 1-3) & Backdoor/Frontdoor Criteria
- 3-Step Counterfactual Regret Engine (Abduction, Action, Prediction) & Causal Thompson Sampling ($TS^C$)
- Cost-Optimal Intervention Planner ($X^* = \arg\min \text{Cost}(X)$)
- Relational Structural Causal Models (RSCMs) & Selection Diagram Transportability
- Continuous DAG Structure Discovery (NOTEARS $h(W) = \text{tr}(e^{W \circ W}) - d = 0$) & PC Algorithm

## Core State Constraint
Causal graph structures are computable projections over $w$ (WorldModel) within $\sigma = (c, w, g, \rho_{\text{ext}})$; formally typed and admissible for inference. Strictly adheres to **Rule 005** (Strict Prohibition of Artificial Human Cognitive Deficiencies) and **Rule 006** (Primacy of Data-Driven YAML Approach).
