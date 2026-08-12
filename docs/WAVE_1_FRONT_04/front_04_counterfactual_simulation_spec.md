# Front 04 Specification — Counterfactual Simulation & Multi-Future Lookahead

**Status**: SPECIFICATION  
**Wave**: Wave 1 (Single-Agent Foundations)  
**Front**: Front 04 — Counterfactual Simulation  
**Target Substrate**: HYPOSTASES Engine ($\sigma = (c, w, g, \rho_{\text{ext}})$)  
**SOTA Reference**: ICML/ICLR 2026 EvoCF (*Evolutionary Counterfactual Planning*) & Memory-Grounded MCTS  

---

## 1. Overview & Theoretical Architecture

Front 04 replaces basic direct single-step action selection with **internal multi-future hypothetical rollouts and Monte Carlo tree evaluation**. Prior to committing an action to physical environment execution, an agent evaluates competing hypothetical action trajectories in an ephemeral virtual sandbox.

```
                           [ Agent State σ_t = (c, w, g, ρ_ext) ]
                                            │
                                            ▼
                           ┌──────────────────────────────────┐
                           │   Virtual Sandbox Ephemeral Clone│
                           │   σ_virtual = clone(σ_physical)  │
                           └────────────────┬─────────────────┘
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               ▼                            ▼                            ▼
     ┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
     │  Trajectory A    │         │   Trajectory B   │         │   Trajectory C   │
     │ K steps rollout  │         │ K steps rollout  │         │ K steps rollout  │
     └────────┬─────────┘         └────────┬─────────┘         └────────┬─────────┘
              │                            │                            │
              ▼                            ▼                            ▼
     ┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
     │ Expected Utility │         │ Expected Utility │         │ Expected Utility │
     │ E[U(A)]          │         │ E[U(B)]          │         │ E[U(C)]          │
     └────────┬─────────┘         └────────┬─────────┘         └────────┬─────────┘
              └────────────────────────────┼────────────────────────────┘
                                           │
                                           ▼
                           ┌──────────────────────────────────┐
                           │   Argmax Expected Utility (EvoCF)│
                           │   Selected Action a* Executed    │
                           └──────────────────────────────────┘
```

---

## 2. Invariance & Rule 005 Compliance

1. **State Invariant Compliance**: The persistent agent state remains strictly the four-tuple $\sigma = (c, w, g, \rho_{\text{ext}})$. Ephemeral trajectory states during counterfactual rollout exist only in transient memory during calculation.
2. **Rule 005 Prohibitions**: Action trajectories are selected strictly using formal game-theoretic expected utility and Monte Carlo value estimation.
3. **Data-Driven Configuration (Rule 006)**: All rollout depth, branching factors, and discount hyperparameters are configured in `schema/counterfactual_config.yaml`.
