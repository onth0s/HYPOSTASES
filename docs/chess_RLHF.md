# Reinforcement Learning from Human Feedback (RLHF) for HYPOSTASES Chess Engine

> **Design Specification & Architectural Proposal**  
> High-level conceptual guide detailing how human expert feedback (RLHF/DPO) can be integrated into the HYPOSTASES Active Sensing & Meta-RL engine to bootstrap early self-play training and eliminate cold-start tactical blunders.

---

## Executive Summary

During early self-play training generations (Gen 0–Gen 5), autonomous agents expend significant computational resources exploring basic tactical anti-patterns (e.g., hanging pieces, premature queen exposure, or repetitive piece shuffling).

Integrating **Reinforcement Learning from Human Feedback (RLHF)** into HYPOSTASES provides a direct mechanism for a human domain expert to intervene, correct, or guide the agent's policy without compromising the core engine's game-theoretic state dynamics ($\sigma = (c, w, g, \rho_{\text{ext}})$).

---

## Architectural Interaction Points

Human feedback interfaces directly with three core components of the HYPOSTASES architecture:

```
                          +-------------------------+
                          |   Human Domain Expert   |
                          +------------+------------+
                                       |
                +----------------------+----------------------+
                |                      |                      |
                v                      v                      v
     [Mode 1: Interactive]    [Mode 2: Preference]   [Mode 3: Prior Initialization]
     Live Trajectory Veto     Pairwise Moves (DPO)   Human-Aligned theta_meta
                |                      |                      |
                +----------------------+----------------------+
                                       |
                                       v
                     +----------------------------------+
                     |  HYPOSTASES Active Sensing Engine |
                     |                                  |
                     |  U_total = (1-β)U_prag + βU_epis  |
                     +-----------------+----------------+
                                       |
                     +-----------------+----------------+
                     |                                  |
                     v                                  v
       +---------------------------+      +---------------------------+
       |   Meta-RL Parameter Vector|      |    NNUENet HalfKP Neural  |
       |       θ_meta Updates      |      |   Accumulator SGD Buffer  |
       +---------------------------+      +---------------------------+
```

---

## 1. Interaction Modes

### Mode 1: Interactive Live Trajectory Intervention (DAgger / TAMER Style)
- **Workflow**: During early self-play training generations, an interactive CLI / UI allows the human expert to monitor live move evaluations.
- **Intervention**: When the agent considers a tactical blunder, the expert can:
  1. **Veto / Correct**: Override the move with an expert choice (e.g., replace `1. e4 e5 2. Qh5` with `2. Nf3`).
  2. **Scalar Feedback**: Assign a real-time scalar reward $+1.0$ (praise) or $-1.0$ (reproof).
- **Engine Mechanics**:
  - Corrected moves are tagged as **High-Priority Expert Demonstrations** inside `train_nnue()` (Supervised Fine-Tuning buffer).
  - Meta-gradients $\Delta \theta_{\text{meta}}$ receive an immediate policy gradient bump toward the expert feature vector.

---

### Mode 2: Pairwise Preference Learning (Direct Preference Optimization - DPO)
- **Workflow**: The engine presents the human expert with two alternative candidate trajectories generated from the same board state:
  - **Variation A**: `1. e4 e5 2. Qh5 Nc6 3. Bc4 g6`
  - **Variation B**: `1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6`
- **Intervention**: The expert indicates a preference ($y_w \succ y_l$).
- **Engine Mechanics**:
  - Evaluated using a Bradley-Terry preference loss:
    $$\mathcal{L}_{\text{DPO}}(\theta) = -\log \sigma \left( U_{\text{EFE}}(y_w) - U_{\text{EFE}}(y_l) \right)$$
  - Adjusts both NNUENet's dense linear layers and the active perception parameter $\beta_{\text{EFE}}$ to align pragmatic utility $U_{\text{pragmatic}}$ with human positional evaluation.

---

### Mode 3: Human Preference Prior Initialization (Soft Anchoring)
- **Workflow**: Instead of real-time move intervention, human opening principles (center control, early piece development, king safety) are formalized into prior initialization weights.
- **Engine Mechanics**:
  - Initial $\theta_{\text{meta}}$ at Gen 0 is initialized with human-anchored priors $\theta_0$ rather than isotropic uniforms `[1.0, 1.0, ...]`.
  - The curriculum position sampler ($25\%$ opening curriculum) draws from master human opening datasets to bootstrap early HalfKP accumulator representations.

---

## 2. Expected Performance Benefits

1. **Elimination of Cold-Start Exploration Wasted Ticks**: Bootstrap-aligns the initial policy within 10–20 human-guided games, skipping hundreds of random exploration generations.
2. **Accelerated Monotonicity**: Guides early $\theta_{\text{meta}}$ trajectory towards stable tactical regimes, preventing anti-chess draw collapse.
3. **Calibrated Active Sensing ($\beta_{\text{EFE}}$)**: Directly tunes the balance between pragmatic tactical safety and epistemic exploration entropy.
