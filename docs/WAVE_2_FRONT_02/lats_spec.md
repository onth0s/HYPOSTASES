# Language Agent Tree Search (LATS) — Literature Reference Spec

**Source**: Zhou et al., ICML (2024)  
**Relevance to HYPOSTASES Wave 2 Front 02**: Provides trajectory reflection, episodic execution memory integration, and contingency tree expansion mechanics.

---

## 1. Core Mechanics

LATS unifies search, action execution, and self-reflection into a tree structure:
1. **Trajectory Exploration**: Explores alternative plan branches under environment stochasticity.
2. **Value Evaluation**: Scores tree nodes using external environment feedback or internal game-theoretic utility $u(\sigma)$.
3. **Episodic Reflection**: Upon execution failure or sub-optimal plan yield, generates structured reflection summaries saved in episodic memory (`Front 03`) to guide future plan searches.

---

## 2. Contingency Tree Integration

Plan nodes maintain explicit branch conditions based on state observation predicates:
- Primary execution path: $\mathcal{P}_{\text{main}}$.
- Contingency sub-trees: $\mathcal{T}_{\text{contingency}}(w, c)$ triggered if observation fails expected precondition bounds.
