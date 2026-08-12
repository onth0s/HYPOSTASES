# Reasoning via Planning (RAP) — Literature Reference Spec

**Source**: Hao et al., NeurIPS (2023–2024)  
**Relevance to HYPOSTASES Wave 2 Front 02**: Mathematical bridge integrating Monte Carlo Tree Search (MCTS) with state evaluation and Front 04 counterfactual simulation sandbox.

---

## 1. Core Formulation

RAP models planning as tree search over an explicit world model, decoupling action generation from state evaluation:
- **World Model**: Simulates internal forward state transitions $s_{t+1} \sim P(s_{t+1} \mid s_t, a_t)$. In HYPOSTASES, this corresponds to `step_env` / `feedback` in `counterfactual.py`.
- **Reasoning Agent**: Proposes candidate action branches $a_t \in \mathcal{A}$.

---

## 2. MCTS & Q-Value Propagation

1. **Selection**: Navigates tree using UCT (Upper Confidence Bound for Trees):
   $$UCT(s, a) = Q(s, a) + c_{\text{puct}} P(a \mid s) \frac{\sqrt{N(s)}}{1 + N(s, a)}$$
2. **Expansion & Simulation**: Evaluates candidate nodes via game-theoretic expected utility $E[u(c, w)]$.
3. **Back-propagation**: Updates node visit counts $N(s, a)$ and action values $Q(s, a)$ up the planning tree.
