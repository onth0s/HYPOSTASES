# Wave 4 Front 12 — Synthesized Pertinent Literature

**Front**: Front 12 — Scientific Discovery Loop  
**Wave**: Wave 4 (Recursive Adaptation & Discovery Loops)  
**Target Substrate**: HYPOSTASES Engine ($\sigma = (c, w, g, \rho_{\text{ext}}$)  
**Rule 005 Invariant**: Pure Game-Theoretic & Probabilistic Rationality (Zero Artificial Human Cognitive Biases)  
**Rule 006 Compliance**: Primacy of Data-Driven YAML Approach (`schema/scientific_discovery_config.yaml`)  
**Rule 009 Compliance**: Default Friston Expected Free Energy (EFE) Integration (`efe_mode: true`)

---

## 1. Executive Synthesis & Architectural Integration

The literature synthesized for **Wave 4 Front 12** incorporates 6 primary ingested PDF papers alongside foundational text references, establishing the theoretical, mathematical, and algorithmic foundations for automated, autonomous scientific discovery loops.

The Scientific Discovery Loop unifies three core cognitive fronts:
- **Front 08 (Causal World Models)**: Structural Causal Models (SCMs) and intervention calculus ($do$-calculus, Pearl 2000/2009).
- **Front 09 (Active Sensing & Information Gathering)**: Epistemic utility maximization and Friston Expected Free Energy (EFE, Lindley 1956, MacKay 1992, Foster 2020).
- **Front 11 (Abductive Reasoning & Hypothesis Objects)**: Explanations represented as explicit computational objects $H_k \in \mathcal{H}$ subject to Elo-based evolutionary debate tournaments (King 2004, Lu 2024, Gottweis 2025).

By closing the loop between hypothesis generation, optimal experiment design, interventional execution in $\rho_{\text{ext}}$, and Bayesian posterior updating over SCM structures, HYPOSTASES agents gain the capability to perform autonomous scientific inquiry.

All operations adhere strictly to **Rule 005**, ensuring that hypothesis evaluation and experiment selection are driven purely by expected information gain, Bayesian likelihoods, Minimum Description Length (MDL) Occam penalties, and optimal game-theoretic decision policies, without artificial human cognitive biases or irrational emotional defects.

---

## 2. Mapping Ingested Literature to State Substrate $\sigma = (c, w, g, \rho_{\text{ext}})$

```mermaid
graph TD
    subgraph Ingested Scientific Discovery Literature
        LIND[Lindley 1956: Expected Information Gain EIG]
        MAC[MacKay 1992: Active Data Selection & Variance Reduction]
        ADAM[King et al. 2004: Robot Scientist Adam & ASE Cost Minimization]
        DAD[Foster et al. 2020: Deep Adaptive Design & ACE Bounds]
        PEARL[Pearl 2000/2009: Structural Causal Models & do-Calculus]
        AISCI[Lu et al. 2024: The AI Scientist Closed-Loop Iteration]
        COSCI[Gottweis et al. 2025: Co-Scientist Elo Tournaments]
        DISC[Discovery Loop / Dean et al. 2026: High-Throughput Science]
    end

    subgraph HYPOSTASES State Tuple σ
        C[c: Cognition & Hypothesis Pool H_k, Elo Tournaments & MDL Penalties]
        W[w: World Model & Ensemble of SCM Causal Graphs]
        G[g: Goal State & Expected Information Gain EIG / EFE]
        R[ρ_ext: External Environment & Interventional Execution do(d*)]
    end

    LIND --> G
    MAC --> G
    ADAM --> C
    ADAM --> R
    DAD --> G
    PEARL --> W
    AISCI --> C
    COSCI --> C
    DISC --> R
```

---

## 3. Mathematical & Algorithmic Substrate Mapping

### 3.1 Cognition State ($c$): Epistemic Updates & Hypothesis Evolutionary Tournaments
- **Bayesian Posterior Update Core (Lindley 1956, MacKay 1992)**:
  Epistemic state updating over hypothesis space $\mathcal{H} = \{H_1, \dots, H_K\}$ obeys exact Bayesian updates:
  $$P(H_k \mid E_{1:t}) \propto P(H_k \mid E_{1:t-1}) \cdot P(E_t \mid H_k, d^*)$$
- **Evolutionary Tournament Dynamics (Gottweis et al. 2025)**:
  Hypothesis generation within $c.m_{\text{procedural}}$ uses multi-agent debate tournaments. A Ranking evaluator computes Elo ratings $R(H_k)$ over pairwise competitive comparisons:
  $$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}, \quad R_A' = R_A + K \left( S_A - E_A \right)$$
  Top-rated hypotheses advance into the active experimental pool, while low-rated hypotheses undergo mutation (Evolution agent) or reflection decomposition (Reflection agent).
- **Abductive ASE Cost Minimization (King et al. 2004)**:
  Under candidate hypotheses $H_k$, the Active Selection of Experiments (ASE) strategy selects experiment designs that minimize expected cost to eliminate invalid hypotheses.
- **Rule 005 Compliance**: Zero artificial human cognitive defects (e.g. confirmation bias, sunk cost fallacy) are introduced. Hypothesis ranking is strictly governed by Elo expected scores, Bayesian posterior probabilities, and Minimum Description Length (MDL) Occam penalties.

### 3.2 World Model ($w$): Structural Causal Models & Amortized Querying
- **SCM Interventions (Pearl 2000/2009)**:
  The environment is represented as an ensemble of Structural Causal Models $\mathcal{M}_{\text{causal}} = \{f_1, \dots, f_K\}$. Experimental interventions $do(X_i = x_i)$ mutate graph topologies $\mathcal{G}_k$ by replacing structural assignment equations $X_i := f_i(\text{Pa}(X_i), \epsilon_i)$ with fixed constant constraints $X_i := x_i$.
- **Amortized BOED via ACE Bounds (Foster et al. 2020)**:
  Sequential experimental design queries into $w$ are optimized using Adaptive Contrastive Estimation (ACE) lower bounds:
  $$I_{\text{ACE}}(\xi, \phi, L) = \mathbb{E} \left[ \log \frac{p(y \mid \theta_0, \xi)}{\frac{1}{L+1}\sum_{\ell=0}^L p(y \mid \theta_\ell, \xi)} \right] \le \text{EIG}(\xi)$$
  allowing real-time EIG optimization across high-dimensional continuous state spaces.

### 3.3 Goal State & Utility ($g$): Friston Expected Free Energy Integration (Rule 009)
- **Friston EFE Active Sensing (`efe_mode: true`)**:
  Action selection for experimental queries $d^* \in \mathcal{D}$ strictly maximizes Expected Information Gain (EIG) under Friston Expected Free Energy mode:
  $$G(d) = -\text{EIG}(d) + \mathbb{E}_{P(o|d)} \left[ D_{\text{KL}}(P(o \mid d) \parallel P_{\text{target}}(o)) \right]$$
  When pragmatic goal constraints are inactive ($P_{\text{target}}$ is uniform), action selection reduces strictly to maximum EIG, eliminating ad-hoc utility mixing heuristics.

### 3.4 External State & Budget Constraints ($\rho_{\text{ext}}$): Cost-Aware Experimentation
- **Budget-Bounded Active Learning (King et al. 2004, Foster et al. 2020)**:
  Physical experimentation in $\rho_{\text{ext}}$ incurs execution costs $C(d)$. The design policy optimizes cost-normalized EIG:
  $$d^* = \arg\max_{d \in \mathcal{D}} \frac{\text{EIG}(d)}{C(d) + \epsilon_{\text{cost}}}$$
  respecting external resource limits $\rho_{\text{ext}}$ while accelerating convergence.

---

## 4. Key Invariants for Implementation Verification (`tests/formal_math/`)

1. **EIG Non-Negativity Invariant**: $\text{EIG}(d) \ge 0 \quad \forall d \in \mathcal{D}$.
2. **Bayesian Posterior Asymptotic Consistency**: $\lim_{t \to \infty} P(H^* \mid E_{1:t}) = 1.0$.
3. **ACE Bound Monotonicity**: $I_{\text{ACE}}(\xi, \phi, L_1) \le I_{\text{ACE}}(\xi, \phi, L_2)$ for $L_1 < L_2$.
4. **Elo Tournament Rank Stability**: Iterative debate tournaments monotonically converge toward Pareto-optimal hypothesis rankings.
5. **Rule 005 Anti-Human Defect Invariant**: Zero cognitive bias penalties or irrational emotional defects in hypothesis scoring.
