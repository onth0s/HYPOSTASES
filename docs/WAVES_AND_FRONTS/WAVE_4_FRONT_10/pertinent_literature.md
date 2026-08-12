# Wave 4 Front 10 — Synthesized Pertinent Literature

**Front**: Front 10 — Mechanism Search  
**Wave**: Wave 4 (Recursive Adaptation & Discovery Loops)  
**Target Substrate**: HYPOSTASES Engine ($\sigma = (c, w, g, \rho_{\text{ext}})$)  
**Rule 005 Invariant**: Pure Game-Theoretic & Probabilistic Rationality (Zero Artificial Human Cognitive Biases)

---

## 1. Executive Synthesis & Architectural Integration

The 12 ingested papers in [`docs/WAVE_4_FRONT_10/papers/`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_10/papers/) establish the theoretical, mathematical, and algorithmic foundations for the **Mechanism Search Layer** in HYPOSTASES.

Mechanism Search transitions the HYPOSTASES engine from evaluating hand-designed static institutions to active computational optimization over the space of institutional rules, market mechanisms, auction designs, tax-subsidy schedules, and governance policies. The complete agent state tuple:

$$\sigma = (c, w, g, \rho_{\text{ext}})$$

is evaluated dynamically via the simulation harness oracle $\mathcal{O}_{\text{sim}}$.

In strict adherence to **Rule 005** (prohibiting artificial human cognitive defects or emotional irrationality hacks), mechanism candidates are evaluated against populations of game-theoretically rational agents operating under Friston Expected Free Energy (EFE) active perception (`efe_mode: true`, Rule 009) and optimal probabilistic planning.

---

## 2. Mapping Ingested Literature to State Substrate $\sigma = (c, w, g, \rho_{\text{ext}})$

```mermaid
graph TD
    subgraph Mechanism Search Layer (Wave 4 Front 10)
        VICK[Vickrey 1961: Second-Price Sealed Bid DSIC]
        CLARKE[Clarke 1971: Pivot Tax Public Goods]
        GROVES[Groves 1973: Incentive Alignment in Teams]
        MYER[Myerson 1981: Virtual Valuations & Optimal Auctions]
        CLIFF[Cliff 1997: ZIP Agents & Double Auctions]
        AMD[Conitzer & Sandholm 2002/2004: Automated Mechanism Design]
        PHELPS[Phelps et al. 2010: Evolutionary Mechanism Design]
        ROUGH[Roughgarden 2010/2014: Price of Anarchy & Stability Bounds]
        REG[Dütting et al. 2019/2024: RegretNet & Differentiable Economics]
        REAL[Real et al. 2020: AutoML-Zero Functional Equivalence]
        ECON[Zheng et al. 2020/2022: AI Economist & Tax Policy Search]
        ALPHA[Novikov et al. 2025: AlphaEvolve Evolutionary Coding Agents]
    end

    subgraph HYPOSTASES State Tuple σ
        C[c: Cognition & Bidding Signals]
        W[w: World Model & Valuation Distributions]
        G[g: Expected Free Energy Utility & Social Welfare]
        R[ρ_ext: Institutional Allocation X & Payment P Schedules]
    end

    VICK --> R
    CLARKE --> R
    GROVES --> R
    MYER --> W
    CLIFF --> C
    AMD --> R
    PHELPS --> R
    ROUGH --> G
    REG --> R
    REAL --> C
    ECON --> R
    ALPHA --> C
```

### 2.1 Component $\rho_{\text{ext}}$ — External Institutional State & Governance Rules
- **Vickrey (1961), Clarke (1971), & Groves (1973)**: Define the VCG mechanism family establishing dominant strategy incentive compatibility (DSIC) for private valuations and public goods:
  $$P_i(b) = h_i(b_{-i}) - \sum_{j \neq i} v_j(x(b))$$
- **Conitzer & Sandholm (2002, 2004) [Automated Mechanism Design]**: Formulates structural optimization over outcome allocations $\mathbf{X}(b)$ and payments $\mathbf{P}(b)$ subject to exact IC and IR constraints, proving NP-completeness for deterministic mechanisms without side payments and polynomial tractability for randomized mechanisms.
- **Dütting et al. (2019, 2024) [RegretNet & Differentiable Economics]**: Formulates continuous mechanism search as an end-to-end differentiable architecture, using augmented Lagrangian penalization for empirical IC regret $\mathcal{R}_{\text{IC}}(\mu)$.
- **Zheng et al. (2020, 2022) [The AI Economist]**: Demonstrates bi-level MARL search over tax-subsidy schedules $\tau(y)$, optimizing social welfare functions combining economic Productivity and Gini Equality.
- **Phelps et al. (2010) [Evolutionary Mechanism Design]**: Reviews genetic algorithms and AST mutations for discovering double auction market rules when analytic gradients are unavailable.

### 2.2 Component $g$ — Goal State & Expected Free Energy Utility
- **Roughgarden (2010, 2014) [Algorithmic Game Theory]**: Establishes Price of Anarchy (PoA) and Price of Stability (PoS) metrics comparing equilibrium multi-agent social welfare under mechanism $\mu$ against social optimum.
- **Zheng et al. (2022)**: Supplies the multi-criteria social welfare aggregator combining Productivity and Gini Equality:
  $$W(\mathbf{\sigma}_{1:T}) = \left( \sum_{i=1}^N u_{i,T} \right) \times \left( 1 - \text{Gini}(u_{1:N, T}) \right)$$

### 2.3 Component $w$ & $c$ — World Model & Cognition/Bidding Signals
- **Myerson (1981) [Optimal Auction Design]**: Defines virtual valuations $\psi_i(v_i) = v_i - \frac{1 - F_i(v_i)}{f_i(v_i)}$ and revenue maximization theorems used as formal math verification baselines in `tests/formal_math/test_mechanism_search_formal.py`.
- **Cliff (1997) [ZIP Agents]**: Demonstrates fast market equilibrium convergence using Zero Intelligence Plus adaptive agents in continuous double auctions.
- **Real et al. (2020) & Novikov et al. (2025) [AutoML-Zero & AlphaEvolve]**: Informs functional equivalence checking (FEC) and multi-component evolutionary AST code mutations for discrete institutional rule search.

---

## 3. Mathematical Unification for HYPOSTASES Engine

### 3.1 Bi-Level Optimization Formulation
$$J(\mu) = \mathbb{E}_{\mathbf{\sigma} \sim \mathcal{O}_{\text{sim}}(\mu)} [W(\mathbf{\sigma}_{1:T})] - \lambda_{\text{IC}} \mathcal{R}_{\text{IC}}(\mu) - \lambda_{\text{IR}} \mathcal{R}_{\text{IR}}(\mu) - \lambda_{\text{budget}} \mathcal{R}_{\text{budget}}(\mu)$$

### 3.2 Key Invariants for Implementation Verification
1. **VCG Incentive Compatibility Invariant**: $\mathcal{R}_{\text{IC}}(\mu_{\text{VCG}}) \equiv 0$.
2. **Clarke Pivot Tax Public Goods Invariant**: $\mathcal{R}_{\text{IC}}(\mu_{\text{Clarke}}) \equiv 0$.
3. **Myerson Revenue Equivalence Bounds**: $\mathbb{E}[P_{\text{discovered}}(b)] \ge \mathbb{E}[P_{\text{Myerson}}(b)] - \epsilon$.
4. **Simplex Projection Conservation**: $\sum_{i=1}^N X_{i,j}(b) \le 1, \quad \forall j \in M$.
5. **Pure Game-Theoretic Rationality Invariant (Rule 005)**: Zero artificial human cognitive defects in agent utility $u_i$.
