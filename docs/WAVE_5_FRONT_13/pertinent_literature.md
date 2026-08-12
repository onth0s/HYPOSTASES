# Wave 5 Front 13 — Synthesized Pertinent Literature

**Front**: Front 13 — Evolutionary Algorithm Discovery (AlphaEvolve Engine)  
**Wave**: Wave 5 (Universal Scaling & Symbolic Generalization)  
**Target Substrate**: HYPOSTASES Engine ($\sigma = (c, w, g, \rho_{\text{ext}}$)  
**Rule 005 Invariant**: Pure Game-Theoretic & Probabilistic Rationality (Zero Artificial Human Cognitive Biases)  
**Rule 006 Compliance**: Primacy of Data-Driven YAML Approach (`schema/alphaevolve_config.yaml`)  
**Rule 009 Compliance**: Default Friston Expected Free Energy (EFE) Integration (`efe_mode: true`)

---

## 1. Executive Synthesis & Architectural Integration

The literature synthesized for **Wave 5 Front 13** incorporates 9 foundational papers establishing the theoretical, algorithmic, and mathematical framework for **Evolutionary Algorithm Discovery (AlphaEvolve Engine)** operating within multi-agent game-theoretic environments.

Front 13 unifies four primary paradigms:
- **LLM Program Search & AST Mutation (AlphaEvolve / FunSearch)**: Code-level program evolution where LLMs and AST mutators manipulate candidate Python functions evaluated against automated evaluation oracles (Novikov et al. 2025, Romera-Paredes et al. 2023).
- **AutoML-Zero & Functional Equivalence Checking (FEC)**: Evolving algorithms from basic mathematical primitives while using functional equivalence testing to prune duplicate AST structures (Real et al. 2020).
- **Quality-Diversity (MAP-Elites) & Behavioral Feature Grids**: Maintaining multi-dimensional feature archives indexed by behavioral descriptors $(\mathcal{C}_{\text{time}}, \mathcal{R}_{\text{IC}}, \Delta \kappa_{\text{scarcity}})$ to prevent premature convergence and deceptive local optima trapping (Mouret & Clune 2015, Lehman & Stanley 2011).
- **Morphological Computation & Reservoir Co-Evolution**: Co-evolving physical/procedural reservoir dynamics in $\rho_{\text{ext}}$ alongside software action primitives in $c.m_{\text{procedural}}$ (Müller & Hoffmann 2017).

By evaluating candidate AST algorithms against multi-agent game-theoretic equilibrium, incentive compatibility regret $\mathcal{R}_{\text{IC}}$, resource scarcity $\kappa$, and Friston EFE active sensing (`efe_mode: true`), HYPOSTASES scales algorithm search beyond single-agent scalar benchmarks.

All evolved code primitives operate strictly as derived projections over the primitive state tuple:

$$\sigma = (c, w, g, \rho_{\text{ext}})$$

---

## 2. Mapping Literature to State Substrate $\sigma = (c, w, g, \rho_{\text{ext}})$

```mermaid
graph TD
    subgraph Foundational Literature (Wave 5 Front 13)
        ALPHA[Novikov et al. 2025: AlphaEvolve Coding Agents]
        FUN[Romera-Paredes et al. 2023: FunSearch Program Search]
        ZERO[Real et al. 2020: AutoML-Zero AST Primitives & FEC]
        MAP[Mouret & Clune 2015: MAP-Elites QD Archives]
        MORPH[Müller & Hoffmann 2017: Morphological Computation]
        EMD[Phelps et al. 2010: Evolutionary Mechanism Search]
        NOV[Lehman & Stanley 2011: Novelty Search]
        REG[Real et al. 2019: Regularized Evolution Aging]
        ZIP[Cliff 1997: Minimal Intelligence ZIP Auction Benchmarks]
    end

    subgraph HYPOSTASES State Substrate σ
        C[c: Cognition State & Compiled SkillArtifact in c.m_procedural]
        W[w: World Model & SCM Ensembles]
        G[g: Goal Hierarchy & EFE Utility Gain ΔU]
        R[ρ_ext: External Environment & Physical Morphology / Resource Scarcity κ]
    end

    ALPHA --> C
    FUN --> C
    ZERO --> C
    MAP --> C
    MORPH --> R
    EMD --> W
    NOV --> C
    REG --> C
    ZIP --> R
```

---

## 3. Mathematical Formulations & Synthesized Invariants

### 3.1 Outer-Loop Quality-Diversity Optimization (MAP-Elites)
Candidates are stored in an archive grid $\mathcal{A}$ indexed by discrete behavioral bins $\mathbf{b}(\pi) \in \mathcal{B}$:
$$\mathbf{b}(\pi) = \left( \lfloor \mathcal{C}_{\text{time}}(\pi) \rfloor, \lfloor \mathcal{R}_{\text{IC}}(\pi) \rfloor, \lfloor \Delta \kappa(\pi) \rfloor \right)$$

For each bin $\mathbf{b}$, the archive retains the elite policy $\pi_{\text{elite}}$ that maximizes the multi-criteria fitness function:
$$\mathcal{F}(\pi) = \mathbb{E}_{\sigma \sim \mathcal{O}_{\text{sim}}}\left[ (1-\beta) U_{\text{pragmatic}} + \beta U_{\text{epistemic}} \right] - \lambda_{\text{IC}} \mathcal{R}_{\text{IC}}(\pi) - \lambda_{\kappa} \Delta \kappa_{\text{scarcity}}$$

### 3.2 Functional Equivalence Checking (FEC)
Before submitting a candidate AST $\pi_{\text{cand}}$ to the simulation oracle $\mathcal{O}_{\text{sim}}$, the engine evaluates $\pi_{\text{cand}}$ on a fixed set of synthetic state probes $\{ \sigma_1^{\text{probe}}, \dots, \sigma_K^{\text{probe}} \}$.
If:
$$\sum_{k=1}^K \|\pi_{\text{cand}}(\sigma_k^{\text{probe}}) - \pi_{\text{existing}}(\sigma_k^{\text{probe}})\|^2 < \epsilon_{\text{FEC}}$$
the candidate is flagged as functionally equivalent and discarded, avoiding redundant multi-agent simulation ticks.

### 3.3 Morphological Computation Co-Evolution
Morphological reservoir co-evolution optimizes the physical dynamics in $\rho_{\text{ext}}$ alongside software control laws in $c.m_{\text{procedural}}$:
$$\max_{\theta_{\text{phys}}, \theta_{\text{ctrl}}} \mathcal{J}(\theta_{\text{phys}}, \theta_{\text{ctrl}}) \quad \text{s.t.} \quad \dot{x}_{\rho} = f_{\theta_{\text{phys}}}(x_{\rho}, a_t), \quad a_t = \pi_{\theta_{\text{ctrl}}}(\sigma_t)$$
reducing active control effort by transferring computational load into physical state dynamics.
