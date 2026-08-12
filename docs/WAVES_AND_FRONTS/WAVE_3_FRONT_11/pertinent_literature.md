# Ingested & Pertinent Literature Inventory — Wave 3 Front 11: Abductive Reasoning & Hypothesis Objects

**Location of Ingested PDFs**: [`docs/WAVE_3_FRONT_11/papers/`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_3_FRONT_11/papers/)  
**Target Substrate**: HYPOSTASES Multi-Agent Engine ($\sigma = (c, w, g, \rho_{\text{ext}})$)  
**Rule 005 Invariant Compliance**: Pure Game-Theoretic & Probabilistic Rationality (Zero Artificial Human Cognitive Biases)

---

## Technical Synthesis across 4 Core Theoretical Pillars

### Pillar 1 — Computational & Philosophical Foundations of Abductive Inference (IBE)

#### 1. Josephson & Josephson (1994)
- **Full Reference**: Josephson, J. R., & Josephson, S. G. (Eds.). (1994). *Abductive Inference: Computation, Philosophy, Technology*. Cambridge University Press.
- **Key Formulations**:
  - Defines formal abductive assembly: Given domain observations $D$, find best explanation hypothesis set $H \subseteq \mathcal{H}$ such that $H$ accounts for $D$ better than alternatives.
  - Plausibility scoring: Score $(H) = \text{Coverage}(H \mid D) - \text{Redundancy}(H) - \text{Contradiction}(H)$.
- **HYPOSTASES Engine Relevance**: Direct architectural model for `AbductiveEngine` candidate ranking and hypothesis assembly over observation trajectories $O_{1:t}$ of $\sigma$.

#### 2. Lipton (2004)
- **Full Reference**: Lipton, P. (2004). *Inference to the Best Explanation* (2nd ed.). Routledge.
- **Key Formulations**:
  - Differentiates "Likeliest Explanation" ($P(O \mid H)$) vs. "Loveliest Explanation" (explanatory depth, elegance, symmetry).
  - Contrastive explanation framework: *"Why observation $P$ rather than default outcome $Q$?"*
- **HYPOSTASES Engine Relevance**: Provides contrastive likelihood formulation for hypothesis score comparisons in `Hypothesis.compute_posterior()`.

---

### Pillar 2 — Bayesian Model Selection & Minimum Description Length (MDL)

#### 3. Rissanen (1978) / Rissanen (1989)
- **Full Reference**: Rissanen, J. (1978). *Modeling by Shortest Data Description*. **Automatica**, 14(5), 465–471.
- **Key Formulations**:
  - Two-part Minimum Description Length (MDL) code length:
    $$L(D, H) = L(H) + L(D \mid H)$$
    where $L(H)$ is the description complexity of hypothesis $H$ in bits, and $L(D \mid H) = -\log_2 P(D \mid H)$ is the log-likelihood error.
  - Regularized Bayesian posterior:
    $$P(H_k \mid D) \propto \exp(-\lambda \cdot L(H_k)) \cdot P(D \mid H_k)$$
- **HYPOSTASES Engine Relevance**: Structural complexity penalty $C(H_k)$ in `Hypothesis` scoring to prevent overfitting and enforce Occam's Razor under Rule 005.

#### 4. MacKay (2003)
- **Full Reference**: MacKay, D. J. (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press. (Chapter 28: Occam's Razor and Evidence).
- **Key Formulations**:
  - Model Evidence / Marginal Likelihood formulation: $P(D \mid \mathcal{M}_k) = \int P(D \mid w, \mathcal{M}_k) P(w \mid \mathcal{M}_k) dw$.
  - Occam factor decomposition: $\text{Occam Factor} = \frac{\sigma_{w|D}}{\sigma_w} \ll 1$ penalizing overly flexible hypothesis spaces.
- **HYPOSTASES Engine Relevance**: Bayesian marginal likelihood evaluation for structural hypothesis selection over causal graphs $w$.

---

### Pillar 3 — Probabilistic Structure Learning & Active Inference Insight

#### 5. Tenenbaum, Kemp, Griffiths, & Goodman (2011)
- **Full Reference**: Tenenbaum, J. B., Kemp, C., Griffiths, T. L., & Goodman, N. D. (2011). *How to Grow a Mind: Statistics, Structure, and Abstraction*. **Science**, 331(6022), 1279–1285.
- **Key Formulations**:
  - Hierarchical Bayesian Modeling over abstract structural hypothesis spaces:
    $$P(\text{Structure } S, \text{Parameters } \theta \mid D) \propto P(D \mid S, \theta) P(\theta \mid S) P(S)$$
  - Rapid inference via structured compositional priors over graph models and state spaces.
- **HYPOSTASES Engine Relevance**: Formal basis for generating hypothesis objects across hierarchical world models (Front 01).

#### 6. Friston, Lin, Frith, Pezzulo, Hobson, & Verbelen (2017)
- **Full Reference**: Friston, K. J., Lin, M., Frith, C. D., Pezzulo, G., Hobson, J. A., & Verbelen, T. (2017). *Active Inference, Curiosity and Insight*. **Neural Computation**, 29(10), 2633–2683.
- **Key Formulations**:
  - Abductive model expansion driven by Free Energy minimization:
    $$F(H) = D_{KL}(q(\sigma) \parallel P(\sigma \mid H)) - \log P(D \mid H)$$
  - Surprise / Anomaly metric: $\text{Surprise}(D) = -\log P(D) \approx F(H^*)$. When surprise exceeds threshold $\tau_{\text{surprise}}$, dynamic hypothesis generation is triggered.
- **HYPOSTASES Engine Relevance**: Design basis for `SurpriseDetector` and active hypothesis expansion under EFE (Front 09 integration).

---

### Pillar 4 — Causal Abduction & Diagnostic Attribution in Multi-Agent Systems

#### 7. Pearl (2000)
- **Full Reference**: Pearl, J. (2000). *Causality: Models, Reasoning and Inference*. Cambridge University Press. (Chapter 7: Structure-Based Counterfactuals).
- **Key Formulations**:
  - Counterfactual Probability of Necessity and Sufficiency (PNS):
    $$\text{PNS} = P(Y_{x=1} = 1, Y_{x=0} = 0)$$
  - Abductive causal attribution: Infer exogenous noise variables $U=u$ given observation $E=e$, then evaluate counterfactual under intervention $do(X=x')$.
- **HYPOSTASES Engine Relevance**: Core attribution algorithm for `HypothesisGenerator` when diagnosing root causes of environmental state anomalies in $w$ (Front 08 integration).

#### 8. Halpern (2015) / Halpern & Pearl (2005)
- **Full Reference**: Halpern, J. Y. (2015). *Actual Causality*. MIT Press.
- **Key Formulations**:
  - Modified HP Definition of Actual Cause: Event $C=c$ is an actual cause of $E=e$ if (AC1) both occur, (AC2) contingency path exists under structural intervention, and (AC3) $C$ is minimal.
  - Explanatory degree metric: Score of hypothesis $H$ as an explanation for $E$ given context.
- **HYPOSTASES Engine Relevance**: Structural causal attribution engine for hypothesis candidate generation.

#### 9. Goodman & Stuhlmüller (2014)
- **Full Reference**: Goodman, N. D., & Stuhlmüller, A. (2014). *The Design and Implementation of Probabilistic Programming Languages*. Church Tutorial (ProbMods v1).
- **Key Formulations**:
  - Programmatic Bayesian abduction over executable computational models (Probabilistic Programs).
  - Conditioning observation traces: `query` / `rejection-sample` / `MCMC` over model structures.
- **HYPOSTASES Engine Relevance**: Operational design for `Hypothesis.predictive_model` as an executable Python/YAML forward sampler over $\sigma$.

#### 10. De Kleer & Williams (1987)
- **Full Reference**: De Kleer, J., & Williams, B. C. (1987). *Diagnosing Multiple Faults*. **Artificial Intelligence**, 32(1), 97–130.
- **Key Formulations**:
  - Model-Based Diagnosis (GDE - General Diagnostic Engine): Conflict candidate generation from minimal conflict sets over system component assumptions.
  - Minimal Diagnosis Hypothesis sets.
- **HYPOSTASES Engine Relevance**: Classical model-based abductive conflict generation for hypothesis pruning and contradiction checking (`Hypothesis.contradicting_evidence`).
