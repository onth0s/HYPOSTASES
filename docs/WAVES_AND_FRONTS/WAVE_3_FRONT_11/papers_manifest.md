# Wave 3 Front 11 — Abductive Reasoning & Hypothesis Objects: Paper Manifest

**Front**: Front 11 — Abductive Reasoning & Hypothesis Objects  
**Wave**: Wave 3 (Social Epistemology & Swarm Mechanics)  
**Target Substrate**: HYPOSTASES Engine ($\sigma = (c, w, g, \rho_{\text{ext}})$)  
**Rule 005 Invariant**: Pure Game-Theoretic & Probabilistic Rationality (Zero Artificial Human Cognitive Biases)  
**Ingestion Directory**: [`papers/`](../../WAVE_3_FRONT_11/papers)

All 6 literature PDF files validated for `%PDF` header magic bytes and text extraction via `pypdf`. In accordance with Rule 010, PDF files are local agent-ingestion assets and remain ignored by Git.

---

## 1. Information Theory, Model Evidence & Occam's Razor

### 1. MacKay (2003)
- **Title**: *Information Theory, Inference, and Learning Algorithms* (Chapter 28: Occam's Razor and Model Comparison)
- **Author**: David J. C. MacKay
- **Venue**: Cambridge University Press, 2003
- **File**: `mackay_information_theory_inference_learning_2003.pdf` (11.7 MB, 640 pp.)
- **Key Concepts**: Model evidence / marginal likelihood $P(D \mid H_k) = \int P(D \mid w, H_k) P(w \mid H_k) dw$; Occam factor $\frac{\sigma_{w \mid D}}{\sigma_w} \ll 1$ penalizing flexible hypotheses; Bayes Factor $B_{12} = \frac{P(D \mid H_1)}{P(D \mid H_2)}$.
- **Engine Relevance**: Mathematical basis for parameter-free complexity regularization in `Hypothesis.compute_posterior()` without arbitrary heuristic penalties.

---

## 2. Model-Based Diagnosis & Minimal Conflict Sets

### 2. De Kleer & Williams (1987)
- **Title**: *Diagnosing Multiple Faults*
- **Authors**: Johan de Kleer, Brian C. Williams
- **Venue**: *Artificial Intelligence*, 32(1), 97–130 (1987)
- **File**: `dekleer_williams_diagnosing_multiple_faults_1987.pdf` (3.1 MB, 34 pp.)
- **Key Concepts**: General Diagnostic Engine (GDE); minimal conflict sets $C \subseteq A$ where component assumptions violate observations; minimal candidate hypothesis generation; entropy-based optimal probe selection $\arg\max_a \Delta H_e(a)$.
- **Engine Relevance**: Core algorithm for conflict detection in `SurpriseDetector` and candidate hypothesis generation in `HypothesisGenerator`. Tracks `contradicting_evidence` assumptions.

---

## 3. Active Inference, Curiosity & Bayesian Model Expansion

### 3. Friston, Lin, Frith, Pezzulo, Hobson, & Verbelen (2017)
- **Title**: *Active Inference, Curiosity and Insight*
- **Authors**: Karl J. Friston, Marco Lin, Christopher D. Frith, Giovanni Pezzulo, J. Allan Hobson, Sasha Ondobaka
- **Venue**: *Neural Computation*, 29(10), 2633–2683 (2017)
- **File**: `friston_active_inference_curiosity_insight_2017.pdf` (1.7 MB, 51 pp.)
- **Key Concepts**: Variational Free Energy $F(q, o) = D_{KL}(q(s) \parallel P(s)) - \mathbb{E}_q[\log P(o \mid s)] = \text{Complexity} - \text{Accuracy}$; surprise bound $-\log P(o) \le F$; abductive insight as Bayesian model reduction / expansion when $F(H_{\text{new}}) < F(H_{\text{current}})$.
- **Engine Relevance**: Surprise threshold $\tau_{\text{surprise}}$ triggering hypothesis expansion in `SurpriseDetector`, integrated directly with Expected Free Energy (EFE) action selection (Front 09 / Rule 009).

---

## 4. Structural Causality, Abductive Counterfactuals & Probability of Causation

### 4. Pearl (2000) — Chapter 7
- **Title**: *Causality: Models, Reasoning and Inference* (Chapter 7: Structure-Based Counterfactuals)
- **Author**: Judea Pearl
- **Venue**: Cambridge University Press, 2000
- **File**: `pearl_causality_ch7_counterfactuals_2000.pdf` (56 KB, 2 pp. excerpt)
- **Key Concepts**: Three-step counterfactual algorithm: (1) Abduction: update exogenous noise $P(U \mid e)$; (2) Action: apply intervention $do(X=x)$; (3) Prediction: compute target $Y_x$.
- **Engine Relevance**: Grounding for causal hypothesis generation in `HypothesisGenerator` when diagnosing root causes of environmental anomalies in $w$ (Front 08 integration).

### 5. Pearl (2000) — Chapter 9
- **Title**: *Causality: Models, Reasoning and Inference* (Chapter 9: Probability of Causation: Bounds and Identification)
- **Author**: Judea Pearl
- **Venue**: Cambridge University Press, 2000
- **File**: `pearl_causality_ch9_probability_of_causation_2000.pdf` (75 KB, 4 pp. excerpt)
- **Key Concepts**: Probability of Necessity (PN), Probability of Sufficiency (PS), Probability of Necessity and Sufficiency (PNS); bounding causal necessity given observational and experimental data.
- **Engine Relevance**: Formulates quantitative hypothesis confidence metrics evaluating whether cause $C$ was necessary/sufficient for anomaly $E$.

---

## 5. Hierarchical Bayesian Structure Learning & Cognitive Abstraction

### 6. Tenenbaum, Kemp, Griffiths, & Goodman (2011)
- **Title**: *How to Grow a Mind: Statistics, Structure, and Abstraction*
- **Authors**: Joshua B. Tenenbaum, Charles Kemp, Thomas L. Griffiths, Noah D. Goodman
- **Venue**: *Science*, 331(6022), 1279–1285 (2011)
- **File**: `tenenbaum_how_to_grow_a_mind_2011.pdf` (1.1 MB, 8 pp.)
- **Key Concepts**: Hierarchical Bayesian modeling $P(S, \theta \mid D) \propto P(D \mid S, \theta) P(\theta \mid S) P(S)$ over compositional structural spaces; top-down constraints guiding rapid hypothesis formation.
- **Engine Relevance**: Guides multi-level hypothesis generation across structural abstract spaces (Front 01) and agent intent classes (Front 06).
