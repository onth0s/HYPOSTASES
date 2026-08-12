# Front 11 Master Specification — Abductive Reasoning & Hypothesis Objects

**Status**: RATIFIED MASTER SPECIFICATION (All 6 Literature PDFs Ingested & Synthesized)  
**Wave**: Wave 3 (Social Epistemology & Swarm Mechanics)  
**Front**: Front 11 — Abductive Reasoning & Hypothesis Objects  
**Target Substrate**: HYPOSTASES Engine ($\sigma = (c, w, g, \rho_{\text{ext}})$)  
**Rule 005 Compliance**: Pure Game-Theoretic & Probabilistic Rationality (Zero Artificial Human Cognitive Biases)

---

## 1. Ingested Literature Foundation (`docs/WAVE_3_FRONT_11/papers/`)

This master specification synthesizes the theoretical mechanisms, mathematical formulations, and computational structures from **all 6 foundational PDF papers** ingested into `docs/WAVE_3_FRONT_11/papers/`:

| Ingested PDF File | Source & Core Theoretical Synthesis |
|---|---|
| [`mackay_information_theory_inference_learning_2003.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_3_FRONT_11/papers/mackay_information_theory_inference_learning_2003.pdf) | **MacKay (2003) — Occam's Razor & Model Evidence**: Model marginal likelihood $P(D \mid H_k) = \int P(D \mid w, H_k) P(w \mid H_k) dw$; Occam factor $\frac{\sigma_{w \mid D}}{\sigma_w} \ll 1$ penalizing overly complex hypothesis spaces; Bayes Factor evidence ratios $B_{12} = \frac{P(D \mid H_1)}{P(D \mid H_2)}$. |
| [`dekleer_williams_diagnosing_multiple_faults_1987.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_3_FRONT_11/papers/dekleer_williams_diagnosing_multiple_faults_1987.pdf) | **De Kleer & Williams (1987) — Model-Based Diagnosis (GDE)**: Minimal conflict sets $C \subseteq A$ violating observation constraints; minimal candidate hypothesis generation; entropy-driven optimal measurement selection $\arg\max_a \Delta H_e(a)$. |
| [`friston_active_inference_curiosity_insight_2017.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_3_FRONT_11/papers/friston_active_inference_curiosity_insight_2017.pdf) | **Friston et al. (2017) — Active Inference & Insight**: Variational Free Energy $F(q, o) = D_{KL}(q(s) \parallel P(s)) - \mathbb{E}_q[\log P(o \mid s)]$; surprise metric $-\log P(o) \le F$; abductive insight as Bayesian model reduction/expansion when $F(H_{\text{new}}) < F(H_{\text{current}})$. |
| [`pearl_causality_ch7_counterfactuals_2000.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_3_FRONT_11/papers/pearl_causality_ch7_counterfactuals_2000.pdf) | **Pearl (2000) Ch. 7 — Abductive Counterfactuals**: 3-step counterfactual algorithm: (1) Abduction: infer exogenous state $P(U \mid e)$; (2) Action: apply intervention $do(X=x)$; (3) Prediction: evaluate counterfactual outcome $P(Y_x \mid e)$. |
| [`pearl_causality_ch9_probability_of_causation_2000.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_3_FRONT_11/papers/pearl_causality_ch9_probability_of_causation_2000.pdf) | **Pearl (2000) Ch. 9 — Probability of Causation**: Probability of Necessity (PN), Probability of Sufficiency (PS), Probability of Necessity and Sufficiency (PNS); bounding causal attribution given observational and experimental evidence. |
| [`tenenbaum_how_to_grow_a_mind_2011.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_3_FRONT_11/papers/tenenbaum_how_to_grow_a_mind_2011.pdf) | **Tenenbaum et al. (2011) — Hierarchical Structure Learning**: Hierarchical Bayesian Modeling $P(S, \theta \mid D) \propto P(D \mid S, \theta) P(\theta \mid S) P(S)$ over compositional structural hypothesis spaces. |

---

## 2. Integrated Theoretical Architecture & Mathematical Formalism

```
            Observation Trajectory O_{1:t} over σ = (c, w, g, ρ_ext)
                                 │
                                 ▼
       ┌───────────────────────────────────────────────────┐
       │ 1. SurpriseDetector (Friston 2017 Free Energy)    │
       │    - Computes F(q, o) = Complexity - Accuracy     │
       │    - Surprise Check: F(q, o) > τ_surprise          │
       └─────────────────────────┬─────────────────────────┘
                                 │ Anomaly Confirmed
                                 ▼
       ┌───────────────────────────────────────────────────┐
       │ 2. HypothesisGenerator (De Kleer + Pearl + Tenenbaum) │
       │    - De Kleer minimal conflict sets C ⊆ A         │
       │    - Pearl 3-step SCM abduction P(U | e)           │
       │    - Tenenbaum compositional priors P(S, θ)       │
       └─────────────────────────┬─────────────────────────┘
                                 │ Candidate Ensemble {H_k}
                                 ▼
       ┌───────────────────────────────────────────────────┐
       │ 3. Hypothesis Evaluator (MacKay 2003 Occam Factor)│
       │    - Model Evidence P(O | H_k)                    │
       │    - Occam Factor = (Δw_post / Δw_prior) << 1     │
       │    - Posterior P(H_k | O) ∝ exp(-λ C(H_k)) P(O|H_k)│
       └─────────────────────────┬─────────────────────────┘
                                 │ Posterior Score & Evidence
                                 ▼
       ┌───────────────────────────────────────────────────┐
       │ 4. AbductiveEngine Management                     │
       │    - Rank {H_k} & Prune low-posterior (P < ε)     │
       │    - Top Hypothesis -> Consolidate to Semantic Mem │
       │    - Discriminating Hypotheses -> Target Probe (Front 09) │
       └───────────────────────────────────────────────────┘
```

### 2.1 Anomaly Detection & Surprise Sensor (`SurpriseDetector`)
Following Friston et al. (2017), observation trajectories $O_{1:t}$ are continuously evaluated against active world model predictions $w$. An anomaly is registered when Variational Free Energy $F(q, o)$ exceeds the configured threshold $\tau_{\text{surprise}}$:

$$F(q, o) = D_{KL}(q(\sigma) \parallel P(\sigma)) - \mathbb{E}_{q}[\log P(O_{1:t} \mid \sigma)] > \tau_{\text{surprise}}$$

### 2.2 Abductive Hypothesis Generation (`HypothesisGenerator`)
Following De Kleer & Williams (1987), Pearl (2000), and Tenenbaum et al. (2011), when an anomaly is detected, candidate hypothesis objects are generated across three distinct modalities:

1. **Environment Anomaly Hypotheses** ($H_{\text{env}}$): Postulates hidden shifts in resource replenishment $\kappa$ or unobserved environmental parameters.
2. **Peer Intent / Alignment Hypotheses** ($H_{\text{peer}}$): Postulates latent preference shifts $g_j = u_j$ or hidden parameter shifts in peer agent $j$.
3. **Causal Graph Mutation Hypotheses** ($H_{\text{causal}}$): Mutates structural causal edges in $w$ via Pearl SCM abduction $P(U \mid e)$ and De Kleer minimal conflict sets.

### 2.3 Hypothesis Object Structure & Occam Marginal Likelihood (`Hypothesis`)
Following MacKay (2003), each hypothesis object explicitly maintains Occam complexity regularization:

```python
@dataclass
class Hypothesis:
    identifier: str
    description: str
    category: HypothesisCategory
    assumptions: dict[str, Any]
    predictive_model: Callable[[AgentState, int], np.ndarray]
    prior: float
    likelihood: float
    posterior: float
    complexity: float  # MDL code length L(H_k) or parameter count
    supporting_evidence: list[int]  # Observation timestamps
    contradicting_evidence: list[int]
    confidence: float
```

The log posterior score is computed as:

$$\ln P(H_k \mid O_{1:t}) = \ln P(O_{1:t} \mid H_k) - \lambda_{\text{MDL}} \cdot C(H_k) + \ln P(H_k) - \ln Z$$

where $C(H_k)$ represents the description complexity (MacKay Occam factor penalty) and $P(O_{1:t} \mid H_k)$ is the empirical predictive likelihood.

---

## 3. Data-Driven Configuration (`schema/abductive_reasoning_config.yaml`)

```yaml
abductive_reasoning:
  enabled: true
  max_hypothesis_pool_size: 16
  surprise_threshold: 1.5  # Tau_surprise (Friston Free Energy bound)
  complexity_penalty_lambda: 0.2  # Lambda_MDL (MacKay Occam factor weight)
  pruning_threshold: 0.05  # Epsilon_prune
  evidence_window_length: 20  # W_evidence (Timestamps)
  consolidation_confidence_threshold: 0.85
  generation_modalities:
    environment_anomalies: true
    peer_intent_anomalies: true
    causal_structural_mutations: true
```

---

## 4. Module Map & Implementation Files

- **Master Specification**: [`docs/WAVE_3_FRONT_11/front_11_abductive_reasoning_hypothesis_objects_spec.md`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_3_FRONT_11/front_11_abductive_reasoning_hypothesis_objects_spec.md)
- **Paper Manifest**: [`docs/WAVE_3_FRONT_11/papers_manifest.md`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_3_FRONT_11/papers_manifest.md)
- **Literature Summary**: [`docs/WAVE_3_FRONT_11/pertinent_literature.md`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_3_FRONT_11/pertinent_literature.md)
- **YAML Config**: [`schema/abductive_reasoning_config.yaml`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/schema/abductive_reasoning_config.yaml)
- **Core Engine Module**: `src/hypostases/abduction/`
  - `types.py`
  - `hypothesis.py`
  - `anomaly_detector.py`
  - `hypothesis_generator.py`
  - `abductive_engine.py`
  - `__init__.py`
- **Pytest Suite**: [`tests/test_abductive_reasoning.py`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/tests/test_abductive_reasoning.py)
