# Wave 4 Front 12 — Scientific Discovery Loop: Ratified Master Specification

**Front**: Front 12 — Scientific Discovery Loop  
**Wave**: Wave 4 (Recursive Adaptation & Discovery Loops)  
**Target Substrate**: HYPOSTASES Engine ($\sigma = (c, w, g, \rho_{\text{ext}}$)  
**Status**: RATIFIED MASTER SPECIFICATION  
**Rule 005 Invariant**: Pure Game-Theoretic & Probabilistic Rationality (Zero Artificial Human Cognitive Biases)  
**Rule 006 Compliance**: Primacy of Data-Driven YAML Approach (`schema/scientific_discovery_config.yaml`)  
**Rule 009 Compliance**: Default Friston Expected Free Energy (EFE) Integration (`efe_mode: true`)  
**Rule 011 Compliance**: Dual Persistence for Discovery State, Hypotheses & Experiment Logs ($\theta_{\text{meta}}$, YAML Snapshots)  
**Rule 012 Compliance**: Formal Mathematical Implementation Verification in `tests/formal_math/test_scientific_discovery_formal.py`

---

## 1. Executive Summary & Core Objective

The **Scientific Discovery Loop** (Front 12) transitions the HYPOSTASES agent from passive belief update (`Observe → Infer → Act`) to a closed-loop, active scientific engine capable of autonomous model refinement, abductive hypothesis generation, optimal experiment design, Bayesian evidence accumulation, and causal discovery.

In complex, non-stationary, or partially observed environments, agents must do more than estimate latent states under a fixed world model $w$. They must continuously hypothesize alternative generative structures $H_i \in \mathcal{H}$, design optimal probing interventions $d^* \in \mathcal{D}$ that maximize Expected Information Gain (EIG), execute experiments in $\rho_{\text{ext}}$, evaluate empirical likelihoods $P(E | H_i, d^*)$, and update belief posteriors over structural causal models (SCMs).

All 8 stages of the Scientific Discovery Loop operate strictly as **computational projections over the core state tuple**:

$$\sigma = (c, w, g, \rho_{\text{ext}})$$

preserving architectural minimality, formal mathematical computability, and strict adherence to **Rule 005** (prohibiting artificial human cognitive flaws or emotional biases).

---

## 2. Scientific Cognitive Pipeline & State Substrate $\sigma = (c, w, g, \rho_{\text{ext}})$

### 2.1 The 8-Stage Closed-Loop Discovery Architecture

```mermaid
graph TD
    S1[1. Observe: Stream Empirical Observations o_t from ρ_ext] --> S2[2. Infer: Update State Beliefs P_t(σ) & Detect Residual Anomalies]
    S2 --> S3[3. Generate Hypotheses: Abductive Synthesis of Candidate Structural Models H_1, ..., H_K]
    S3 --> S4[4. Rank Explanations: Compute Prior P_0(H_k) & Complexity Penalties L(H_k)]
    S4 --> S5[5. Design Experiment: Solve Optimal Bayesian Experimental Design d* = argmax EIG(d)]
    S5 --> S6[6. Collect Evidence: Execute Probing Intervention do(d*) in ρ_ext & Record Evidence E_t]
    S6 --> S7[7. Update Hypotheses: Compute Bayesian Likelihood P(E_t | H_k, d*) & Posterior P(H_k | E_1:t)]
    S7 --> S8[8. Act: Execute Goal-Directed Policy π(a | P(H), g) under Refined World Model w]
    S8 -->|Iterative Refinement Loop| S1
```

### 2.2 Functional Mapping to State Primitives

1. **Cognition State $c$ ($c.m_{\text{semantic}}, c.m_{\text{procedural}}$)**:
   - Maintains the candidate hypothesis pool $\mathcal{H} = \{H_1, H_2, \dots, H_K\}$, structural graph representations, hypothesis priors $P(H_k)$, complexity scores $\mathcal{C}(H_k)$, and historical experiment-evidence logs.
   - Stores learned optimal experimental design heuristics $\theta_{\text{exp}} \in \theta_{\text{meta}}$.
2. **World Model $w$**:
   - Represents the agent's current ensemble of Structural Causal Models (SCMs) $\mathcal{M}_{\text{causal}} = \{f_1, \dots, f_K\}$ and DAG conditional independence relations (Front 08 integration).
3. **Goal State & Utility $g$**:
   - Integrates epistemic utility gain $\Delta U_{\text{epistemic}} = \text{EIG}(d)$ alongside pragmatic utility $U_{\text{pragmatic}}$ under Friston EFE mode (`efe_mode: true`, Rule 009):
     $$G(d) = -\text{EIG}(d) + \mathbb{E}_{P(o|d)} [D_{\text{KL}}(P(o|d) \parallel P_{\text{target}}(o))]$$
4. **External State & Environment $\rho_{\text{ext}}$**:
   - Accepts active experimental interventions $do(X = x)$ and provides multi-modal observation vectors $o_t \in \mathbb{R}^{d_{\text{obs}}}$.

---

## 3. Formal Mathematical Architecture & State Dynamics

### 3.1 Abductive Hypothesis Generation & Representation

When belief update anomaly $\mathcal{A}_t = D_{\text{KL}}(P(o_t | \sigma_t) \parallel P(o_t | w)) > \eta_{\text{anomaly}}$ is detected:
1. **Hypothesis Representation $H_k$**: Each hypothesis is represented as an explicit tuple:
   $$H_k = \langle \mathcal{G}_k, \Theta_k, P_0(H_k), \mathcal{C}(H_k) \rangle$$
   where $\mathcal{G}_k = (\mathcal{V}, \mathcal{E}_k)$ is a directed acyclic causal graph over state variables, $\Theta_k$ is the parameter set for functional structural equations $X_v = f_v(\text{Pa}(X_v), \epsilon_v)$, $P_0(H_k)$ is the normalized prior, and $\mathcal{C}(H_k)$ is the Minimum Description Length (MDL) complexity score:
   $$\mathcal{C}(H_k) = |\mathcal{E}_k| \log_2 |\mathcal{V}| + \frac{|\Theta_k|}{2} \log_2 N_{\text{obs}}$$

2. **Prior Probability & Occam's Razor Penalty**:
   $$P_0(H_k) = \frac{\exp\left(-\beta_{\text{MDL}} \mathcal{C}(H_k)\right)}{\sum_{j=1}^K \exp\left(-\beta_{\text{MDL}} \mathcal{C}(H_j)\right)}$$

### 3.2 Optimal Bayesian Experimental Design (BED) & Expected Information Gain (EIG)

To select the most informative experimental intervention $d^* \in \mathcal{D}_{\text{exp}}$ (where $d = do(X_i = x_i)$):

1. **Expected Information Gain Formulation**:
   $$\text{EIG}(d) = \mathbb{E}_{P(E | d)} \left[ D_{\text{KL}}\left( P(H | E, d) \;\parallel\; P(H) \right) \right] = H(H) - \mathbb{E}_{P(E | d)} [H(H | E, d)]$$
   where $H(H) = -\sum_{k=1}^K P(H_k) \log P(H_k)$ is the Shannon entropy over hypothesis space $\mathcal{H}$.

2. **Marginal Evidence Predictive Distribution**:
   $$P(E | d) = \sum_{k=1}^K P(E | H_k, d) P(H_k)$$

3. **Optimal Design Selection under Friston EFE (Rule 009)**:
   $$d^* = \arg\max_{d \in \mathcal{D}_{\text{exp}}} \left[ (1 - \beta_{\text{efe}}) \cdot \text{EIG}(d) + \beta_{\text{efe}} \cdot \mathbb{E}_{P(E|d)} [U_{\text{pragmatic}}(E)] \right]$$

### 3.3 Bayesian Posterior Updating & Hypothesis Convergence

Upon executing design $d^*$ and observing empirical evidence $E_t$:

1. **Likelihood Computation under Structural Causal Model**:
   $$P(E_t | H_k, d^*) = \prod_{v \in \mathcal{V}_{\text{obs}}} P_{H_k}\left(X_v = e_{v,t} \mid \text{Pa}(X_v) = e_{\text{Pa}(v), t}, do(d^*)\right)$$

2. **Bayesian Posterior Update**:
   $$P(H_k \mid E_{1:t}) = \frac{P(E_t \mid H_k, d^*) P(H_k \mid E_{1:t-1})}{\sum_{j=1}^K P(E_t \mid H_j, d^*) P(H_j \mid E_{1:t-1})}$$

3. **Asymptotic Convergence Property**:
   $$\lim_{t \to \infty} P(H^* \mid E_{1:t}) = 1 \quad \text{for } H^* = \arg\min_{H_k \in \mathcal{H}} D_{\text{KL}}(P_{\text{true}}(E) \parallel P(E \mid H_k))$$

---

## 4. Software Architecture & Module Interface

### 4.1 Directory Structure & File Organization

```
src/hypostases/scientific_discovery/
├── __init__.py
├── pipeline.py                 # Core ScientificDiscoveryPipeline manager
├── hypothesis_manager.py       # Hypothesis representation, priors, and MDL complexity
├── experimental_design.py      # BED engine, EIG calculator, and EFE design selector
├── evidence_collector.py       # Interventional execution do(d*) & evidence recording
├── bayesian_updater.py        # Bayesian posterior updating & ensemble pruning
└── schemas.py                  # Pydantic/dataclass state schemas for H_k, d*, E_t
```

### 4.2 Configuration Schema (`schema/scientific_discovery_config.yaml`)

```yaml
scientific_discovery:
  version: "1.0.0"
  enabled: true
  efe_mode: true                      # Rule 009 Friston Expected Free Energy mode
  anomaly_threshold_eta: 0.15          # KL divergence threshold for hypothesis generation
  max_hypotheses_K: 16                # Max candidate hypothesis space capacity
  mdl_complexity_penalty_beta: 0.5    # Occam's razor weight parameter
  eig_monte_carlo_samples: 1000       # MC sampling resolution for EIG calculation
  pruning_threshold_epsilon: 1e-4      # Posterior probability threshold for hypothesis elimination
  dual_persistence:
    enabled: true                     # Rule 011 dual persistence
    snapshot_interval_ticks: 10
    storage_format: "yaml"
```

---

## 5. Verification & Mathematical Testing Plan (`tests/formal_math/`)

In compliance with **Rule 012**, Front 12 includes rigorous formal mathematical verification tests in `tests/formal_math/test_scientific_discovery_formal.py`:

1. **EIG Non-Negativity Invariant**: Verifies $\text{EIG}(d) \ge 0, \forall d \in \mathcal{D}$, proving experiment design never decreases expected information.
2. **Bayesian Posterior Convergence Theorem**: Simulates $t=1..200$ discovery steps and verifies $P(H^* | E_{1:t}) \to 1.0 \pm 1e-3$ for the true data-generating SCM $H^*$.
3. **MDL Occam Invariant**: Verifies that given two empirically equivalent hypotheses $P(E|H_1) = P(E|H_2)$, the simpler graph $\mathcal{C}(H_1) < \mathcal{C}(H_2)$ maintains higher prior and posterior.
4. **Friston EFE Equivalence Invariant**: Verifies that when $U_{\text{pragmatic}} \equiv 0$, design selection $d^*$ under EFE mode reduces exactly to maximum EIG.
5. **Rule 005 Anti-Human Defect Invariant**: Asserts zero emotional heuristics, cognitive bias penalties, or irrational stopping criteria in the discovery loop.

---

## 6. Execution Status & Roadmap Alignment

- **Front ID**: Wave 4 Front 12 (`Front 12 — Scientific Discovery Loop`)
- **Dependencies**: 
  - Front 08 (Causal World Models) — SCM & Interventions
  - Front 09 (Active Sensing) — Epistemic Utility & EFE
  - Front 11 (Abductive Reasoning & Hypothesis Objects) — $H_k$ representation
- **Status**: RATIFIED MASTER SPECIFICATION
