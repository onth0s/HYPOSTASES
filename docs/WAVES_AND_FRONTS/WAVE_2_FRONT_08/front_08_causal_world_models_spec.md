# Front 08 Master Specification — Causal World Models & Structural Causal Graphs

**Status**: RATIFIED SPECIFICATION (Ingested Literature Synthesized)  
**Wave**: Wave 2 (Structural Abstraction & Metacognitive Planning)  
**Front**: Front 08 — Causal World Models & Structural Causal Graphs  
**Target Substrate**: HYPOSTASES Engine ($\sigma = (c, w, g, \rho_{\text{ext}})$)  

---

## 1. Ingested Literature Foundation (`docs/WAVE_2_FRONT_08/papers/`)

This master specification synthesizes the theoretical mechanisms, mathematical formulations, and algorithmic structures from all 11 foundational papers ingested into `docs/WAVE_2_FRONT_08/papers/`:

| Source Paper | Key Architectural Synthesis |
|---|---|
| [`seven_tools_causal_inference_pearl_2019.pdf`](../../WAVE_2_FRONT_08/papers/seven_tools_causal_inference_pearl_2019.pdf) | 3-rung Causal Hierarchy (Association, Intervention, Counterfactuals), 7 core causal tasks, $do$-calculus rules 1-3, transportability under domain drift ($\rho_{\text{ext}}$). |
| [`causal_inference_in_statistics_primer_pearl_2016.pdf`](../../WAVE_2_FRONT_08/papers/causal_inference_in_statistics_primer_pearl_2016.pdf) | Nonparametric SCM framework, Markov product decomposition, truncated product formula ($g$-formula), frontdoor/backdoor adjustment, 3-step Abduction-Action-Prediction counterfactual cycle. |
| [`designing_optimal_interventions_zhang_bareinboim_2019.pdf`](../../WAVE_2_FRONT_08/papers/designing_optimal_interventions_zhang_bareinboim_2019.pdf) | Cost-optimal intervention target selection $X^* = \arg\min_{X \subseteq \mathcal{I}} \text{Cost}(X)$ satisfying interventional utility thresholds under resource constraints ($\rho_{\text{ext}}$). |
| [`bandits_unobserved_confounders_bareinboim_2015.pdf`](../../WAVE_2_FRONT_08/papers/bandits_unobserved_confounders_bareinboim_2015.pdf) | Regret Decision Criterion (RDC), Effect of Treatment on the Treated (ETT) $E(Y_{X=a} \mid X=x)$, Causal Thompson Sampling ($TS^C$) using observational data to seed counterfactual action selection. |
| [`causal_inference_data_fusion_bareinboim_pearl_2016.pdf`](../../WAVE_2_FRONT_08/papers/causal_inference_data_fusion_bareinboim_pearl_2016.pdf) | Data Fusion theory, selection diagrams ($S$-nodes), transportability across domains $\pi \to \pi^*$, sample selection bias recovery. |
| [`relational_structural_causal_models_ejaz_bareinboim_2026.pdf`](../../WAVE_2_FRONT_08/papers/relational_structural_causal_models_ejaz_bareinboim_2026.pdf) | Relational Structural Causal Models (RSCMs), relational schema $\mathcal{S} = \langle \mathcal{E}, \mathcal{R}, \mathcal{A} \rangle$, template structural equations $O.A \leftarrow f_{O.A}$, zero-shot cross-skeleton causal transfer. |
| [`toward_causal_representation_learning_scholkopf_2021.pdf`](../../WAVE_2_FRONT_08/papers/toward_causal_representation_learning_scholkopf_2021.pdf) | Independent Causal Mechanisms (ICM) Principle, Sparse Mechanism Shift (SMS) hypothesis, Causal representation encoder $q(\sigma)$, OOD risk min-max. |
| [`dags_with_no_tears_zheng_2018.pdf`](../../WAVE_2_FRONT_08/papers/dags_with_no_tears_zheng_2018.pdf) | NOTEARS continuous DAG structure learning, smooth equality acyclicity constraint $h(W) = \text{tr}(e^{W \circ W}) - d = 0$, augmented Lagrangian method. |
| [`learning_sparse_nonparametric_dags_zheng_2020.pdf`](../../WAVE_2_FRONT_08/papers/learning_sparse_nonparametric_dags_zheng_2020.pdf) | Nonparametric extension of NOTEARS to continuous state vectors using Sobolev norm penalties and MLP mechanism parameterization. |
| [`causation_prediction_search_spirtes_2000.pdf`](../../WAVE_2_FRONT_08/papers/causation_prediction_search_spirtes_2000.pdf) | Constraint-based causal discovery, the PC Algorithm, conditional independence testing, V-structure orientation, Meek rules. |
| [`elements_of_causal_inference_peters_2017.pdf`](../../WAVE_2_FRONT_08/papers/elements_of_causal_inference_peters_2017.pdf) | Additive Noise Models (ANM), structural equation models, bivariate causal directionality tests via independence of residuals. |

---

## 2. Integrated Theoretical Architecture & Mathematical Formalism

```
Rung 3: Counterfactuals ──► Abduction-Action-Prediction (Y_{X=x'} | X=x, Y=y) & TS^C ETT Evaluation
   ▲
Rung 2: Interventions   ──► do-Calculus Engine (Rules 1-3) & Cost-Optimal Target Planner X*
   ▲
Rung 1: Associations    ──► Markov Product Decomposition & SCM Functional Assignment V_i = f_i(PA_i, U_i)
   ▲
Structural Discovery    ──► NOTEARS Smooth Acyclicity h(W) = tr(exp(W ∘ W)) - d = 0 & PC Algorithm
   ▲
Relational Substrate    ──► RSCM Template Equations O.A = f_{O.A}(PA, U, PA^r, U^r) & Selection Diagrams S
```

### 2.1 Pearl's Three-Rung Causal Hierarchy in $w$
Following Pearl (2016, 2019) and Bareinboim et al. (2015, 2016):
- **Rung 1: Association (Observational)**:
  $$P(Y=y \mid X=x) = \frac{P(X=x, Y=y)}{P(X=x)}$$
- **Rung 2: Intervention ($do$-calculus)**:
  $$P(Y=y \mid \text{do}(X=x)) = \sum_{\mathbf{z}} P(Y=y \mid X=x, \mathbf{Z}=\mathbf{z}) \, P(\mathbf{Z}=\mathbf{z})$$
- **Rung 3: Counterfactuals**:
  $$P(Y_{X=x'} = y \mid X=x, Y=y') = P^{M_{X=x'}}(Y(u) = y \mid U \sim P(U \mid X=x, Y=y'))$$

### 2.2 Symbolic $do$-Calculus Engine
Following Pearl (2016, 2019) and Bareinboim & Pearl (2016), the engine enforces the 3 rules of $do$-calculus over directed acyclic graphs $G$:
1. **Rule 1 (Insertion/deletion of observations)**:
   $$P(y \mid \text{do}(x), z, w) = P(y \mid \text{do}(x), w) \quad \text{if } (Y \perp\!\!\!\perp Z \mid X, W)_{G_{\overline{X}}}$$
2. **Rule 2 (Action/observation exchange)**:
   $$P(y \mid \text{do}(x), \text{do}(z), w) = P(y \mid \text{do}(x), z, w) \quad \text{if } (Y \perp\!\!\!\perp Z \mid X, W)_{G_{\overline{X},\underline{Z}}}$$
3. **Rule 3 (Insertion/deletion of actions)**:
   $$P(y \mid \text{do}(x), \text{do}(z), w) = P(y \mid \text{do}(x), w) \quad \text{if } (Y \perp\!\!\!\perp Z \mid X, W)_{G_{\overline{X},\overline{Z(W)}}}$$

### 2.3 Structural Causal Model (SCM) & Counterfactual Regret Engine
Following Pearl et al. (2016) and Bareinboim et al. (2015):
- **SCM Definition**: $M = \langle \mathbf{U}, \mathbf{V}, \mathbf{F}, P(\mathbf{U}) \rangle$, where $V_i = f_i(\text{PA}_i, U_i)$ and $U_i$ are independent exogenous noise terms.
- **Three-Step Counterfactual Algorithm**:
  1. **Abduction**: Update exogenous noise prior $P(\mathbf{U})$ given observation $e = (\mathbf{x}, \mathbf{y})$ to obtain posterior $P(\mathbf{U} \mid \mathbf{e})$.
  2. **Action**: Surgerize $M$ by replacing structural equation for $X$ with $X = x'$, forming interventional sub-model $M_{x'}$.
  3. **Prediction**: Compute outcome distribution for target variable $Y$ in $M_{x'}$ using updated noise posterior $P(\mathbf{U} \mid \mathbf{e})$.
- **Effect of Treatment on the Treated (ETT)**:
  $$\text{ETT} = E(Y_{X=a} \mid X=x)$$
  used in **Causal Thompson Sampling ($TS^C$)** to prevent linear cumulative regret under unobserved confounders.

### 2.4 Cost-Optimal Intervention Planner
Following Zhang & Bareinboim (2019):
- Computes minimum-cost intervention set $X^* \subseteq \mathcal{I} \subseteq V$ for goal $g$:
  $$X^* = \arg\min_{X \subseteq \mathcal{I}} \sum_{X_i \in X} c_i(x_i) \quad \text{s.t.} \quad \mathcal{U}(P(Y \mid \text{do}(X=x))) \ge \tau$$
  ensuring action paths respect resource constraints $\rho_{\text{ext}}$.

### 2.5 Relational Structural Causal Models (RSCMs) & Transportability
Following Ejaz & Bareinboim (2026) and Bareinboim & Pearl (2016):
- **Relational Schema**: $\mathcal{S} = \langle \mathcal{E}, \mathcal{R}, \mathcal{A} \rangle$.
- **Template Mechanism**: $O.A \leftarrow f_{O.A}(\mathbf{Pa}_{O.A}, \mathbf{U}_{O.A}, \mathbf{Pa}^r_{O.A}, \mathbf{U}^r_{O.A})$ using permutation-invariant aggregators (attention/sum).
- **Zero-Shot Transfer**: Enables causal reasoning across dynamic team sizes, unseen object counts, and target environments $\pi^*$ via Selection Diagrams ($S$-nodes).

### 2.6 Continuous Structure Discovery (NOTEARS)
Following Zheng et al. (2018, 2020) and Spirtes et al. (2000):
- **NOTEARS Acyclicity Constraint**:
  $$h(W) = \text{tr}(e^{W \circ W}) - d = 0$$
- **Gradient**:
  $$\nabla h(W) = (e^{W \circ W})^T \circ 2W$$
- **Augmented Lagrangian Formulation**:
  $$\min_{W \in \mathbb{R}^{d \times d}} F(W) + \frac{\rho}{2} |h(W)|^2 + \alpha h(W)$$
  solved via L-BFGS / Proximal Quasi-Newton to learn SCM DAG graphs directly from state interaction logs.

---

## 3. Core Module Specifications (`src/hypostases/causal/`)

1. [`causal_types.py`](../../../src/hypostases/causal/causal_types.py): Dataclasses for `CausalNode`, `CausalEdge`, `StructuralEquation`, `Intervention`, `CounterfactualQuery`, and `RSCMSchema`.
2. [`structural_causal_model.py`](../../../src/hypostases/causal/structural_causal_model.py): Core `StructuralCausalModel` class supporting DAG topological sort, d-separation, structural equation evaluation, graph surgery, and 3-step counterfactual reasoning (`abduce_act_predict`).
3. [`do_calculus_engine.py`](../../../src/hypostases/causal/do_calculus_engine.py): Symbolic graph separation engine checking $do$-calculus Rules 1-3, backdoor adjustment, frontdoor adjustment, and transportability selection diagrams ($S$-nodes).
4. [`causal_discovery.py`](../../../src/hypostases/causal/causal_discovery.py): `CausalDiscoveryEngine` implementing NOTEARS smooth continuous optimization ($h(W) = \text{tr}(e^{W \circ W}) - d = 0$) and constraint-based PC algorithm.
5. [`causal_policy_evaluator.py`](../../../src/hypostases/causal/causal_policy_evaluator.py): `CausalPolicyEvaluator` and `CostOptimalPlanner` evaluating interventional distributions $P(g \mid \text{do}(a))$, computing ETT counterfactual regret, and finding minimal cost intervention targets $X^*$.
6. [`rscm_engine.py`](../../../src/hypostases/causal/rscm_engine.py): `RelationalSCMEngine` instantiating template structural equations $O.A \leftarrow f_{O.A}$ over dynamic multi-agent schemas $\mathcal{S} = \langle \mathcal{E}, \mathcal{R}, \mathcal{A} \rangle$.

---

## 4. Invariant & Safety Guarantees (Rule 005 & Rule 006/007)

- **Rule 005**: All causal graph operations, $do$-calculus transformations, NOTEARS optimization steps, and counterfactual evaluations represent formal mathematical state projections over $\sigma = (c, w, g, \rho_{\text{ext}})$. Zero artificial human cognitive defects or emotional irrationality.
- **Rule 006**: Default SCM graph structures, exogenous noise distributions $P(U)$, structural equation parameters, and NOTEARS discovery hyperparameters reside in declarative YAML configuration files (`schema/causal_world_model_config.yaml`).
- **Rule 007**: YAML serialization performance of SCM graph structures and learned mechanisms is verified during simulation execution.
