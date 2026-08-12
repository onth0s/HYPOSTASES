# Front 09 Master Specification — Active Information Gathering & Active Perception

**Status**: RATIFIED SPECIFICATION (Ingested Literature Synthesized)  
**Wave**: Wave 1 (Single-Agent Foundations: Memory, Lookahead & Active Sensing)  
**Front**: Front 09 — Active Information Gathering & Active Perception  
**Target Substrate**: HYPOSTASES Engine ($\sigma = (c, w, g, \rho_{\text{ext}})$)  

---

## 1. Ingested Literature Foundation (`docs/WAVE_1_FRONT_09/papers/`)

This master specification synthesizes the theoretical mechanisms, mathematical formulations, and algorithmic structures from all 9 foundational papers ingested into `docs/WAVE_1_FRONT_09/papers/`:

| Source Paper | Key Architectural Synthesis |
|---|---|
| [`dodig_crnkovic_cognition_morphological_2022.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_1_FRONT_09/papers/dodig_crnkovic_cognition_morphological_2022.pdf) | Info-computational paradigm: physical/abstract state transitions represent computation while environment structures represent information. Theoretical foundation for Variational Free Energy $F = D_{KL}(q(w) \parallel p(w)) - \mathbb{E}_q[\ln p(o \mid w)]$. |
| [`friston_active_inference_curiosity_insight_2017.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_1_FRONT_09/papers/friston_active_inference_curiosity_insight_2017.pdf) | Active inference under expected free energy (EFE) minimization: resolving ambiguity vs. pragmatic risk minimization, epistemic affordances, and posterior belief precision updates. |
| [`bajcsy_active_perception_1988.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_1_FRONT_09/papers/bajcsy_active_perception_1988.pdf) | Active Perception framework: modeling observation as an explicit control process with goal-directed exploratory actions (`INSPECT`, `PROBE`, `MONITOR`, `SPY`) rather than passive sensory ingestion. |
| [`aloimonos_active_vision_1988.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_1_FRONT_09/papers/aloimonos_active_vision_1988.pdf) | Control-theoretic active vision: converting ill-posed passive inference problems into well-posed active optimization problems via deliberate sensor parameter adjustments. |
| [`lindley_measure_information_experiment_1956.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_1_FRONT_09/papers/lindley_measure_information_experiment_1956.pdf) | Bayesian experimental design: expected Kullback-Leibler information gain $\Delta H(w) = \frac{1}{2} \ln\left(1 + \frac{\sigma^2_{\text{prior}}}{\sigma^2_{\text{obs}}}\right)$, decision-theoretic observation utility. |
| [`mackay_information_based_objective_1992.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_1_FRONT_09/papers/mackay_information_based_objective_1992.pdf) | Information-based objective functions for active data selection: variance minimization over model parameters, optimal query selection under resource constraints. |
| [`schmidhuber_formal_theory_creativity_2010.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_1_FRONT_09/papers/schmidhuber_formal_theory_creativity_2010.pdf) | Formal theory of artificial curiosity and intrinsic motivation: compression progress drives, mathematical epistemic reward trading material utility for variance reduction. |
| [`oudeyer_intrinsic_motivation_2007.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_1_FRONT_09/papers/oudeyer_intrinsic_motivation_2007.pdf) | Intelligent Adaptive Curiosity (IAC): autonomous mental development, active variance reduction, dynamically balancing exploration vs. exploitation parameters. |
| [`settles_active_learning_survey_2009.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_1_FRONT_09/papers/settles_active_learning_survey_2009.pdf) | Comprehensive active learning taxonomy: uncertainty sampling, query-by-committee, expected model change, structured epistemic action modalities (`QUERY`, `EXPERIMENT`, `VERIFY`). |

---

## 2. Integrated Theoretical Architecture & Mathematical Formalism

```
+-----------------------------------------------------------------------------------+
|                           Info-Computational Active Inference                      |
|      Variational Free Energy F = D_KL(q(w) || p(w)) - E_q[ln p(o|w)] (Dodig-Crnkovic) |
+-----------------------------------------------------------------------------------+
                                         ▲
                                         │
+-----------------------------------------------------------------------------------+
|                        Bayesian Epistemic Utility Mechanics                       |
|           U_total = (1 - β) U_pragmatic + β U_epistemic  (Schmidhuber, Friston)    |
|        Information Gain ΔH = H(σ²_prior) - H(σ²_post) (Lindley, MacKay, Oudeyer)  |
+-----------------------------------------------------------------------------------+
                                         ▲
                                         │
+-----------------------------------------------------------------------------------+
|                     Active Perception & Sensing Affordances                       |
|     Action Types: INSPECT, PROBE, MONITOR, QUERY, EXPERIMENT, VERIFY, OBSERVE, SPY    |
|                    Control-Theoretic Active Sensing (Bajcsy, Aloimonos)           |
+-----------------------------------------------------------------------------------+
                                         ▲
                                         │
+-----------------------------------------------------------------------------------+
|                         HYPOSTASES Primitive State Substrate                      |
|                  σ = (c, w, g, ρ_ext) with Rule 005 Game-Theoretic Rationality       |
+-----------------------------------------------------------------------------------+
```

### 2.1 Info-Computational Active Inference & Variational Free Energy
Following **Dodig-Crnkovic (2022)** and **Friston et al. (2017)**:
- **Info-Computational Paradigm**: Physical and mental state transitions compute updates over environmental information structures.
- **Variational Free Energy**:
  $$F(q, o) = D_{\text{KL}}\big(q(w) \parallel p(w)\big) - \mathbb{E}_{q(w)}[\ln p(o \mid w)]$$
- **Free Energy Decomposition**:
  - *Complexity Penalty*: $D_{\text{KL}}(q(w) \parallel p(w))$ penalizes deviation from prior belief $p(w)$.
  - *Accuracy Mismatch*: $-\mathbb{E}_{q(w)}[\ln p(o \mid w)]$ measures negative log-likelihood of observation $o$.

### 2.2 Active Sensing Control Loops & Action Affordances
Following **Bajcsy (1988)**, **Aloimonos et al. (1988)**, and **Settles (2009)**:
- Observation is an active execution branch in the environment feedback loop (`execute_epistemic_action`).
- **Epistemic Action Modalities (`EPISTEMIC_ACTION_TYPES`)**:
  1. `INSPECT`: Directed visual/spatial observation of a target entity or canvas region.
  2. `PROBE`: Active stimulus injection to measure environment feedback variance.
  3. `MONITOR`: Continuous low-cost sensory tracking over a designated region.
  4. `QUERY`: Directed epistemic query emitted to peer agents or institutional entities.
  5. `EXPERIMENT`: Systematic multi-step empirical test of competing hypotheses.
  6. `VERIFY`: Epistemic confirmation check against ground truth.
  7. `OBSERVE`: Passive baseline environmental observation.
  8. `SPY`: Passive covert information gathering on peer reserves and states.

### 2.3 Bayesian Experimental Design & Information Gain
Following **Lindley (1956)** and **MacKay (1992)**:
- **Continuous Gaussian Entropy**: For belief distribution $\mathcal{N}(\mu, \sigma^2)$:
  $$H(S) = \frac{1}{2} \ln(2\pi e \sigma^2)$$
- **Precision Update**:
  $$\frac{1}{\sigma^2_{\text{post}}} = \frac{1}{\sigma^2_{\text{prior}}} + \frac{1}{\sigma^2_{\text{obs}}}$$
- **Expected Information Gain**:
  $$\Delta H(w) = H(\sigma^2_{\text{prior}}) - H(\sigma^2_{\text{post}}) = \frac{1}{2} \ln\left(1 + \frac{\sigma^2_{\text{prior}}}{\sigma^2_{\text{obs}}}\right)$$

### 2.4 Intrinsic Motivation & Epistemic Utility Integration
Following **Schmidhuber (2010)**, **Oudeyer et al. (2007)**, and **Friston et al. (2017)**:
- **Combined Pragmatic-Epistemic Objective**:
  $$U_{\text{total}}(a, \sigma) = (1 - \beta) U_{\text{pragmatic}}(a, \sigma) + \beta U_{\text{epistemic}}(a, \sigma)$$
- **Resource Allocation Costs**: Epistemic actions trade physical reserve ($c_{\text{reserve}}$) and time budget ($\rho_{\text{ext.time}}$) for variance reduction ($\Delta \sigma^2$).
- **Rule 005 Compliance**: Epistemic actions operate under computable mathematical state dynamics without introducing artificial human irrationality hacks.

---

## 3. Engine Implementation Architecture (`src/hypostases/`)

The theoretical principles are fully implemented in the engine codebase:

1. [`active_perception.py`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/src/hypostases/active_perception.py): Core dynamics routine `execute_epistemic_action()` evaluating state costs ($\Delta c, \Delta \rho_{\text{ext}}$), Bayesian precision updates ($\Delta \mu, \Delta \sigma^2$), and peer belief updates.
2. [`epistemic_utility.py`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/src/hypostases/epistemic_utility.py): Functions for continuous Shannon entropy (`compute_shannon_entropy`), Gaussian KL-divergence (`compute_kl_divergence_gaussian`), Variational Free Energy (`compute_variational_free_energy`), Expected Information Gain (`compute_expected_information_gain`), and combined pragmatic-epistemic utility (`compute_epistemic_utility`).
3. [`active_sensing_config.yaml`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/schema/active_sensing_config.yaml): Rule 006 data-driven specification defining base reserve/time costs, target properties, and observation variances for all 8 epistemic action types.
4. [`test_active_perception.py`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/tests/test_active_perception.py): 18 comprehensive unit tests verifying Bayesian precision updates, free energy formulas, state-dependent resilience scaling, and utility trade-offs.
