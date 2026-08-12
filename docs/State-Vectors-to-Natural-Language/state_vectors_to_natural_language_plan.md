# State Vectors to Natural Language Text Generation — Working Architecture Specification & Plan (v4.2 - RATIFIED)

**Spec Ref**: Wave 5 Front 14 (`docs/WAVES_AND_FRONTS/WAVE_5_FRONT_14/front_14_natural_language_symbolic_compression_spec.md`)  
**Target Substrate**: HYPOSTASES Engine v0.4.0 ($\sigma = (c, w, g, \rho_{\text{ext}})$)  
**Status**: RATIFIED IMPLEMENTATION-READY SPECIFICATION & PLAN  
**Rule 005 Compliance**: Regularized Weight Calibration & Cold-Start Seed Corpus Disclosure  
**Rule 006 Compliance**: Primacy of Data-Driven YAML Configuration (`schema/nlp_decoder_config.yaml`)  
**Rule 009 Compliance**: Default Friston Expected Free Energy (EFE) Active Sensing Integration (`efe_mode: true`)  
**Rule 011 Compliance**: Dual Persistence for Lexicon Mappers, PCFG Rules, Meta-Parameters, and Snapshots  
**Rule 012 Compliance**: Mandatory Formal Mathematical Implementation Verification (`tests/formal_math/test_nlp_generation_formal.py`)

---

## 1. Architectural Audit & Final Refinements (Pass 5 - Ratification)

Following final mathematical audit, the weight optimization identifiability gap has been closed:

1. **Non-Degenerate Regularized Weight Calibration**:  
   Prevented weight collapse ($w_i \to 0$ for high-variance components like $d_w$) by enforcing a minimum weight floor $w_i \ge w_{\min} = 0.10$ and adding an entropy penalty $-\lambda_H H(\mathbf{w})$ to the calibration objective.
2. **Held-Out Split Verification (Rule 012 Compliance)**:  
   Calibration of $\mathbf{w}$ and $\boldsymbol{\tau}$ is performed on training split $\mathcal{D}_{\text{train}}$, while final Rule 012 round-trip verification metrics are reported exclusively on an independent **held-out evaluation split $\mathcal{D}_{\text{eval}}$**.
3. **YAML Constant Disclosure**:  
   Explicitly declared $c_{\text{dummy}} = 1.0$ (insertion/deletion penalty for Hungarian GED padding) as a YAML-configured parameter in `schema/nlp_decoder_config.yaml` (Rule 006).

---

## 2. Granular 5-Step Architecture

```
Continuous State σ = (c, w, g, ρ_ext)
            │
            ▼
┌─────────────────────────────────────────┐
│ Step 1: VQ Corpus-Derived Lexicon       │  <-- Stage 0 Seed -> Stage 1+ Data-Derived Lexicon
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Step 2: Priority Waterfall Decoder     │  <-- Mode A (PCFG) -> Mode C (MDL) -> Mode B (SLM) -> Fallback
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Step 3: Multi-Criteria Causal Sanitizer │  <-- SCM Audit + Trust-Discounted g.u Utility Promotion
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Step 4: Calibrated Fano Uncertainty     │  <-- Regularized Fano token budget policy
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Step 5: Held-Out Offline Verification   │  <-- Held-Out Split D_eval Metric Loss & Fuzzing Corpus
└─────────────────────────────────────────┘
```

---

### Step 1: Vocabulary Bootstrapping & Lexicon Mapping (`src/hypostases/nlp/lexicon_mapper.py`)

* **Cold-Start & Bootstrapping Procedure**:
  - **Stage 0 (Seed Corpus)**: Bootstrapped from a domain-general frequency lexicon $\mathcal{V}_{\text{seed}}$.
  - **Stage 1+ (Outcome-Correlated Clustering)**: Codebook slots ($k \in K_{\text{codebook}}$) map to semantic tokens via outcome correlation over trace logs:
    $$\text{LexiconToken}(k) = \arg\max_{t \in \mathcal{V}} P(\Delta \text{Outcome} \mid \sigma \in \text{Cluster}(k))$$
* **Key Classes**:
  - `DataDerivedLexiconMapper`: Loads VQ cluster-outcome correlation tables from `schema/nlp_decoder_config.yaml`.
  - `ConceptCompositionEngine`: Assembles discrete codebook tokens into compositional semantic feature tuples without hardcoded human label hacks.

---

### Step 2: Multi-Mode Decoder Engine & Priority Waterfall Arbitration (`src/hypostases/nlp/generative_decoder.py`)

* **Decoder Modes**:
  - **Mode A: Compositional PCFG Synthesizer (Zero-Latency / Deterministic)**
  - **Mode B: Local SLM Engine (Phi-3 / Llama-3 / Qwen / Ollama)**
  - **Mode C: Minimum Description Length (MDL) Neural Autoencoder**

* **Priority Waterfall Arbitration Policy**:
  $$\text{Mode}(\sigma) = \begin{cases} 
  \text{Mode A (PCFG)}, & \text{if } \text{is\_structured}(\sigma) \land \text{latency\_bound\_ms} < 5.0 \text{ (Priority 1: Real-Time Governance)} \\ 
  \text{Mode C (MDL)}, & \text{else if } B_{\text{channel}} < \tau_{\text{bandwidth}} \text{ (Priority 2: Bandwidth Constraint)} \\ 
  \text{Mode B (SLM)}, & \text{else if } H(w) > \tau_{\text{uncert}} \land \text{compute\_available} \text{ (Priority 3: Complex Negotiation)} \\ 
  \text{Mode A (PCFG)}, & \text{otherwise (Fallback Default: Deterministic Template)}
  \end{cases}$$

---

### Step 3: Adversarial Threat Model, Multi-Criteria Sanitizer, & Belief Updater (`src/hypostases/nlp/text_belief_updater.py`)

* **Adversarial Threat Model**:
  Peer agents may attempt **Prompt Injection / Deceptive Text Manipulation** to inject adversarial goals into $g$ or corrupt $w$.
* **Multi-Criteria Promotion Acceptance Rule**:
  $$\text{Promote}(\sigma_{\text{sandbox}} \to \sigma) \iff \text{Audit}_{\text{format}} \land \left( \text{Audit}_{\text{SCM}}(w, w_{\text{sandbox}}) \le \tau_{\text{causal}} \right) \land \left( \text{Trust}(\text{peer}) \cdot \Delta U_{\text{expected}} \ge \tau_{\text{promotion}} \right)$$
  where expected utility gain uses the receiving agent's endogenous Goal Hierarchy $g.u$:
  $$\Delta U_{\text{expected}} = \mathbb{E}_{\sigma_{\text{sandbox}}}[g.u(\sigma_{\text{sandbox}})] - \mathbb{E}_{\sigma}[g.u(\sigma)]$$

---

### Step 4: Regularized Calibrated Fano Uncertainty Router (`src/hypostases/nlp/clsr_text_router.py`)

* **Theoretical Bound**: Grounded in Fano's Inequality and Shannon Rate-Distortion Theory:
  $$\mathbb{E}[N_{\text{tokens}}] \ge \frac{H(Y \mid X) - h_2(P_{\text{error}})}{\kappa_{\text{active}}}$$
* **Non-Degenerate Regularized Calibration Objective**:
  On training split $\mathcal{D}_{\text{train}}$, optimize thresholds $(\tau_1, \tau_2)$ and component weights $\mathbf{w} = (w_c, w_w, w_g, w_\rho)$ by solving:
  $$\min_{\mathbf{w}, \tau_1, \tau_2} \mathbb{E}_{\mathcal{D}_{\text{train}}}\left[ \text{TokenCost}(N_{\text{tokens}}) + \lambda_{\text{task}} \mathcal{L}_{\text{roundtrip}}(\sigma, \sigma'; \mathbf{w}) - \lambda_H H(\mathbf{w}) \right]$$
  $$\text{s.t.} \quad P_{\text{error}} \le \delta_{\text{max}}, \quad \sum_{i} w_i = 1, \quad w_i \ge w_{\min} = 0.10$$

---

### Step 5: Held-Out Offline Formal Verification (`tests/formal_math/test_nlp_generation_formal.py`)

* **Rule 012 Compliance**: Executed **offline at test/calibration time** (never on the $<5\text{ms}$ hot path):
  1. **Held-Out Component-Wise Round-Trip Distance**:
     Evaluate $\mathcal{L}_{\text{roundtrip}}(\sigma, \sigma'; \mathbf{w}) = \sum w_i d_i(\sigma_i, \sigma_i')$ on **held-out evaluation split $\mathcal{D}_{\text{eval}}$** using Hungarian GED ($d_w$, $c_{\text{dummy}} = 1.0$) and Zhang-Shasha Tree Edit ($d_g$).
  2. **Fano Bound Rate Minimization**:
     Empirically verify word allocation bounds across varying state entropy levels.
  3. **Adversarial Fuzzing Test Corpus**:
     Verify state invariant preservation across a $10^5$-input adversarial fuzzing corpus $\mathcal{C}_{\text{fuzz}}$.

---

## 3. Execution Plan & Verification

1. **Step 1 & Step 2 Implementation**: Build `lexicon_mapper.py` (with Stage 0 cold-start seed handling) and priority waterfall `generative_decoder.py`.
2. **Step 3 Implementation**: Build `text_belief_updater.py` with Front 06/08 multi-criteria sanitizer and $g.u$ evaluation.
3. **Step 4 & Step 5 Implementation**: Build regularized `clsr_text_router.py` and run offline held-out test suite `test_nlp_generation_formal.py`.
