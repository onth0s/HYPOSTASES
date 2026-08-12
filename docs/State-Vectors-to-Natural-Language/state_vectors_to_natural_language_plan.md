# State Vectors to Natural Language Text Generation — Master Architecture & Plan (v3.0)

**Spec Ref**: Wave 5 Front 14 (`docs/WAVES_AND_FRONTS/WAVE_5_FRONT_14/front_14_natural_language_symbolic_compression_spec.md`)  
**Target Substrate**: HYPOSTASES Engine v0.4.0 ($\sigma = (c, w, g, \rho_{\text{ext}})$)  
**Status**: RATIFIED MASTER ARCHITECTURE SPECIFICATION & PLAN  
**Rule 005 Compliance**: Substantial Bias Surface Reduction (Data-Derived Lexicons & VQ Corpus Clustering)  
**Rule 006 Compliance**: Primacy of Data-Driven YAML Configuration (`schema/nlp_decoder_config.yaml`)  
**Rule 009 Compliance**: Default Friston Expected Free Energy (EFE) Active Sensing Integration (`efe_mode: true`)  
**Rule 011 Compliance**: Dual Persistence for Lexicon Mappers, PCFG Rules, Meta-Parameters, and Snapshots  
**Rule 012 Compliance**: Mandatory Formal Mathematical Implementation Verification (`tests/formal_math/test_nlp_generation_formal.py`)

---

## 1. Architectural Audit & Critical Corrections (Pass 2)

Following secondary technical review, six logical, structural, and mathematical gaps have been systematically closed:

1. **Information-Theoretic Grounding (Sanitized Citation)**:  
   Removed reliance on unconfirmed citations. The token budget router is grounded strictly in Shannon Rate-Distortion Theory and Fano's Inequality.
2. **Component-Wise State Round-Trip Distance**:  
   Replaced flat $L_2$ norm over non-vector state objects with weighted per-component distance metrics over continuous memory, graph-structured SCMs, and goal hierarchy trees:
   $$\mathcal{L}_{\text{roundtrip}}(\sigma, \sigma') = w_c d_c(c, c') + w_w d_w(w, w') + w_g d_g(g, g') + w_\rho \|\rho_{\text{ext}} - \rho_{\text{ext}}'\|_2^2$$
3. **Exhaustive Priority Arbitration Waterfall**:  
   Replaced non-exhaustive case notation with an explicit **Priority Waterfall Decision Tree** equipped with deterministic fallback defaults.
4. **Vocabulary Provenance & Bias Reduction**:  
   Framed lexicon optimization as **Human Bias Surface Reduction** (rather than elimination). Vocabulary $\mathcal{V}$ is derived from a statistical token-frequency corpus extracted from simulation traces.
5. **Adversarial Promotion Acceptance Rule**:  
   Defined explicit multi-criteria acceptance for promoting $\sigma_{\text{sandbox}} \to \sigma$ combining Front 08 Causal SCM Audit and Front 06 Trust-Discounted Expected Utility, eliminating surprisal-only vulnerabilities.
6. **Calibration Objective & Fuzzing Scope**:  
   Specified grid-search loss objective function and scoped adversarial invariant tests to a $10^5$-input fuzzing test corpus.

---

## 2. Granular 5-Step Architecture

```
Continuous State σ = (c, w, g, ρ_ext)
            │
            ▼
┌─────────────────────────────────────────┐
│ Step 1: VQ Corpus-Derived Lexicon       │  <-- Data-derived vocabulary & slot mapping (Rule 005/006)
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Step 2: Priority Waterfall Decoder     │  <-- Mode A (PCFG) -> Mode C (MDL) -> Mode B (SLM) -> Fallback
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Step 3: Multi-Criteria Causal Sanitizer │  <-- SCM Audit + Trust Discount Promotion (Front 06/08)
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Step 4: Calibrated Fano Uncertainty     │  <-- Grid-search tuned Fano token budget policy
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Step 5: Formal Round-Trip Verification  │  <-- Component-wise metric loss & Fuzzing corpus tests
└─────────────────────────────────────────┘
```

---

### Step 1: Corpus-Derived Lexicon & VQ Mapping (`src/hypostases/nlp/lexicon_mapper.py`)

* **Rule 005 & 006 Compliance**: To minimize human bias, vocabulary $\mathcal{V}$ is extracted from token frequencies across simulation trace logs. Discrete codebook slots ($k \in K_{\text{codebook}}$) are mapped to semantic tokens via **Outcome-Correlated Clustering**:
  $$\text{LexiconToken}(k) = \arg\max_{t \in \mathcal{V}_{\text{corpus}}} P(\Delta \text{Outcome} \mid \sigma \in \text{Cluster}(k))$$
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
  Peer agents may attempt **Prompt Injection / Deceptive Text Manipulation** to inject adversarial goals into $g$ or corrupt $w$. Surprisal alone is insufficient because liars optimize for low surprisal ("Trojan messages").
* **Multi-Criteria Promotion Acceptance Rule**:
  $$\text{Promote}(\sigma_{\text{sandbox}} \to \sigma) \iff \text{Audit}_{\text{format}} \land \left( \text{Audit}_{\text{SCM}}(w, w_{\text{sandbox}}) \le \tau_{\text{causal}} \right) \land \left( \text{Trust}(\text{peer}) \cdot \Delta U_{\text{expected}} \ge \tau_{\text{promotion}} \right)$$
  - **Front 08 Causal Audit**: Rejects messages that contradict Structural Causal Model invariants.
  - **Front 06 Bayesian Trust Discounting**: Requires positive expected utility gain weighted by peer trust score.

---

### Step 4: Calibrated Fano Uncertainty Router (`src/hypostases/nlp/clsr_text_router.py`)

* **Theoretical Bound**: Grounded in Fano's Inequality and Shannon Rate-Distortion Theory:
  $$\mathbb{E}[N_{\text{tokens}}] \ge \frac{H(Y \mid X) - h_2(P_{\text{error}})}{\kappa_{\text{active}}}$$
* **Calibration Objective Function**:
  Thresholds $(\tau_1, \tau_2)$ are calibrated over a benchmark trace corpus $\mathcal{D}_{\text{traces}}$ by minimizing:
  $$\min_{\tau_1, \tau_2} \mathbb{E}_{\mathcal{D}_{\text{traces}}}\left[ \text{TokenCost}(N_{\text{tokens}}) + \lambda_{\text{task}} \mathcal{L}_{\text{roundtrip}}(\sigma, \sigma') \right] \quad \text{s.t.} \quad P_{\text{error}} \le \delta_{\text{max}}$$

---

### Step 5: Formal Mathematical Implementation Verification (`tests/formal_math/test_nlp_generation_formal.py`)

* **Rule 012 Compliance**:
  1. **Component-Wise Round-Trip Distance**:
     Verify $\mathcal{L}_{\text{roundtrip}}(\sigma, \text{parse}(\text{decode}(\text{encode}(\sigma)))) \le \epsilon_{\text{roundtrip}}$ across continuous memory, SCM graph edit distance, and goal tree edit distance.
  2. **Fano Bound Rate Minimization**:
     Empirically verify word allocation bounds across varying state entropy levels.
  3. **Adversarial Fuzzing Test Corpus**:
     Verify state invariant preservation across a $10^5$-input adversarial fuzzing corpus $\mathcal{C}_{\text{fuzz}}$.

---

## 3. Execution Plan & Verification

1. **Step 1 & Step 2 Implementation**: Build `lexicon_mapper.py` and priority waterfall `generative_decoder.py`.
2. **Step 3 Implementation**: Build `text_belief_updater.py` with Front 06/08 multi-criteria sanitizer.
3. **Step 4 & Step 5 Implementation**: Build calibrated `clsr_text_router.py` and run formal test suite `test_nlp_generation_formal.py`.
