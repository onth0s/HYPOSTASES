# State Vectors to Natural Language Text Generation — Refined Master Architecture & Plan

**Spec Ref**: Wave 5 Front 14 (`docs/WAVES_AND_FRONTS/WAVE_5_FRONT_14/front_14_natural_language_symbolic_compression_spec.md`)  
**Target Substrate**: HYPOSTASES Engine v0.4.0 ($\sigma = (c, w, g, \rho_{\text{ext}})$)  
**Status**: RATIFIED REFINED ARCHITECTURE PLAN  
**Rule 005 Compliance**: Pure Game-Theoretic & Probabilistic Rationality (Zero Artificial Human Cognitive Biases; Data-Derived Lexicons)  
**Rule 006 Compliance**: Primacy of Data-Driven YAML Configuration (`schema/nlp_decoder_config.yaml`)  
**Rule 009 Compliance**: Default Friston Expected Free Energy (EFE) Active Sensing Integration (`efe_mode: true`)  
**Rule 011 Compliance**: Dual Persistence for Lexicon Profilers, Decoders, Meta-Parameters, and Snapshots  
**Rule 012 Compliance**: Mandatory Formal Mathematical Implementation Verification (`tests/formal_math/test_nlp_generation_formal.py`)

---

## 1. Architectural Audit & Critical Corrections

Following technical review, five critical vulnerabilities and formal gaps in the initial draft have been systematically resolved:

1. **Unconfirmed Citation Sanitation**:  
   Removed reliance on unconfirmed pre-print citations ("Pei et al. ICML 2026"). The information-theoretic token budget router is grounded strictly in proven Shannon Rate-Distortion Theory and Fano's Inequality.
2. **Correct Mathematical Round-Trip Formulation**:  
   Replaced improper continuous-to-string KL divergence with formal **State Round-Trip Reconstruction Loss**:
   $$\mathcal{L}_{\text{roundtrip}}(\sigma) = \mathbb{E}_{\sigma}\left[ \|\sigma - \text{parse}(\text{decode}(\text{encode}(\sigma)))\|_2^2 \right]$$
3. **Explicit Decoder Selection & Fallback Arbitration**:  
   Defined clear deterministic arbitration rules specifying when Mode A (PCFG), Mode B (Local SLM), or Mode C (MDL Neural Autoencoder) is active based on state complexity and compute latency bounds.
4. **Elimination of Rule 005 Bias in Lexicon Design**:  
   Replaced hand-glossed human label mappings (`Slot 12 -> "resource_scarcity"`) with **Outcome-Correlated Vector Quantization (VQ)**. Lexicons are statistically derived from cluster-outcome correlations and loaded via YAML schema (Rule 006).
5. **Adversarial Threat Model & Injection Filtering**:  
   Introduced a mandatory **Peer Message Adversarial Filtering & Trust Sanitizer** (Front 06 integration) to prevent peer prompt injection attacks from corrupting $w$ (`WorldModel`) or $g$ (`GoalHierarchy`).

---

## 2. Granular 5-Step Architecture

```
Continuous State σ = (c, w, g, ρ_ext)
            │
            ▼
┌─────────────────────────────────────────┐
│ Step 1: VQ Outcome-Correlated Lexicon   │  <-- Data-derived state cluster mapping (Rule 005/006)
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Step 2: Arbitrated Decoder Engine       │  <-- Mode A (PCFG), Mode B (SLM), Mode C (MDL)
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Step 3: Adversarial Sanitizer & Parser  │  <-- Front 06 Threat Model & Belief Updates
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Step 4: Calibrated Uncertainty Router   │  <-- Fano / Shannon Token Budget Policy
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Step 5: Formal Round-Trip Verification  │  <-- Round-Trip Fidelity & Fano Bounds (Rule 012)
└─────────────────────────────────────────┘
```

---

### Step 1: Outcome-Correlated Lexicon Derivation (`src/hypostases/nlp/lexicon_mapper.py`)

* **Rule 005 & 006 Compliance**: To prevent hand-authored human cognitive bias, discrete codebook slots ($k \in K_{\text{codebook}}$) are mapped to semantic tokens via **Statistical Outcome Clustering**:
  $$\text{LexiconToken}(k) = \arg\max_{t \in \mathcal{V}} P(\Delta \text{Outcome} \mid \sigma \in \text{Cluster}(k))$$
* **Key Classes**:
  - `DataDerivedLexiconMapper`: Loads VQ cluster-outcome correlation tables from `schema/nlp_decoder_config.yaml`.
  - `ConceptCompositionEngine`: Assembles discrete codebook tokens into compositional semantic feature tuples without hardcoded human label hacks.

---

### Step 2: Multi-Mode Decoder Engine & Selection Arbitration (`src/hypostases/nlp/generative_decoder.py`)

* **Decoder Mode Specifications**:
  - **Mode A: Compositional PCFG Synthesizer (Zero-Latency / Deterministic)**
    Used for institutional treaties (Front 05), explicit planning strategies (Front 02), and strict real-time actions.
  - **Mode B: Local SLM Engine (Phi-3 / Llama-3 / Qwen / Ollama)**
    Used for complex multi-agent negotiations when state configurations exceed PCFG grammar coverage.
  - **Mode C: Minimum Description Length (MDL) Neural Autoencoder**
    Used for ultra-low bandwidth, highly compressed peer signaling over noisy channels.

* **Selection & Arbitration Policy**:
  $$\text{Mode}(\sigma) = \begin{cases} 
  \text{Mode A (PCFG)}, & \text{if } \sigma \in \mathcal{S}_{\text{structured}} \text{ (High state structure, latency } < 5\text{ms}) \\ 
  \text{Mode B (SLM)}, & \text{if } \text{Uncertainty } H(w) > \tau_{\text{uncert}} \text{ and compute budget permitted} \\ 
  \text{Mode C (MDL)}, & \text{if Channel Bandwidth } B_{\text{channel}} < \tau_{\text{bandwidth}} 
  \end{cases}$$

---

### Step 3: Adversarial Threat Model, Peer Message Sanitizer, & Belief Updater (`src/hypostases/nlp/text_belief_updater.py`)

* **Adversarial Threat Model**:
  Peer agents may attempt **Prompt Injection / Deceptive Text Manipulation** to inject adversarial goals into $g$ or corrupt $w$. Simple surprisal calculation is insufficient because a skilled adversary optimizes for low surprisal ("Trojan messages").
* **Sanitizer Pipeline**:
  ```
  Peer Text Message
        │
        ▼
  1. Structural Integrity Check (Format Verification)
        │
        ▼
  2. Front 06 Bayesian Trust Discounting: P_trusted(text) = P(text) * Trust(peer_id)
        │
        ▼
  3. Causal Consistency Audit (Front 08 SCM Check): Verify message against w.SCM
        │
        ▼
  4. Sandboxed Belief Update: Apply candidate update to σ_sandbox before mutating σ
  ```

---

### Step 4: Calibrated Information-Theoretic Uncertainty Router (`src/hypostases/nlp/clsr_text_router.py`)

* **Theoretical Foundation**: Grounded in Fano's Inequality and Shannon Rate-Distortion Theory. For state uncertainty $H(Y \mid X)$, the lower bound on generated words/tokens is:
  $$\mathbb{E}[N_{\text{tokens}}] \ge \frac{H(Y \mid X) - h_2(P_{\text{error}})}{\kappa_{\text{active}}}$$
* **Empirical Calibration**:
  Thresholds ($\tau_1, \tau_2$) are not fixed invariant constants; they are configurable hyperparameters tuned via grid-search profiling in `schema/natural_language_compression_config.yaml`.

---

### Step 5: Formal Mathematical Implementation Verification (`tests/formal_math/test_nlp_generation_formal.py`)

* **Rule 012 Compliance**: Replaces superficial checks with rigorous mathematical verification:
  1. **State Round-Trip Reconstruction Error**:
     Verify end-to-end reconstruction loss $\| \sigma - \text{parse}(\text{decode}(\text{encode}(\sigma))) \|_2^2 \le \epsilon_{\text{roundtrip}}$.
  2. **Fano Bound Rate Minimization**:
     Empirically verify that word allocations strictly satisfy Fano's lower bound across varying state entropy levels.
  3. **Adversarial Invariant Verification**:
     Verify that malformed or deceptive injection text cannot mutate state tuple invariants $\sigma = (c, w, g, \rho_{\text{ext}})$.

---

## 3. Execution Plan & Verification

1. **Step 1 & Step 2 Implementation**: Build `lexicon_mapper.py` and arbitrated `generative_decoder.py`.
2. **Step 3 Implementation**: Build `text_belief_updater.py` with Front 06 adversarial sanitizer.
3. **Step 4 & Step 5 Implementation**: Build calibrated `clsr_text_router.py` and run formal mathematical test suite `test_nlp_generation_formal.py`.
