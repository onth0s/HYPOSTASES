# State Vectors to Natural Language Text Generation — Working Architecture Specification & Plan (v4.0)

**Spec Ref**: Wave 5 Front 14 (`docs/WAVES_AND_FRONTS/WAVE_5_FRONT_14/front_14_natural_language_symbolic_compression_spec.md`)  
**Target Substrate**: HYPOSTASES Engine v0.4.0 ($\sigma = (c, w, g, \rho_{\text{ext}})$)  
**Status**: WORKING ARCHITECTURE SPECIFICATION (PENDING IMPLEMENTATION BINDINGS)  
**Rule 005 Compliance**: Inverse-Variance Weighting & Cold-Start Seed Corpus Disclosure  
**Rule 006 Compliance**: Primacy of Data-Driven YAML Configuration (`schema/nlp_decoder_config.yaml`)  
**Rule 009 Compliance**: Default Friston Expected Free Energy (EFE) Active Sensing Integration (`efe_mode: true`)  
**Rule 011 Compliance**: Dual Persistence for Lexicon Mappers, PCFG Rules, Meta-Parameters, and Snapshots  
**Rule 012 Compliance**: Mandatory Formal Mathematical Implementation Verification (`tests/formal_math/test_nlp_generation_formal.py`)

---

## 1. Architectural Audit & Critical Corrections (Pass 3)

Following tertiary technical review, five second-order mathematical and operational gaps have been explicitly resolved:

1. **Polynomial-Time Component Metric Definitions**:  
   Replaced generic NP-hard Graph Edit Distance with polynomial-time algorithms:
   - SCM distance $d_w(w, w')$ uses **Bipartite Assignment GED Approximation (Hungarian Algorithm $O(V^3)$)** + Adjacency Matrix Frobenius Norm $\|A_w - A_{w'}\|_F$.
   - Goal Tree distance $d_g(g, g')$ uses **Zhang-Shasha Tree Edit Distance** ($O(|V_1||V_2| \cdot \text{depth} \cdot \text{leaves})$).
2. **Data-Driven Weight Calibration Procedure (Inverse Variance)**:  
   Eliminated hand-tuned weight bias by defining weight assignment via normalized **Inverse-Variance Information Weighting**:
   $$w_i = \frac{1 / \text{Var}(d_i)}{\sum_{j \in \{c, w, g, \rho\}} 1 / \text{Var}(d_j)}$$
   where variances are computed over baseline simulation trace rollouts $\mathcal{D}_{\text{baseline}}$.
3. **Formal Definition of Expected Utility Gain $\Delta U_{\text{expected}}$**:  
   Explicitly defined evaluation using the receiving agent's endogenous Goal Hierarchy $g.u$ evaluated *after* passing the Front 08 Causal SCM Audit:
   $$\Delta U_{\text{expected}} = \mathbb{E}_{\sigma_{\text{sandbox}}}[g.u(\sigma_{\text{sandbox}})] - \mathbb{E}_{\sigma}[g.u(\sigma)]$$
4. **Explicit Cold-Start Vocabulary Bootstrapping Procedure**:  
   Acknowledged a 2-stage vocabulary initialization:
   - **Stage 0 (Cold-Start Seed Corpus)**: Initialized from a domain-general token frequency list (WordNet / Universal Dependencies).
   - **Stage 1+ (Data-Driven Refinement)**: Dynamically refined as simulation traces accumulate via VQ cluster outcome mapping.
5. **Execution Latency Scoping**:  
   Explicitly scoped $\mathcal{L}_{\text{roundtrip}}$ as an **offline test-time / calibration-time evaluation metric**, keeping it completely off the $<5\text{ms}$ real-time inference hot path.

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
│ Step 4: Calibrated Fano Uncertainty     │  <-- Grid-search tuned Fano token budget policy
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Step 5: Offline Formal Verification     │  <-- Offline Round-Trip Metric Loss & Fuzzing Corpus
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

### Step 4: Calibrated Fano Uncertainty Router (`src/hypostases/nlp/clsr_text_router.py`)

* **Theoretical Bound**: Grounded in Fano's Inequality and Shannon Rate-Distortion Theory:
  $$\mathbb{E}[N_{\text{tokens}}] \ge \frac{H(Y \mid X) - h_2(P_{\text{error}})}{\kappa_{\text{active}}}$$
* **Calibration Objective Function**:
  Thresholds $(\tau_1, \tau_2)$ are calibrated over trace corpus $\mathcal{D}_{\text{traces}}$ by minimizing:
  $$\min_{\tau_1, \tau_2} \mathbb{E}_{\mathcal{D}_{\text{traces}}}\left[ \text{TokenCost}(N_{\text{tokens}}) + \lambda_{\text{task}} \mathcal{L}_{\text{roundtrip}}(\sigma, \sigma') \right] \quad \text{s.t.} \quad P_{\text{error}} \le \delta_{\text{max}}$$

---

### Step 5: Offline Formal Verification (`tests/formal_math/test_nlp_generation_formal.py`)

* **Rule 012 Compliance**: Executed **offline at test/calibration time** (never on the $<5\text{ms}$ hot path):
  1. **Polynomial Component-Wise Round-Trip Distance**:
     Evaluate $\mathcal{L}_{\text{roundtrip}}(\sigma, \sigma') = \sum w_i d_i(\sigma_i, \sigma_i')$ using Hungarian GED ($d_w$) and Zhang-Shasha Tree Edit ($d_g$), with inverse-variance weights $w_i = \frac{1/\text{Var}(d_i)}{\sum 1/\text{Var}(d_j)}$.
  2. **Fano Bound Rate Minimization**:
     Empirically verify word allocation bounds across varying state entropy levels.
  3. **Adversarial Fuzzing Test Corpus**:
     Verify state invariant preservation across a $10^5$-input adversarial fuzzing corpus $\mathcal{C}_{\text{fuzz}}$.

---

## 3. Execution Plan & Verification

1. **Step 1 & Step 2 Implementation**: Build `lexicon_mapper.py` (with Stage 0 cold-start seed handling) and priority waterfall `generative_decoder.py`.
2. **Step 3 Implementation**: Build `text_belief_updater.py` with Front 06/08 multi-criteria sanitizer and $g.u$ evaluation.
3. **Step 4 & Step 5 Implementation**: Build calibrated `clsr_text_router.py` and run offline formal test suite `test_nlp_generation_formal.py`.
