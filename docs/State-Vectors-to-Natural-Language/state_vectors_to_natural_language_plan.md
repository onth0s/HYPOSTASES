# State Vectors to Natural Language Text Generation — Master Architecture & Implementation Plan

**Spec Ref**: Wave 5 Front 14 (`docs/WAVE_5_FRONT_14/front_14_natural_language_symbolic_compression_spec.md`)  
**Target Substrate**: HYPOSTASES Engine v0.4.0 ($\sigma = (c, w, g, \rho_{\text{ext}})$)  
**Status**: RATIFIED MASTER ARCHITECTURE PLAN  
**Rule 005 Invariant**: Pure Game-Theoretic & Probabilistic Rationality (Zero Artificial Human Cognitive Biases)  
**Rule 006 Invariant**: Primacy of Data-Driven YAML Configuration (`schema/nlp_decoder_config.yaml`)  
**Rule 009 Invariant**: Default Friston Expected Free Energy (EFE) Active Sensing Integration (`efe_mode: true`)  
**Rule 011 Invariant**: Dual Persistence for Lexicon Mappers, PCFG Rules, Meta-Parameters, and Snapshots  
**Rule 012 Invariant**: Formal Mathematical Implementation Verification (`tests/formal_math/test_nlp_generation_formal.py`)

---

## 1. Executive Summary & Core Objective

While **Wave 5 Front 14** established the formal mathematical state compression pipeline—discretizing high-dimensional continuous state manifolds $\sigma = (c, w, g, \rho_{\text{ext}})$ into token indices under Minimum Description Length (MDL) rate-distortion bounds—the engine requires a dedicated **Natural Language Generative Bridge** to emit and ingest actual natural language text strings.

This plan details the high-granular 5-stage architecture to connect `SymbolicCompressionEngine` to **actual English prose, formal protocol strings, and natural language dialogs**.

```
Continuous State σ = (c, w, g, ρ_ext)
            │
            ▼
┌─────────────────────────────────────────┐
│ Step 1: Codebook-to-Lexicon Mapper      │  <-- Map state clusters to semantic concepts
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Step 2: Generative Decoder Layer        │  <-- Produce real English text (PCFGs or SLMs)
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Step 3: Text Belief Updater & Parser    │  <-- Parse incoming text back into Bayesian likelihoods
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Step 4: CLSR Text-Budget Router         │  <-- Scale text length based on query uncertainty
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Step 5: Data-Driven YAML & Formal Math  │  <-- Enforce Rule 006 & Rule 012 Verification
└─────────────────────────────────────────┘
```

---

## 2. Detailed 5-Step Architectural Components

### Step 1: Codebook-to-Lexicon Mapper (`src/hypostases/nlp/lexicon_mapper.py`)

* **Objective**: Create a deterministic, mathematically grounded bridge between discrete codebook slots ($K_{\text{codebook}}$) and natural language semantic primitives.
* **Key Classes**:
  - `SemanticLexicon`: A structured dictionary mapping discrete cluster IDs to core lexical concepts:
    - `Slot 12` $\to$ `"resource_scarcity"`
    - `Slot 45` $\to$ `"altruistic_punishment"`
    - `Slot 88` $\to$ `"institutional_treaty"`
    - `Slot 102` $\to$ `"cooperative_strategy"`
  - `ConceptCompositionEngine`: Combines multiple active slot tokens into structured semantic tuples:
    - Input slots: `[12, 102]` $\implies$ Tuple: `(SUBJECT: self, ACTION: propose_cooperation, CONDITION: resource_scarcity)`.

---

### Step 2: Generative Text Decoder Engine (`src/hypostases/nlp/generative_decoder.py`)

* **Objective**: Convert continuous state projections and semantic tuples into actual natural language text strings.
* **Three Complementary Decoder Modes**:
  1. **Mode A: Compositional PCFG Synthesizer (Zero-Latency / Deterministic)**
     - Uses Probabilistic Context-Free Grammars (PCFGs) over the state tuple to generate grammatically crisp, predictable English text without requiring heavy neural networks.
     - *Example Output*: `"Agent 1 proposes mutual cooperation under severe resource scarcity."`
  2. **Mode B: Local SLM Engine (Phi-3 / Llama-3 / Qwen / VLLM / Ollama)**
     - Feeds the compressed symbol token stream $L$ into a local Small Language Model (SLM) prompt.
     - *System Prompt*: `"You are an agent in a game-theoretic simulation. Convert the following state compression into a 1-sentence strategic message: {tokens}"`
  3. **Mode C: Minimum Description Length (MDL) Neural Autoencoder**
     - Employs a tiny, fine-tuned neural decoder that directly minimizes the MDL loss $\mathcal{L}_{\text{MDL}} = |H| + \lambda_{\text{MDL}} D_{\text{KL}}$ over text sequences.

---

### Step 3: Text Belief Updater & Parser (`src/hypostases/nlp/text_belief_updater.py`)

* **Objective**: Complete the loop by taking natural language text messages received from peer agents and updating the local agent's $w$ (`WorldModel`) and $g$ (`GoalHierarchy`).
* **Key Classes**:
  - `NaturalLanguageSemanticParser`: Extracts structured semantic tuples `(actor, intent, proposed_action, claimed_state)` from raw English text.
  - `TextLikelihoodConverter`: Converts parsed text into Bayesian likelihood evidence matrix updates $P(o \mid s_{\text{text}})$.
  - `TextSurprisalCalculator`: Computes Shannon surprisal $-\log_2 P(\text{text} \mid w)$ of incoming messages to adjust trust scores and detect deception (Front 06 integration).

---

### Step 4: CLSR Text-Budget Router (`src/hypostases/nlp/clsr_text_router.py`)

* **Objective**: Enforce Pei et al. (ICML 2026) Theorem 3.2 lower bounds ($\mathbb{E}[|\mathcal{T}|] \ge \frac{I_{\text{req}}}{\kappa_\theta}$) on real word counts.
* **Routing Policies**:
  - **Low Uncertainty ($H(Y \mid X) \le 0.5$)**: Emit a 3-word shorthand command (e.g. `"ACCEPT_COOPERATION_PLAN"`).
  - **Medium Uncertainty ($0.5 < H \le 2.0$)**: Emit a single concise English sentence (e.g. `"I accept the treaty provided resource extraction is capped at 50%."`).
  - **High Uncertainty ($H > 2.0$)**: Emit a full multi-sentence rationale outlining contingencies.

---

### Step 5: Data-Driven YAML Preset & Formal Verification

1. **YAML Configuration** (`schema/nlp_decoder_config.yaml`):
   - Stores PCFG grammar rules, lexicon lookup paths, SLM model parameters, and CLSR active-token rate constants ($\kappa_\theta$).
2. **Formal Math Verification** (`tests/formal_math/test_nlp_generation_formal.py`):
   - **Reconstruction Fidelity Bound**: Verifies $D_{\text{KL}}(\sigma \parallel \text{decode}(\text{encode}(\sigma))) \le \epsilon$.
   - **PCFG Grammar Completeness**: Verifies that 100% of state tuple projections map to syntactically valid sentences.
   - **MDL Codelength Equivalence**: Verifies that text code lengths strictly obey Kraft's Inequality and Shannon source coding bounds.

---

## 3. Verification Plan

### Automated Tests
1. **Rule 001 Verification**: Run `ruff check .` and `ruff format --check .`
2. **Rule 002 & Rule 012 Verification**: Run `pytest` and `pytest tests/formal_math/test_nlp_generation_formal.py`
3. **State Invariant Assertions**: Ensure full state tuple compatibility with $\sigma = (c, w, g, \rho_{\text{ext}})$.
