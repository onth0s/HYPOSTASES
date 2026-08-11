  # HYPOSTASES — Cognitive Expansion High-Level Specification & Compass

  **Status**: RATIFIED GROUND-TRUTH COMPASS  
  **Target Substrate**: HYPOSTASES v0.4.0 Engine ($\sigma = (c, w, g, \rho_{\text{ext}})$)  
  **Spec Version**: v1.0-Cognitive-Roadmap (2026 SOTA Aligned)

  ---

  ## Executive Summary & Architectural Integrity

  This document serves as the **Ratified High-Level Specification and Strategic Compass** for expanding HYPOSTASES from an agent simulation engine into a unified generative engine for **intelligent reasoning, multi-agent swarm distillation, and evolutionary algorithm discovery**.

  All higher-order cognitive capabilities across all 14 expansion fronts obey the **Core State Invariant**:

  $$\sigma = (c, w, g, \rho_{\text{ext}})$$

  Higher-order cognitive structures (memories, planners, causal models, hypothesis managers, evolutionary search, natural language symbol streams) operate strictly as **computational projections and functional layers** over this persistent primitive state tuple, preserving mathematical minimality, formal computability, and Rule 005 integrity.

  ---

  ## SOTA 2026 Wave Dependency Sequence

  ```
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ WAVE 1: Single-Agent Foundations (Memory, Lookahead & Active Sensing)                     │
  │ [Front 03: Memory] ──► [Front 04: Counterfactual Simulation] ──► [Front 09: Active Sensing] │
  └───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                              │
                                              ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ WAVE 2: Structural Abstraction & Metacognitive Planning                                  │
  │ [Front 02: Explicit Planning] ──► [Front 01: Hierarchical Models] ──► [Front 08: Causal]  │
  └───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                              │
                                              ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ WAVE 3: Multi-Agent Epistemology & Swarm Mechanics                                       │
  │ [Front 06: Bayesian Comm] ──► [Front 11: Abduction & Hypotheses] ──► [Front 05: Inst.]    │
  └───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                              │
                                              ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ WAVE 4: Recursive Adaptation & Discovery Loops                                           │
  │ [Front 07: Meta-Learning] ──► [Front 10: Mechanism Search] ──► [Front 12: Discovery Loop] │
  └───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                              │
                                              ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ WAVE 5: Universal Scaling & Symbolic Generalization                                      │
  │ [Front 13: AlphaEvolve Engine] ──► [Front 14: Natural Language Compression]               │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
  ```

  ---

  ## Detailed Wave & Front Specifications

  ### Wave 1 — Single-Agent Cognition Foundations

  #### 1. Front 03 — Memory Architecture (`docs/front_03_memory_architecture.md`)
  - **Core Concept**: Explicit separation of memory into Working, Episodic, Semantic, and Procedural layers.
  - **Key Artifact**: `SkillArtifact` (consolidated reusable macro-action patterns with trigger conditions and expected utility gain $\Delta g$).
  - **State Coupling**: Expands read-only derived views over $\sigma$ without introducing non-computable state mutations.

  #### 2. Front 04 — Counterfactual Simulation (`docs/front_04_counterfactual_simulation.md`)
  - **Core Concept**: Memory-grounded multi-future hypothetical rollouts and Monte Carlo tree evaluation (aligned with ICML/ICLR 2026 EvoCF).
  - **Key Mechanism**: Internal forward rollouts using `step_env` / `feedback` / `evolve` without mutating physical state until selection.

  #### 3. Front 09 — Active Information Gathering (`docs/front_09_active_information_gathering.md`)
  - **Core Concept**: Epistemic actions (`INSPECT`, `PROBE`, `MONITOR`) where agents intentionally trade immediate utility for belief variance reduction ($\Delta \sigma^2$).
  - **State Coupling**: Information gain enters the decision calculus as explicit epistemic utility.

  ---

  ### Wave 2 — Structural Abstraction & Metacognitive Planning

  #### 4. Front 02 — Explicit Planning Layer (`docs/front_02_explicit_planning_layer.md`)
  - **Core Concept**: Transition from direct `Goal -> Action` to structured `Goal -> Plan -> Action`.
  - **Key Mechanics**: Reusable strategies, plan libraries, plan interruption, contingency branching, and plan repair.

  #### 5. Front 01 — Hierarchical World Models (`docs/front_01_hierarchical_world_models.md`)
  - **Core Concept**: Multi-level semantic representations of the environment.
  - **Hierarchy**: `Environment -> Objects -> Relations -> Institutions -> Norms -> Meta-models`.

  #### 6. Front 08 — Causal World Models (`docs/front_08_causal_world_models.md`)
  - **Core Concept**: Explicit representation of Structural Causal Models (SCMs) and intervention reasoning ($do$-calculus) inside $w$.
  - **Epistemic Shift**: Replaces predictive correlation ($A \text{ predicts } B$) with explicit causality ($A \text{ causes } B$).

  ---

  ### Wave 3 — Social Epistemology & Swarm Mechanics

  #### 7. Front 06 — Communication as Bayesian Evidence (`docs/front_06_communication_bayesian_evidence.md`)
  - **Core Concept**: Messages act as probabilistic likelihood evidence updating peer belief posteriors $P(\sigma_{\text{peer}})$.
  - **Key Capabilities**: Trust modeling, deception detection, reputation scoring, misinformation filtering.

  #### 8. Front 11 — Abductive Reasoning & Hypothesis Objects (`docs/front_11_abductive_reasoning_hypothesis_objects.md`)
  - **Core Concept**: Explicit `Hypothesis` objects ($H_1, H_2, \dots$) identifying the best explanations for environment/peer anomalies.
  - **Question Answered**: *"What explanation best accounts for these observations?"*

  #### 9. Front 05 — Institution Layer (`docs/front_05_institution_layer.md`)
  - **Core Concept**: First-class institutional entities (governments, markets, guilds, courts) operating as agents with authority and governance rules.
  - **Accountability**: Altruistic punishment costs (`PUNISH_RESERVE_COST`) ensure non-tyrannical, self-limiting governance enforcement.

  ---

  ### Wave 4 — Recursive Adaptation & Discovery Loops

  #### 10. Front 07 — Meta-Learning (`docs/front_07_meta_learning.md`)
  - **Core Concept**: Agents adapt their own internal search heuristics, exploration parameters ($\xi$), learning rates, and policy updates across experience ticks.
  - **Target**: Learning how to learn.

  #### 11. Front 10 — Mechanism Search (`docs/front_10_mechanism_search.md`)
  - **Core Concept**: Optimization over the space of institutional designs, auction rules, and governance policies using the simulation harness as a black-box oracle.

  #### 12. Front 12 — Scientific Discovery Loop (`docs/front_12_scientific_discovery_loop.md`)
  - **Core Concept**: Full cognitive discovery cycle: `Observe -> Infer -> Generate Hypotheses -> Rank Explanations -> Design Experiment -> Collect Evidence -> Update Hypotheses -> Act`.

  ---

  ### Wave 5 — Universal Scaling & Symbolic Generalization

  #### 13. Front 13 — Evolutionary Algorithm Discovery (AlphaEvolve Engine) (`docs/front_13_evolutionary_algorithm_discovery_alphaevolve.md`)
  - **Core Concept**: Mutating and evolving code/heuristics evaluated against multi-agent game-theoretic equilibrium and endogenous scarcity ($\kappa$) feedback oracles.

  #### 14. Front 14 — Natural Language as Symbolic Compression (`docs/front_14_natural_language_symbolic_compression.md`)
  - **Core Concept**: Natural language as a lossy, high-density symbolic compression operator over continuous state $\sigma \in \mathbb{R}^d$, enabling tractable swarm communication and LLM reasoning.

  ---

  ## Ground-Truth Alignment & Safety Invariant (`src/hypostases/schemas`)

  All 14 fronts are subject to strict programmatic enforcement:
  1. `assert_invariants(agent_state)`: Rejects non-physical or invalid state bounds.
  2. `assert_schema_completeness()`: Prevents degenerate state-independent shortcuts.
  3. `Rule 005 (AGENTS.md)`: Prohibits artificial human cognitive defects or emotional irrationality hacks; enforces pure game-theoretic rationality.

  ---

  ## Execution Status

  - **Status**: RATIFIED & COMMITTED AS COMPASS (`docs/roadmap_compass.md`).
  - **Implementation State**: DEFERRED (Pending pre-requisite infrastructure tasks prior to Wave 1 kickoff).
