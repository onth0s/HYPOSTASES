# HYPOSTASES — Cognitive Expansion Fronts Index

Spec Ref: `misc/next-steps.md` | Ratified Compass: [`docs/roadmap_compass.md`](roadmap_compass.md)

All 14 cognitive expansion specs have been spliced into individual, self-contained Front files adhering to project naming conventions (`front_XX_<name>.md`).

---

## Spliced Research Fronts Summary

| Front ID | Title | Status | Source Sections | Key Concept | Document Link |
|---|---|---|---|---|---|
| **Front 01** | Hierarchical World Models | `IMPLEMENTED` | Section I | Multi-level semantic representations, Gärdenfors Conceptual Spaces, TEM & CAN grid-cell factorizations | [front_01_hierarchical_world_models.md](front_01_hierarchical_world_models.md) |
| **Front 02** | Explicit Planning Layer | `IMPLEMENTED` | Section II | Reusable strategies, plans as first-class objects, plan repair & interruption | [front_02_explicit_planning_layer.md](front_02_explicit_planning_layer.md) |
| **Front 03** | Memory Architecture | `IMPLEMENTED` | Section III | Working, Episodic, Semantic, Procedural memory separation, kMP primitives & consolidation | [front_03_memory_architecture.md](front_03_memory_architecture.md) |
| **Front 04** | Counterfactual Simulation | `IMPLEMENTED` | Section IV | Internal multi-future hypothetical simulations, elastica PDE rollouts & Monte Carlo search evaluation | [front_04_counterfactual_simulation.md](front_04_counterfactual_simulation.md) |
| **Front 05** | Institution Layer | `IMPLEMENTED` | Section V | First-class institutional entities operating as agents with governance & rules | [front_05_institution_layer.md](front_05_institution_layer.md) |
| **Front 06** | Communication as Bayesian Evidence | `IMPLEMENTED` | Section VI | Messages as probabilistic likelihood evidence for belief posterior updates | [front_06_communication_bayesian_evidence_spec.md](WAVE_3_FRONT_06/front_06_communication_bayesian_evidence_spec.md) |
| **Front 07** | Meta-Learning | `IMPLEMENTED` | Section VII | Adapting internal reasoning mechanisms, learning rates, and policy updates | [front_07_meta_learning.md](front_07_meta_learning.md) |
| **Front 08** | Causal World Models | `IMPLEMENTED` | Section VIII | Structural Causal Models (SCMs), interventions (do-calculus), causal diagnosis, and verified NOTEARS structure recovery | [front_08_causal_world_models.md](front_08_causal_world_models.md) |
| **Front 09** | Active Information Gathering | `IMPLEMENTED` | Section IX | Epistemic actions (probe, inspect, monitor) trading material utility for entropy reduction | [front_09_active_information_gathering.md](front_09_active_information_gathering.md) |
| **Front 10** | Mechanism Search | `IMPLEMENTED` | Section X | Optimization over the space of institutional designs and governance rules | [front_10_mechanism_search_spec.md](WAVE_4_FRONT_10/front_10_mechanism_search_spec.md) |
| **Front 11** | Abductive Reasoning & Hypotheses | `IMPLEMENTED` | Sections XI & XII | Explanations as explicit Hypothesis objects ($H_1, H_2, \dots$) and abductive inference | [front_11_abductive_reasoning_hypothesis_objects.md](front_11_abductive_reasoning_hypothesis_objects.md) |
| **Front 12** | Scientific Discovery Loop | `IMPLEMENTED` | Section XIII | Iterative hypothesis generation, experimental design, and empirical model refinement | [front_12_scientific_discovery_loop_spec.md](WAVE_4_FRONT_12/front_12_scientific_discovery_loop_spec.md) |
| **Front 13** | Evolutionary Algorithm Discovery | `IMPLEMENTED` | AlphaEvolve Spec | Mutating and evolving code/heuristics evaluated against multi-agent equilibrium & game-theoretic oracles | [front_13_evolutionary_algorithm_discovery_alphaevolve_spec.md](WAVE_5_FRONT_13/front_13_evolutionary_algorithm_discovery_alphaevolve_spec.md) |
| **Front 14** | Natural Language & Symbolic Compression | `IMPLEMENTED` | Epistemic Symbol Spec | Visual-Epistemic Duality: Natural language & symbols as lossy compression of spatial state $\sigma \in \mathbb{R}^d$ | [front_14_natural_language_symbolic_compression.md](front_14_natural_language_symbolic_compression.md) |

---

## State Invariant
All fronts respect the core architectural invariant:
$$\sigma = (c, w, g, \rho_{\text{ext}})$$
Higher-order cognitive capabilities are modeled as computational layers operating over this persistent state tuple rather than adding non-computable state parameters.
