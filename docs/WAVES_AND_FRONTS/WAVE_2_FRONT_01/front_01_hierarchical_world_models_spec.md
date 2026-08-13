# Front 01 Master Specification — Hierarchical World Models & Conceptual Spaces

**Status**: RATIFIED SPECIFICATION (Ingested Literature Synthesized)  
**Wave**: Wave 2 (Structural Abstraction & Metacognitive Planning)  
**Front**: Front 01 — Hierarchical World Models & Conceptual Spaces  
**Target Substrate**: HYPOSTASES Engine ($\sigma = (c, w, g, \rho_{\text{ext}})$)  

---

## 1. Ingested Literature Foundation (`docs/WAVE_2_FRONT_01/papers/`)

This specification synthesizes the theoretical mechanisms, mathematical formulations, and algorithmic structures from all 8 foundational papers ingested into `docs/WAVE_2_FRONT_01/papers/`:

| Source Paper | Key Architectural Synthesis |
|---|---|
| [`tolman_eichenbaum_machine_space_relational_memory_2020.pdf`](../../WAVE_2_FRONT_01/papers/tolman_eichenbaum_machine_space_relational_memory_2020.pdf) | Entorhinal grid-cell structural basis representations $G$, sensory binding $X \otimes G$, path integration via action transition matrices $g_{t+1} = W_{a_t} g_t$, and relational memory generalization across environments. |
| [`what_is_a_cognitive_map_behrens_2018.pdf`](../../WAVE_2_FRONT_01/papers/what_is_a_cognitive_map_behrens_2018.pdf) | Formalizing cognitive maps as relational graphs mapping non-spatial state topologies into structural vector spaces for fast downstream computation and zero-shot inference. |
| [`how_to_build_a_cognitive_map_whittington_2022.pdf`](../../WAVE_2_FRONT_01/papers/how_to_build_a_cognitive_map_whittington_2022.pdf) | Higher-order abstraction mechanics in entorhinal-hippocampal networks: transitive inference, compositional structure representation, and structural graph transfer. |
| [`thorough_formalization_of_conceptual_spaces_bechberger_2017.pdf`](../../WAVE_2_FRONT_01/papers/thorough_formalization_of_conceptual_spaces_bechberger_2017.pdf) | Mathematical formalization of Gärdenfors conceptual spaces: metric quality dimensions, prototype vectors $\vec{\mu}_k$, Mahalanobis distance metric $d_{\mathbf{M}}$, and Voronoi region boundary equations. |
| [`towards_a_definition_of_disentangled_representations_2018.pdf`](../../WAVE_2_FRONT_01/papers/towards_a_definition_of_disentangled_representations_2018.pdf) | Group-theoretic formulation of disentangled representations: quality dimensions as invariant subspace projections under symmetry transformations $g \in G$. |
| [`recurrent_world_models_facilitate_policy_evolution_ha_2018.pdf`](../../WAVE_2_FRONT_01/papers/recurrent_world_models_facilitate_policy_evolution_ha_2018.pdf) | Two-phase world model architecture separating spatial state encoding (V-model) from latent temporal dynamics predictions (M-model). |
| [`dreamerv3_mastering_diverse_domains_world_models_2023.pdf`](../../WAVE_2_FRONT_01/papers/dreamerv3_mastering_diverse_domains_world_models_2023.pdf) | Multi-scale discrete latent space representation, KL-balancing, and robust world model predictions across heterogeneous domains without hyperparameter tuning. |
| [`discovery_of_structural_form_kemp_tenenbaum_2008.pdf`](../../WAVE_2_FRONT_01/papers/discovery_of_structural_form_kemp_tenenbaum_2008.pdf) | Generative structural form discovery (trees, grids, rings, hierarchies, clusters) parsing relational observation matrices into topological graphs. |

---

## 2. Integrated Theoretical Architecture & Mathematical Formalism

```
Level 6: Meta-models (SCM Structural Causal Models & Dynamic Belief Priors)
   ▲
Level 5: Institutions & Norms (Governance Rules & Strategic Payoff/Penalty Contours)
   ▲
Level 4: Conceptual Spaces (Convex Voronoi Regions C_k, Prototype μ_k, Mahalanobis Distance d_M)
   ▲
Level 3: Relations (TEM Entorhinal Structural Basis G ⊗ Sensory Binding X, Transitions g_{t+1} = W_a g_t)
   ▲
Level 2: Objects (Bound Spatial Clusters & Disentangled Quality Vectors x ∈ ℝ^D)
   ▲
Level 1: Environment (Raw Sensory/Spatial Array V_canvas)
```

### 2.1 Level 4: Gärdenfors Conceptual Spaces & Metric Semantics
Following Bechberger & Kühnberger (2017) and Gärdenfors (2000, 2014):
- **Quality Dimensions**: Metric axes $D_1, D_2, \dots, D_d$ spanning quality domain $\Omega \subseteq \mathbb{R}^D$.
- **Prototype Vectors**: Focal point exemplars $\vec{\mu}_k \in \mathbb{R}^D$ for category $k$.
- **Mahalanobis Distance Metric**:
  $$d_{\mathbf{M}}(x, \mu_k) = \sqrt{(x - \mu_k)^T \mathbf{M} (x - \mu_k)}$$
- **Similarity Operator**:
  $$\text{Sim}(x, \mu_k) = \exp\left(-\gamma \, d_{\mathbf{M}}(x, \mu_k)\right)$$
- **Voronoi Partition Convexity**: Concept region $C_k = \{x \in \Omega \mid d_{\mathbf{M}}(x, \mu_k) \le d_{\mathbf{M}}(x, \mu_j) \; \forall j \neq k\}$ guarantees $O(1)$ categorization and spatial neighborhood lookup.

### 2.2 Level 3: Tolman-Eichenbaum Machine (TEM) Relational Factorization
Following Whittington et al. (2020, 2022) and Behrens et al. (2018):
- **Structural Basis Matrix $G$**: Entorhinal grid-cell activation pattern representing structural task graph layout.
- **Sensory Binding Matrix $X$**: Grounded environment observations bound to structural positions.
- **Action Transition Predictor**: Given action $a_t$, structural location updates via linear transformation:
  $$g_{t+1} = W_{a_t} g_t$$
- **Factorized State**: Relational state representation $r_t = g_t \otimes x_t$, enabling structural generalization across non-spatial domains.

### 2.3 Level 1-2 & 5-6: Visual-Dynamics & Structural Form Discovery
Following Ha & Schmidhuber (2018), Hafner et al. (2023), Kemp & Tenenbaum (2008), and Higgins et al. (2018):
- **V-Model Encoding**: Spatial array $V_{\text{canvas}}$ to object clusters and disentangled quality vectors $x \in \mathbb{R}^D$.
- **Structural Form Discovery**: Relational matrices parsed into optimal graph structures (trees, grids, institutional hierarchies).

---

## 3. Core Module Specifications (`src/hypostases/world_model/`)

1. [`hierarchical_types.py`](../../../src/hypostases/world_model/hierarchical_types.py): Dataclasses for `QualityDimension`, `ConceptualRegion`, `TEMBasis`, `AbstractionLevel`, and `HierarchicalState`.
2. [`conceptual_spaces.py`](../../../src/hypostases/world_model/conceptual_spaces.py): Gärdenfors Mahalanobis distance calculator, Voronoi tessellation engine, and $O(1)$ gist categorization.
3. [`tem_factorization.py`](../../../src/hypostases/world_model/tem_factorization.py): TEM structural basis matrix $G$, sensory binding $X \otimes G$, and action transition predictor $W_a$.
4. [`hierarchical_world_model.py`](../../../src/hypostases/world_model/hierarchical_world_model.py): Master `HierarchicalWorldModel` engine integrating all 6 abstraction levels into state component $w$.

---

## 4. Invariant & Safety Guarantees (Rule 005 & Rule 006/007)

- **Rule 005**: All conceptual region categorizations, Mahalanobis metric calculations, and TEM transition updates represent formal mathematical state projections. Zero artificial human cognitive defects or emotional irrationality.
- **Rule 006**: Default quality dimensions, prototype vectors, and TEM transition matrices reside in `schema/hierarchical_world_model_config.yaml`.
- **Rule 007**: YAML serialization performance of conceptual space persistence formats is benchmarked during simulation runs.
