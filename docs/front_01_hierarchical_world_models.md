# Front 01 — Hierarchical World Models & Conceptual Spaces

Spec Ref: `misc/next-steps.md` Section I | Gärdenfors Conceptual Spaces & TEM Factorization

## Overview
The World Model ($w \in W$) maintains multi-level semantic world representations, allowing beliefs to exist across several spatial and abstract layers.

Front 01 integrates **Gärdenfors Conceptual Spaces** (geometric semantics) and the **Tolman-Eichenbaum Machine (TEM)** (entorhinal grid-cell structural basis representations) into the world model architecture.

## Abstraction Hierarchy
```
Environment (Raw Sensor & Spatial Array V_canvas)
    ↓
Objects (Bound Spatial Clusters & Quality Dimensions)
    ↓
Relations (TEM Grid-Cell Tensor Factorization G ⊗ X)
    ↓
Conceptual Spaces (Convex Voronoi Gist Regions in Ω ⊂ ℝ^D)
    ↓
Institutions & Norms (Governance Rules & Strategic Contours)
    ↓
Meta-models (SCM Structural Causal Models)
```

## Conceptual Spaces & Metric Semantics (Peter Gärdenfors)
Instead of purely Boolean predicate logic, semantic concepts are formalized as **convex regions within multidimensional quality spaces** ($\Omega \subset \mathbb{R}^D$):
- **Quality Dimensions**: Metric axes representing physical or abstract properties (e.g. reserve capacity, pool density, peer trust, scarcity, uncertainty).
- **Convex Voronoi Regions**: A concept or Gist is defined as a convex region centered around prototype vector $\vec{\mu}_k$.
- **Mahalanobis Similarity Metric**:
  $$d_{\mathbf{M}}(x, \mu_k) = \sqrt{(x - \mu_k)^T \mathbf{M} (x - \mu_k)}$$
  $$\text{Similarity}(x, \mu_k) = \exp\left(-\gamma \, d_{\mathbf{M}}(x, \mu_k)\right)$$
- **Geometric Free Lunch**: Instant $O(1)$ categorization, spatial neighborhood lookup, and generalization across continuous state space without combinatorial logical checks.

## Tolman-Eichenbaum Machine (TEM) Structural Basis
Following Whittington et al. (*Cell* 2020), entorhinal grid-cell representations are formalized as structural basis matrices $G$ that factorize relational environments:
- **Structural Basis Matrix $G$**: Invariant grid-cell spatial/relational layout.
- **Sensory Binding Matrix $X$**: Specific environment observations bound to grid positions.
- Enables the agent to use its spatial navigation hardware to navigate non-spatial conceptual and relational hierarchies.

## Targeted Capabilities
- Geometric conceptual spaces & Voronoi tessellation
- Grid-cell relational tensor factorization (TEM)
- Semantic abstraction & spatial reasoning
- Institutional reasoning & nested representations
- Abstract situation understanding

## Core State Constraint
Operates over persistent primitive state $\sigma = (c, w, g, \rho_{\text{ext}})$ without introducing non-computable primitives or Rule 005 violations.
