# Front 03 — Memory Architecture

Spec Ref: `misc/next-steps.md` Section III

## Overview
Investigates explicit separation of agent memory systems beyond transient scalar decay into structured memory stores.

## Memory Systems Decomposition
- **Working Memory**: Transient, high-priority state buffers for ongoing goal evaluation $g$.
- **Episodic Memory**: Sequential logs of historical state-action-reward experiences $\langle \sigma_t, a_t, r_t, \sigma_{t+1} \rangle$ supporting counterfactual replay.
- **Semantic Memory**: Abstracted Gärdenfors conceptual space prototypes $\vec{\mu}_k$ and relational topologies $G \otimes X$.
- **Procedural Memory (`SkillArtifact`)**: Consolidated reusable macro-action patterns and continuous Kinematic Motion Primitives (kMPs).

## Procedural Skill Decomposition & Kinematic Motion Primitives (kMPs)
Following **Zelman et al. (2013)** (*Front Comput Neurosci*) and **Gutfreund et al. (1998)** (*J Neurosci*), continuous procedural skill execution is modeled as a weighted linear combination of spatiotemporal Gaussian basis functions:

$$\text{Skill}(s, t) = \sum_{k=1}^K w_k \cdot \exp\left( -\frac{(s - \mu_{s,k})^2}{2\sigma_{s,k}^2} - \frac{(t - \mu_{t,k})^2}{2\sigma_{t,k}^2} \right)$$

- **Spatiotemporal Primitives (Zelman et al. 2013)**: Decomposes continuous spatial curvature and velocity trajectories into primitive Gaussian basis blocks, enabling smooth composition of high-DOF motion primitives.
- **Stiffening & Propagation Waves (Gutfreund et al. 1998)**: Models traveling wave primitives along physical/abstract execution chains, offloading multi-joint motor control into single primitive trigger actions.
- **`SkillArtifact` Interface**: Each acquired skill artifact stores trigger preconditions, Gaussian basis weights $\mathbf{w}$, and expected utility gain $\Delta g$.

## Implementation Status
- **Status**: **IMPLEMENTED** (`test_memory.py`, episodic memory stores, procedural skill acquisition logic).

## Targeted Capabilities
- Structured retrieval
- Selective forgetting & consolidation
- Abstraction & analogical recall
- Episodic replay
- Skill acquisition & kMP continuous primitive synthesis

## Core State Constraint
Memory mechanisms operate over primitive state $\sigma = (c, w, g, \rho_{\text{ext}})$ without introducing non-computable state mutations.
