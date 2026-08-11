# Front 13 — Evolutionary Algorithm Discovery (AlphaEvolve Engine)

Spec Ref: Post-Refactor Cognitive Expansion

## Overview
Integrates evolutionary algorithm search (inspired by Google DeepMind's AlphaEvolve) directly into the HYPOSTASES mechanism design loop (Front 10). Instead of evaluating static code or hardcoded heuristics, the agent swarm autonomously generates, mutates, and refines executable algorithms against the environment oracle.

## Evolutionary Architecture
```
Initial Seed Algorithm / Heuristic
    ↓
LLM / Heuristic Mutator (Generates Candidate Code)
    ↓
HYPOSTASES Multi-Agent Simulation Engine (Evaluator Oracle)
    ↓
Game-Theoretic & Scarcity Feedback (step_env → feedback → evolve)
    ↓
Meta-Learning Selection (Front 07)
    ↓
Consolidated Skill Artifact (Front 03)
```

## Unique Advantage over AlphaEvolve
Standard AlphaEvolve evaluates code against single-agent scalar benchmarks. A HYPOSTASES-powered AlphaEvolve engine evaluates candidate algorithms against:
- Multi-agent game-theoretic equilibrium
- Endogenous resource scarcity ($\kappa$)
- Adversarial peer dynamics & institutional crowding-out
- Dynamic goal hierarchies ($g.u$)

## Core State Constraint
Evolved algorithm candidates manipulate primitive state $\sigma = (c, w, g, \rho_{\text{ext}})$ and are stored as compiled `SkillArtifact` objects within procedural memory.
