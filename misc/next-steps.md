# HYPOSTASES — Cognitive Expansion Seeds (vNext)

This document is **not** a formal specification.

Its purpose is to capture high-level architectural directions for future iterations of HYPOSTASES. These are intentionally broad research vectors rather than implementation details.

---

# I. Hierarchical World Models

The current World Model primarily maintains beliefs over the environment and peer latent states.

Future iterations should investigate **multi-level semantic world representations**, allowing beliefs to exist across several abstraction layers.

Example hierarchy:

```
Environment
    ↓
Objects
    ↓
Relations
    ↓
Institutions
    ↓
Norms
    ↓
Meta-models
```

Potential capabilities:

- semantic abstraction
- relational reasoning
- institutional reasoning
- nested representations
- abstract situation understanding

---

# II. Explicit Planning Layer

Current architecture:

```
Goal
    ↓
Action
```

Future architecture:

```
Goal
    ↓
Plan
    ↓
Action
```

Plans become first-class reusable objects.

Potential capabilities:

- hierarchical planning
- reusable strategies
- contingency plans
- plan interruption
- plan repair
- plan libraries
- long-horizon reasoning

---

# III. Memory Architecture

Investigate explicit separation of memory systems.

Possible decomposition:

- Working Memory
- Episodic Memory
- Semantic Memory
- Procedural Memory

Potential capabilities:

- retrieval
- forgetting
- abstraction
- episodic replay
- skill acquisition
- memory consolidation
- analogical recall

---

# IV. Counterfactual Simulation

Rather than directly selecting an action, agents may internally simulate multiple hypothetical futures.

Conceptually:

```
Current State

↓

Future A

Future B

Future C

↓

Evaluation

↓

Execution
```

Potential capabilities:

- lookahead search
- planning under uncertainty
- hypothetical reasoning
- branch evaluation
- expected utility estimation
- Monte Carlo search

---

# V. Institution Layer

Institutions become explicit entities rather than purely emergent behavior.

Examples:

- governments
- markets
- guilds
- corporations
- courts
- protocols
- treaties

Institutions may themselves operate as agents with:

- goals
- resources
- governance rules
- memory
- authority
- decision policies

---

# VI. Communication as Bayesian Evidence

Messages should become probabilistic observations rather than deterministic information transfers.

Communication updates belief.

```
Message

↓

Likelihood

↓

Posterior Belief
```

Potential capabilities:

- trust
- deception
- reputation
- uncertainty
- misinformation
- evidence accumulation

---

# VII. Meta-Learning

Agents should eventually adapt not only their state but also their own internal reasoning mechanisms.

Potential adaptation targets:

- policy
- planner
- utility update
- learning rates
- inference parameters
- decision heuristics

Goal:

Learning how to learn.

---

# VIII. Causal World Models

Move beyond predictive relationships.

Represent causal structure explicitly.

Instead of

```
A predicts B
```

reason about

```
A causes B
```

Potential capabilities:

- intervention reasoning
- causal diagnosis
- counterfactual causality
- structural causal models
- policy evaluation

---

# IX. Active Information Gathering

Observation becomes an intentional action.

Possible epistemic actions:

- inspect
- observe
- query
- experiment
- probe
- monitor
- spy
- verify

Objective:

Trade immediate utility for uncertainty reduction.

---

# X. Mechanism Search

Move beyond simulating systems.

Search for better systems.

Instead of evaluating one institutional design:

```
Mechanism

↓

Simulation

↓

Evaluation
```

search over mechanism space:

```
Desired Objective

↓

Generate Mechanism

↓

Simulate

↓

Evaluate

↓

Modify

↓

Repeat
```

Potential applications:

- governance
- economics
- coordination
- institutional optimization
- public policy
- resource allocation

---

# XI. Abductive Reasoning

Current inverse inference estimates latent state.

Future work should investigate explicit abductive reasoning.

Question answered:

> "What explanation best accounts for these observations?"

Instead of storing only beliefs over state:

```
P(state)
```

reason over competing explanatory models:

```
H₁
H₂
H₃
...
```

Example:

Observation:

```
Resource pool shrinking
```

Possible explanations:

- overconsumption
- environmental degradation
- hidden agent
- sensor failure

Future observations refine belief over explanations.

---

# XII. Hypothesis Objects

Represent explanations as explicit computational objects.

Possible fields:

```
Hypothesis

identifier

description

assumptions

predictive model

prior

likelihood

posterior

complexity

supporting evidence

contradicting evidence

confidence
```

This enables reasoning over models rather than only states.

---

# XIII. Scientific Discovery Loop

Extend the cognitive cycle.

Current:

```
Observe

↓

Infer

↓

Act
```

Future:

```
Observe

↓

Infer

↓

Generate Hypotheses

↓

Rank Explanations

↓

Design Experiment

↓

Collect Evidence

↓

Update Hypotheses

↓

Act
```

Agents become capable of iterative model refinement.

---

# XIV. Design Philosophy

These directions intentionally preserve the current foundational architecture.

Persistent primitive state remains:

```
σ = (c, w, g, ρ_ext)
```

The proposed additions are **higher-order cognitive capabilities** built on top of this state rather than new persistent primitives.

Examples:

- planners
- memories
- hypothesis managers
- causal models
- mechanism search
- institution models

should be viewed as computational layers operating over the existing state representation.

This preserves architectural minimality while significantly increasing expressive power.

---

# Guiding Principle

HYPOSTASES should continue evolving from:

> "An engine that simulates agents."

toward

> "An engine that models intelligent reasoning, explanation, planning, institutional dynamics, and scientific discovery within a unified generative framework."
