# Wave 3 Front 05 — Pertinent Literature & Synthesis

This document provides an exhaustive theoretical synthesis of the literature ingested into `docs/WAVE_3_FRONT_05/papers/` for building **Front 05: Institution Layer** in the HYPOSTASES engine.

---

## 1. Theoretical Foundations & Taxonomy of Institutions

### 1.1 The ADICO Grammar of Institutions (Crawford & Ostrom 1995; Ostrom et al. 1992)
Ostrom's Institutional Analysis and Development (IAD) framework categorizes institutional statements into five syntax components (ADICO):

$$\text{Institutional Statement} = \langle \mathbf{A}, \mathbf{D}, \mathbf{I}, \mathbf{C}, \mathbf{O} \rangle$$

- **Attributes ($\mathbf{A}$)**: The subset of agents or roles to which the rule applies ($\text{proj}_{\text{role}}(i) \in \mathbf{A}$).
- **Deontic ($\mathbf{D}$)**: The modal operator governing the action ($\mathbf{D} \in \{\text{MUST}, \text{MAY}, \text{MUST\_NOT}\}$).
- **Aim ($\mathbf{I}$)**: The target action or physical state outcome specified in action space $A$.
- **Condition ($\mathbf{C}$)**: The state predicate condition $C(\sigma)$ under which the rule is active.
- **Or Else ($\mathbf{O}$)**: The formal sanction function executed upon deontic violation, levying reserve penalties or authority downgrades.

Institutional statements are classified according to their component composition:
1. **Shared Strategies ($\mathbf{AIC}$)**: Regularized behavioral patterns followed out of individual interest without deontic operators or sanctions.
2. **Norms ($\mathbf{ADIC}$)**: Shared moral/social obligations enforced via informal social disapproval without explicit centralized sanction clauses.
3. **Rules ($\mathbf{ADICO}$)**: Enforceable formal prescriptions containing explicit sanction clauses $\mathbf{O}$.

### 1.2 Common Pool Resource (CPR) Dilemmas & Endogenous Covenants (Ostrom et al. 1992)
In a CPR game with $n$ agents investing extraction effort $x_i$, individual payoff is defined as:

$$u_i(x) = w(e - x_i) + \frac{x_i}{\sum_{j=1}^n x_j} F\left(\sum_{j=1}^n x_j\right)$$

Unregulated Nash equilibrium leads to severe over-harvesting ($x_i^* = \frac{a - w}{(n+1)b}$), resulting in dissipation of economic rent. Ostrom et al. empirically establish that:
- Centralized top-down coercion without communication ("sword without covenants") leads to blind retaliation and economic loss (~37% net yield).
- Endogenous governance ("covenants with an internal sword") combined with face-to-face or protocol-driven communication achieves ~93% net yield with minimal defection.

---

## 2. Peer Sanctioning & Altruistic Punishment Dynamics (Fehr & Gächter 2002; Bowles & Gintis 2002)

### 2.1 Quantitative Altruistic Punishment Calibration
Fehr & Gächter (2002) demonstrate that cooperation in public goods dilemmas requires explicit sanctioning mechanisms. Key parameters ingested into the engine:
- **Cost-to-Impact Ratio**: $1 : 3$ (each unit spent by punisher reduces defector's payoff by $3$ units).
- **`PUNISH_RESERVE_COST`**: Represented as $\gamma_{\text{punish}} = 0.333$.
- **Engine State Coupling**: Executing a sanction depletes internal reserve $c_{\text{enforcer}}$ while reducing the target's capacity $c_{\text{target}}$:

$$\Delta c_{\text{enforcer}} = -\text{cost}, \quad \Delta c_{\text{target}} = -3 \cdot \text{cost}$$

Crucially, under Rule 005 (AGENTS.md), altruistic punishment is NOT modeled as emotional irrationality, but as game-theoretic equilibrium selection where punishers enforce long-term collective governance state $g$ to protect future expected utility.

---

## 3. Formal Mechanism Design & Algorithmic Enforcement (Hurwicz 1973; Nisan & Ronen 2001)

### 3.1 Informational Decentralization & Privacy (Hurwicz 1973)
An institution $I_k$ is formalized as a mechanism $(M, f, h)$:
- $M = \prod_{i=1}^n M_i$: Combined message space available to participants.
- $f_i: M \times S \to M_i$: Agent strategy response rule.
- $h: M \to \Sigma$: Outcome function updating system state.

Hurwicz proves that informationally decentralized mechanisms must respect participant privacy (agents know only their private state $\sigma_i$).

### 3.2 Algorithmic Verification & Payments (Nisan & Ronen 2001)
Nisan & Ronen extend mechanism design to computational environments through:
- **Vickrey-Groves-Clarke (VGC) Payment Rules**:
  
  $$p^i(t) = \sum_{j \neq i} v^j(t^j, o(t)) + h^i(t^{-i})$$

  guaranteeing dominant-strategy incentive compatibility for truthful reporting.
- **Verification Loops**: Post-execution checking enables compensation-and-bonus functions $p^i(t, \tilde{t}) = c^i(t, \tilde{t}) + b^i(t, \tilde{t})$, ensuring that physical actions $\tilde{t}$ match declared strategies.

---

## 4. Synthesis for HYPOSTASES Wave 3 Front 05

The ingested literature yields three unified architectural pillars:
1. **ADICO Executable Rule Engine**: Direct implementation of Crawford & Ostrom's ADICO syntax for evaluating norms and rules over agent state $\sigma$.
2. **`InstitutionAgent` Abstraction**: Hurwicz mechanism wrapper managing private message spaces $M$, roles, authority matrices $\rho_{\text{auth}}$, and VCG allocations.
3. **Altruistic Sanction & Governance Manager**: Fehr & Gächter 1:3 cost-to-impact sanction levying, enforcing CPR optimal yields and preventing free-rider decay without violating Rule 005.
