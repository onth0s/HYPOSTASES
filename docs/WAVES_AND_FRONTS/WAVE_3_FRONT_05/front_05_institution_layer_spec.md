# Front 05 Master Specification — Institution Layer

**Status**: RATIFIED MASTER SPECIFICATION (All 5 Literature PDFs Ingested & Synthesized)  
**Wave**: Wave 3 (Social Epistemology & Swarm Mechanics)  
**Front**: Front 05 — Institution Layer  
**Target Substrate**: HYPOSTASES Engine ($\sigma = (c, w, g, \rho_{\text{ext}})$)  
**Rule 005 Compliance**: Pure Game-Theoretic & Probabilistic Rationality (Zero Artificial Human Cognitive Biases)

---

## 1. Ingested Literature Foundation (`docs/WAVE_3_FRONT_05/papers/`)

This master specification synthesizes theoretical mechanisms, mathematical formulations, and computational structures from **all 5 foundational PDF papers** ingested into `docs/WAVE_3_FRONT_05/papers/`:

| Ingested PDF File | Source & Core Theoretical Synthesis |
|---|---|
| [`ostrom_walker_gardner_covenants_sword_1992.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_3_FRONT_05/papers/ostrom_walker_gardner_covenants_sword_1992.pdf) | **Ostrom et al. (1992) — CPR Games & Endogenous Covenants**: Common Pool Resource extraction game; symmetric Nash dissipation vs. group optimal extraction; ADICO grammar syntax ($\mathbf{A-D-I-C-O}$); endogenous covenants with internal sanctioning achieving ~93% net yield. |
| [`fehr_gaechter_altruistic_punishment_2002.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_3_FRONT_05/papers/fehr_gaechter_altruistic_punishment_2002.pdf) | **Fehr & Gächter (2002) — Altruistic Peer Punishment**: 1:3 cost-to-impact sanction ratio; empirical calibration for `PUNISH_RESERVE_COST` ($\gamma_{\text{punish}} = 0.333$); contribution decay without sanctioning vs. sustainable high cooperation under active sanction options. |
| [`bowles_gintis_social_capital_community_governance_2002.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_3_FRONT_05/papers/bowles_gintis_social_capital_community_governance_2002.pdf) | **Bowles & Gintis (2002) — Community Governance & Strong Reciprocity**: Community governance solving incomplete contract failures; peer monitoring; informal Norms vs. formal Rules; authority distribution across agent networks. |
| [`hurwicz_mechanisms_for_resource_allocation_1973.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_3_FRONT_05/papers/hurwicz_mechanisms_for_resource_allocation_1973.pdf) | **Hurwicz (1973) — Formal Mechanism Design**: Mechanism triad $(M, f, h)$; message space privacy; Pareto-satisfactoriness; incentive compatibility bounds in decentralized allocation. |
| [`nisan_ronen_algorithmic_mechanism_design_2001.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_3_FRONT_05/papers/nisan_ronen_algorithmic_mechanism_design_2001.pdf) | **Nisan & Ronen (2001) — Algorithmic Mechanism Design & Verification**: VCG truthful payment mechanisms; task scheduling allocation; execution verification loops enabling post-execution bonus/penalty functions. |

---

## 2. Integrated Theoretical Architecture & Mathematical Formalism

```
             Observation & Communication Stream over σ = (c, w, g, ρ_ext)
                                  │
                                  ▼
        ┌───────────────────────────────────────────────────┐
        │ 1. ADICO Grammar Engine (Ostrom 1992/1995)         │
        │    - Rule: <Attributes, Deontic, Aim, Condition, OrElse>│
        │    - Deontics: MUST, MAY, MUST_NOT               │
        │    - Evaluates Condition C(σ) -> Target Deontic   │
        └─────────────────────────┬─────────────────────────┘
                                  │ Violation Detected (Or Else Clause)
                                  ▼
        ┌───────────────────────────────────────────────────┐
        │ 2. Altruistic Sanction Engine (Fehr & Gächter 2002)│
        │    - 1:3 Cost-to-Impact Ratio (PUNISH_RESERVE_COST)│
        │    - Δc_enforcer = -cost                          │
        │    - Δc_target = -3 * cost                       │
        │    - Δg = +governance_recovery                   │
        └─────────────────────────┬─────────────────────────┘
                                  │ Authority & Mechanism State
                                  ▼
        ┌───────────────────────────────────────────────────┐
        │ 3. InstitutionAgent (Hurwicz 1973 Mechanism Design)│
        │    - Message Space M, Response Rules f_i          │
        │    - Resources r_inst, Membership Roster           │
        │    - VCG / Algorithmic Allocation (Nisan & Ronen) │
        │    - Authority Matrix ρ_auth                       │
        └─────────────────────────┬─────────────────────────┘
                                  │ Presets & Swarm State
                                  ▼
        ┌───────────────────────────────────────────────────┐
        │ 4. GovernanceManager                              │
        │    - Orchestrates Government, Market, Guild, Court│
        │    - Manages Charters, Dispute Resolution          │
        │    - Prevents Tyranny & Crowding-Out              │
        └───────────────────────────────────────────────────┘
```

### 2.1 ADICO Grammar & Executable Rules
Following Ostrom et al. (1992, 1995), an institutional rule is formally represented as:

$$\mathbf{R} = \langle \mathbf{A}, \mathbf{D}, \mathbf{I}, \mathbf{C}, \mathbf{O} \rangle$$

- **Attribute Predicate $\mathbf{A}(i)$**: True iff agent $i$ possesses the designated role or membership in institution $I_k$.
- **Deontic Operator $\mathbf{D} \in \{\text{MUST}, \text{MAY}, \text{MUST\_NOT}\}$**: Dictates required, permissible, or prohibited action categories in action space $A$.
- **Aim $\mathbf{I}$**: Action matching pattern or target physical state update.
- **Condition $\mathbf{C}(\sigma)$**: Boolean predicate over current state $\sigma = (c, w, g, \rho_{\text{ext}})$.
- **Or Else $\mathbf{O}$**: Execution function levying sanction penalties when $\mathbf{A}(i) \land \mathbf{C}(\sigma) \land \neg \text{Complies}(\mathbf{D}, \mathbf{I})$.

### 2.2 Altruistic Punishment Calibration
Following Fehr & Gächter (2002), sanction enforcement incurs a reserve cost on the enforcer/authority defined by `PUNISH_RESERVE_COST` ($\gamma_{\text{punish}} = 0.333$):

$$\Delta c_i = -x, \quad \Delta c_j = -3x$$

where $i$ is the enforcing entity and $j$ is the violating defector. Altruistic punishment increases collective governance stability $g$ by preventing free-rider decay in public goods games.

### 2.3 `InstitutionAgent` State & Authority Matrix
Following Hurwicz (1973) and Nisan & Ronen (2001), an institution operates as a specialized agent:

```python
@dataclass
class InstitutionState:
    institution_id: str
    archetype: InstitutionArchetype  # GOVERNMENT, MARKET, GUILD, COURT, PROTOCOL
    resources: float  # Capital pool r_inst
    authority_matrix: dict[str, float]  # rho_auth mapping roles/agents to authority scores
    members: set[str]
    roles: dict[str, InstitutionalRole]
    rules: list[ADICORule]
```

---

## 3. Data-Driven Configuration (`schema/institution_layer_config.yaml`)

Presets for Government, Market, Guild, Court, Protocol, and Treaty archetypes reside in `schema/institution_layer_config.yaml`.

---

## 4. Module Structure & Implementation Files

- **Master Specification**: [`docs/WAVE_3_FRONT_05/front_05_institution_layer_spec.md`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_3_FRONT_05/front_05_institution_layer_spec.md)
- **Paper Manifest**: [`docs/WAVE_3_FRONT_05/papers_manifest.md`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_3_FRONT_05/papers_manifest.md)
- **Literature Summary**: [`docs/WAVE_3_FRONT_05/pertinent_literature.md`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_3_FRONT_05/pertinent_literature.md)
- **YAML Preset Schema**: [`schema/institution_layer_config.yaml`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/schema/institution_layer_config.yaml)
- **Core Engine Module**: `src/hypostases/institutions/`
  - `types.py`
  - `adico_engine.py`
  - `institution_agent.py`
  - `governance_manager.py`
  - `__init__.py`
- **Pytest Suite**: [`tests/test_institution_layer.py`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/tests/test_institution_layer.py)
