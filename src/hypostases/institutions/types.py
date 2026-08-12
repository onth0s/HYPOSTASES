"""Data types, enums, and dataclasses for the Institution Layer (Front 05)."""

from dataclasses import dataclass, field
from enum import Enum


class DeonticModality(str, Enum):
    """Ostrom ADICO Deontic Modalities."""

    MUST = "MUST"
    MAY = "MAY"
    MUST_NOT = "MUST_NOT"


class InstitutionalRole(str, Enum):
    """Roles within institutional governance structures."""

    CITIZEN = "CITIZEN"
    MEMBER = "MEMBER"
    TRADER = "TRADER"
    OFFICER = "OFFICER"
    JUDGE = "JUDGE"
    LITIGANT = "LITIGANT"


class InstitutionArchetype(str, Enum):
    """Institutional entity archetypes."""

    GOVERNMENT = "GOVERNMENT"
    MARKET = "MARKET"
    GUILD = "GUILD"
    COURT = "COURT"
    PROTOCOL = "PROTOCOL"
    TREATY = "TREATY"


@dataclass
class ADICORule:
    """Ostrom ADICO Institutional Rule representation.

    Syntax: <Attributes, Deontic, Aim, Condition, OrElse>
    """

    rule_id: str
    attribute: str  # InstitutionalRole or attribute predicate
    deontic: DeonticModality
    aim: str  # Action identifier or state outcome aim
    condition: str  # Predicate name or expression evaluated over state
    or_else_penalty: float  # Sanction penalty levied on violation
    active: bool = True


@dataclass
class SanctionRecord:
    """Record of a levied sanction under ADICO governance."""

    sanction_id: str
    rule_id: str
    target_agent_id: str
    enforcer_id: str
    cost_to_enforcer: float
    penalty_to_target: float
    timestamp: int


@dataclass
class DisputeCase:
    """Dispute arbitration case managed by Court/Arbitrator institutions."""

    case_id: str
    complainant_id: str
    respondent_id: str
    claim_amount: float
    status: str = "OPEN"  # OPEN, RESOLVED, DISMISSED
    ruling_amount: float = 0.0


@dataclass
class InstitutionState:
    """State representation for an Institutional Agent operating over sigma."""

    institution_id: str
    name: str
    archetype: InstitutionArchetype
    resources: float  # Capital pool r_inst
    authority_matrix: dict[str, float] = field(default_factory=dict)  # agent_id -> rho_auth
    members: set[str] = field(default_factory=set)
    roles: dict[str, InstitutionalRole] = field(default_factory=dict)  # agent_id -> Role
    rules: list[ADICORule] = field(default_factory=list)
    sanctions_history: list[SanctionRecord] = field(default_factory=list)
    disputes: list[DisputeCase] = field(default_factory=list)
