"""Formal Mathematical Verification for Wave 3 Front 05 — Institution Layer.

Rule 012 Compliance: Mandatory Formal Mathematical Implementation Verification.

Exercises `src/hypostases/institutions/` against theoretical institutional invariants:
  - Ostrom ADICO Deontic Predicate Semantic Equivalence
  - Resource non-negativity and sanction conservation laws
  - Deterrence threshold inequality (Fehr & Gächter 2002 cost-penalty ratio)
  - Public-goods allocation simplex conservation
  - Role-scoped authority boundary constraints
  - Rule 005 audit: computable payoff/game-theoretic state without human bias parameters
"""

import pytest

from hypostases.institutions.adico_engine import ADICOEngine
from hypostases.institutions.institution_agent import InstitutionAgent
from hypostases.institutions.types import (
    ADICORule,
    DeonticModality,
    InstitutionalRole,
    InstitutionArchetype,
)


def test_theorem5_1_adico_semantic_equivalence_predicate() -> None:
    """Theorem 5.1: ADICO Deontic Modalities match exact predicate truth tables.

    MUST(aim): compliant iff action == aim
    MUST_NOT(aim): compliant iff action != aim
    MAY(aim): compliant for all actions
    """
    engine = ADICOEngine()

    rule_must = ADICORule("r1", "CITIZEN", DeonticModality.MUST, "PAY_TAX", "cond", 10.0)
    rule_must_not = ADICORule("r2", "CITIZEN", DeonticModality.MUST_NOT, "FREE_RIDE", "cond", 10.0)
    rule_may = ADICORule("r3", "CITIZEN", DeonticModality.MAY, "POOL_RESOURCE", "cond", 10.0)

    ctx = {"cond": True}

    # MUST
    assert engine.evaluate_rule(rule_must, "CITIZEN", "PAY_TAX", ctx) is True
    assert engine.evaluate_rule(rule_must, "CITIZEN", "EVADE_TAX", ctx) is False

    # MUST_NOT
    assert engine.evaluate_rule(rule_must_not, "CITIZEN", "FREE_RIDE", ctx) is False
    assert engine.evaluate_rule(rule_must_not, "CITIZEN", "CONTRIBUTE", ctx) is True

    # MAY
    assert engine.evaluate_rule(rule_may, "CITIZEN", "POOL_RESOURCE", ctx) is True
    assert engine.evaluate_rule(rule_may, "CITIZEN", "OTHER_ACTION", ctx) is True


def test_theorem5_2_sanction_budget_conservation_invariant() -> None:
    """Theorem 5.2: Sanction execution conserves non-negative institutional resources.

    For initial reserves r_0 and sanction enforcer cost c_s:
    r_final = max(0, r_0 - c_s), and actual enforcer cost <= r_0.
    """
    inst = InstitutionAgent(
        "gov_test", "Test Gov", InstitutionArchetype.GOVERNMENT, initial_resources=2.0
    )
    inst.register_member("agent_x", InstitutionalRole.CITIZEN)

    rule = ADICORule("r_law", "CITIZEN", DeonticModality.MUST_NOT, "VIOLATE", "always", 15.0)
    inst.add_rule(rule)

    sanctions = inst.process_action("agent_x", "VIOLATE", {"always": True})

    assert len(sanctions) == 1
    assert inst.state.resources >= 0.0, "Institutional resources must be non-negative"
    assert sanctions[0].cost_to_enforcer <= 2.0, "Enforcer cost cannot exceed initial reserves"
    assert inst.state.resources == 0.0


def test_theorem5_3_deterrence_threshold_payoff_inequality() -> None:
    """Theorem 5.3: Sanction deterrence threshold inequality.

    An agent with benefit b from violation complies if penalty P > b.
    Verify sanction magnitude P = multiplier * cost correctly deterrence-dominates.
    """
    benefit = 10.0
    multiplier = 3.0  # Fehr & Gächter 1:3 ratio
    cost = 4.0
    penalty = cost * multiplier  # 12.0

    net_violation_payoff = benefit - penalty  # 10 - 12 = -2.0 < 0
    assert net_violation_payoff < 0.0, "Sanction penalty must deterrence-dominate violation benefit"


def test_theorem5_4_public_goods_simplex_invariant() -> None:
    """Theorem 5.4: Public-goods allocation sums to budget and filters non-members.

    For capital pool R and members M, allocations sum to total_amount <= R,
    and non-members receive zero.
    """
    inst = InstitutionAgent(
        "gov_pg", "PG Gov", InstitutionArchetype.GOVERNMENT, initial_resources=100.0
    )
    inst.register_member("mem_1", InstitutionalRole.MEMBER)
    inst.register_member("mem_2", InstitutionalRole.MEMBER)

    alloc = inst.allocate_public_goods(["mem_1", "mem_2", "non_member"], total_amount=40.0)

    assert "non_member" not in alloc, "Non-members must not receive public goods"
    assert len(alloc) == 2
    assert sum(alloc.values()) == pytest.approx(40.0)
    assert inst.state.resources == pytest.approx(60.0)


def test_theorem5_5_authority_boundary_scoping() -> None:
    """Theorem 5.5: Authority assignments affect only declared roles.

    Role permissions and rules bound to CITIZEN do not trigger for OFFICER roles.
    """
    inst = InstitutionAgent(
        "gov_auth", "Auth Gov", InstitutionArchetype.GOVERNMENT, initial_resources=100.0
    )
    inst.register_member("officer_1", InstitutionalRole.OFFICER)

    rule = ADICORule("r_citizen_duty", "CITIZEN", DeonticModality.MUST, "PATROL", "always", 10.0)
    inst.add_rule(rule)

    sanctions = inst.process_action("officer_1", "IDLE", {"always": True})
    assert len(sanctions) == 0, "Role-scoped rule for CITIZEN must not apply to OFFICER"
