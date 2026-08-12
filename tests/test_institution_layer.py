"""Unit test suite for Wave 3 Front 05 (Institution Layer)."""

import os

import pytest
import yaml

from src.hypostases.institutions import (
    ADICOEngine,
    ADICORule,
    DeonticModality,
    GovernanceManager,
    InstitutionAgent,
    InstitutionalRole,
    InstitutionArchetype,
)


def test_adico_rule_evaluation():
    engine = ADICOEngine(punish_reserve_cost=0.333)

    rule = ADICORule(
        rule_id="r1",
        attribute="CITIZEN",
        deontic=DeonticModality.MUST_NOT,
        aim="FREE_RIDE",
        condition="resource_scarce",
        or_else_penalty=12.0,
    )

    ctx_scarce = {"resource_scarce": True}
    ctx_abundant = {"resource_scarce": False}

    # Violation: Attempting FREE_RIDE when scarce
    assert not engine.evaluate_rule(rule, "CITIZEN", "FREE_RIDE", ctx_scarce)

    # Compliant: Doing something else when scarce
    assert engine.evaluate_rule(rule, "CITIZEN", "CONTRIBUTE", ctx_scarce)

    # Compliant: Condition does not apply
    assert engine.evaluate_rule(rule, "CITIZEN", "FREE_RIDE", ctx_abundant)


def test_altruistic_punishment_cost_ratio():
    engine = ADICOEngine(punish_reserve_cost=0.333)

    rule = ADICORule(
        rule_id="r_tax",
        attribute="CITIZEN",
        deontic=DeonticModality.MUST,
        aim="PAY_TAX",
        condition="has_income",
        or_else_penalty=15.0,
    )

    sanction = engine.execute_sanction(
        rule, target_agent_id="agent_1", enforcer_id="gov_1", timestamp=1
    )

    assert sanction.penalty_to_target == 15.0
    assert pytest.approx(sanction.cost_to_enforcer, 0.01) == 4.995  # 15.0 * 0.333


def test_institution_agent_membership_and_sanctions():
    gov = InstitutionAgent(
        "gov_1", "Test Government", InstitutionArchetype.GOVERNMENT, initial_resources=100.0
    )
    gov.register_member("agent_a", role=InstitutionalRole.CITIZEN)

    rule = ADICORule(
        rule_id="r_law",
        attribute="CITIZEN",
        deontic=DeonticModality.MUST_NOT,
        aim="STEAL",
        condition="always",
        or_else_penalty=10.0,
    )
    gov.add_rule(rule)

    sanctions = gov.process_action("agent_a", "STEAL", {"always": True}, timestamp=5)
    assert len(sanctions) == 1
    assert sanctions[0].target_agent_id == "agent_a"
    assert pytest.approx(gov.state.resources, 0.1) == 96.67  # 100 - (10 * 0.333)


def test_governance_manager_preset_loading():
    config_path = "schema/institution_layer_config.yaml"
    assert os.path.exists(config_path)

    with open(config_path, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    mgr = GovernanceManager(config=config_data)
    mgr.load_presets_from_config()

    assert "gov_primary" in mgr.institutions
    assert "mkt_primary" in mgr.institutions

    gov = mgr.get_institution("gov_primary")
    assert gov is not None
    assert gov.state.archetype == InstitutionArchetype.GOVERNMENT
    assert len(gov.state.rules) > 0


def test_dispute_resolution():
    court = InstitutionAgent("court_1", "Supreme Court", InstitutionArchetype.COURT)
    court.register_member("plaintiff", InstitutionalRole.LITIGANT)
    court.register_member("defendant", InstitutionalRole.LITIGANT)

    dispute = court.resolve_dispute(
        "plaintiff", "defendant", claim_amount=50.0, ruling_in_favor_of_complainant=True
    )
    assert dispute.status == "RESOLVED"
    assert dispute.ruling_amount == 50.0
    assert len(court.state.disputes) == 1
