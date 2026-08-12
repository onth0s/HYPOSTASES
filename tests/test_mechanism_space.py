"""Unit tests for Mechanism Space & Candidates (Wave 4 Front 10)."""

from hypostases.mechanism_search.mechanism_space import (
    AllocationRule,
    GovernanceRule,
    MechanismCandidate,
    PaymentRule,
)


def test_mechanism_candidate_creation():
    alloc = AllocationRule(rule_type="highest_bidder", parameters={"reserve_price": 1.0})
    pay = PaymentRule(rule_type="second_highest_price")
    gov = GovernanceRule(punish_reserve_cost=0.1)

    cand = MechanismCandidate(
        candidate_id="mu_test_1",
        name="Test Vickrey Candidate",
        allocation_rule=alloc,
        payment_rule=pay,
        governance_rule=gov,
    )

    assert cand.candidate_id == "mu_test_1"
    assert cand.allocation_rule.rule_type == "highest_bidder"
    assert cand.payment_rule.rule_type == "second_highest_price"
    assert cand.governance_rule.punish_reserve_cost == 0.1


def test_allocation_and_payment_logic():
    alloc_rule = AllocationRule(rule_type="highest_bidder")
    payment_rule = PaymentRule(rule_type="second_highest_price")

    bids = [10.0, 5.0, 2.0]
    state = {}
    allocs = alloc_rule.allocate(bids, state)
    assert allocs == [1.0, 0.0, 0.0]

    payments = payment_rule.calculate_payments(bids, allocs, state)
    assert payments == [5.0, 0.0, 0.0]


def test_yaml_dual_persistence_roundtrip():
    alloc = AllocationRule(rule_type="virtual_valuation_max", parameters={"reserve_price": 2.0})
    pay = PaymentRule(rule_type="progressive_tax")
    gov = GovernanceRule(punish_reserve_cost=0.15)

    cand = MechanismCandidate(
        candidate_id="mu_yaml_1",
        name="YAML Persistence Test",
        allocation_rule=alloc,
        payment_rule=pay,
        governance_rule=gov,
        fitness_score=42.5,
        meta_parameters={"efe_mode": True, "aggregator": "productivity_gini"},
    )

    yaml_str = cand.to_yaml()
    deserialized = MechanismCandidate.from_yaml(yaml_str)

    assert deserialized.candidate_id == cand.candidate_id
    assert deserialized.allocation_rule.rule_type == "virtual_valuation_max"
    assert deserialized.payment_rule.rule_type == "progressive_tax"
    assert deserialized.governance_rule.punish_reserve_cost == 0.15
    assert deserialized.fitness_score == 42.5
    assert deserialized.meta_parameters["efe_mode"] is True
