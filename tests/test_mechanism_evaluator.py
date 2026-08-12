"""Unit tests for MechanismEvaluator and Welfare Aggregators (Wave 4 Front 10)."""

import pytest

from hypostases.mechanism_search.evaluator import MechanismEvaluator
from hypostases.mechanism_search.mechanism_space import (
    AllocationRule,
    MechanismCandidate,
    PaymentRule,
)


def test_gini_calculation():
    evaluator = MechanismEvaluator()
    # Equal distribution -> Gini = 0
    assert evaluator.compute_gini([10.0, 10.0, 10.0, 10.0]) == pytest.approx(0.0, abs=1e-5)
    # High inequality
    gini_high = evaluator.compute_gini([0.0, 0.0, 0.0, 100.0])
    assert gini_high > 0.70


def test_welfare_aggregators_all_modes():
    utilities = [10.0, 20.0, 30.0, 40.0]

    # 1. Productivity x Gini Equality
    eval_gini = MechanismEvaluator(aggregator_type="productivity_gini")
    res_gini = eval_gini.compute_social_welfare(utilities)
    assert res_gini["productivity"] == 100.0
    assert res_gini["welfare"] == pytest.approx(100.0 * res_gini["equality"], abs=1e-4)

    # 2. Rawlsian Max-Min
    eval_rawls = MechanismEvaluator(aggregator_type="rawlsian_maxmin")
    res_rawls = eval_rawls.compute_social_welfare(utilities)
    assert res_rawls["welfare"] == 10.0

    # 3. Pareto Efficiency
    eval_pareto = MechanismEvaluator(aggregator_type="pareto_efficiency")
    res_pareto = eval_pareto.compute_social_welfare(utilities, pareto_max=200.0)
    assert res_pareto["welfare"] == pytest.approx(0.5, abs=1e-4)

    # 4. Weighted Linear
    eval_linear = MechanismEvaluator(aggregator_type="weighted_linear")
    res_linear = eval_linear.compute_social_welfare(utilities)
    assert res_linear["welfare"] > 0.0


def test_ic_regret_vickrey_is_zero():
    evaluator = MechanismEvaluator()
    cand = MechanismCandidate(
        candidate_id="vickrey_cand",
        name="Vickrey Second Price",
        allocation_rule=AllocationRule(rule_type="highest_bidder"),
        payment_rule=PaymentRule(rule_type="second_highest_price"),
    )

    valuations = [10.0, 8.0, 5.0]
    bids = [10.0, 8.0, 5.0]
    state = {}

    regret = evaluator.compute_ic_regret(cand, bids, valuations, state)
    # DSIC mechanism -> zero IC regret
    assert regret == pytest.approx(0.0, abs=1e-3)


def test_evaluator_complete_scoring():
    evaluator = MechanismEvaluator(aggregator_type="productivity_gini")
    cand = MechanismCandidate(
        candidate_id="cand_score",
        name="Scored Candidate",
        allocation_rule=AllocationRule(rule_type="highest_bidder"),
        payment_rule=PaymentRule(rule_type="second_highest_price"),
    )

    valuations = [10.0, 8.0, 5.0]
    bids = [10.0, 8.0, 5.0]
    state = {}

    score = evaluator.evaluate_candidate(cand, bids, valuations, state)
    assert score > 0.0
    assert cand.fitness_score == score
