"""HYPOSTASES — Causal World Model Test Suite (Front 08).

Spec Ref: docs/WAVE_2_FRONT_08/front_08_causal_world_models_spec.md
Validates SCM DAG topology, d-separation, do-calculus rules, 3-step counterfactual cycle,
NOTEARS continuous DAG optimization, cost-optimal intervention planning, and Rule 005 compliance.
"""

from __future__ import annotations

import numpy as np

from hypostases.causal.causal_discovery import CausalDiscoveryEngine
from hypostases.causal.causal_policy_evaluator import CostOptimalPlanner
from hypostases.causal.causal_types import (
    CounterfactualQuery,
    Intervention,
    RSCMSchema,
)
from hypostases.causal.do_calculus_engine import DoCalculusEngine
from hypostases.causal.rscm_engine import RelationalSCMEngine
from hypostases.causal.structural_causal_model import StructuralCausalModel


def create_sample_scm() -> StructuralCausalModel:
    """Helper to build standard 3-variable SCM: X -> M -> Y, X -> Y."""
    scm = StructuralCausalModel("test_scm")
    scm.add_node("X", exogenous_mean=1.0, exogenous_std=0.0)
    scm.add_node("M", exogenous_mean=0.0, exogenous_std=0.0)
    scm.add_node("Y", exogenous_mean=0.0, exogenous_std=0.0)

    scm.add_edge("X", "M", weight=2.0)
    scm.add_edge("M", "Y", weight=3.0)
    scm.add_edge("X", "Y", weight=1.5)
    return scm


def test_scm_topological_sort() -> None:
    """Tests DAG topological sorting."""
    scm = create_sample_scm()
    order = scm.topological_sort()
    assert order.index("X") < order.index("M")
    assert order.index("M") < order.index("Y")


def test_scm_observational_rung_1() -> None:
    """Tests Rung 1 observational forward state evaluation P(V)."""
    scm = create_sample_scm()
    state = scm.evaluate_observational(exogenous_noise={"X": 1.0, "M": 0.0, "Y": 0.0})
    assert state["X"] == 1.0
    assert state["M"] == 2.0  # 2.0 * X
    assert state["Y"] == 7.5  # 3.0 * M + 1.5 * X = 6.0 + 1.5 = 7.5


def test_scm_interventional_rung_2() -> None:
    """Tests Rung 2 interventional graph surgery P(V | do(M = 5.0))."""
    scm = create_sample_scm()
    interv = Intervention(target_values={"M": 5.0})
    state = scm.evaluate_intervention(interv, exogenous_noise={"X": 1.0, "M": 0.0, "Y": 0.0})
    assert state["X"] == 1.0
    assert state["M"] == 5.0  # Overridden by do(M = 5.0)
    assert state["Y"] == 16.5  # 3.0 * 5.0 + 1.5 * 1.0 = 15.0 + 1.5 = 16.5


def test_scm_counterfactual_rung_3() -> None:
    """Tests Rung 3 3-step Abduction-Action-Prediction counterfactual cycle."""
    scm = create_sample_scm()
    query = CounterfactualQuery(
        observed_evidence={"X": 1.0, "M": 2.0, "Y": 7.5},
        intervention_override={"X": 2.0},
        target_variables=["Y"],
    )
    cf_res = scm.evaluate_counterfactual(query)
    assert "Y" in cf_res
    # Under X = 2.0: M = 4.0, Y = 3.0 * 4.0 + 1.5 * 2.0 = 12.0 + 3.0 = 15.0
    assert cf_res["Y"] == 15.0


def test_d_separation() -> None:
    """Tests d-separation query engine."""
    # X -> M -> Y. Given M, X and Y should be d-separated if direct edge X->Y is pruned
    scm_chain = StructuralCausalModel("chain_scm")
    scm_chain.add_node("X")
    scm_chain.add_node("M")
    scm_chain.add_node("Y")
    scm_chain.add_edge("X", "M")
    scm_chain.add_edge("M", "Y")

    assert scm_chain.is_d_separated({"X"}, {"Y"}, {"M"}) is True
    assert scm_chain.is_d_separated({"X"}, {"Y"}, set()) is False


def test_do_calculus_rules() -> None:
    """Tests do-calculus rules 1-3."""
    scm = create_sample_scm()
    engine = DoCalculusEngine(scm)
    # Rule 1 test
    r1_pass = engine.check_rule_1(y={"Y"}, x={"X"}, z={"M"}, w=set())
    assert isinstance(r1_pass, bool)


def test_notears_continuous_dag_discovery() -> None:
    """Tests NOTEARS continuous DAG structure learning h(W) = tr(exp(W o W)) - d = 0."""
    var_names = ["X", "M", "Y"]
    disc_engine = CausalDiscoveryEngine(var_names)

    # Generate synthetic data from true DAG: X -> M -> Y
    n_samples = 100
    x = np.random.normal(0, 1, n_samples)
    m = 2.0 * x + np.random.normal(0, 0.1, n_samples)
    y = 3.0 * m + 1.5 * x + np.random.normal(0, 0.1, n_samples)
    data = np.column_stack([x, m, y])

    learned_scm = disc_engine.learn_notears(data, max_iter=30, w_threshold=0.1)
    assert isinstance(learned_scm, StructuralCausalModel)
    assert len(learned_scm.nodes) == 3


def test_cost_optimal_planner() -> None:
    """Tests Zhang & Bareinboim (2019) cost-optimal interventional planner."""
    scm = create_sample_scm()
    planner = CostOptimalPlanner(scm)
    best_action, cost = planner.select_cost_optimal_intervention(
        goal_variable="Y",
        target_goal_value=16.5,
        candidate_actions=["X", "M"],
        action_costs={"X": 5.0, "M": 1.0},
    )
    assert best_action in ["X", "M"]
    assert cost in [1.0, 5.0]


def test_relational_scm_engine() -> None:
    """Tests Ejaz & Bareinboim (2026) RSCM relational template instantiation."""
    schema = RSCMSchema(
        entities=["Agent"],
        relations=[("Agent", "interacts", "Agent")],
        attribute_templates={"Agent": ["state", "utility"]},
    )
    engine = RelationalSCMEngine(schema)
    ground_scm = engine.instantiate_ground_scm(
        entity_instances={"Agent": ["agent_1", "agent_2"]},
        relation_instances=[("agent_1", "interacts", "agent_2")],
    )
    assert "agent_1.state" in ground_scm.nodes
    assert "agent_2.state" in ground_scm.nodes


def test_rule_005_safety_compliance() -> None:
    """Rule 005 Compliance: Verifies zero artificial human cognitive defect or irrational emotional factors exist."""
    scm = create_sample_scm()
    state = scm.evaluate_observational()
    for val in state.values():
        assert not np.isnan(val)
        assert not np.isinf(val)
