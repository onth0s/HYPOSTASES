"""Targeted regression and validation tests for code audit refactorings."""

import numpy as np
import pytest

import hypostases.engine.constants as const
from hypostases.engine import (
    Action,
    ActionType,
    AgentState,
    Characteristics,
    DeltaLog,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
    dynamic_action_costs,
    evolve,
    evolve_rb,
    feedback,
)
from hypostases.inference import goal_posterior
from hypostases.inference.particle_filter import Particle
from hypostases.schemas import assert_invariants
from hypostases.simulation.scenarios import create_scenario_agents


def test_scarcity_kappa_reads_const_directly(monkeypatch):
    """C-1 regression: dynamic_action_costs directly reads const.SCARCITY_COST_KAPPA."""
    base_costs = dynamic_action_costs(pool_belief=1.0)
    monkeypatch.setattr(const, "SCARCITY_COST_KAPPA", 2.0)
    high_kappa_costs = dynamic_action_costs(pool_belief=1.0)
    assert high_kappa_costs[0] > base_costs[0]


def test_evolve_and_evolve_rb_agree_on_non_world_fields():
    """S-1 regression: evolve and evolve_rb apply identical updates to c, g, rho_ext, peer_beliefs."""
    ag1 = AgentState(
        c=Characteristics(reserve=10.0, mood=0.5),
        w=WorldModel(),
        g=GoalHierarchy(u=np.array([1.0, 1.0, 1.0, 1.0])),
        rho_ext=PowerExternal(social_capital=1.0, time_budget=10.0),
    )
    ag2 = ag1.clone()

    act = Action(ActionType.SHARE, amount=2.0)
    delta_log: DeltaLog = {
        "pool_before": 10.0,
        "pool_after_shares": 10.0,
        "pool_after": 10.0,
        "shares_total": 2.0,
        "requests_total": 0.0,
        "granted": {},
    }
    phi = feedback(ag1, pool_before=10.0, pool_after=10.0, action=act, delta_log=delta_log)

    evolve(ag1, phi)
    evolve_rb(ag2, phi, surprise=0.0)

    # Non-world fields must match exactly
    assert ag1.c.reserve == pytest.approx(ag2.c.reserve)
    assert ag1.c.mood == pytest.approx(ag2.c.mood)
    assert np.allclose(ag1.g.u, ag2.g.u)
    assert ag1.rho_ext.social_capital == pytest.approx(ag2.rho_ext.social_capital)
    assert ag1.rho_ext.time_budget == pytest.approx(ag2.rho_ext.time_budget)


def test_status_withdraw_fee_negated_semantics():
    """C-4 verification: WITHDRAW feedback under active fee negates STATUS utility gain."""
    ag = AgentState(
        c=Characteristics(sociality=0.2),
        w=WorldModel(),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )
    act = Action(ActionType.WITHDRAW)

    log_no_fee: DeltaLog = {"enable_withdraw_fee": False}
    log_fee: DeltaLog = {"enable_withdraw_fee": True}

    phi_no_fee = feedback(ag, 10.0, 10.0, act, log_no_fee)
    phi_fee = feedback(ag, 10.0, 10.0, act, log_fee)

    # _STATUS_IDX is 3
    assert phi_no_fee.delta_g[3] > 0.0
    assert phi_fee.delta_g[3] < 0.0
    assert phi_fee.delta_g[3] == pytest.approx(-phi_no_fee.delta_g[3])


def test_scenario_registry_all_keys():
    """S-5 regression: All registered scenarios instantiate valid agent states."""
    scenarios = [
        "tragedy",
        "altruism",
        "freerider",
        "punishment",
        "inequity",
        "deceptive",
        "crowding_out",
    ]
    for name in scenarios:
        agents = create_scenario_agents(name)
        assert len(agents) > 0
        for ag_state in agents.values():
            assert_invariants(ag_state)


def test_scenario_registry_unknown_raises():
    """S-5 regression: Unknown scenario name raises ValueError with valid choices list."""
    with pytest.raises(ValueError, match="Unknown scenario 'nonexistent'"):
        create_scenario_agents("nonexistent")


def test_goal_posterior_uses_goal_probs():
    """S-6 verification: goal_posterior accepts xi and returns valid category probabilities."""
    p1 = Particle(
        sigma=AgentState(
            Characteristics(),
            WorldModel(),
            GoalHierarchy(u=np.array([10.0, 0.0, 0.0, 0.0])),
            PowerExternal(),
        ),
        weight=1.0,
    )
    post = goal_posterior([p1], xi=np.array([0.2, 0.2, 0.2, 0.2]), pool_belief=10.0)
    assert post["SURVIVAL"] == pytest.approx(1.0)
    assert pytest.approx(sum(post.values())) == 1.0
