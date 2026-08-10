"""Tests for HYPOSTASES Core Loop Dynamics."""

import numpy as np
import pytest

from hypostases.engine import (
    Action,
    ActionType,
    AgentState,
    Characteristics,
    DeltaLog,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
    evolve,
    feedback,
    pi_decision,
    step_env,
)


def test_step_env_shares_apply_first():
    actions = [
        ("Agent_A", Action(ActionType.SHARE, amount=5.0)),
        ("Agent_B", Action(ActionType.REQUEST, amount=12.0)),
    ]
    pool_before = 10.0
    pool_after, delta_log = step_env(pool_before, actions)

    # Pool after shares: 10 + 5 = 15. Request 12 granted in full.
    assert delta_log["pool_after_shares"] == 15.0
    assert delta_log["granted"]["Agent_B"] == 12.0
    assert pool_after == 3.0


def test_step_env_pro_rata_rationing():
    actions = [
        ("Agent_A", Action(ActionType.REQUEST, amount=10.0)),
        ("Agent_B", Action(ActionType.REQUEST, amount=10.0)),
    ]
    pool_before = 10.0
    pool_after, delta_log = step_env(pool_before, actions)

    # Pool 10 oversubscribed by requests 20. Rationing factor 0.5.
    assert delta_log["granted"]["Agent_A"] == 5.0
    assert delta_log["granted"]["Agent_B"] == 5.0
    assert pool_after == 0.0


def test_feedback_withdraw_branch_consequences():
    agent = AgentState(
        c=Characteristics(sociality=0.8, mood=0.0),
        w=WorldModel(),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(social_capital=1.0),
    )
    act = Action(ActionType.WITHDRAW)
    delta_log: DeltaLog = {
        "pool_before": 10.0,
        "pool_after_shares": 10.0,
        "pool_after": 10.0,
        "shares_total": 0.0,
        "requests_total": 0.0,
        "granted": {},
    }
    phi = feedback(agent, pool_before=10.0, pool_after=10.0, action=act, delta_log=delta_log)

    # WITHDRAW: reserve delta = 0, mood delta = -0.03 * sociality
    assert phi.delta_c["reserve"] == 0.0
    assert phi.delta_c["mood"] == -0.03 * 0.8
    assert phi.delta_rho_ext["social_capital"] == -0.01


def test_evolve_integrates_deltas_and_clamps():
    agent = AgentState(
        c=Characteristics(reserve=1.0, mood=-0.95),
        w=WorldModel(mu=5.0),
        g=GoalHierarchy(u=np.array([1.0, 1.0, 1.0, 1.0])),
        rho_ext=PowerExternal(social_capital=0.005),
    )
    act = Action(ActionType.SHARE, amount=5.0)
    delta_log: DeltaLog = {
        "pool_before": 10.0,
        "pool_after_shares": 10.0,
        "pool_after": 10.0,
        "shares_total": 0.0,
        "requests_total": 0.0,
        "granted": {},
    }
    phi = feedback(agent, 10.0, 10.0, act, delta_log)
    evolve(agent, phi)

    # Reserve cannot drop below zero
    assert agent.c.reserve == 0.0
    # Social capital integrated
    assert agent.rho_ext.social_capital > 0.0
    # Mood clamped in [-1.0, 1.0]
    assert -1.0 <= agent.c.mood <= 1.0


def test_pi_decision_deterministic_with_rng():
    agent = AgentState(
        c=Characteristics(),
        w=WorldModel(),
        g=GoalHierarchy(u=np.array([10.0, 0.0, 0.0, 0.0])),  # SURVIVAL dominant
        rho_ext=PowerExternal(),
    )
    rng = np.random.default_rng(123)
    xi = np.array([0.1, 0.1, 0.1, 0.1])
    action = pi_decision(agent, pool_belief=10.0, xi=xi, rng=rng)

    assert action.action_type == ActionType.REQUEST


def test_time_budget_decrements_on_evolve():
    agent = AgentState(
        c=Characteristics(),
        w=WorldModel(),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(time_budget=12.0),
    )
    act = Action(ActionType.WITHDRAW)
    phi = feedback(agent, 10.0, 10.0, act, {})
    evolve(agent, phi)
    assert agent.rho_ext.time_budget == 11.0


def test_peer_beliefs_updated_from_delta_log():
    agent_a = AgentState(
        c=Characteristics(),
        w=WorldModel(),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )
    delta_log = {"granted": {"Agent_A": 2.0, "Agent_B": 4.0}}
    act = Action(ActionType.REQUEST, amount=2.0)
    phi = feedback(agent_a, 10.0, 10.0, act, delta_log, agent_name="Agent_A")
    evolve(agent_a, phi)

    assert "Agent_B" in agent_a.w.peer_beliefs
    assert agent_a.w.peer_beliefs["Agent_B"] > 0.0


def test_world_model_sigma2_updated_and_clamped():
    agent = AgentState(
        c=Characteristics(),
        w=WorldModel(sigma2=1.0),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )
    act = Action(ActionType.WITHDRAW)
    # High surprise (observed pool change 5.0 vs predicted 1.0)
    phi = feedback(agent, pool_before=10.0, pool_after=15.0, action=act, delta_log={})
    evolve(agent, phi)
    assert agent.w.sigma2 > 0.0


def test_step_env_no_requests():
    actions = [
        ("Agent_A", Action(ActionType.SHARE, amount=5.0)),
        ("Agent_B", Action(ActionType.WITHDRAW)),
    ]
    pool_before = 10.0
    pool_after, delta_log = step_env(pool_before, actions)

    # 10 + 5 = 15. No requests granted.
    assert delta_log["pool_after_shares"] == 15.0
    assert delta_log["pool_after"] == 15.0
    assert not delta_log["granted"]
    assert pool_after == 15.0


def test_feedback_zero_amount_request():
    agent = AgentState(
        c=Characteristics(sociality=0.5, mood=0.0, resilience=0.5),
        w=WorldModel(),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )
    act = Action(ActionType.REQUEST, amount=0.0)
    delta_log = {"granted": {"agent": 0.0}}
    phi = feedback(agent, pool_before=10.0, pool_after=10.0, action=act, delta_log=delta_log)
    assert phi.delta_c["reserve"] == 0.0
    assert phi.delta_c["mood"] == 0.0


def test_evolve_peer_beliefs_ema_update():
    agent = AgentState(
        c=Characteristics(),
        w=WorldModel(peer_beliefs={"Agent_B": 10.0}),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )
    delta_log = {"granted": {"Agent_A": 2.0, "Agent_B": 0.0}}
    act = Action(ActionType.REQUEST, amount=2.0)
    phi = feedback(agent, 10.0, 10.0, act, delta_log, agent_name="Agent_A")
    evolve(agent, phi)
    # EMA update: PEER_BELIEF_ALPHA * 0.0 + (1 - PEER_BELIEF_ALPHA) * 10.0 = 0.3 * 0.0 + 0.7 * 10.0 = 7.0
    assert agent.w.peer_beliefs["Agent_B"] == pytest.approx(7.0)


def test_sigma2_clamps_to_minimum():
    from hypostases.engine.constants import SIGMA2_MIN

    agent = AgentState(
        c=Characteristics(memory_decay=1.0),
        w=WorldModel(sigma2=1e-5),  # very small
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )
    # Evolve with negative delta for sigma2 to drive it lower
    from hypostases.engine.types import FeedbackDelta

    phi = FeedbackDelta(delta_w={"sigma2": -1.0})
    evolve(agent, phi)
    assert agent.w.sigma2 == SIGMA2_MIN


def test_time_budget_reaches_zero():
    agent = AgentState(
        c=Characteristics(),
        w=WorldModel(),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(time_budget=2.0),
    )
    from hypostases.engine.types import FeedbackDelta

    phi = FeedbackDelta()
    evolve(agent, phi)
    assert agent.rho_ext.time_budget == 1.0
    evolve(agent, phi)
    assert agent.rho_ext.time_budget == 0.0
    evolve(agent, phi)
    assert agent.rho_ext.time_budget == 0.0  # Clamped to 0.0


def test_step_env_concurrency_operators():
    actions = [
        ("Agent_A", Action(ActionType.SHARE, amount=5.0)),
        ("Agent_B", Action(ActionType.REQUEST, amount=12.0)),
    ]
    p1, log1 = step_env(10.0, actions, concurrency_operator="shares-first")
    assert p1 == pytest.approx(3.0)
    assert log1["granted"]["Agent_B"] == pytest.approx(12.0)

    p2, log2 = step_env(10.0, actions, concurrency_operator="pro-rata")
    assert p2 == pytest.approx(5.0)
    assert log2["granted"]["Agent_B"] == pytest.approx(10.0)

    priorities = {"Agent_A": 10.0, "Agent_B": 5.0}
    p3, log3 = step_env(10.0, actions, concurrency_operator="priority", priorities=priorities)
    assert p3 == pytest.approx(3.0)
    assert log3["granted"]["Agent_B"] == pytest.approx(12.0)
