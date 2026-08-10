"""Tests for HYPOSTASES Engine Types and v4 State Space Representations."""

import numpy as np
import pytest

from hypostases.engine import (
    Action,
    ActionType,
    Agent,
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)


def test_agent_state_four_tuple_primitives():
    c = Characteristics(reserve=10.0, mood=0.5)
    w = WorldModel(mu=8.0, sigma2=1.5)
    g = GoalHierarchy(u=np.array([1.0, 1.0, 1.0, 0.5]))
    rho_ext = PowerExternal(social_capital=2.0)

    agent = AgentState(c=c, w=w, g=g, rho_ext=rho_ext)

    assert agent.c.reserve == 10.0
    assert agent.w.mu == 8.0
    assert np.allclose(agent.g.u, [1.0, 1.0, 1.0, 0.5])
    assert agent.rho_ext.social_capital == 2.0


def test_agent_wrapper_identity():
    sigma = AgentState(
        c=Characteristics(reserve=10.0),
        w=WorldModel(),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )
    agent = Agent(name="Agent_A", sigma=sigma)
    assert agent.name == "Agent_A"
    assert agent.sigma.c.reserve == 10.0

    clone = agent.clone()
    clone.sigma.c.reserve = 99.0
    assert agent.sigma.c.reserve == 10.0
    assert clone.name == "Agent_A"


def test_goal_hierarchy_v4_transient_pi():
    g = GoalHierarchy(u=np.array([2.0, 0.0, 0.0, 0.0]))
    pi = g.pi
    assert len(pi) == 4
    assert pytest.approx(np.sum(pi)) == 1.0
    assert pi[0] > pi[1]  # Highest utility has highest softmax mass


def test_goal_hierarchy_invalid_shape():
    with pytest.raises(ValueError, match="u must be a vector of length 4"):
        GoalHierarchy(u=np.array([1.0, 2.0]))


def test_power_internal_is_derived_view():
    c = Characteristics(reserve=15.5)
    agent = AgentState(
        c=c,
        w=WorldModel(),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )
    p_int = agent.power_internal()
    assert p_int["reserve_capacity"] == 15.5

    # Mutating c.reserve updates derived view dynamically
    agent.c.reserve = 20.0
    assert agent.power_internal()["reserve_capacity"] == 20.0


def test_clone_independence():
    agent = AgentState(
        c=Characteristics(reserve=10.0),
        w=WorldModel(mu=5.0),
        g=GoalHierarchy(u=np.array([1.0, 1.0, 1.0, 1.0])),
        rho_ext=PowerExternal(social_capital=1.0),
    )
    clone = agent.clone()
    clone.c.reserve = 99.0
    clone.g.u[0] = 50.0

    assert agent.c.reserve == 10.0
    assert agent.g.u[0] == 1.0
    assert clone.c.reserve == 99.0


def test_action_representation():
    a_withdraw = Action(ActionType.WITHDRAW)
    a_req = Action(ActionType.REQUEST, amount=3.5)
    assert repr(a_withdraw) == "WITHDRAW"
    assert "REQUEST" in repr(a_req)
