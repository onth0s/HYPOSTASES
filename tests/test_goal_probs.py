"""Tests for HYPOSTASES Goal Probabilities Generative Model."""

import numpy as np
import pytest

from hypostases.engine import (
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
    goal_probs,
)


def test_goal_probs_simplex_invariant():
    agent = AgentState(
        c=Characteristics(reserve=10.0),
        w=WorldModel(),
        g=GoalHierarchy(u=np.array([1.0, 2.0, 0.5, 3.0])),
        rho_ext=PowerExternal(),
    )
    xi = np.array([0.2, 0.2, 0.2, 0.2])
    probs = goal_probs(agent, xi)

    assert len(probs) == 4
    assert (probs >= 0.0).all()
    assert pytest.approx(np.sum(probs)) == 1.0


def test_goal_probs_status_reserve_coupling():
    ag_low = AgentState(
        c=Characteristics(reserve=2.0),
        w=WorldModel(),
        g=GoalHierarchy(u=np.array([1.0, 1.0, 1.0, 1.0])),
        rho_ext=PowerExternal(),
    )
    ag_high = AgentState(
        c=Characteristics(reserve=20.0),
        w=WorldModel(),
        g=GoalHierarchy(u=np.array([1.0, 1.0, 1.0, 1.0])),
        rho_ext=PowerExternal(),
    )
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    p_low = goal_probs(ag_low, xi)
    p_high = goal_probs(ag_high, xi)

    # High reserve increases effective utility of STATUS (index 3)
    assert p_high[3] > p_low[3]
