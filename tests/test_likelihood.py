"""Tests for HYPOSTASES Action Likelihood & Generative Evaluation."""

import numpy as np
import pytest

from hypostases.engine import (
    Action,
    ActionType,
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
    action_likelihood,
    expected_action_type,
    predict_amount,
)
from hypostases.engine.constants import LIKELIHOOD_MIN


def make_agent(reserve: float = 10.0, u: np.ndarray | None = None) -> AgentState:
    if u is None:
        u = np.array([1.0, 1.0, 1.0, 1.0])
    return AgentState(
        c=Characteristics(reserve=reserve, sociality=0.5),
        w=WorldModel(),
        g=GoalHierarchy(u=u),
        rho_ext=PowerExternal(),
    )


def test_expected_action_type_all_goals():
    assert expected_action_type("SURVIVAL") == ActionType.REQUEST
    assert expected_action_type("ACQUISITION") == ActionType.REQUEST
    assert expected_action_type("RELATIONAL") == ActionType.SHARE
    assert expected_action_type("STATUS") == ActionType.WITHDRAW


@pytest.mark.parametrize(
    "goal,pool_belief,expected",
    [
        ("SURVIVAL", 10.0, 6.0),
        ("ACQUISITION", 10.0, 3.0),
        ("RELATIONAL", 10.0, 0.8),
        ("STATUS", 10.0, 0.0),
    ],
)
def test_predict_amount_parametrized(goal, pool_belief, expected):
    ag = make_agent(reserve=4.0)
    assert predict_amount(ag, goal, pool_belief=pool_belief) == pytest.approx(expected)


def test_action_likelihood_request():
    xi = np.array([0.2, 0.2, 0.2, 0.2])
    ag = make_agent(reserve=4.0, u=np.array([10.0, 0.0, 0.0, 0.0]))  # SURVIVAL dominant
    obs_action = Action(ActionType.REQUEST, amount=6.0)
    lik = action_likelihood(ag, obs_action, xi, pool_belief=10.0)
    assert lik > 0.3


def test_action_likelihood_share():
    xi = np.array([0.2, 0.2, 0.2, 0.2])
    ag = make_agent(reserve=4.0, u=np.array([0.0, 0.0, 10.0, 0.0]))  # RELATIONAL dominant
    obs_action = Action(ActionType.SHARE, amount=0.8)
    lik = action_likelihood(ag, obs_action, xi, pool_belief=10.0)
    assert lik > 0.3


def test_action_likelihood_withdraw():
    xi = np.array([0.2, 0.2, 0.2, 0.2])
    ag = make_agent(reserve=4.0, u=np.array([0.0, 0.0, 0.0, 10.0]))  # STATUS dominant
    obs_action = Action(ActionType.WITHDRAW, amount=0.0)
    lik = action_likelihood(ag, obs_action, xi, pool_belief=10.0)
    # WITHDRAW likelihood bypasses Gaussian and is 1.0 (multiplied by goal prob)
    assert lik == pytest.approx(1.0, rel=1e-2)


def test_action_likelihood_floor():
    xi = np.array([0.2, 0.2, 0.2, 0.2])
    ag = make_agent()
    # Mismatched extreme action
    obs_action = Action(ActionType.REQUEST, amount=999.0)
    lik = action_likelihood(ag, obs_action, xi)
    assert lik >= LIKELIHOOD_MIN


# --- Phase 2: _is_infeasible pruning gate ---


def test_infeasible_share_returns_likelihood_min():
    """SHARE amount exceeding reserve is physically impossible → LIKELIHOOD_MIN."""
    xi = np.array([0.2, 0.2, 0.2, 0.2])
    ag = make_agent(reserve=1.0)  # reserve too low to share 5.0
    obs_action = Action(ActionType.SHARE, amount=5.0)
    lik = action_likelihood(ag, obs_action, xi, pool_belief=10.0)
    assert lik == LIKELIHOOD_MIN


def test_feasible_share_exceeds_likelihood_min():
    """SHARE within reserve produces a real Gaussian score, above LIKELIHOOD_MIN."""
    xi = np.array([0.2, 0.2, 0.2, 0.2])
    ag = make_agent(reserve=10.0, u=np.array([0.0, 0.0, 10.0, 0.0]))  # RELATIONAL dominant
    obs_action = Action(ActionType.SHARE, amount=1.0)  # reserve=10 can afford 1.0
    lik = action_likelihood(ag, obs_action, xi, pool_belief=10.0)
    assert lik > LIKELIHOOD_MIN


def test_withdraw_always_feasible_regardless_of_reserve():
    """WITHDRAW passes the feasibility gate even with zero reserve."""
    xi = np.array([0.2, 0.2, 0.2, 0.2])
    ag = make_agent(reserve=0.0, u=np.array([0.0, 0.0, 0.0, 10.0]))  # STATUS dominant
    obs_action = Action(ActionType.WITHDRAW, amount=0.0)
    lik = action_likelihood(ag, obs_action, xi, pool_belief=0.0)
    # Should NOT be pruned — WITHDRAW is always feasible; expect real likelihood
    assert lik > LIKELIHOOD_MIN
