"""Tests — Full Goal-Dimension Feedback Coverage (Gate tests).

Verifies that all four goal-hierarchy dimensions receive nonzero delta_g
updates from their respective action types in feedback().

Must pass before the full 3x3 grid sweep is run.
"""

from __future__ import annotations

import numpy as np

from hypostases.engine.constants import (
    ACQUISITION_U_GAIN,
    STATUS_U_GAIN,
    SURVIVAL_U_GAIN,
)
from hypostases.engine.dynamics import _ACQUISITION_IDX, _STATUS_IDX, _SURVIVAL_IDX, feedback
from hypostases.engine.types import (
    Action,
    ActionType,
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)


def _make_agent(sociality: float = 0.3, reserve: float = 10.0) -> AgentState:
    return AgentState(
        c=Characteristics(reserve=reserve, sociality=sociality),
        w=WorldModel(),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )


def _base_delta_log(
    pool_before: float = 10.0, pool_after: float = 10.0, granted: dict | None = None
) -> dict:
    return {
        "pool_before": pool_before,
        "pool_after_shares": pool_after,
        "pool_after": pool_after,
        "shares_total": 0.0,
        "requests_total": 2.0,
        "granted": granted or {},
        "punishments": {},
        "enable_withdraw_fee": False,
        "actions_log": {},
    }


class TestFullGoalFeedback:
    def test_request_full_grant_gives_positive_survival(self):
        """Full grant on REQUEST yields positive delta_g[SURVIVAL]."""
        agent = _make_agent()
        action = Action(ActionType.REQUEST, amount=2.0)
        dl = _base_delta_log(granted={"agent": 2.0})

        phi = feedback(agent, 10.0, 10.0, action, dl, agent_name="agent")

        assert phi.delta_g[_SURVIVAL_IDX] > 0, (
            f"Full grant should give positive SURVIVAL delta, got {phi.delta_g[_SURVIVAL_IDX]}"
        )
        assert np.isclose(phi.delta_g[_SURVIVAL_IDX], SURVIVAL_U_GAIN * (2.0 * 1.0 - 1.0))

    def test_request_zero_grant_gives_negative_survival(self):
        """Zero grant on REQUEST yields negative delta_g[SURVIVAL]."""
        agent = _make_agent()
        action = Action(ActionType.REQUEST, amount=2.0)
        dl = _base_delta_log(granted={"agent": 0.0})

        phi = feedback(agent, 10.0, 10.0, action, dl, agent_name="agent")

        assert phi.delta_g[_SURVIVAL_IDX] < 0, (
            f"Zero grant should give negative SURVIVAL delta, got {phi.delta_g[_SURVIVAL_IDX]}"
        )
        assert np.isclose(phi.delta_g[_SURVIVAL_IDX], SURVIVAL_U_GAIN * (2.0 * 0.0 - 1.0))

    def test_request_full_grant_gives_positive_acquisition(self):
        """Full grant on REQUEST yields positive delta_g[ACQUISITION]."""
        agent = _make_agent()
        action = Action(ActionType.REQUEST, amount=2.0)
        dl = _base_delta_log(granted={"agent": 2.0})

        phi = feedback(agent, 10.0, 10.0, action, dl, agent_name="agent")

        assert phi.delta_g[_ACQUISITION_IDX] > 0, (
            f"Full grant should give positive ACQUISITION delta, got {phi.delta_g[_ACQUISITION_IDX]}"
        )
        assert np.isclose(phi.delta_g[_ACQUISITION_IDX], ACQUISITION_U_GAIN * 1.0)

    def test_withdraw_no_fee_gives_positive_status(self):
        """WITHDRAW without fee gives positive delta_g[STATUS] for low-sociality agent."""
        agent = _make_agent(sociality=0.1)
        action = Action(ActionType.WITHDRAW, amount=2.0)
        dl = _base_delta_log()
        dl["enable_withdraw_fee"] = False

        phi = feedback(agent, 10.0, 10.0, action, dl, agent_name="agent")

        assert phi.delta_g[_STATUS_IDX] > 0, (
            f"WITHDRAW without fee should give positive STATUS delta, got {phi.delta_g[_STATUS_IDX]}"
        )
        expected = STATUS_U_GAIN * (1.0 - agent.c.sociality)
        assert np.isclose(phi.delta_g[_STATUS_IDX], expected)

    def test_withdraw_with_fee_gives_zero_status(self):
        """WITHDRAW with active governance fee cancels STATUS delta (positive - positive = 0)."""
        agent = _make_agent(sociality=0.1)
        action = Action(ActionType.WITHDRAW, amount=2.0)
        dl = _base_delta_log()
        dl["enable_withdraw_fee"] = True

        phi = feedback(agent, 10.0, 10.0, action, dl, agent_name="agent")

        assert np.isclose(phi.delta_g[_STATUS_IDX], 0.0, atol=1e-9), (
            f"WITHDRAW with fee should give ~0 STATUS delta, got {phi.delta_g[_STATUS_IDX]}"
        )
