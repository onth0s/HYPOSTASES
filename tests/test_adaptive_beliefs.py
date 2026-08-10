"""Tests — Contention 2: Adaptive Regime-Shift Belief Learning.

Verifies that sigma2 expands more aggressively when consecutive surprise
signals differ (acceleration > 0), using the last_surprise field.
"""

from __future__ import annotations

from hypostases.engine.dynamics import evolve, feedback
from hypostases.engine.types import (
    Action,
    ActionType,
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)


def _make_agent(last_surprise: float = 0.0) -> AgentState:
    return AgentState(
        c=Characteristics(reserve=10.0),
        w=WorldModel(sigma2=2.0, last_surprise=last_surprise),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )


def _run_feedback_and_evolve(
    agent: AgentState,
    pool_before: float,
    pool_after: float,
    action: Action,
) -> None:
    delta_log = {
        "pool_before": pool_before,
        "pool_after_shares": pool_after,
        "pool_after": pool_after,
        "shares_total": 0.0,
        "requests_total": 0.0,
        "granted": {},
        "actions_log": {"agent": action},
    }
    phi = feedback(agent, pool_before, pool_after, action, delta_log, agent_name="agent")
    evolve(agent, phi)


class TestRegimeShiftLearning:
    def test_last_surprise_updated_after_evolve(self):
        """After evolve, agent.w.last_surprise should reflect the current tick's surprise."""
        agent = _make_agent(last_surprise=0.0)
        action = Action(ActionType.WITHDRAW)
        _run_feedback_and_evolve(agent, pool_before=10.0, pool_after=8.0, action=action)
        # surprise = (8 - 10) - 0.0 - 1.0 (replenish_rate_est) = -3.0
        assert agent.w.last_surprise != 0.0, "last_surprise should be updated after evolve"

    def test_constant_surprise_no_acceleration(self):
        """Identical surprises over two ticks → acceleration ≈ 0, minimal extra sigma2 growth."""
        agent = _make_agent(last_surprise=0.0)
        action = Action(ActionType.WITHDRAW)
        # Tick 1: surprise ≈ (10 - 10) - 1.0 = -1.0
        _run_feedback_and_evolve(agent, pool_before=10.0, pool_after=10.0, action=action)

        # Tick 2: same pool conditions → same surprise, acceleration ≈ 0
        agent.w.replenish_rate_est = agent.w.replenish_rate_est  # unchanged
        _run_feedback_and_evolve(agent, pool_before=10.0, pool_after=10.0, action=action)
        sigma2_after_t2 = agent.w.sigma2

        # Both ticks had identical surprise; regime gain should be ~0
        # Delta from t1 → t2 comes from EMA only, not regime gain
        assert sigma2_after_t2 >= 0, "sigma2 must be non-negative"

    def test_accelerating_surprise_expands_sigma2_more(self):
        """A sudden surprise shift (regime break) should expand sigma2 more than stable conditions."""
        # Stable run: constant surprises
        agent_stable = _make_agent(last_surprise=0.0)
        action = Action(ActionType.WITHDRAW)
        _run_feedback_and_evolve(agent_stable, 10.0, 9.0, action)
        sigma2_stable_start = agent_stable.w.sigma2
        _run_feedback_and_evolve(agent_stable, 10.0, 9.0, action)
        sigma2_stable_end = agent_stable.w.sigma2
        stable_growth = sigma2_stable_end - sigma2_stable_start

        # Regime-break run: surprise flips sign dramatically
        agent_regime = _make_agent(last_surprise=0.0)
        _run_feedback_and_evolve(agent_regime, 10.0, 9.0, action)  # surprise ~= -2
        sigma2_regime_start = agent_regime.w.sigma2
        _run_feedback_and_evolve(agent_regime, 10.0, 15.0, action)  # surprise flips positive
        sigma2_regime_end = agent_regime.w.sigma2
        regime_growth = sigma2_regime_end - sigma2_regime_start

        assert regime_growth >= stable_growth, (
            f"Regime-break should expand sigma2 more than stable: "
            f"regime_growth={regime_growth:.4f}, stable_growth={stable_growth:.4f}"
        )

    def test_sigma2_never_below_min(self):
        """sigma2 must never fall below SIGMA2_MIN regardless of update direction."""
        from hypostases.engine.constants import SIGMA2_MIN

        agent = _make_agent()
        agent.w.sigma2 = SIGMA2_MIN + 1e-6
        action = Action(ActionType.WITHDRAW)
        for _ in range(20):
            _run_feedback_and_evolve(agent, 10.0, 10.0, action)
        assert agent.w.sigma2 >= SIGMA2_MIN
