"""Tests for Tier-0 Continuous Substrate Dynamics (Phase 1)."""

import numpy as np

from hypostases.engine import (
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
    step_continuous_agent,
    step_continuous_substrate,
)


def make_agent(reserve: float = 10.0) -> AgentState:
    return AgentState(
        c=Characteristics(reserve=reserve),
        w=WorldModel(),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )


def test_continuous_substrate_zero_dt():
    """dt <= 0 produces exact identity."""
    assert step_continuous_substrate(10.0, 0.0) == 10.0
    assert step_continuous_substrate(10.0, -1.0) == 10.0


def test_continuous_agent_zero_dt():
    """dt <= 0 produces no state mutation."""
    ag = make_agent(10.0)
    step_continuous_agent(ag, 0.0)
    assert ag.c.reserve == 10.0


def test_continuous_substrate_drift_and_noise():
    """Positive dt causes stochastic pool evolution bounded below by 0."""
    rng = np.random.default_rng(42)
    p1 = step_continuous_substrate(10.0, dt=1.0, rng=rng)
    assert p1 >= 0.0
    assert p1 != 10.0


def test_continuous_agent_reserve_decay():
    """Positive dt causes reserve decay over continuous time."""
    ag = make_agent(10.0)
    rng = np.random.default_rng(42)
    step_continuous_agent(ag, dt=2.0, rng=rng)
    # Reserve should decay from 10.0
    assert ag.c.reserve < 10.0
    assert ag.c.reserve >= 0.0
