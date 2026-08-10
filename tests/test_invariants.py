"""Tests for HYPOSTASES Invariants & Schema Validation."""

import numpy as np
import pytest

from hypostases.engine import AgentState, Characteristics, GoalHierarchy, PowerExternal, WorldModel
from hypostases.schemas import InvariantViolationError, assert_invariants, validate_agent_state


def test_valid_agent_state_passes():
    agent = AgentState(
        c=Characteristics(skill=0.5, reserve=10.0, mood=0.2),
        w=WorldModel(sigma2=1.0),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(social_capital=1.0),
    )
    violations = validate_agent_state(agent)
    assert len(violations) == 0
    assert_invariants(agent)  # Should not raise


def test_negative_reserve_violates_invariant():
    agent = AgentState(
        c=Characteristics(reserve=-5.0),
        w=WorldModel(),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )
    violations = validate_agent_state(agent)
    assert any("c.reserve must be non-negative" in v for v in violations)

    with pytest.raises(InvariantViolationError, match="failed invariant validation"):
        assert_invariants(agent)


def test_boundary_reserve_zero_is_valid():
    agent = AgentState(
        c=Characteristics(reserve=0.0),
        w=WorldModel(),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )
    assert_invariants(agent)  # Should not raise


def test_boundary_mood_at_extremes():
    agent = AgentState(
        c=Characteristics(mood=-1.0),
        w=WorldModel(),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )
    assert_invariants(agent)

    agent.c.mood = 1.0
    assert_invariants(agent)


def test_non_positive_sigma2_violates_invariant():
    agent = AgentState(
        c=Characteristics(),
        w=WorldModel(sigma2=0.0),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )
    violations = validate_agent_state(agent)
    assert any("w.sigma2 must be strictly positive" in v for v in violations)


def test_invalid_mood_range_violates_invariant():
    agent = AgentState(
        c=Characteristics(mood=1.5),
        w=WorldModel(),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )
    violations = validate_agent_state(agent)
    assert any("c.mood must be in" in v for v in violations)


def test_negative_peer_belief_violates_invariant():
    agent = AgentState(
        c=Characteristics(),
        w=WorldModel(peer_beliefs={"peer1": -0.5}),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )
    violations = validate_agent_state(agent)
    assert any("w.peer_beliefs['peer1'] must be non-negative" in v for v in violations)
    with pytest.raises(InvariantViolationError):
        assert_invariants(agent)


def test_nan_utility_violates_invariant():
    agent = AgentState(
        c=Characteristics(),
        w=WorldModel(),
        g=GoalHierarchy(u=np.array([np.nan, 1.0, 1.0, 1.0])),
        rho_ext=PowerExternal(),
    )
    violations = validate_agent_state(agent)
    assert any("g.u must not contain NaN or Inf values" in v for v in violations)


def test_inf_replenish_violates_invariant():
    agent = AgentState(
        c=Characteristics(),
        w=WorldModel(replenish_rate_est=float("inf")),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )
    violations = validate_agent_state(agent)
    assert any("w.replenish_rate_est must be finite" in v for v in violations)
