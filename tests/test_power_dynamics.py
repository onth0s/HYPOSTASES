"""Unit tests for HYPOSTASES Power Dynamics and Projections (Part I §2.2.2, Part IV §6.6b)."""

from __future__ import annotations

import pytest

from hypostases.engine.power_dynamics import (
    compute_external_power_projection,
    compute_internal_power_projection,
    compute_total_power,
)
from hypostases.engine.types import (
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)


@pytest.fixture
def sample_agent() -> AgentState:
    return AgentState(
        c=Characteristics(reserve=12.5, skill=0.8, resilience=0.6, sociality=0.4, mood=0.1),
        w=WorldModel(mu=10.0, sigma2=2.0),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(time_budget=8.0, social_capital=4.0),
    )


def test_internal_power_projection(sample_agent: AgentState) -> None:
    """Verifies that proj_int(c) reads directly from c.reserve per §6.6b."""
    p_int = compute_internal_power_projection(sample_agent)
    assert p_int == 12.5
    assert p_int == sample_agent.c.reserve
    assert sample_agent.power_internal()["reserve_capacity"] == p_int


def test_external_power_projection(sample_agent: AgentState) -> None:
    """Verifies proj_ext(rho_ext) as convex combination of time_budget and social_capital."""
    # Default weights: 0.5 * 8.0 + 0.5 * 4.0 = 4.0 + 2.0 = 6.0
    p_ext = compute_external_power_projection(sample_agent)
    assert pytest.approx(p_ext) == 6.0

    # Custom weights
    p_ext_custom = compute_external_power_projection(sample_agent, w_time=0.25, w_social=0.75)
    # 0.25 * 8.0 + 0.75 * 4.0 = 2.0 + 3.0 = 5.0
    assert pytest.approx(p_ext_custom) == 5.0


def test_total_power_composition(sample_agent: AgentState) -> None:
    """Verifies total power projection interpolation P_total = alpha * P_int + (1 - alpha) * P_ext."""
    # alpha = 0.5: 0.5 * 12.5 + 0.5 * 6.0 = 6.25 + 3.0 = 9.25
    p_total = compute_total_power(sample_agent, alpha=0.5)
    assert pytest.approx(p_total) == 9.25

    # alpha boundary clamping
    p_int_only = compute_total_power(sample_agent, alpha=1.5)  # Clamped to 1.0
    assert pytest.approx(p_int_only) == 12.5

    p_ext_only = compute_total_power(sample_agent, alpha=-0.5)  # Clamped to 0.0
    assert pytest.approx(p_ext_only) == 6.0
