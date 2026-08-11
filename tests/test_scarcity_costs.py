"""Tests — Contention 1: Endogenous Scarcity Action Costs.

Verifies that dynamic_action_costs and compute_omega correctly inflate
costs and reduce willingness when pool_belief is scarce.
"""

from __future__ import annotations

import numpy as np

from hypostases.engine._math import compute_omega, dynamic_action_costs
from hypostases.engine.constants import (
    ACTION_COSTS,
    SCARCITY_POOL_THRESHOLD,
)


class TestDynamicActionCosts:
    def test_above_threshold_returns_base_costs(self):
        """When pool >= threshold, costs are identical to ACTION_COSTS."""
        costs = dynamic_action_costs(SCARCITY_POOL_THRESHOLD)
        np.testing.assert_array_almost_equal(costs, ACTION_COSTS)

    def test_well_above_threshold_no_inflation(self):
        """Large pool produces negligible or zero inflation."""
        costs = dynamic_action_costs(1000.0)
        np.testing.assert_array_almost_equal(costs, ACTION_COSTS, decimal=4)

    def test_below_threshold_inflates_costs(self):
        """Scarcity below threshold inflates all costs by > 1×."""
        costs = dynamic_action_costs(1.0)
        assert np.all(costs > ACTION_COSTS), "All costs should inflate under scarcity"

    def test_zero_pool_inflates_maximally(self):
        """At pool ≈ 0, inflation is bounded but large."""
        costs_zero = dynamic_action_costs(0.0)
        costs_high = dynamic_action_costs(SCARCITY_POOL_THRESHOLD)
        assert np.all(costs_zero > costs_high)

    def test_inflation_monotone_in_scarcity(self):
        """Costs are monotonically increasing as pool decreases below threshold."""
        pools = [4.0, 3.0, 2.0, 1.0, 0.5]
        prev = ACTION_COSTS.copy()
        for p in pools:
            c = dynamic_action_costs(p)
            assert np.all(c >= prev), f"Costs should not decrease as pool drops: pool={p}"
            prev = c

    def test_kappa_zero_yields_base_costs(self, monkeypatch):
        """κ=0 disables scarcity inflation entirely."""
        import hypostases.engine.constants as const

        monkeypatch.setattr(const, "SCARCITY_COST_KAPPA", 0.0)
        costs = dynamic_action_costs(0.001)
        np.testing.assert_array_almost_equal(costs, ACTION_COSTS)


class TestComputeOmegaScarcity:
    def test_scarce_pool_reduces_willingness(self):
        """omega with scarce pool should be <= omega with abundant pool."""
        u = np.array([1.0, 1.0, 1.0, 1.0])
        reserve = 10.0
        omega_abundant = compute_omega(u, reserve, pool_belief=100.0)
        omega_scarce = compute_omega(u, reserve, pool_belief=0.5)
        assert np.all(omega_scarce <= omega_abundant + 1e-9)

    def test_omega_pool_belief_default_is_abundant(self):
        """Default pool_belief=10.0 >= threshold, so behavior is unchanged."""
        u = np.array([1.0, 1.0, 1.0, 1.0])
        reserve = 10.0
        omega_default = compute_omega(u, reserve)
        omega_explicit = compute_omega(u, reserve, pool_belief=10.0)
        np.testing.assert_array_almost_equal(omega_default, omega_explicit)

    def test_goal_probs_with_scarce_pool(self):
        """goal_probs with scarce pool_belief produces valid distribution."""
        from hypostases.engine.dynamics import goal_probs
        from hypostases.engine.types import (
            AgentState,
            Characteristics,
            GoalHierarchy,
            PowerExternal,
            WorldModel,
        )

        agent = AgentState(
            c=Characteristics(reserve=5.0),
            w=WorldModel(),
            g=GoalHierarchy(),
            rho_ext=PowerExternal(),
        )
        xi = np.array([0.2, 0.2, 0.2, 0.2])
        probs = goal_probs(agent, xi, pool_belief=1.0)
        assert abs(probs.sum() - 1.0) < 1e-6
        assert np.all(probs >= 0.0)
