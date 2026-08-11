"""Tests — Softmax Jacobian Attenuation on delta_g.

Verifies that delta_g updates are attenuated by policy sensitivity pi_k * (1 - pi_k):
1. Dominant utility dimension (pi_k -> 1) yields attenuated delta_g -> 0.
2. Uniform utility distribution yields expected un-attenuated sensitivity (~0.1875).
"""

from __future__ import annotations

import numpy as np

from hypostases.engine.dynamics import feedback
from hypostases.engine.types import (
    Action,
    ActionType,
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)


def _make_agent(u_vec: np.ndarray) -> AgentState:
    gh = GoalHierarchy()
    gh.u = u_vec.copy()
    return AgentState(
        c=Characteristics(reserve=10.0, sociality=0.5),
        w=WorldModel(),
        g=gh,
        rho_ext=PowerExternal(),
    )


def _base_delta_log(agent_name: str = "agent") -> dict:
    return {
        "pool_before": 10.0,
        "pool_after_shares": 10.0,
        "pool_after": 10.0,
        "shares_total": 0.0,
        "requests_total": 2.0,
        "granted": {agent_name: 2.0},
        "punishments": {},
        "enable_withdraw_fee": False,
        "actions_log": {},
    }


class TestJacobianAttenuation:
    def test_dominant_utility_attenuates_delta_g(self):
        """When one utility dimension is strongly dominant, delta_g updates attenuate near zero."""
        # u with dimension 2 (RELATIONAL) extremely dominant (100.0)
        u_dominant = np.array([0.0, 0.0, 100.0, 0.0])
        agent_dom = _make_agent(u_dominant)

        # u uniform
        u_uniform = np.array([1.0, 1.0, 1.0, 1.0])
        agent_uni = _make_agent(u_uniform)

        action = Action(ActionType.SHARE, amount=2.0)
        dl = _base_delta_log("agent")

        phi_dom = feedback(agent_dom, 10.0, 10.0, action, dl, agent_name="agent")
        phi_uni = feedback(agent_uni, 10.0, 10.0, action, dl, agent_name="agent")

        # Attenuated delta_g under dominant u should be substantially smaller than under uniform u
        norm_dom = float(np.linalg.norm(phi_dom.delta_g))
        norm_uni = float(np.linalg.norm(phi_uni.delta_g))

        assert norm_dom < norm_uni * 0.1, (
            f"Dominant u should attenuate delta_g: norm_dom={norm_dom:.6f}, norm_uni={norm_uni:.6f}"
        )

    def test_sensitivity_scaling_magnitude(self):
        """Uniform u=[0,0,0,0] yields sensitivity pi_k*(1-pi_k) = 0.25*0.75 = 0.1875."""
        agent = _make_agent(np.zeros(4))
        action = Action(ActionType.SHARE, amount=2.0)
        dl = _base_delta_log("agent")

        phi = feedback(agent, 10.0, 10.0, action, dl, agent_name="agent")

        # Un-attenuated delta_g[RELATIONAL] would be RELATIONAL_U_GAIN * sociality = 0.1 * 0.5 = 0.05
        # Attenuated: 0.05 * 0.1875 = 0.009375
        expected_relational_dg = 0.1 * 0.5 * (0.25 * 0.75)
        assert np.isclose(phi.delta_g[2], expected_relational_dg, atol=1e-5)
