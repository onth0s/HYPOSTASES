"""Tests — Condition 2: Bayesian Trust & Peer Belief Dynamics under Theory of Mind.

Verifies that observing peer actions updates w.peer_beliefs directly, modifying
state-dependent action evaluation without hardcoded scenario rules.
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


class TestBayesianTrustUpdating:
    def test_observed_peer_grant_updates_peer_beliefs(self):
        """Observing a peer receive grants updates w.peer_beliefs via EMA filtering."""
        agent = AgentState(
            c=Characteristics(reserve=10.0),
            w=WorldModel(peer_beliefs={"Peer_A": 0.0}),
            g=GoalHierarchy(),
            rho_ext=PowerExternal(),
        )

        action = Action(ActionType.REQUEST, amount=2.0)
        delta_log = {
            "pool_before": 10.0,
            "pool_after_shares": 10.0,
            "pool_after": 10.0,
            "shares_total": 0.0,
            "requests_total": 2.0,
            "granted": {"agent": 2.0, "Peer_A": 5.0},
            "actions_log": {"agent": action},
        }

        phi = feedback(agent, 10.0, 10.0, action, delta_log, agent_name="agent")
        assert "Peer_A" in phi.delta_peer_beliefs
        assert phi.delta_peer_beliefs["Peer_A"] == 5.0

        evolve(agent, phi)
        # PEER_BELIEF_ALPHA * 5.0 + (1 - ALPHA) * 0.0 = 0.3 * 5.0 = 1.5
        assert agent.w.peer_beliefs["Peer_A"] > 0.0
