"""Tests — Inequity Aversion & Relative Deprivation.

Verifies that relative wealth disparities between peer beliefs and own reserves
trigger mood decay and alter goal dynamics.
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
from hypostases.simulation.scenarios import create_scenario_agents


class TestInequityAversion:
    def test_scenario_inequity_loading(self):
        agents = create_scenario_agents("inequity")
        assert "Wealthy" in agents
        assert "Deprived" in agents
        assert agents["Wealthy"].c.reserve > agents["Deprived"].c.reserve

    def test_relative_deprivation_triggers_mood_decay(self):
        # Deprived agent has reserve=3.0, but peer belief for Wealthy is 25.0
        deprived = AgentState(
            c=Characteristics(reserve=3.0, mood=0.0),
            w=WorldModel(peer_beliefs={"Wealthy": 25.0}),
            g=GoalHierarchy(),
            rho_ext=PowerExternal(),
        )
        action = Action(ActionType.WITHDRAW)
        delta_log = {
            "pool_before": 10.0,
            "pool_after_shares": 10.0,
            "pool_after": 10.0,
            "shares_total": 0.0,
            "requests_total": 0.0,
            "granted": {},
            "actions_log": {"Deprived": action},
        }
        phi = feedback(deprived, 10.0, 10.0, action, delta_log, agent_name="Deprived")
        # Inequity mood penalty should be negative
        assert phi.delta_c["mood"] < 0.0
        evolve(deprived, phi)
        assert deprived.c.mood < 0.0

    def test_no_deprivation_when_richer_than_peers(self):
        wealthy = AgentState(
            c=Characteristics(reserve=25.0, mood=0.0),
            w=WorldModel(peer_beliefs={"Deprived": 3.0}),
            g=GoalHierarchy(),
            rho_ext=PowerExternal(),
        )
        action = Action(ActionType.WITHDRAW)
        delta_log = {
            "pool_before": 10.0,
            "pool_after_shares": 10.0,
            "pool_after": 10.0,
            "shares_total": 0.0,
            "requests_total": 0.0,
            "granted": {},
            "actions_log": {"Wealthy": action},
        }
        phi = feedback(wealthy, 10.0, 10.0, action, delta_log, agent_name="Wealthy")
        # Should not suffer relative deprivation mood penalty
        assert phi.delta_c.get("mood", 0.0) >= -0.05
