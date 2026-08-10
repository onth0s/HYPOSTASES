"""Tests — Second-Order Altruistic Punishment & Vigilantism.

Verifies that ActionType.PUNISH correctly deducts reserve cost from the punisher,
applies penalty to the target agent, and integrates into the inference feasibility gate.
"""

from __future__ import annotations

from hypostases.engine.constants import PUNISH_RESERVE_COST, PUNISH_TARGET_PENALTY
from hypostases.engine.dynamics import evolve, feedback, step_env
from hypostases.engine.likelihood import _is_infeasible
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


class TestAltruisticPunishment:
    def test_scenario_punishment_loading(self):
        agents = create_scenario_agents("punishment")
        assert "Defector" in agents
        assert "PassiveCoop" in agents
        assert "Vigilante" in agents

    def test_step_env_records_punishment_penalties(self):
        actions = [
            ("Vigilante", Action(ActionType.PUNISH, target="Defector")),
            ("Defector", Action(ActionType.WITHDRAW)),
        ]
        _, delta_log = step_env(10.0, actions)
        assert "punishments" in delta_log
        assert delta_log["punishments"].get("Defector") == PUNISH_TARGET_PENALTY

    def test_feedback_deducts_cost_from_punisher(self):
        agent = AgentState(
            c=Characteristics(reserve=10.0, sociality=0.8),
            w=WorldModel(),
            g=GoalHierarchy(),
            rho_ext=PowerExternal(),
        )
        action = Action(ActionType.PUNISH, target="Defector")
        delta_log = {
            "pool_before": 10.0,
            "pool_after_shares": 10.0,
            "pool_after": 10.0,
            "shares_total": 0.0,
            "requests_total": 0.0,
            "granted": {},
            "punishments": {"Defector": PUNISH_TARGET_PENALTY},
            "actions_log": {"Vigilante": action},
        }
        phi = feedback(agent, 10.0, 10.0, action, delta_log, agent_name="Vigilante")
        assert phi.delta_c["reserve"] == -PUNISH_RESERVE_COST
        evolve(agent, phi)
        assert agent.c.reserve == 10.0 - PUNISH_RESERVE_COST

    def test_feedback_applies_penalty_to_target(self):
        defector = AgentState(
            c=Characteristics(reserve=15.0),
            w=WorldModel(),
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
            "punishments": {"Defector": PUNISH_TARGET_PENALTY},
            "actions_log": {"Defector": action},
        }
        phi = feedback(defector, 10.0, 10.0, action, delta_log, agent_name="Defector")
        assert phi.delta_c["reserve"] == -PUNISH_TARGET_PENALTY
        evolve(defector, phi)
        assert defector.c.reserve == 15.0 - PUNISH_TARGET_PENALTY

    def test_feasibility_gate_prunes_low_reserve_punisher(self):
        agent_poor = AgentState(
            c=Characteristics(reserve=1.0),
            w=WorldModel(),
            g=GoalHierarchy(),
            rho_ext=PowerExternal(),
        )
        punish_action = Action(ActionType.PUNISH, target="Defector")
        assert _is_infeasible(agent_poor, punish_action, pool_belief=10.0)
