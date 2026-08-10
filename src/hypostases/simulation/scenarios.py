"""HYPOSTASES Simulation — Multi-Agent Preset Scenarios (Phase 2)."""

from __future__ import annotations

import numpy as np

from hypostases.engine import (
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)


def create_scenario_agents(scenario: str) -> dict[str, AgentState]:
    """Generates named agent states for pre-defined multi-agent simulation scenarios.

    Scenarios:
      - 'tragedy': Tragedy of the Commons (greedy survival & acquisition agents).
      - 'altruism': Cooperative pool maintenance (relational-dominant agents).
      - 'freerider': Mixed population (cooperators vs. defector status-seeking withdrawers).
    """
    scenario = scenario.lower()

    if scenario == "tragedy":
        # 3 greedy agents with high SURVIVAL/ACQUISITION weights
        return {
            "Greedy_1": AgentState(
                c=Characteristics(reserve=5.0, sociality=0.1),
                w=WorldModel(mu=10.0),
                g=GoalHierarchy(u=np.array([5.0, 5.0, 0.0, 0.0])),
                rho_ext=PowerExternal(),
            ),
            "Greedy_2": AgentState(
                c=Characteristics(reserve=4.0, sociality=0.2),
                w=WorldModel(mu=10.0),
                g=GoalHierarchy(u=np.array([6.0, 4.0, 0.0, 0.0])),
                rho_ext=PowerExternal(),
            ),
            "Greedy_3": AgentState(
                c=Characteristics(reserve=6.0, sociality=0.1),
                w=WorldModel(mu=10.0),
                g=GoalHierarchy(u=np.array([4.0, 6.0, 0.0, 0.0])),
                rho_ext=PowerExternal(),
            ),
        }

    if scenario == "altruism":
        # 3 cooperative agents with high RELATIONAL weights
        return {
            "Coop_1": AgentState(
                c=Characteristics(reserve=15.0, sociality=0.9),
                w=WorldModel(mu=10.0),
                g=GoalHierarchy(u=np.array([1.0, 1.0, 8.0, 0.0])),
                rho_ext=PowerExternal(),
            ),
            "Coop_2": AgentState(
                c=Characteristics(reserve=12.0, sociality=0.8),
                w=WorldModel(mu=10.0),
                g=GoalHierarchy(u=np.array([1.0, 1.0, 7.0, 0.0])),
                rho_ext=PowerExternal(),
            ),
            "Coop_3": AgentState(
                c=Characteristics(reserve=14.0, sociality=0.85),
                w=WorldModel(mu=10.0),
                g=GoalHierarchy(u=np.array([1.0, 1.0, 9.0, 0.0])),
                rho_ext=PowerExternal(),
            ),
        }

    if scenario == "freerider":
        # 2 cooperators + 1 defector (status-seeking withdrawer)
        return {
            "Coop_A": AgentState(
                c=Characteristics(reserve=12.0, sociality=0.8),
                w=WorldModel(mu=10.0),
                g=GoalHierarchy(u=np.array([1.0, 1.0, 6.0, 0.0])),
                rho_ext=PowerExternal(),
            ),
            "Coop_B": AgentState(
                c=Characteristics(reserve=11.0, sociality=0.75),
                w=WorldModel(mu=10.0),
                g=GoalHierarchy(u=np.array([1.0, 1.0, 5.0, 0.0])),
                rho_ext=PowerExternal(),
            ),
            "FreeRider": AgentState(
                c=Characteristics(reserve=8.0, sociality=0.1),
                w=WorldModel(mu=10.0),
                g=GoalHierarchy(u=np.array([2.0, 2.0, 0.0, 8.0])),
                rho_ext=PowerExternal(),
            ),
        }

    if scenario == "punishment":
        # 1 Defector + 1 Passive Cooperator + 1 Active Vigilante/Punisher
        return {
            "Defector": AgentState(
                c=Characteristics(reserve=15.0, sociality=0.1),
                w=WorldModel(mu=10.0),
                g=GoalHierarchy(u=np.array([1.0, 2.0, 0.0, 8.0])),
                rho_ext=PowerExternal(),
            ),
            "PassiveCoop": AgentState(
                c=Characteristics(reserve=10.0, sociality=0.8),
                w=WorldModel(mu=10.0),
                g=GoalHierarchy(u=np.array([1.0, 1.0, 6.0, 0.0])),
                rho_ext=PowerExternal(),
            ),
            "Vigilante": AgentState(
                c=Characteristics(reserve=12.0, sociality=0.9, resilience=0.8),
                w=WorldModel(mu=10.0, peer_beliefs={"Defector": 15.0}),
                g=GoalHierarchy(u=np.array([1.0, 1.0, 5.0, 2.0])),
                rho_ext=PowerExternal(),
            ),
        }

    if scenario == "inequity":
        # 1 Wealthy Agent + 1 Disadvantaged Agent with peer awareness
        return {
            "Wealthy": AgentState(
                c=Characteristics(reserve=25.0, sociality=0.5),
                w=WorldModel(mu=10.0, peer_beliefs={"Deprived": 3.0}),
                g=GoalHierarchy(u=np.array([1.0, 4.0, 2.0, 1.0])),
                rho_ext=PowerExternal(),
            ),
            "Deprived": AgentState(
                c=Characteristics(reserve=3.0, sociality=0.5, mood=0.0),
                w=WorldModel(mu=10.0, peer_beliefs={"Wealthy": 25.0}),
                g=GoalHierarchy(u=np.array([3.0, 3.0, 1.0, 1.0])),
                rho_ext=PowerExternal(),
            ),
        }

    if scenario == "deceptive":
        # 1 Deceptive Agent (high true reserve, claims scarcity) + 1 Observer
        return {
            "DeceptiveAgent": AgentState(
                c=Characteristics(reserve=18.0, sociality=0.2),
                w=WorldModel(mu=10.0),
                g=GoalHierarchy(u=np.array([6.0, 1.0, 0.0, 0.0])),
                rho_ext=PowerExternal(),
            ),
            "Observer": AgentState(
                c=Characteristics(reserve=10.0, sociality=0.7),
                w=WorldModel(mu=10.0, peer_beliefs={"DeceptiveAgent": 2.0}),
                g=GoalHierarchy(u=np.array([1.0, 1.0, 5.0, 0.0])),
                rho_ext=PowerExternal(),
            ),
        }

    if scenario == "crowding_out":
        # Population with balanced initial utilities for fee-toggling studies
        return {
            "Agent_1": AgentState(
                c=Characteristics(reserve=10.0, sociality=0.5),
                w=WorldModel(mu=10.0),
                g=GoalHierarchy(u=np.array([2.0, 2.0, 4.0, 2.0])),
                rho_ext=PowerExternal(),
            ),
            "Agent_2": AgentState(
                c=Characteristics(reserve=10.0, sociality=0.4),
                w=WorldModel(mu=10.0),
                g=GoalHierarchy(u=np.array([2.0, 2.0, 4.0, 2.0])),
                rho_ext=PowerExternal(),
            ),
        }

    valid_opts = (
        "'tragedy', 'altruism', 'freerider', 'punishment', 'inequity', 'deceptive', 'crowding_out'"
    )
    raise ValueError(f"Unknown scenario '{scenario}'. Valid options: {valid_opts}.")
