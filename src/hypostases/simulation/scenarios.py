"""HYPOSTASES Simulation — Multi-Agent Preset Scenarios (Phase 2)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from hypostases.engine import (
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)


def _make_tragedy() -> dict[str, AgentState]:
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


def _make_altruism() -> dict[str, AgentState]:
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


def _make_freerider() -> dict[str, AgentState]:
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


def _make_punishment() -> dict[str, AgentState]:
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


def _make_inequity() -> dict[str, AgentState]:
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


def _make_deceptive() -> dict[str, AgentState]:
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


def _make_crowding_out() -> dict[str, AgentState]:
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


_SCENARIO_REGISTRY: dict[str, Callable[[], dict[str, AgentState]]] = {
    "tragedy": _make_tragedy,
    "altruism": _make_altruism,
    "freerider": _make_freerider,
    "punishment": _make_punishment,
    "inequity": _make_inequity,
    "deceptive": _make_deceptive,
    "crowding_out": _make_crowding_out,
}


def create_scenario_agents(scenario: str) -> dict[str, AgentState]:
    """Generates named agent states for pre-defined multi-agent simulation scenarios."""
    key = scenario.lower()
    if key not in _SCENARIO_REGISTRY:
        valid_opts = ", ".join(f"'{k}'" for k in _SCENARIO_REGISTRY)
        raise ValueError(f"Unknown scenario '{scenario}'. Valid options: {valid_opts}.")
    return _SCENARIO_REGISTRY[key]()
