"""HYPOSTASES Simulation — Multi-Agent Preset Scenarios (Data-Driven YAML, Directive 006)."""

from __future__ import annotations

import numpy as np

from hypostases.engine import (
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)
from hypostases.schemas.loader import load_yaml


def _build_agent_state_from_dict(cfg: dict) -> AgentState:
    c_cfg = cfg.get("c", {})
    w_cfg = cfg.get("w", {})
    g_cfg = cfg.get("g", {})
    rho_cfg = cfg.get("rho_ext", {})

    c = Characteristics(
        skill=c_cfg.get("skill", 0.6),
        resilience=c_cfg.get("resilience", 0.5),
        sociality=c_cfg.get("sociality", 0.5),
        memory_decay=c_cfg.get("memory_decay", 0.9),
        reserve=c_cfg.get("reserve", 10.0),
        mood=c_cfg.get("mood", 0.0),
    )
    w = WorldModel(
        mu=w_cfg.get("mu", 10.0),
        sigma2=w_cfg.get("sigma2", 2.0),
        replenish_rate_est=w_cfg.get("replenish_rate_est", 1.0),
        peer_beliefs=dict(w_cfg.get("peer_beliefs", {})),
    )
    u_val = g_cfg.get("u", [1.0, 1.0, 1.0, 1.0])
    g = GoalHierarchy(u=np.array(u_val, dtype=float))
    rho_ext = PowerExternal(
        social_capital=rho_cfg.get("social_capital", 1.0),
        time_budget=rho_cfg.get("time_budget", 12.0),
    )
    return AgentState(c=c, w=w, g=g, rho_ext=rho_ext)


def load_scenario_definitions() -> dict[str, dict[str, AgentState]]:
    """Loads all pre-defined multi-agent simulation scenarios from schema/scenarios.yaml (Directive 006)."""
    raw = load_yaml("scenarios.yaml")
    scenarios_raw = raw.get("scenarios", {})

    out = {}
    for sc_key, sc_data in scenarios_raw.items():
        agents_map = {}
        for ag_name, ag_cfg in sc_data.get("agents", {}).items():
            agents_map[ag_name] = _build_agent_state_from_dict(ag_cfg)
        out[sc_key.lower()] = agents_map
    return out


def create_scenario_agents(scenario: str) -> dict[str, AgentState]:
    """Generates named agent states for pre-defined multi-agent simulation scenarios."""
    scenarios = load_scenario_definitions()
    key = scenario.lower()
    if key not in scenarios:
        valid_opts = ", ".join(f"'{k}'" for k in sorted(scenarios.keys()))
        raise ValueError(f"Unknown scenario '{scenario}'. Valid options: {valid_opts}.")
    return scenarios[key]
