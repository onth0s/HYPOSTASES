"""Tests for Multi-Agent Preset Scenarios (Phase 2)."""

import pytest

from hypostases.simulation import create_scenario_agents


def test_create_scenario_tragedy():
    agents = create_scenario_agents("tragedy")
    assert len(agents) == 3
    assert "Greedy_1" in agents
    assert agents["Greedy_1"].c.sociality == 0.1


def test_create_scenario_altruism():
    agents = create_scenario_agents("altruism")
    assert len(agents) == 3
    assert "Coop_1" in agents
    assert agents["Coop_1"].c.sociality == 0.9


def test_create_scenario_freerider():
    agents = create_scenario_agents("freerider")
    assert len(agents) == 3
    assert "FreeRider" in agents
    # FreeRider has STATUS dominant (index 3)
    assert agents["FreeRider"].g.u[3] == 8.0


def test_create_scenario_invalid():
    with pytest.raises(ValueError, match="Unknown scenario"):
        create_scenario_agents("nonexistent")
