"""Tests for HYPOSTASES Simulation Harness and Benchmark Module."""

from hypostases.simulation import (
    generate_forced_withdraw_trace,
    generate_sample_trace,
    make_agent,
    make_test_agent,
    run_simulation_trace,
)


def test_make_agent_and_test_agent():
    ag1 = make_agent("Agent1", sociality=0.7, status_u=0.5, reserve=8.0)
    assert ag1.name == "Agent1"
    assert ag1.sigma.c.sociality == 0.7
    assert ag1.sigma.c.reserve == 8.0
    assert ag1.sigma.g.u[3] == 0.5

    ag2 = make_test_agent("TestAgent", reserve0=12.0)
    assert ag2.name == "TestAgent"
    assert ag2.sigma.c.reserve == 12.0


def test_generate_sample_trace():
    actions, pools = generate_sample_trace(n_steps=5, seed=123)
    assert len(actions) == 5
    assert len(pools) == 5


def test_generate_forced_withdraw_trace():
    ag = make_test_agent("W_Agent", reserve0=10.0)
    actions, pools = generate_forced_withdraw_trace(ag, pool0=10.0, n_steps=4)
    assert len(actions) == 4
    assert len(pools) == 4
    assert all(a.action_type.value == "WITHDRAW" for a in actions)


def test_run_simulation_trace():
    res = run_simulation_trace(steps=3, seed=42)
    assert res["steps"] == 3
    assert len(res["trace"]) == 3
    assert "Agent_A" in res["agents"]
    assert "Agent_B" in res["agents"]
