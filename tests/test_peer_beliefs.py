"""Tests for HYPOSTASES Theory of Mind Peer Belief Dynamics."""

from hypostases.engine import (
    Action,
    ActionType,
    evolve,
    feedback,
    step_env,
)
from hypostases.engine.constants import PEER_BELIEF_ALPHA
from hypostases.simulation import make_agent


def test_peer_beliefs_feedback_generation():
    ag_a = make_agent("Agent_A", sociality=0.8, status_u=0.2)

    act_a = Action(ActionType.REQUEST, amount=3.0)
    act_b = Action(ActionType.REQUEST, amount=2.0)

    _, delta_log = step_env(10.0, [("Agent_A", act_a), ("Agent_B", act_b)])

    phi_a = feedback(ag_a.sigma, 10.0, 5.0, act_a, delta_log, agent_name="Agent_A")

    assert "Agent_B" in phi_a.delta_peer_beliefs
    assert phi_a.delta_peer_beliefs["Agent_B"] == 2.0
    assert "Agent_A" not in phi_a.delta_peer_beliefs


def test_peer_beliefs_evolution():
    ag_a = make_agent("Agent_A", sociality=0.8, status_u=0.2)

    act_a = Action(ActionType.REQUEST, amount=3.0)
    act_b = Action(ActionType.REQUEST, amount=2.0)

    _, delta_log = step_env(10.0, [("Agent_A", act_a), ("Agent_B", act_b)])

    phi_a = feedback(ag_a.sigma, 10.0, 5.0, act_a, delta_log, agent_name="Agent_A")
    evolve(ag_a.sigma, phi_a)

    expected_peer_belief = PEER_BELIEF_ALPHA * 2.0 + (1.0 - PEER_BELIEF_ALPHA) * 2.0
    assert "Agent_B" in ag_a.sigma.w.peer_beliefs
    assert ag_a.sigma.w.peer_beliefs["Agent_B"] == expected_peer_belief
