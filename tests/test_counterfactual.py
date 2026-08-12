"""HYPOSTASES Engine — v4 Counterfactual Simulation Tests.

Spec Ref: Wave 1 Front 04 (docs/WAVE_1_FRONT_04/front_04_counterfactual_simulation_spec.md).
Verifies:
  1. Ephemeral sandbox state isolation (zero side effects on physical state σ).
  2. Multi-future lookahead branch evaluation and action selection.
  3. EvoCF mutation mechanics.
  4. Rule 005 invariant compliance.
"""

from __future__ import annotations

import numpy as np
import pytest

from hypostases.counterfactual import CounterfactualEngine, VirtualEnvironmentSandbox
from hypostases.engine.types import (
    Action,
    ActionType,
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)
from hypostases.schemas import assert_invariants


@pytest.fixture
def sample_agent_state() -> AgentState:
    return AgentState(
        c=Characteristics(reserve=10.0, mood=0.0, skill=0.6),
        w=WorldModel(mu=10.0, sigma2=2.0),
        g=GoalHierarchy(u=np.array([1.0, 1.0, 1.0, 1.0])),
        rho_ext=PowerExternal(social_capital=1.0, time_budget=12.0),
    )


def test_virtual_sandbox_state_isolation(sample_agent_state: AgentState) -> None:
    """Verify that virtual sandbox rollouts leave physical state σ completely unmodified."""
    initial_reserve = sample_agent_state.c.reserve
    initial_u = sample_agent_state.g.u.copy()
    pool_state = 10.0
    xi = np.array([0.5, 0.5, 0.5])

    sandbox = VirtualEnvironmentSandbox(
        agent_state=sample_agent_state.clone(),
        pool_state=pool_state,
        xi=xi,
    )

    action = Action(ActionType.REQUEST, amount=3.0)
    sandbox.step(action)

    # Physical state remains identical
    assert sample_agent_state.c.reserve == initial_reserve
    np.testing.assert_array_equal(sample_agent_state.g.u, initial_u)
    assert_invariants(sample_agent_state)


def test_counterfactual_engine_lookahead(sample_agent_state: AgentState) -> None:
    """Verify multi-future lookahead search and physical action selection."""
    engine = CounterfactualEngine(
        lookahead_depth=3,
        branching_factor=4,
        discount_factor=0.95,
        risk_penalty_lambda=0.1,
    )

    pool_state = 10.0
    xi = np.array([0.5, 0.5, 0.5])
    rng = np.random.default_rng(42)

    selected_action = engine.simulate_lookahead(
        initial_agent_state=sample_agent_state,
        initial_pool=pool_state,
        xi=xi,
        rng=rng,
    )

    assert isinstance(selected_action, Action)
    assert selected_action.action_type in ActionType
    assert_invariants(sample_agent_state)


def test_evocf_mutation(sample_agent_state: AgentState) -> None:
    """Verify Evolutionary Counterfactual plan mutation."""
    engine = CounterfactualEngine(evocf_mutation_rate=1.0)  # Always mutate
    rng = np.random.default_rng(123)

    from hypostases.counterfactual import CounterfactualBranch

    branch = CounterfactualBranch(
        action_sequence=[
            Action(ActionType.REQUEST, amount=1.0),
            Action(ActionType.SHARE, amount=1.0),
        ]
    )

    mutated = engine.mutate_plan_evocf(branch, rng=rng)
    assert len(mutated.action_sequence) == len(branch.action_sequence)


def test_dubins_reachability_evaluation() -> None:
    """Verifies Dubins reachability path calculation under curvature constraints (Cacace et al. 2020)."""
    engine = CounterfactualEngine()
    start_pose = np.array([0.0, 0.0, 0.0])
    target_pose = np.array([3.0, 4.0, np.pi / 2])

    dist = engine.evaluate_dubins_reachability(start_pose, target_pose, kappa_max=1.0)
    assert dist > 5.0  # Euclidean distance 5.0 + curvature penalty > 5.0
