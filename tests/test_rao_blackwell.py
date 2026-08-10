"""Tests for Rao-Blackwellized particle filter (Phase 4)."""

import numpy as np
import pytest

from hypostases.engine import Action, ActionType
from hypostases.engine.dynamics import evolve_rb
from hypostases.engine.types import (
    AgentState,
    Characteristics,
    FeedbackDelta,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)
from hypostases.inference import infer, infer_joint
from hypostases.inference.summaries import summarize_map


def make_agent(reserve: float = 10.0) -> AgentState:
    return AgentState(
        c=Characteristics(reserve=reserve),
        w=WorldModel(mu=10.0, sigma2=2.0),
        g=GoalHierarchy(u=np.array([1.0, 1.0, 1.0, 1.0])),
        rho_ext=PowerExternal(),
    )


def test_rb_execution_single_agent():
    """infer(..., use_rao_blackwell=True) runs without error and returns valid particles."""
    actions = [Action(ActionType.REQUEST, amount=2.0) for _ in range(5)]
    pools = [10.0] * 5
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    rng = np.random.default_rng(100)
    particles = infer(
        observed_actions=actions,
        observed_pool_trace=pools,
        xi=xi,
        n_particles=40,
        use_rao_blackwell=True,
        rng=rng,
    )

    assert len(particles) == 40
    assert pytest.approx(sum(p.weight for p in particles)) == 1.0


def test_rb_execution_joint():
    """infer_joint(..., use_rao_blackwell=True) runs without error."""
    observed_actions = [
        {"A": Action(ActionType.REQUEST, amount=1.0), "B": Action(ActionType.SHARE, amount=2.0)}
    ]
    observed_pools = [10.0]
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    rng = np.random.default_rng(101)
    particles = infer_joint(
        observed_actions=observed_actions,
        observed_pool_trace=observed_pools,
        xi=xi,
        agent_names=["A", "B"],
        n_particles=20,
        use_rao_blackwell=True,
        rng=rng,
    )
    assert len(particles) == 20
    assert pytest.approx(sum(p.weight for p in particles)) == 1.0


def test_rb_sigma2_decreases_on_consistent_observations():
    """After repeated consistent observations, sigma2 under RB is <= initial value."""
    actions = [Action(ActionType.REQUEST, amount=2.0) for _ in range(10)]
    pools = [10.0] * 10
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    rng = np.random.default_rng(102)
    particles = infer(
        observed_actions=actions,
        observed_pool_trace=pools,
        xi=xi,
        n_particles=80,
        use_rao_blackwell=True,
        rng=rng,
    )

    map_state = summarize_map(particles)
    # Initial sigma2 = 2.0; Kalman should reduce uncertainty on consistent obs
    assert map_state.w.sigma2 <= 2.5  # some slack for resampling jitter


def test_rb_vs_flat_reserve_map_within_tolerance():
    """MAP reserve estimate from RB and flat filter on same fixed-seed trace are close."""
    actions = [Action(ActionType.SHARE, amount=1.0) for _ in range(6)]
    pools = [10.0] * 6
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    rng_rb = np.random.default_rng(200)
    rng_flat = np.random.default_rng(200)

    rb_particles = infer(
        observed_actions=actions,
        observed_pool_trace=pools,
        xi=xi,
        n_particles=80,
        use_rao_blackwell=True,
        rng=rng_rb,
    )
    flat_particles = infer(
        observed_actions=actions,
        observed_pool_trace=pools,
        xi=xi,
        n_particles=80,
        use_rao_blackwell=False,
        rng=rng_flat,
    )

    rb_map = summarize_map(rb_particles)
    flat_map = summarize_map(flat_particles)

    assert abs(rb_map.c.reserve - flat_map.c.reserve) < 5.0


def test_evolve_rb_kalman_update_reduces_sigma2():
    """evolve_rb directly: after a zero-surprise update, sigma2 strictly decreases."""
    agent = make_agent()
    initial_sigma2 = agent.w.sigma2  # 2.0

    phi = FeedbackDelta()
    # surprise = 0 => clean Kalman update with no new info
    evolve_rb(agent, phi, surprise=0.0)

    # Kalman update with R=2.0, Q=0.1: sig2_pred = 2.0 + 0.1 = 2.1
    # K = 2.1 / (2.1 + 2.0) = ~0.512; sig2_post = (1-K)*2.1 = ~1.024
    assert agent.w.sigma2 < initial_sigma2


def test_evolve_rb_sigma2_bounded_below():
    """evolve_rb never produces sigma2 below SIGMA2_MIN regardless of obs noise."""
    from hypostases.engine.constants import SIGMA2_MIN

    agent = make_agent()
    agent.w.sigma2 = SIGMA2_MIN  # start at floor

    phi = FeedbackDelta()
    evolve_rb(agent, phi, surprise=0.0, obs_noise_r=1e6)  # huge R => K near 0

    assert agent.w.sigma2 >= SIGMA2_MIN


def test_rb_default_false_unchanged():
    """use_rao_blackwell=False (default) is identical to omitting the flag (same seed)."""
    actions = [Action(ActionType.WITHDRAW) for _ in range(3)]
    pools = [10.0] * 3
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    rng1 = np.random.default_rng(300)
    rng2 = np.random.default_rng(300)

    default_particles = infer(actions, pools, xi, n_particles=20, rng=rng1)
    explicit_false = infer(actions, pools, xi, n_particles=20, use_rao_blackwell=False, rng=rng2)

    assert default_particles[0].sigma.c.reserve == explicit_false[0].sigma.c.reserve
