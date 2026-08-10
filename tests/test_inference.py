"""Tests for HYPOSTASES Inverse Inference Engine."""

import numpy as np
import pytest

from hypostases.engine import Action, ActionType
from hypostases.inference import (
    goal_posterior,
    infer,
    sample_prior,
    summarize_kalman,
    summarize_map,
)


def test_sample_prior_generates_valid_particle():
    rng = np.random.default_rng(42)
    state = sample_prior(reserve_range=(1.0, 20.0), rng=rng)

    assert 1.0 <= state.c.reserve <= 20.0
    assert len(state.g.u) == 4


def test_sample_prior_truncated_normal():
    rng = np.random.default_rng(42)
    state = sample_prior(reserve_range=(2.0, 15.0), prior_type="truncated_normal", rng=rng)
    assert 2.0 <= state.c.reserve <= 15.0


def test_sample_prior_log_normal():
    rng = np.random.default_rng(42)
    state = sample_prior(reserve_range=(2.0, 15.0), prior_type="log_normal", rng=rng)
    assert 2.0 <= state.c.reserve <= 15.0


def test_infer_particle_filter_execution():
    actions = [Action(ActionType.REQUEST, amount=2.0), Action(ActionType.SHARE, amount=1.0)]
    pools = [10.0, 8.0]
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    rng = np.random.default_rng(7)
    particles = infer(
        observed_actions=actions,
        observed_pool_trace=pools,
        xi=xi,
        n_particles=50,
        rng=rng,
    )

    assert len(particles) == 50
    total_weight = sum(p.weight for p in particles)
    assert pytest.approx(total_weight) == 1.0


def test_inference_summaries():
    actions = [Action(ActionType.SHARE, amount=2.0) for _ in range(5)]
    pools = [10.0] * 5
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    rng = np.random.default_rng(42)
    particles = infer(actions, pools, xi, n_particles=50, validate_invariants=True, rng=rng)

    map_state = summarize_map(particles)
    assert map_state.c.reserve > 0.0

    kalman = summarize_kalman(particles)
    assert "reserve_mean" in kalman
    assert "reserve_var" in kalman
    assert "u_mean" in kalman
    assert "u_var" in kalman
    assert len(kalman["u_mean"]) == 4
    assert len(kalman["u_var"]) == 4
    assert kalman["reserve_var"] >= 0.0

    g_post = goal_posterior(particles)
    assert pytest.approx(sum(g_post.values())) == 1.0
    # Continuous share actions should strongly weight RELATIONAL
    assert g_post["RELATIONAL"] > 0.5


def test_deterministic_reproducibility():
    actions = [Action(ActionType.WITHDRAW)] * 3
    pools = [9.0] * 3
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)

    res1 = infer(actions, pools, xi, n_particles=30, rng=rng1)
    res2 = infer(actions, pools, xi, n_particles=30, rng=rng2)

    assert res1[0].sigma.c.reserve == res2[0].sigma.c.reserve


def test_goal_posterior_direct():
    from hypostases.engine import (
        AgentState,
        Characteristics,
        GoalHierarchy,
        PowerExternal,
        WorldModel,
    )
    from hypostases.inference.particle_filter import Particle

    p1 = Particle(
        sigma=AgentState(
            Characteristics(),
            WorldModel(),
            GoalHierarchy(u=np.array([10.0, 0.0, 0.0, 0.0])),
            PowerExternal(),
        ),
        weight=0.6,
    )
    p2 = Particle(
        sigma=AgentState(
            Characteristics(),
            WorldModel(),
            GoalHierarchy(u=np.array([0.0, 0.0, 10.0, 0.0])),
            PowerExternal(),
        ),
        weight=0.4,
    )
    post = goal_posterior([p1, p2])
    assert post["SURVIVAL"] == pytest.approx(0.6)
    assert post["RELATIONAL"] == pytest.approx(0.4)
    assert post["ACQUISITION"] == pytest.approx(0.0)
    assert post["STATUS"] == pytest.approx(0.0)


def test_summarize_kalman_value_correctness():
    from hypostases.engine import (
        AgentState,
        Characteristics,
        GoalHierarchy,
        PowerExternal,
        WorldModel,
    )
    from hypostases.inference.particle_filter import Particle

    p1 = Particle(
        sigma=AgentState(
            Characteristics(reserve=2.0, mood=-0.5),
            WorldModel(),
            GoalHierarchy(u=np.array([1.0, 2.0, 3.0, 4.0])),
            PowerExternal(),
        ),
        weight=0.7,
    )
    p2 = Particle(
        sigma=AgentState(
            Characteristics(reserve=12.0, mood=0.5),
            WorldModel(),
            GoalHierarchy(u=np.array([5.0, 6.0, 7.0, 8.0])),
            PowerExternal(),
        ),
        weight=0.3,
    )

    res = summarize_kalman([p1, p2])
    assert res["reserve_mean"] == pytest.approx(2.0 * 0.7 + 12.0 * 0.3)
    assert res["mood_mean"] == pytest.approx(-0.5 * 0.7 + 0.5 * 0.3)

    # u_mean calculation
    expected_u_mean = np.array([1.0, 2.0, 3.0, 4.0]) * 0.7 + np.array([5.0, 6.0, 7.0, 8.0]) * 0.3
    assert np.allclose(res["u_mean"], expected_u_mean)


def test_infer_empty_observations():
    xi = np.array([0.2, 0.2, 0.2, 0.2])
    res = infer(observed_actions=[], observed_pool_trace=[], xi=xi, n_particles=10)
    assert len(res) == 10
    assert pytest.approx(sum(p.weight for p in res)) == 1.0


def test_weight_collapse_warning():
    from unittest.mock import patch

    actions = [Action(ActionType.REQUEST, amount=1.0)]
    pools = [10.0]
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    with (
        patch("hypostases.inference.particle_filter.action_likelihood", return_value=0.0),
        pytest.warns(UserWarning, match="Particle weight collapse detected"),
    ):
        infer(actions, pools, xi, n_particles=5)


def test_infer_joint_execution():
    from hypostases.inference import infer_joint

    observed_actions = [
        {
            "Agent_A": Action(ActionType.REQUEST, amount=1.0),
            "Agent_B": Action(ActionType.SHARE, amount=2.0),
        }
    ]
    observed_pools = [10.0]
    xi = np.array([0.2, 0.2, 0.2, 0.2])
    res = infer_joint(
        observed_actions,
        observed_pools,
        xi,
        agent_names=["Agent_A", "Agent_B"],
        n_particles=15,
    )
    assert len(res) == 15
    assert "Agent_A" in res[0].sigmas
    assert "Agent_B" in res[0].sigmas


def test_infer_mean_field_execution():
    from hypostases.inference import infer_mean_field

    observed_actions = [
        {
            "Agent_A": Action(ActionType.REQUEST, amount=1.0),
            "Agent_B": Action(ActionType.SHARE, amount=2.0),
        }
    ]
    observed_pools = [10.0]
    xi = np.array([0.2, 0.2, 0.2, 0.2])
    res = infer_mean_field(
        observed_actions,
        observed_pools,
        xi,
        agent_names=["Agent_A", "Agent_B"],
        n_particles=15,
    )
    assert "Agent_A" in res
    assert "Agent_B" in res
    assert len(res["Agent_A"]) == 15


def test_infer_lag_window_truncates_trace():
    """lag_window slices the trace before filtering; result is still a valid particle set."""
    # 5-step trace but window of 1 — filter only sees the last observation
    actions = [
        Action(ActionType.REQUEST, amount=3.0),
        Action(ActionType.SHARE, amount=1.0),
        Action(ActionType.WITHDRAW),
        Action(ActionType.REQUEST, amount=2.0),
        Action(ActionType.SHARE, amount=0.5),
    ]
    pools = [10.0, 9.0, 9.5, 8.0, 8.5]
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    rng = np.random.default_rng(77)
    particles = infer(
        observed_actions=actions,
        observed_pool_trace=pools,
        xi=xi,
        n_particles=30,
        lag_window=1,
        rng=rng,
    )

    assert len(particles) == 30
    assert pytest.approx(sum(p.weight for p in particles)) == 1.0


def test_infer_lag_window_none_unchanged():
    """lag_window=None produces the same result as omitting the parameter (same seed)."""
    actions = [Action(ActionType.SHARE, amount=1.0) for _ in range(4)]
    pools = [10.0] * 4
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    res_default = infer(actions, pools, xi, n_particles=20, rng=rng1)
    res_none = infer(actions, pools, xi, n_particles=20, lag_window=None, rng=rng2)

    assert res_default[0].sigma.c.reserve == res_none[0].sigma.c.reserve
