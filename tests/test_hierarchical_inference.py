"""Tests for infer_hierarchical — two-pass hierarchical particle filter (Phase 3)."""

import numpy as np
import pytest

from hypostases.engine import Action, ActionType
from hypostases.inference import infer_hierarchical, sample_prior
from hypostases.inference.summaries import goal_posterior


def test_hierarchical_execution_valid_particles():
    """infer_hierarchical returns n_particles valid, normalised particles."""
    actions = [Action(ActionType.SHARE, amount=1.0) for _ in range(5)]
    pools = [10.0] * 5
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    rng = np.random.default_rng(11)
    particles = infer_hierarchical(
        observed_actions=actions,
        observed_pool_trace=pools,
        xi=xi,
        n_particles=40,
        macro_n_particles=15,
        rng=rng,
    )

    assert len(particles) == 40
    assert pytest.approx(sum(p.weight for p in particles)) == 1.0


def test_hierarchical_goal_bias_shifts_posterior_toward_relational():
    """After 10 SHARE observations, RELATIONAL posterior > 0.4 for hierarchical filter."""
    actions = [Action(ActionType.SHARE, amount=1.0) for _ in range(10)]
    pools = [10.0] * 10
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    rng = np.random.default_rng(22)
    particles = infer_hierarchical(
        observed_actions=actions,
        observed_pool_trace=pools,
        xi=xi,
        n_particles=100,
        macro_n_particles=30,
        goal_bias_strength=3.0,
        rng=rng,
    )

    post = goal_posterior(particles)
    assert post["RELATIONAL"] > 0.4


def test_hierarchical_vs_flat_convergence():
    """MAP reserve from hierarchical and flat filters on same trace are within tolerance."""
    from hypostases.inference import infer
    from hypostases.inference.summaries import summarize_map

    actions = [Action(ActionType.REQUEST, amount=2.0) for _ in range(6)]
    pools = [10.0] * 6
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    rng_hier = np.random.default_rng(33)
    rng_flat = np.random.default_rng(33)

    hier_particles = infer_hierarchical(
        observed_actions=actions,
        observed_pool_trace=pools,
        xi=xi,
        n_particles=80,
        macro_n_particles=20,
        rng=rng_hier,
    )
    flat_particles = infer(
        observed_actions=actions,
        observed_pool_trace=pools,
        xi=xi,
        n_particles=80,
        rng=rng_flat,
    )

    hier_map = summarize_map(hier_particles)
    flat_map = summarize_map(flat_particles)

    assert abs(hier_map.c.reserve - flat_map.c.reserve) < 5.0


def test_sample_prior_goal_bias_applied():
    """goal_bias shifts u values in the generated prior."""
    rng = np.random.default_rng(99)
    biased = sample_prior(goal_bias={"RELATIONAL": 10.0}, rng=rng)

    relational_idx = 2  # K = (SURVIVAL, ACQUISITION, RELATIONAL, STATUS)
    assert biased.g.u[relational_idx] > 2.0


def test_sample_prior_unknown_goal_bias_silently_ignored():
    """Unknown goal names in goal_bias do not raise errors."""
    rng = np.random.default_rng(55)
    state = sample_prior(goal_bias={"NONEXISTENT_GOAL": 5.0}, rng=rng)
    assert state is not None
    assert len(state.g.u) == 4


def test_hierarchical_with_lag_window():
    """infer_hierarchical respects lag_window forwarding to both passes."""
    actions = [Action(ActionType.WITHDRAW) for _ in range(8)]
    pools = [10.0] * 8
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    rng = np.random.default_rng(44)
    particles = infer_hierarchical(
        observed_actions=actions,
        observed_pool_trace=pools,
        xi=xi,
        n_particles=30,
        macro_n_particles=10,
        lag_window=3,
        rng=rng,
    )

    assert len(particles) == 30
    assert pytest.approx(sum(p.weight for p in particles)) == 1.0
