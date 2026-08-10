"""Tests for HYPOSTASES Engine Constants Integrity."""

from hypostases.engine import N_K
from hypostases.engine.constants import (
    ACTION_COSTS,
    LIKELIHOOD_MIN,
    PEER_BELIEF_ALPHA,
    RELATIONAL_U_GAIN,
    REQUEST_MOOD_PENALTY,
    REQUEST_SOCIAL_COST,
    ROUGHEN_RESERVE_SD,
    SHARE_MOOD_BONUS,
    SHARE_SOCIAL_GAIN,
    SIGMA2_MIN,
    SOFTMAX_EPSILON,
    STATUS_COUPLING,
    STATUS_RESERVE_THRESHOLD,
    TEMPERATURE_OFFSET,
    WITHDRAW_MOOD_PENALTY,
    WITHDRAW_SOCIAL_COST,
    WORLD_MU_GAIN,
    WORLD_REPLENISH_GAIN,
    WORLD_SIGMA2_UPDATE_GAIN,
)


def test_action_costs_shape_and_values():
    assert ACTION_COSTS.shape == (N_K,)
    assert (ACTION_COSTS > 0).all()


def test_positive_constants():
    assert SOFTMAX_EPSILON > 0
    assert TEMPERATURE_OFFSET > 0
    assert STATUS_COUPLING > 0
    assert STATUS_RESERVE_THRESHOLD > 0
    assert LIKELIHOOD_MIN > 0
    assert ROUGHEN_RESERVE_SD > 0
    assert SIGMA2_MIN > 0
    assert 0 < PEER_BELIEF_ALPHA <= 1.0


def test_feedback_and_world_gain_coefficients():
    assert REQUEST_MOOD_PENALTY > 0
    assert SHARE_MOOD_BONUS > 0
    assert WITHDRAW_MOOD_PENALTY > 0
    assert SHARE_SOCIAL_GAIN > 0
    assert REQUEST_SOCIAL_COST > 0
    assert WITHDRAW_SOCIAL_COST > 0
    assert RELATIONAL_U_GAIN > 0
    assert WORLD_MU_GAIN > 0
    assert WORLD_REPLENISH_GAIN > 0
    assert WORLD_SIGMA2_UPDATE_GAIN > 0


def test_mood_decay_rate_calibrated_bounds():
    from hypostases.engine.constants import MOOD_DECAY_RATE

    assert 0.05 <= MOOD_DECAY_RATE <= 0.25
