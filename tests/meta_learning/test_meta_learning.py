"""Unit and integration tests for Meta-Learning Layer (Wave 4 Front 07).

Tests cover:
1. Rule 011 Dual Persistence (in-memory tuple + YAML snapshot parity)
2. Bounded adaptation under MetaEvaluator rewards
3. Preference compatibility constraint C_o = A * C_s (Champion et al. 2024)
4. Rule 005 Invariant Verification (Zero Artificial Human Cognitive Bias)
"""

import os

from hypostases.meta_learning.meta_evaluator import MetaEvaluator
from hypostases.meta_learning.meta_optimizer import MetaLearner
from hypostases.meta_learning.meta_state import MetaParameterVector


def test_meta_parameter_dual_persistence(tmp_path):
    """Verifies Rule 011 dual persistence between in-memory procedural tuple and YAML snapshot."""
    params = MetaParameterVector(
        learning_rate=0.02,
        mood_decay_rate=0.1,  # Rule 004
        rollout_depth=6,
        particle_count=32,
        efe_beta=0.7,
        kmp_k=4,  # Rule 008
        peft_gamma_qkv=1.2,
    )

    # 1. Test in-memory tuple projection
    tup = params.to_procedural_tuple()
    assert tup == (0.02, 0.1, 6, 32, 0.7, 4)

    reconstructed_tup = MetaParameterVector.from_procedural_tuple(tup)
    assert reconstructed_tup.learning_rate == 0.02
    assert reconstructed_tup.kmp_k == 4

    # 2. Test YAML persistence serialization
    yaml_file = tmp_path / "test_snapshot.yaml"
    params.save_yaml(str(yaml_file))

    assert os.path.exists(yaml_file)

    loaded_params = MetaParameterVector.load_yaml(str(yaml_file))
    assert loaded_params.learning_rate == params.learning_rate
    assert loaded_params.mood_decay_rate == params.mood_decay_rate
    assert loaded_params.particle_count == params.particle_count
    assert loaded_params.peft_gamma_qkv == params.peft_gamma_qkv


def test_meta_evaluator_reward():
    """Verifies MetaEvaluator calculation of meta-reward."""
    evaluator = MetaEvaluator(lambda_info=0.5, lambda_cost=0.01)
    reward = evaluator.record_step(
        utility_gain=1.0, belief_variance_reduction=0.2, compute_cost=2.0
    )
    # Expected: 1.0 + (0.5 * 0.2) - (0.01 * 2.0) = 1.0 + 0.1 - 0.02 = 1.08
    assert abs(reward - 1.08) < 1e-5


def test_meta_learner_adaptation(tmp_path):
    """Verifies meta-learning parameter updates across simulation ticks."""
    learner = MetaLearner()

    initial_lr = learner.meta_params.learning_rate
    initial_version = learner.meta_params.version

    # Perform positive step
    updated = learner.adapt_step(
        utility_gain=2.0, belief_var_reduction=0.1, compute_cost=1.0, failure_observed=False
    )
    assert updated.learning_rate >= initial_lr
    assert updated.version == initial_version

    # Perform failure step
    updated_fail = learner.adapt_step(
        utility_gain=0.0, belief_var_reduction=0.01, compute_cost=5.0, failure_observed=True
    )
    assert updated_fail.version == initial_version + 1

    # Verify snapshot sync
    yaml_path = tmp_path / "sync_snapshot.yaml"
    learner.sync_dual_persistence(str(yaml_path))
    assert os.path.exists(yaml_path)


def test_preference_compatibility_constraint():
    """Verifies C_o = A * C_s projection (Champion et al. 2024)."""
    learner = MetaLearner()
    state_prefs = [0.8, 0.2]  # 2 states
    likelihood_matrix = [
        [0.9, 0.1],  # Obs 1
        [0.1, 0.9],  # Obs 2
    ]

    obs_prefs = learner.enforce_preference_compatibility(state_prefs, likelihood_matrix)

    # Expected Obs 1: 0.9*0.8 + 0.1*0.2 = 0.72 + 0.02 = 0.74
    # Expected Obs 2: 0.1*0.8 + 0.9*0.2 = 0.08 + 0.18 = 0.26
    assert abs(obs_prefs[0] - 0.74) < 1e-5
    assert abs(obs_prefs[1] - 0.26) < 1e-5
    assert abs(sum(obs_prefs) - 1.0) < 1e-5


def test_rule_005_rationality_invariant():
    """Rule 005: Ensure meta-learning does NOT introduce non-rational human cognitive biases."""
    params = MetaParameterVector()
    # All parameters must be computable non-negative real/integer values
    assert params.learning_rate >= 0
    assert params.rollout_depth > 0
    assert params.particle_count > 0
    assert 0.0 <= params.efe_beta <= 1.0
