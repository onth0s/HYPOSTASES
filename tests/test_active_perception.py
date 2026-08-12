"""Unit Tests for Wave 1 Front 09 — Active Information Gathering & Active Perception.

Spec Ref: Front 09 (Wave 1) — Active Perception & Information Theory Dynamics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hypostases.active_perception import (
    execute_epistemic_action,
    execute_multivariate_epistemic_action,
    load_active_sensing_config,
)
from hypostases.engine.dynamics import feedback
from hypostases.engine.likelihood import action_likelihood
from hypostases.engine.types import (
    EPISTEMIC_ACTION_TYPES,
    Action,
    ActionType,
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)
from hypostases.epistemic_utility import (
    compute_epistemic_utility,
    compute_expected_free_energy,
    compute_expected_information_gain,
    compute_kl_divergence_gaussian,
    compute_learning_progress,
    compute_multivariate_information_gain,
    compute_multivariate_shannon_entropy,
    compute_shannon_entropy,
    compute_variational_free_energy,
)
from hypostases.schemas.audit_schemas import run_completeness_audit


def make_test_state(
    reserve: float = 10.0, sigma2: float = 2.0, resilience: float = 0.5
) -> AgentState:
    return AgentState(
        c=Characteristics(reserve=reserve, resilience=resilience, skill=0.6),
        w=WorldModel(mu=10.0, sigma2=sigma2, replenish_rate_est=1.0),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(time_budget=12.0),
    )


def test_load_active_sensing_config():
    config = load_active_sensing_config()
    assert "epistemic_actions" in config
    assert "INSPECT" in config["epistemic_actions"]
    assert "PROBE" in config["epistemic_actions"]
    assert config["epistemic_weight"] > 0.0


def test_shannon_entropy():
    h1 = compute_shannon_entropy(1.0)
    h2 = compute_shannon_entropy(4.0)
    # H = 0.5 * ln(2 * pi * e * sigma2); higher variance means higher entropy
    assert h2 > h1
    expected_h1 = 0.5 * math.log(2.0 * math.pi * math.e * 1.0)
    assert abs(h1 - expected_h1) < 1e-6


def test_kl_divergence_gaussian():
    kl_same = compute_kl_divergence_gaussian(10.0, 2.0, 10.0, 2.0)
    assert abs(kl_same) < 1e-6

    kl_diff = compute_kl_divergence_gaussian(12.0, 1.0, 10.0, 2.0)
    assert kl_diff > 0.0


def test_expected_information_gain():
    state = make_test_state(sigma2=2.0)
    ig_high_obs_noise = compute_expected_information_gain(
        ActionType.INSPECT, state, obs_variance=1.0
    )
    ig_low_obs_noise = compute_expected_information_gain(
        ActionType.EXPERIMENT, state, obs_variance=0.05
    )
    # Lower observation variance yields higher expected information gain
    assert ig_low_obs_noise > ig_high_obs_noise


def test_epistemic_utility_balancing():
    state = make_test_state(sigma2=2.0)
    act = Action(ActionType.INSPECT)
    cfg = {"efe_mode": False}
    u_pure_pragmatic = compute_epistemic_utility(
        act, state, pragmatic_utility=5.0, beta=0.0, config=cfg
    )
    assert abs(u_pure_pragmatic - 5.0) < 1e-6

    u_balanced = compute_epistemic_utility(act, state, pragmatic_utility=5.0, beta=0.5, config=cfg)
    assert u_balanced != u_pure_pragmatic


def test_execute_epistemic_actions():
    state = make_test_state(reserve=10.0, sigma2=2.0, resilience=0.5)
    act = Action(ActionType.INSPECT)

    delta = execute_epistemic_action(state, act, ground_truth_val=10.0)
    assert "reserve" in delta.delta_c
    assert delta.delta_c["reserve"] < 0.0  # Reserve cost incurred
    assert "time_budget" in delta.delta_rho_ext
    assert delta.delta_rho_ext["time_budget"] < 0.0  # Time budget cost
    assert "sigma2" in delta.delta_w
    assert delta.delta_w["sigma2"] < 0.0  # Variance reduced via precision update


@pytest.mark.parametrize("act_type", list(EPISTEMIC_ACTION_TYPES))
def test_all_epistemic_action_types(act_type: ActionType):
    state = make_test_state(reserve=10.0, sigma2=3.0)
    act = Action(
        act_type, target="peer_1" if act_type in (ActionType.QUERY, ActionType.SPY) else None
    )
    delta = execute_epistemic_action(state, act, ground_truth_val=10.0)
    assert delta.delta_c["reserve"] < 0.0


def test_feedback_epistemic_integration():
    state = make_test_state(reserve=10.0, sigma2=2.5)
    act = Action(ActionType.PROBE)
    delta_log = {}

    delta = feedback(state, pool_before=10.0, pool_after=10.0, action=act, delta_log=delta_log)
    assert "reserve" in delta.delta_c
    assert delta.delta_c["reserve"] < 0.0
    assert "sigma2" in delta.delta_w
    assert delta.delta_w["sigma2"] < 0.0


def test_action_likelihood_epistemic():
    state_high_unc = make_test_state(reserve=10.0, sigma2=5.0)
    state_low_unc = make_test_state(reserve=10.0, sigma2=0.01)

    act = Action(ActionType.INSPECT)
    xi = np.array([0.25, 0.25, 0.25, 0.25])

    lik_high = action_likelihood(state_high_unc, act, xi)
    lik_low = action_likelihood(state_low_unc, act, xi)

    # High state uncertainty gives higher likelihood of epistemic inspection
    assert lik_high > lik_low


def test_completeness_audit():
    assert run_completeness_audit() is True


def test_variational_free_energy():
    """Verifies Variational Free Energy calculation F = D_KL - ln p(o|w) (Dodig-Crnkovic 2022)."""
    f_val = compute_variational_free_energy(
        q_mu=10.0, q_sigma2=1.0, p_mu=10.0, p_sigma2=1.0, log_likelihood=-0.5
    )
    assert f_val == 0.5  # D_KL = 0.0, F = 0.0 - (-0.5) = 0.5


def test_expected_free_energy_and_efe_mode():
    """Verifies Friston Expected Free Energy (EFE) calculation and efe_mode routing (AGENTS.md Rule 009)."""
    state = make_test_state()
    act = Action(ActionType.INSPECT, target="env")

    # Test explicit compute_expected_free_energy
    efe_u = compute_expected_free_energy(act, state, pragmatic_utility=1.0)
    assert efe_u > 1.0  # Pragmatic 1.0 + positive information gain

    # Test routing via compute_epistemic_utility with efe_mode=True
    u_efe = compute_epistemic_utility(act, state, pragmatic_utility=1.0, config={"efe_mode": True})
    assert u_efe == efe_u

    # Test routing via compute_epistemic_utility with efe_mode=False (linear weighted mixing)
    u_linear = compute_epistemic_utility(
        act, state, pragmatic_utility=1.0, beta=0.3, config={"efe_mode": False}
    )
    assert u_linear != u_efe


def test_multivariate_shannon_entropy_and_information_gain():
    """Verifies MacKay (1992) multivariate covariance entropy and log-det information gain."""
    prior_cov = np.diag([1.0, 2.0])
    obs_cov = np.diag([0.2, 0.2])

    h_val = compute_multivariate_shannon_entropy(prior_cov)
    assert isinstance(h_val, float)
    assert h_val > 0.0

    ig_val = compute_multivariate_information_gain(prior_cov, obs_cov)
    assert ig_val > 0.0


def test_learning_progress_iac():
    """Verifies Oudeyer et al. (2007) Intelligent Adaptive Curiosity (IAC) learning progress."""
    lp_val = compute_learning_progress(prev_errors=[2.5, 2.0], current_error=0.5)
    assert lp_val == 2.0  # 2.5 - 0.5 = 2.0 error reduction


def test_morphological_distance_cost():
    """Verifies Dodig-Crnkovic (2022) morphological spatial distance energy drain."""
    state = make_test_state()
    act = Action(ActionType.INSPECT, target="env")

    # Direct observation without distance
    delta_no_dist = execute_epistemic_action(state, act)

    # Observation with target_pos at distance 10.0 from (0,0)
    delta_dist = execute_epistemic_action(state, act, target_pos=(10.0, 0.0), agent_pos=(0.0, 0.0))

    # Reserve cost with distance drain should be strictly greater (more negative delta_c)
    assert delta_dist.delta_c["reserve"] < delta_no_dist.delta_c["reserve"]


def test_execute_multivariate_epistemic_action_option_a():
    """Verifies Option A full joint covariance precision matrix update."""
    prior_mean = np.array([5.0, 10.0])
    prior_cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    obs = np.array([5.5, 10.2])
    obs_cov = np.array([[0.2, 0.0], [0.0, 0.2]])

    mu_post, cov_post, delta_h = execute_multivariate_epistemic_action(
        prior_mean, prior_cov, obs, obs_cov
    )

    assert mu_post.shape == (2,)
    assert cov_post.shape == (2, 2)
    assert delta_h > 0.0  # Variance reduced
