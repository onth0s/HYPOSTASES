"""HYPOSTASES Engine — Epistemic Utility & Information Gain Dynamics.

Spec Ref: Front 09 (Wave 1) — Active Information Gathering.
Quantifies epistemic utility via Shannon Entropy reduction, Gaussian KL-Divergence,
and Bayesian Mutual Information over state uncertainty in w.
"""

from __future__ import annotations

import math
from typing import Any

from hypostases.engine.types import EPISTEMIC_ACTION_TYPES, Action, ActionType, AgentState


def compute_shannon_entropy(sigma2: float) -> float:
    """Computes Shannon entropy H(S) for a continuous Gaussian belief distribution N(mu, sigma2).

    H(S) = 0.5 * ln(2 * pi * e * sigma2)
    """
    sigma2_safe = max(float(sigma2), 1e-8)
    return 0.5 * math.log(2.0 * math.pi * math.e * sigma2_safe)


def compute_kl_divergence_gaussian(
    mu1: float, sigma2_1: float, mu2: float, sigma2_2: float
) -> float:
    """Computes D_KL( N(mu1, sigma2_1) || N(mu2, sigma2_2) )."""
    s1 = max(float(sigma2_1), 1e-8)
    s2 = max(float(sigma2_2), 1e-8)
    term1 = (s1 + (mu1 - mu2) ** 2) / s2
    term2 = math.log(s2 / s1)
    return 0.5 * (term1 - 1.0 + term2)


def compute_expected_information_gain(
    action_type: ActionType,
    agent_state: AgentState,
    obs_variance: float = 0.25,
) -> float:
    """Computes expected information gain (entropy reduction) for an active sensing action.

    Bayesian precision update: 1 / sigma2_post = 1 / sigma2_prior + 1 / obs_variance
    Information Gain Delta H = H(sigma2_prior) - H(sigma2_post) = 0.5 * ln(1 + sigma2_prior / obs_variance)
    """
    prior_var = max(agent_state.w.sigma2, 1e-8)
    obs_var = max(obs_variance, 1e-8)
    post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
    h_prior = compute_shannon_entropy(prior_var)
    h_post = compute_shannon_entropy(post_var)
    return max(0.0, h_prior - h_post)


def compute_epistemic_utility(
    action: Action,
    agent_state: AgentState,
    pragmatic_utility: float = 0.0,
    beta: float = 0.3,
    config: dict[str, Any] | None = None,
) -> float:
    """Calculates combined pragmatic and epistemic utility:

    U_total = (1 - beta) * U_pragmatic + beta * U_epistemic
    """
    if action.action_type not in EPISTEMIC_ACTION_TYPES:
        return pragmatic_utility

    obs_var = 0.25
    if config and "epistemic_actions" in config:
        act_cfg = config["epistemic_actions"].get(action.action_type.value, {})
        obs_var = act_cfg.get("observation_variance", 0.25)

    info_gain = compute_expected_information_gain(
        action.action_type, agent_state, obs_variance=obs_var
    )
    u_epistemic = info_gain

    # Bound beta in [0, 1]
    beta_clamped = max(0.0, min(1.0, float(beta)))
    return (1.0 - beta_clamped) * pragmatic_utility + beta_clamped * u_epistemic
