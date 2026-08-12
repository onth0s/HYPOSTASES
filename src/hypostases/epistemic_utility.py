"""HYPOSTASES Engine — Epistemic Utility & Information Gain Dynamics.

Spec Ref: Front 09 (Wave 1) — Active Information Gathering.
Quantifies epistemic utility via Shannon Entropy reduction, Gaussian KL-Divergence,
and Bayesian Mutual Information over state uncertainty in w.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

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


def compute_variational_free_energy(
    q_mu: float,
    q_sigma2: float,
    p_mu: float,
    p_sigma2: float,
    log_likelihood: float,
) -> float:
    """Computes Variational Free Energy F = D_KL(q(w) || p(w)) - E_q[ln p(o|w)] (Dodig-Crnkovic 2022).

    Measures complexity penalty plus accuracy mismatch for info-computational active inference.
    """
    kl_complexity = compute_kl_divergence_gaussian(q_mu, q_sigma2, p_mu, p_sigma2)
    return float(kl_complexity - log_likelihood)


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


def compute_multivariate_shannon_entropy(cov_matrix: np.ndarray) -> float:
    """Computes continuous Shannon entropy for a d-dimensional Gaussian distribution N(mu, Sigma) (MacKay 1992).

    H(S) = 0.5 * ln( (2 * pi * e)^d * det(Sigma) )
    """
    cov = np.asarray(cov_matrix, dtype=float)
    d = cov.shape[0]
    # Ensure numerical stability via positive semi-definite regularization
    cov_reg = cov + np.eye(d) * 1e-8
    det_val = float(np.linalg.det(cov_reg))
    det_safe = max(det_val, 1e-12)
    return 0.5 * float(np.log(((2.0 * math.pi * math.e) ** d) * det_safe))


def compute_multivariate_information_gain(prior_cov: np.ndarray, obs_cov: np.ndarray) -> float:
    """Computes Expected Information Gain (entropy reduction) for multivariate Gaussian beliefs.

    Log-det entropy reduction: Delta H = 0.5 * ln det( I + Sigma_prior * Sigma_obs^-1 ) (MacKay 1992).
    """
    p_cov = np.asarray(prior_cov, dtype=float)
    o_cov = np.asarray(obs_cov, dtype=float)
    d = p_cov.shape[0]
    p_reg = p_cov + np.eye(d) * 1e-8
    o_reg = o_cov + np.eye(d) * 1e-8
    inv_o = np.linalg.inv(o_reg)
    mat_ratio = np.eye(d) + p_reg @ inv_o
    det_ratio = float(np.linalg.det(mat_ratio))
    det_safe = max(det_ratio, 1.0)
    return 0.5 * float(np.log(det_safe))


def compute_learning_progress(
    prev_errors: list[float] | np.ndarray, current_error: float, decay: float = 0.1
) -> float:
    """Computes Oudeyer et al. (2007) Intelligent Adaptive Curiosity (IAC) Learning Progress (LP_t).

    LP_t = Error(t - tau) - Error(t). Positive value indicates learning velocity / error reduction.
    """
    if not prev_errors:
        return 0.0
    baseline_error = float(prev_errors[0])
    raw_lp = max(0.0, baseline_error - float(current_error))
    return float(raw_lp)


def compute_expected_free_energy(
    action: Action,
    agent_state: AgentState,
    pragmatic_utility: float = 0.0,
    config: dict[str, Any] | None = None,
) -> float:
    """Calculates Friston et al. (2017) Expected Free Energy (EFE) Action Utility.

    Expected Free Energy G(a) = - EpistemicValue(a) - PragmaticValue(a).
    Action utility U_EFE(a) = -G(a) = InformationGain(a) + PragmaticUtility(a).
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
    # Friston EFE unifies epistemic information gain and pragmatic goal utility
    return float(pragmatic_utility + info_gain)


def compute_epistemic_utility(
    action: Action,
    agent_state: AgentState,
    pragmatic_utility: float = 0.0,
    beta: float = 0.3,
    config: dict[str, Any] | None = None,
) -> float:
    """Calculates combined pragmatic and epistemic utility.

    If efe_mode is enabled in config (AGENTS.md Rule 009), routes directly to Friston EFE.
    Otherwise, computes linear weighted utility U_total = (1 - beta) * U_pragmatic + beta * U_epistemic.
    """
    if action.action_type not in EPISTEMIC_ACTION_TYPES:
        return pragmatic_utility

    cfg = config or {}
    use_efe = bool(cfg.get("efe_mode", True))

    if use_efe:
        return compute_expected_free_energy(
            action, agent_state, pragmatic_utility=pragmatic_utility, config=cfg
        )

    obs_var = 0.25
    if "epistemic_actions" in cfg:
        act_cfg = cfg["epistemic_actions"].get(action.action_type.value, {})
        obs_var = act_cfg.get("observation_variance", 0.25)

    info_gain = compute_expected_information_gain(
        action.action_type, agent_state, obs_variance=obs_var
    )
    u_epistemic = info_gain

    # Bound beta in [0, 1]
    beta_clamped = max(0.0, min(1.0, float(beta)))
    return (1.0 - beta_clamped) * pragmatic_utility + beta_clamped * u_epistemic
