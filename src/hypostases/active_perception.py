"""HYPOSTASES Engine — Active Perception & Active Information Gathering.

Spec Ref: Front 09 (Wave 1) — Active Sensing Dynamics.
Implements active perception routines: executing epistemic actions (INSPECT, PROBE,
MONITOR, QUERY, EXPERIMENT, VERIFY, OBSERVE, SPY), performing Bayesian belief state
updates over w, and allocating epistemic costs across primitive state variables.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hypostases.engine.types import (
    EPISTEMIC_ACTION_TYPES,
    Action,
    AgentState,
    DeltaCharacteristics,
    DeltaPowerExternal,
    DeltaWorldModel,
    FeedbackDelta,
)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "schema" / "active_sensing_config.yaml"


def load_active_sensing_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Loads active sensing YAML configuration file (Rule 006 data-driven approach)."""
    target_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not target_path.exists():
        # Fallback default configuration
        return {
            "epistemic_weight": 0.3,
            "min_variance_threshold": 0.05,
            "epistemic_actions": {
                "INSPECT": {
                    "cost_reserve": 0.2,
                    "cost_time": 0.5,
                    "observation_variance": 0.25,
                    "target_property": "environment_mu",
                },
                "PROBE": {
                    "cost_reserve": 0.5,
                    "cost_time": 1.0,
                    "observation_variance": 0.10,
                    "target_property": "replenish_rate",
                },
                "MONITOR": {
                    "cost_reserve": 0.1,
                    "cost_time": 0.2,
                    "observation_variance": 0.50,
                    "target_property": "environment_mu",
                },
                "QUERY": {
                    "cost_reserve": 0.1,
                    "cost_time": 0.3,
                    "observation_variance": 0.30,
                    "target_property": "peer_belief",
                },
                "EXPERIMENT": {
                    "cost_reserve": 1.2,
                    "cost_time": 2.0,
                    "observation_variance": 0.05,
                    "target_property": "replenish_rate",
                },
                "VERIFY": {
                    "cost_reserve": 0.3,
                    "cost_time": 0.4,
                    "observation_variance": 0.15,
                    "target_property": "environment_mu",
                },
                "OBSERVE": {
                    "cost_reserve": 0.05,
                    "cost_time": 0.1,
                    "observation_variance": 0.60,
                    "target_property": "environment_mu",
                },
                "SPY": {
                    "cost_reserve": 0.8,
                    "cost_time": 1.5,
                    "observation_variance": 0.20,
                    "target_property": "peer_reserve",
                },
            },
        }

    with open(target_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def execute_epistemic_action(
    agent: AgentState,
    action: Action,
    ground_truth_val: float = 10.0,
    config: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
    target_pos: tuple[float, float] | None = None,
    agent_pos: tuple[float, float] = (0.0, 0.0),
) -> FeedbackDelta:
    """Executes an active sensing action, calculating Bayesian updates and state costs.

    Parameters:
        agent: Current primitive state σ = (c, w, g, ρ_ext).
        action: Epistemic action (e.g. INSPECT, PROBE, MONITOR, etc.).
        ground_truth_val: Physical true state value being observed.
        config: Loaded active sensing config dict.
        rng: Optional random generator for noisy observation sampling.
        target_pos: Optional spatial location (x, y) of target for Dodig-Crnkovic distance cost.
        agent_pos: Spatial location (x, y) of sensing agent.

    Returns:
        FeedbackDelta containing delta_c, delta_w, delta_rho_ext, etc.
    """
    if action.action_type not in EPISTEMIC_ACTION_TYPES:
        raise ValueError(f"Action {action.action_type} is not an epistemic active sensing action.")

    cfg = config or load_active_sensing_config()
    act_name = action.action_type.value
    act_spec = cfg.get("epistemic_actions", {}).get(act_name, {})

    base_cost_reserve = float(act_spec.get("cost_reserve", 0.2))
    base_cost_time = float(act_spec.get("cost_time", 0.5))
    obs_variance = float(act_spec.get("observation_variance", 0.25))
    target_prop = str(act_spec.get("target_property", "environment_mu"))
    kappa = float(cfg.get("distance_cost_coefficient", 0.1))

    # Dodig-Crnkovic (2022) Morphological Spatial Distance Cost
    dist_cost = 0.0
    if target_pos is not None:
        dx = float(target_pos[0] - agent_pos[0])
        dy = float(target_pos[1] - agent_pos[1])
        distance = math.sqrt(dx * dx + dy * dy)
        dist_cost = kappa * distance

    # State-dependent efficiency scaling: higher resilience reduces reserve cost
    resilience_factor = max(0.1, agent.c.resilience)
    actual_reserve_cost = (base_cost_reserve + dist_cost) / (0.5 + 0.5 * resilience_factor)

    delta_c: DeltaCharacteristics = {"reserve": -actual_reserve_cost}
    delta_rho_ext: DeltaPowerExternal = {"time_budget": -base_cost_time}

    # Bayesian Precision Update for N(mu, sigma2)
    generator = rng or np.random.default_rng()
    noise_std = np.sqrt(obs_variance)
    sample_obs = float(generator.normal(loc=ground_truth_val, scale=noise_std))

    prior_mu = agent.w.mu
    prior_var = max(agent.w.sigma2, 1e-8)

    precision_prior = 1.0 / prior_var
    precision_obs = 1.0 / obs_variance
    precision_post = precision_prior + precision_obs

    post_var = 1.0 / precision_post
    post_mu = post_var * (precision_prior * prior_mu + precision_obs * sample_obs)

    delta_mu = post_mu - prior_mu
    delta_var = post_var - prior_var

    delta_w: DeltaWorldModel = {}
    delta_peer_beliefs: dict[str, float] = {}

    if target_prop == "replenish_rate":
        prior_rep = agent.w.replenish_rate_est
        delta_w["replenish_rate_est"] = 0.5 * (sample_obs - prior_rep)
        delta_w["sigma2"] = delta_var
    elif target_prop in ("peer_belief", "peer_reserve"):
        if action.target:
            delta_peer_beliefs[action.target] = sample_obs
    else:  # environment_mu
        delta_w["mu"] = delta_mu
        delta_w["sigma2"] = delta_var

    # State-dependent goal utility delta (curiosity satisfaction boosts mood slightly)
    delta_g = np.zeros(4)

    return FeedbackDelta(
        delta_c=delta_c,
        delta_w=delta_w,
        delta_g=delta_g,
        delta_rho_ext=delta_rho_ext,
        delta_peer_beliefs=delta_peer_beliefs,
    )


def execute_multivariate_epistemic_action(
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    observation: np.ndarray,
    obs_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Executes Option A Full Joint Covariance Matrix Precision Update (MacKay 1992).

    Lambda_post = Lambda_prior + Lambda_obs
    Sigma_post = Lambda_post^-1
    mu_post = Sigma_post * ( Lambda_prior * mu_prior + Lambda_obs * observation )

    Returns:
        tuple (mu_post, Sigma_post, delta_H_logdet)
    """
    mu_p = np.asarray(prior_mean, dtype=float)
    sigma_p = np.asarray(prior_cov, dtype=float)
    obs = np.asarray(observation, dtype=float)
    sigma_o = np.asarray(obs_cov, dtype=float)

    d = mu_p.shape[0]
    reg = np.eye(d) * 1e-8

    lambda_p = np.linalg.inv(sigma_p + reg)
    lambda_o = np.linalg.inv(sigma_o + reg)

    lambda_post = lambda_p + lambda_o
    sigma_post = np.linalg.inv(lambda_post)

    mu_post = sigma_post @ (lambda_p @ mu_p + lambda_o @ obs)

    # Log-det Shannon entropy reduction
    det_prior = max(float(np.linalg.det(sigma_p + reg)), 1e-12)
    det_post = max(float(np.linalg.det(sigma_post + reg)), 1e-12)
    delta_h = 0.5 * float(np.log(det_prior / det_post))

    return mu_post, sigma_post, delta_h
