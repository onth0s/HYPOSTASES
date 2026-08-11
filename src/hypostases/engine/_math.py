"""HYPOSTASES Engine — Internal Mathematical Helpers (Private)."""

from __future__ import annotations

import numpy as np

import hypostases.engine.constants as const
from hypostases.engine.constants import (
    ACTION_COSTS,
    SOFTMAX_EPSILON,
)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax operator over 1D vector x."""
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def compute_temperature(xi: np.ndarray | None, offset: float = 0.0) -> float:
    """Computes the scaling temperature using the exploration index xi and offset (Part I §2.2.2).

    Parameters:
        xi: The Index of Exploration context vector (1D array). Its mean acts as a baseline temperature.
        offset: Temperature offset added to the baseline mean.
    """
    base = 1.0 if xi is None else float(np.mean(xi))
    return max(offset + base, SOFTMAX_EPSILON)


def dynamic_action_costs(pool_belief: float) -> np.ndarray:
    """Contention 1: Endogenous Scarcity Action Costs (2025 mechanism design alignment).

    Scales ACTION_COSTS upward as pool_belief drops below SCARCITY_POOL_THRESHOLD:
        C_k(S_t) = C_base_k * (1 + κ * max(0, (S_thresh - S_t) / (S_t + ε)))

    When pool_belief >= SCARCITY_POOL_THRESHOLD, returns unmodified ACTION_COSTS.
    """
    thresh = globals().get("SCARCITY_POOL_THRESHOLD", const.SCARCITY_POOL_THRESHOLD)
    kappa = globals().get("SCARCITY_COST_KAPPA", const.SCARCITY_COST_KAPPA)

    scarcity_pressure = max(0.0, thresh - pool_belief) / (pool_belief + SOFTMAX_EPSILON)
    multiplier = 1.0 + kappa * scarcity_pressure
    return ACTION_COSTS * multiplier


def compute_omega(
    u: np.ndarray,
    reserve: float,
    xi: np.ndarray | None = None,
    pool_belief: float = 10.0,
) -> np.ndarray:
    """ω = derive_Ω(u, ρ_ext, ρ_int, c) (Part I §2.2.2, Part IV §6.5).

    Canonical calculation of willingness scaling transient softmax policy allocation π
    by reserve affordability against dynamically scarcity-adjusted action costs.

    Parameters:
        pool_belief: Current pool belief S_t used to compute dynamic action costs (Contention 1).
            When pool is scarce (< SCARCITY_POOL_THRESHOLD), action costs inflate, reducing
            willingness for expensive actions proportionally.
    """
    temp = compute_temperature(xi, offset=0.0)
    logits = u / temp
    pi_transient = softmax(logits)

    costs = dynamic_action_costs(pool_belief)
    affordability = np.minimum(1.0, reserve / costs)
    return pi_transient * affordability
