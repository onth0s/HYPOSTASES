"""HYPOSTASES Engine — Internal Mathematical Helpers (Private)."""

from __future__ import annotations

import numpy as np

from hypostases.engine.constants import ACTION_COSTS, SOFTMAX_EPSILON


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


def compute_omega(u: np.ndarray, reserve: float, xi: np.ndarray | None = None) -> np.ndarray:
    """ω = derive_Ω(u, ρ_ext, ρ_int, c) (Part I §2.2.2, Part IV §6.5).

    Canonical calculation of willingness scaling transient softmax policy allocation π
    by reserve affordability.
    """
    temp = compute_temperature(xi, offset=0.0)
    logits = u / temp
    pi_transient = softmax(logits)

    affordability = np.minimum(1.0, reserve / ACTION_COSTS)
    return pi_transient * affordability
