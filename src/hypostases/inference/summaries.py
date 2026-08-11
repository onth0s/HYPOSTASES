"""HYPOSTASES Inference — Read-out Summaries of Particle Posteriors.

Spec Ref: Part VII §10.1, §11.
MAP and Kalman-style summaries are read-outs of the SAME particle set,
not separate estimators.
"""

from __future__ import annotations

import numpy as np

from hypostases.engine import AgentState, K, goal_probs
from hypostases.inference.particle_filter import Particle


def summarize_map(particles: list[Particle]) -> AgentState:
    """Part VII §10.1: MAP estimate (highest-weight single particle)."""
    return max(particles, key=lambda p: p.weight).sigma


def summarize_kalman(particles: list[Particle]) -> dict[str, float | list[float]]:
    """Part VII §10.1: Gaussian mean/variance summary over continuous sub-blocks (c, u)."""
    weights = np.array([p.weight for p in particles])
    reserves = np.array([p.sigma.c.reserve for p in particles])
    moods = np.array([p.sigma.c.mood for p in particles])
    u_matrix = np.array([p.sigma.g.u for p in particles])  # Shape (N, N_K)

    reserve_mean = float(np.average(reserves, weights=weights))
    reserve_var = float(np.average((reserves - reserve_mean) ** 2, weights=weights))
    mood_mean = float(np.average(moods, weights=weights))

    u_mean = np.average(u_matrix, axis=0, weights=weights)
    u_var = np.average((u_matrix - u_mean) ** 2, axis=0, weights=weights)

    return {
        "reserve_mean": reserve_mean,
        "reserve_var": reserve_var,
        "mood_mean": mood_mean,
        "u_mean": [float(val) for val in u_mean],
        "u_var": [float(val) for val in u_var],
    }


def goal_posterior(
    particles: list[Particle],
    xi: np.ndarray | None = None,
    pool_belief: float = 10.0,
) -> dict[str, float]:
    """Part VII §10.1: Multimodal goal posterior over category dominant goals.

    Dominant goal is argmax of transient softmax policy allocation computed via goal_probs(sigma, xi, pool_belief).
    """
    if xi is None:
        xi = np.array([0.2, 0.2, 0.2, 0.2])
    weights = np.array([p.weight for p in particles])
    dominant = [
        K[int(np.argmax(goal_probs(p.sigma, xi, pool_belief=pool_belief)))].value for p in particles
    ]
    out = {k.value: 0.0 for k in K}
    for d, w in zip(dominant, weights, strict=True):
        out[d] += w
    return out
