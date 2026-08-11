"""HYPOSTASES Inference — Systematic Resampling & Reserve Roughening Algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hypostases.engine.constants import ROUGHEN_RESERVE_SD

if TYPE_CHECKING:
    from hypostases.inference.particle_filter import JointParticle, Particle


def systematic_resample_indices(
    weights: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Low-variance systematic resampling indices from particle weight distribution."""
    cumsum = np.cumsum(weights)
    u0 = rng.uniform(0.0, 1.0 / n)
    positions = u0 + np.arange(n) / n
    idx = np.searchsorted(cumsum, positions)
    return np.clip(idx, 0, len(weights) - 1)


def resample_particles(
    particles: list[Particle],
    n: int,
    roughen_reserve_sd: float = ROUGHEN_RESERVE_SD,
    rng: np.random.Generator | None = None,
) -> list[Particle]:
    """Part VII §12.7: Systematic resampling with post-resample reserve roughening."""
    from hypostases.inference.particle_filter import Particle

    if rng is None:
        rng = np.random.default_rng()

    reserves = np.array([p.sigma.c.reserve for p in particles])
    std_reserve = float(np.std(reserves))
    weights = np.array([p.weight for p in particles])

    idx = systematic_resample_indices(weights, n, rng)
    resampled = [Particle(sigma=particles[i].sigma.clone(), weight=1.0 / n) for i in idx]

    dynamic_sd = max(0.05, roughen_reserve_sd * std_reserve)
    if dynamic_sd > 0:
        for p in resampled:
            p.sigma.c.reserve = max(0.0, p.sigma.c.reserve + float(rng.normal(0, dynamic_sd)))

    return resampled


def resample_joint_particles(
    particles: list[JointParticle],
    n: int,
    roughen_reserve_sd: float = ROUGHEN_RESERVE_SD,
    rng: np.random.Generator | None = None,
) -> list[JointParticle]:
    """Resamples joint particles and applies adaptive roughening per agent state."""
    from hypostases.inference.particle_filter import JointParticle

    if rng is None:
        rng = np.random.default_rng()

    weights = np.array([p.weight for p in particles])
    idx = systematic_resample_indices(weights, n, rng)

    resampled = []
    for i in idx:
        sigmas_clone = {name: state.clone() for name, state in particles[i].sigmas.items()}
        resampled.append(JointParticle(sigmas=sigmas_clone, weight=1.0 / n))

    for name in particles[0].sigmas:
        reserves = np.array([p.sigmas[name].c.reserve for p in particles])
        std_reserve = float(np.std(reserves))
        dynamic_sd = max(0.05, roughen_reserve_sd * std_reserve)

        if dynamic_sd > 0:
            for p in resampled:
                p.sigmas[name].c.reserve = max(
                    0.0, p.sigmas[name].c.reserve + float(rng.normal(0, dynamic_sd))
                )

    return resampled
