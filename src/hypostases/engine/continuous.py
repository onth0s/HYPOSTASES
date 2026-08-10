"""HYPOSTASES Engine — Tier-0 Continuous Substrate Integration.

Spec Ref: Part I §1.2 (Time Model: Tier 0 Substrate), §1.3.
Continuous-time stochastic processes for substrate pool and agent physical metabolism
integrated via Euler-Maruyama step over time interval dt.
"""

from __future__ import annotations

import numpy as np

from hypostases.engine.constants import (
    CONTINUOUS_POOL_DRIFT,
    CONTINUOUS_POOL_NOISE_SD,
    CONTINUOUS_RESERVE_DECAY,
    CONTINUOUS_RESERVE_NOISE_SD,
)
from hypostases.engine.types import AgentState


def step_continuous_substrate(
    pool_before: float,
    dt: float,
    drift_rate: float = CONTINUOUS_POOL_DRIFT,
    noise_sd: float = CONTINUOUS_POOL_NOISE_SD,
    rng: np.random.Generator | None = None,
) -> float:
    """Performs Euler-Maruyama integration step over continuous substrate pool.

    dS_t = drift * dt + noise_sd * dW_t
    """
    if dt <= 0.0:
        return pool_before
    if rng is None:
        rng = np.random.default_rng()

    dw = float(rng.normal(0.0, np.sqrt(dt)))
    delta = drift_rate * dt + noise_sd * dw
    return max(0.0, pool_before + delta)


def step_continuous_agent(
    agent: AgentState,
    dt: float,
    decay_rate: float = CONTINUOUS_RESERVE_DECAY,
    noise_sd: float = CONTINUOUS_RESERVE_NOISE_SD,
    rng: np.random.Generator | None = None,
) -> None:
    """Performs Euler-Maruyama integration step over continuous agent reserve state.

    dc_{reserve} = -decay_rate * dt + noise_sd * dW_t
    """
    if dt <= 0.0:
        return
    if rng is None:
        rng = np.random.default_rng()

    dw = float(rng.normal(0.0, np.sqrt(dt)))
    delta = -decay_rate * dt + noise_sd * dw
    agent.c.reserve = max(0.0, agent.c.reserve + delta)
