"""HYPOSTASES Engine — Power Dynamics and Projections (Part I §2.2.2, Part IV §6.6b)."""

from __future__ import annotations

import numpy as np

from hypostases.engine.types import AgentState


def compute_internal_power_projection(agent: AgentState) -> float:
    """Computes proj_int(c): Internal capability power projection (Part I §2.2.2, Part IV §6.6b).

    Per Specification v4 §6.6b, internal Power is a derived read-only view read directly
    from Characteristics: reserve_capacity = c.reserve.
    """
    return float(agent.c.reserve)


def compute_external_power_projection(
    agent: AgentState,
    w_time: float = 0.5,
    w_social: float = 0.5,
) -> float:
    """Computes proj_ext(rho_ext): External institutional power projection (Part I §2.2.2, Part IV §6.6a).

    proj_ext(rho_ext) = w_time * rho_ext.time_budget + w_social * rho_ext.social_capital
    """
    weights = np.array([float(w_time), float(w_social)])
    rho_vec = np.array([agent.rho_ext.time_budget, agent.rho_ext.social_capital])
    return float(np.dot(weights, rho_vec))


def compute_total_power(
    agent: AgentState,
    alpha: float = 0.5,
    w_time: float = 0.5,
    w_social: float = 0.5,
) -> float:
    """Computes total agent power projection P_total = alpha * P_int + (1 - alpha) * P_ext."""
    p_int = compute_internal_power_projection(agent)
    p_ext = compute_external_power_projection(agent, w_time=w_time, w_social=w_social)
    alpha_clamped = max(0.0, min(1.0, float(alpha)))
    return alpha_clamped * p_int + (1.0 - alpha_clamped) * p_ext
