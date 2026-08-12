"""HYPOSTASES Engine — Power Dynamics and Projections (Part V §8)."""

from __future__ import annotations

import numpy as np

from hypostases.engine.types import AgentState


def compute_internal_power_projection(agent: AgentState) -> float:
    """Computes proj_int(c): Internal capability power projection (Part V §8.1).

    proj_int(c) = w_r * c.reserve + w_res * c.resilience + w_s * c.sociality + w_stat * c.social_status
    """
    weights = np.array([0.4, 0.3, 0.15, 0.15])
    c_vec = np.array(
        [agent.c.reserve, agent.c.resilience, agent.c.sociality, agent.c.social_status]
    )
    return float(np.dot(weights, c_vec))


def compute_external_power_projection(agent: AgentState) -> float:
    """Computes proj_ext(rho_ext): External institutional power projection (Part V §8.2).

    proj_ext(rho_ext) = w_t * rho_ext.time_budget + w_i * rho_ext.institutional_access + w_sc * rho_ext.social_capital
    """
    weights = np.array([0.33, 0.33, 0.34])
    rho_vec = np.array(
        [
            agent.rho_ext.time_budget,
            agent.rho_ext.institutional_access,
            agent.rho_ext.social_capital,
        ]
    )
    return float(np.dot(weights, rho_vec))


def compute_total_power(agent: AgentState, alpha: float = 0.5) -> float:
    """Computes total agent power projection P_total = alpha * P_int + (1 - alpha) * P_ext."""
    p_int = compute_internal_power_projection(agent)
    p_ext = compute_external_power_projection(agent)
    alpha_clamped = max(0.0, min(1.0, float(alpha)))
    return alpha_clamped * p_int + (1.0 - alpha_clamped) * p_ext
