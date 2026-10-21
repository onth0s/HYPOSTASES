"""HYPOSTASES Engine — Active Perception & Active Information Gathering."""

from __future__ import annotations

from hypostases.active_perception.engine import (
    execute_epistemic_action,
    execute_multivariate_epistemic_action,
    load_active_sensing_config,
)
from hypostases.engine.dynamics import register_epistemic_handler

# Register default active perception dispatcher with engine (Phase 3 Dependency Inversion)
register_epistemic_handler(
    lambda agent, action, pool_val: execute_epistemic_action(
        agent, action, ground_truth_val=pool_val
    )
)

__all__ = [
    "execute_epistemic_action",
    "execute_multivariate_epistemic_action",
    "load_active_sensing_config",
]
