"""HYPOSTASES Schemas — Invariants Validator.

Spec Ref: Part I §2, Part II §3, invariants.yaml.
Programmatic validation of hard constraints on AgentState.
"""

from __future__ import annotations

import math

import numpy as np

from hypostases.engine.types import N_K, AgentState


class InvariantViolationError(ValueError):
    """Raised when an AgentState violates hard invariants specified in invariants.yaml."""


def validate_agent_state(agent: AgentState) -> list[str]:
    """Validates an AgentState instance against all invariants.yaml constraints.

    Returns a list of violation messages (empty list means valid).
    """
    violations: list[str] = []

    # Non-negativity constraints (invariants.yaml: reserve_non_negativity)
    if agent.c.reserve < 0:
        violations.append(f"c.reserve must be non-negative, got {agent.c.reserve}")

    if agent.rho_ext.social_capital < 0:
        violations.append(
            f"rho_ext.social_capital must be non-negative, got {agent.rho_ext.social_capital}"
        )

    if agent.rho_ext.time_budget < 0:
        violations.append(
            f"rho_ext.time_budget must be non-negative, got {agent.rho_ext.time_budget}"
        )

    # Bounded characteristics
    if not (-1.0 <= agent.c.mood <= 1.0):
        violations.append(f"c.mood must be in [-1.0, 1.0], got {agent.c.mood}")

    for trait_name in ("skill", "resilience", "sociality", "memory_decay"):
        val = getattr(agent.c, trait_name)
        if not (0.0 <= val <= 1.0):
            violations.append(f"c.{trait_name} must be in [0.0, 1.0], got {val}")

    # Belief validity (invariants.yaml: belief_validity)
    if agent.w.sigma2 <= 0:
        violations.append(f"w.sigma2 must be strictly positive (> 0), got {agent.w.sigma2}")

    if not math.isfinite(agent.w.replenish_rate_est):
        violations.append(f"w.replenish_rate_est must be finite, got {agent.w.replenish_rate_est}")

    for peer_name, p_val in agent.w.peer_beliefs.items():
        if isinstance(p_val, int | float) and p_val < 0:
            violations.append(f"w.peer_beliefs['{peer_name}'] must be non-negative, got {p_val}")

    # Goal hierarchy dimensionality & validity
    if agent.g.u.shape != (N_K,):
        violations.append(f"g.u must be shape ({N_K},), got {agent.g.u.shape}")

    if np.isnan(agent.g.u).any() or np.isinf(agent.g.u).any():
        violations.append(f"g.u must not contain NaN or Inf values, got {agent.g.u}")

    return violations


def assert_invariants(agent: AgentState) -> None:
    """Raises InvariantViolationError if any invariants are violated."""
    violations = validate_agent_state(agent)
    if violations:
        msg = "AgentState failed invariant validation:\n  - " + "\n  - ".join(violations)
        raise InvariantViolationError(msg)


def assert_schema_completeness() -> None:
    """Audits schema functions and raises InvariantViolationError if degenerate branches are found."""
    from hypostases.schemas.audit_schemas import run_completeness_audit

    if not run_completeness_audit():
        raise InvariantViolationError("Schema completeness audit failed with degenerate branches.")
