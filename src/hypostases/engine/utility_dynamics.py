"""HYPOSTASES Engine — Utility Dynamics and Goal Probabilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np

from hypostases.engine._math import compute_temperature, softmax
from hypostases.engine.constants import (
    STATUS_COUPLING,
    STATUS_RESERVE_THRESHOLD,
    TEMPERATURE_OFFSET,
)
from hypostases.engine.types import (
    EPISTEMIC_ACTION_TYPES,
    Action,
    ActionType,
    AgentState,
    GoalCategory,
    K,
)
from hypostases.schemas import declared_simplification


def _survival_amount(ag: AgentState, pool: float) -> float:
    return max(0.5, 10.0 - ag.c.reserve)


@declared_simplification("amount_acquisition")
def _acquisition_amount(ag: AgentState, pool: float) -> float:
    return min(5.0, max(1.0, pool * 0.3))


def _relational_amount(ag: AgentState, pool: float) -> float:
    return min(ag.c.reserve * 0.2, 3.0)


@declared_simplification("amount_status")
def _status_amount(ag: AgentState, pool: float) -> float:
    return 0.0


@dataclass(frozen=True)
class GoalBranch:
    action_type: ActionType
    amount_fn: Callable[[AgentState, float], float]


# Directive 003 Branch Audit (Part III §5.8):
#   - SURVIVAL: ActionType.REQUEST, state-dependent on reserve deficit (10.0 - c.reserve).
#   - ACQUISITION: ActionType.REQUEST, state-dependent on pool_belief (pool * 0.3).
#   - RELATIONAL: ActionType.SHARE, state-dependent on reserve capacity (c.reserve * 0.2).
#   - STATUS: ActionType.WITHDRAW, zero resource exchange (declared simplification for status signaling).
GOAL_SPEC: dict[GoalCategory, GoalBranch] = {
    GoalCategory.SURVIVAL: GoalBranch(ActionType.REQUEST, _survival_amount),
    GoalCategory.ACQUISITION: GoalBranch(ActionType.REQUEST, _acquisition_amount),
    GoalCategory.RELATIONAL: GoalBranch(ActionType.SHARE, _relational_amount),
    GoalCategory.STATUS: GoalBranch(ActionType.WITHDRAW, _status_amount),
}

_STATUS_IDX: Final[int] = K.index(GoalCategory.STATUS)
_RELATIONAL_IDX: Final[int] = K.index(GoalCategory.RELATIONAL)
_ACQUISITION_IDX: Final[int] = K.index(GoalCategory.ACQUISITION)
_SURVIVAL_IDX: Final[int] = K.index(GoalCategory.SURVIVAL)


def _effective_utilities(agent: AgentState, coupling: float = STATUS_COUPLING) -> np.ndarray:
    """Computes effective goal utilities u_eff with reserve sensitivity on STATUS (Part VII §12.5)."""
    u_eff = agent.g.u.copy()

    reserve_factor = 1.0 + coupling * max(0.0, agent.c.reserve - STATUS_RESERVE_THRESHOLD)
    u_eff[_STATUS_IDX] *= reserve_factor
    return u_eff


def goal_probs(
    agent: AgentState,
    xi: np.ndarray,
    coupling: float = STATUS_COUPLING,
    pool_belief: float = 10.0,
) -> np.ndarray:
    """Computes transient goal probabilities π ∈ Δ(K) dynamically from latent utilities u (v4).

    Shared generative goal distribution between pi_decision and action_likelihood.

    Parameters:
        agent: The AgentState containing primitives and weights.
        xi: The Index of Exploration context vector used to compute logit scaling temperature.
        coupling: Status-reserve coupling coefficient.
        pool_belief: Current pool estimate S_t used for endogenous scarcity action cost scaling (Contention 1).
    """
    u_eff = _effective_utilities(agent, coupling=coupling)
    omega = agent.omega(xi, pool_belief=pool_belief)
    logits = omega * u_eff
    temperature = compute_temperature(xi, offset=TEMPERATURE_OFFSET)
    return softmax(logits / temperature)


def action_likelihood(
    agent: AgentState,
    action: Action,
    xi: np.ndarray,
    pool_belief: float = 10.0,
) -> float:
    """Part II §3.1, Part IV §6.7a: Computes marginal likelihood P(a | σ, ξ).

    Directive 003 Branch Audit (Part III §5.8):
    Maps observed action type to generating goal categories π ∈ Δ(K).
      - ActionType.REQUEST -> Sum(π_SURVIVAL, π_ACQUISITION)
      - ActionType.SHARE -> π_RELATIONAL
      - ActionType.WITHDRAW -> π_STATUS
      - Epistemic actions -> Epistemic utility softmax probability
    """
    if action.action_type in EPISTEMIC_ACTION_TYPES:
        from hypostases.epistemic_utility import compute_epistemic_utility

        u_epi = compute_epistemic_utility(action, agent)
        temperature = compute_temperature(xi, offset=TEMPERATURE_OFFSET)
        return float(softmax(np.array([u_epi, 0.0]) / temperature)[0])

    probs = goal_probs(agent, xi, pool_belief=pool_belief)

    if action.action_type == ActionType.REQUEST:
        return float(probs[_SURVIVAL_IDX] + probs[_ACQUISITION_IDX])
    elif action.action_type == ActionType.SHARE:
        return float(probs[_RELATIONAL_IDX])
    elif action.action_type == ActionType.WITHDRAW:
        return float(probs[_STATUS_IDX])
    elif action.action_type == ActionType.PUNISH:
        return 0.05  # Declared baseline probability simplification for altruistic punishment
    return 0.01
