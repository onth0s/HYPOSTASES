"""HYPOSTASES Engine — Action Likelihood & Generative Goal Evaluation.

Spec Ref: Part II §4.3 (Inverse Inference), Part VII §10.2, §11, §12.4.
Enforces the "one generative model, two directions" invariant by sharing
goal_probs with pi_decision.

Phase 2 addition: ``_is_infeasible`` provides a fast feasibility gate that
short-circuits the Gaussian likelihood computation for physically impossible
particles, saving float arithmetic before weights collapse to LIKELIHOOD_MIN.
"""

from __future__ import annotations

import numpy as np

from hypostases.engine.constants import LIKELIHOOD_MIN, PUNISH_RESERVE_COST, SOFTMAX_EPSILON
from hypostases.engine.dynamics import GOAL_SPEC, goal_probs
from hypostases.engine.types import Action, ActionType, AgentState, GoalCategory, K
from hypostases.schemas import declared_simplification


def _is_infeasible(agent: AgentState, action: Action, pool_belief: float) -> bool:
    """Fast feasibility gate: returns True when the observed action is physically
    impossible for a particle in state ``agent`` given ``pool_belief``.

    Rules (conservative — only prune clear physical violations):
      - REQUEST: infeasible when pool_belief is too small to plausibly grant anything
        meaningful (pool < 10% of requested amount).
      - SHARE: infeasible when agent.c.reserve < action.amount (can't give what you
        don't have).
      - PUNISH: infeasible when agent.c.reserve < PUNISH_RESERVE_COST.
      - WITHDRAW: always feasible (zero-amount status signal; spec §5.8).
    """
    if action.action_type == ActionType.REQUEST:
        return pool_belief < action.amount * 0.1
    if action.action_type == ActionType.SHARE:
        return agent.c.reserve < action.amount
    if action.action_type == ActionType.PUNISH:
        return agent.c.reserve < PUNISH_RESERVE_COST
    # WITHDRAW — always feasible
    return False


def predict_amount(agent: AgentState, goal: GoalCategory, pool_belief: float = 10.0) -> float:
    """Predicts expected action amount for a candidate goal category (Part VII §12.4)."""
    return GOAL_SPEC[goal].amount_fn(agent, pool_belief)


def expected_action_type(goal: GoalCategory) -> ActionType:
    """Maps goal category to expected primary action type."""
    return GOAL_SPEC[goal].action_type


@declared_simplification("likelihood_punish")
def action_likelihood(
    agent: AgentState,
    observed_action: Action,
    xi: np.ndarray,
    pool_belief: float = 10.0,
    amount_sd: float = 1.0,
) -> float:
    """Part VII §10.2, §11, §12.4: Likelihood of observed action given agent hypothesis.

    Evaluates action type match AND amount consistency against each candidate goal branch.

    Directive 003 Branch Audit (Part III §5.8):
      - ActionType match branch: Filters goal hypothesis by action category matching.
      - WITHDRAW & PUNISH amount likelihood: Evaluates as 1.0 (amount-independent signaling/punishment).
      - REQUEST / SHARE amount likelihood: Gaussian error model N(pred_amt, amount_sd^2) state-dependent.

    Phase 2 addition: ``_is_infeasible`` short-circuits to LIKELIHOOD_MIN before the
    Gaussian evaluation when the action is physically impossible for this particle.
    """
    amount_sd = max(amount_sd, SOFTMAX_EPSILON)
    if _is_infeasible(agent, observed_action, pool_belief):
        return LIKELIHOOD_MIN
    probs = goal_probs(agent, xi, pool_belief=pool_belief)
    total_lik = 0.0

    for i, goal in enumerate(K):
        exp_type = expected_action_type(goal)
        if observed_action.action_type == exp_type:
            g_prob = probs[i]
            if observed_action.action_type in (ActionType.WITHDRAW, ActionType.PUNISH):
                amount_lik = 1.0
            else:
                pred_amt = predict_amount(agent, goal, pool_belief)
                diff = observed_action.amount - pred_amt
                amount_lik = (1.0 / (amount_sd * np.sqrt(2 * np.pi))) * np.exp(
                    -0.5 * (diff / amount_sd) ** 2
                )
            total_lik += g_prob * amount_lik

    return float(max(total_lik, LIKELIHOOD_MIN))
