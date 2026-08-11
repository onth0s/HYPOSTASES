"""Completeness Auditing Tool for HYPOSTASES Schemas.

Spec Ref: Part III §5.8, Directive 003.
Audits schema-level functions (feedback, action_likelihood, amount_fn) for degenerate/constant branches.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from hypostases.engine.types import (
    Action,
    ActionType,
    AgentState,
    Characteristics,
    FeedbackDelta,
    GoalCategory,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)

# Registry of declared simplifications
_DECLARED_SIMPLIFICATIONS: set[str] = set()


def declared_simplification(name: str) -> Callable[[Any], Any]:
    """Decorator to declare a function or method contains an intentional simplification."""

    def decorator(func: Any) -> Any:
        _DECLARED_SIMPLIFICATIONS.add(name)
        if not hasattr(func, "_declared_simplifications"):
            func._declared_simplifications = set()
        func._declared_simplifications.add(name)
        return func

    return decorator


def is_simplification_declared(name: str) -> bool:
    """Returns True if the given simplification name is declared."""
    return name in _DECLARED_SIMPLIFICATIONS


def generate_test_states(seed: int = 42) -> list[AgentState]:
    """Generates a diverse list of 10 AgentState instances to check function dependency on state."""
    rng = np.random.default_rng(seed)
    states: list[AgentState] = []
    for _ in range(10):
        c = Characteristics(
            skill=float(rng.uniform(0.1, 1.0)),
            resilience=float(rng.uniform(0.1, 1.0)),
            sociality=float(rng.uniform(0.1, 1.0)),
            reserve=float(rng.uniform(1.0, 20.0)),
            mood=float(rng.uniform(-0.9, 0.9)),
        )
        w = WorldModel(
            mu=float(rng.uniform(5.0, 15.0)),
            sigma2=float(rng.uniform(0.5, 5.0)),
            replenish_rate_est=float(rng.uniform(0.5, 2.0)),
        )
        g = GoalHierarchy(u=rng.normal(loc=[1.0, 1.0, 1.0, 1.0], scale=0.5))
        rho = PowerExternal(
            social_capital=float(rng.uniform(0.1, 5.0)),
            time_budget=float(rng.uniform(1.0, 12.0)),
        )
        states.append(AgentState(c=c, w=w, g=g, rho_ext=rho))
    return states


def audit_feedback_branch(
    feedback_fn: Callable[..., FeedbackDelta], states: list[AgentState]
) -> list[str]:
    """Audits the feedback function for degenerate branches."""
    violations = []

    # We test for each action type
    for act_type in ActionType:
        # Generate dummy actions
        action = Action(act_type, amount=2.0)

        # Call feedback for all states
        deltas = []
        for state in states:
            # Mock delta log
            delta_log = {
                "pool_before": 10.0,
                "pool_after_shares": 10.0,
                "pool_after": 10.0,
                "shares_total": 0.0,
                "requests_total": 0.0,
                "granted": {"agent": 2.0 if act_type == ActionType.REQUEST else 0.0},
            }
            res = feedback_fn(
                state,
                pool_before=10.0,
                pool_after=10.0,
                action=action,
                delta_log=delta_log,
                agent_name="agent",
            )
            deltas.append(res)

        # Check if feedback output is completely identical across all states
        def to_key(d: FeedbackDelta) -> tuple:
            c_items = tuple(sorted(d.delta_c.items()))
            w_items = tuple(sorted(d.delta_w.items()))
            g_items = tuple(d.delta_g)
            rho_items = tuple(sorted(d.delta_rho_ext.items()))
            return (c_items, w_items, g_items, rho_items)

        keys = [to_key(d) for d in deltas]
        if len(set(keys)) == 1:
            simpl_name = f"feedback_{act_type.value.lower()}"
            if not is_simplification_declared(simpl_name):
                violations.append(
                    f"Feedback branch '{act_type.value}' is degenerate (returns state-independent constant) "
                    f"but is not declared as a simplification via @declared_simplification('{simpl_name}')."
                )
    return violations


def audit_amount_branches(
    goal_spec: dict[GoalCategory, Any], states: list[AgentState]
) -> list[str]:
    """Audits the goal SPEC amount functions for degenerate state-independent constants."""
    violations = []
    for goal, branch in goal_spec.items():
        amounts = [branch.amount_fn(state, 10.0) for state in states]
        if len(set(amounts)) == 1:
            simpl_name = f"amount_{goal.value.lower()}"
            if not is_simplification_declared(simpl_name):
                violations.append(
                    f"Goal '{goal.value}' amount function is degenerate (returns state-independent constant {amounts[0]}) "
                    f"but is not declared as a simplification via @declared_simplification('{simpl_name}')."
                )
    return violations


def audit_likelihood_branch(
    likelihood_fn: Callable[..., float], states: list[AgentState]
) -> list[str]:
    """Audits action_likelihood for state-independent action types."""
    violations = []
    xi = np.array([0.25, 0.25, 0.25, 0.25])

    for act_type in ActionType:
        action = Action(act_type, amount=2.0)
        liks = [likelihood_fn(state, action, xi, pool_belief=10.0) for state in states]

        if len(set(liks)) == 1:
            simpl_name = f"likelihood_{act_type.value.lower()}"
            if not is_simplification_declared(simpl_name):
                violations.append(
                    f"Action likelihood for '{act_type.value}' is state-independent ({liks[0]}) "
                    f"but is not declared as a simplification via @declared_simplification('{simpl_name}')."
                )
    return violations


def run_completeness_audit() -> bool:
    """Runs all completeness audit checks. Returns True if passed, False otherwise."""
    from hypostases.engine.dynamics import GOAL_SPEC, feedback
    from hypostases.engine.likelihood import action_likelihood

    states = generate_test_states()
    violations = []

    violations.extend(audit_feedback_branch(feedback, states))
    violations.extend(audit_amount_branches(GOAL_SPEC, states))
    violations.extend(audit_likelihood_branch(action_likelihood, states))

    if violations:
        for v in violations:
            warnings.warn(v, UserWarning, stacklevel=2)
        return False
    return True
