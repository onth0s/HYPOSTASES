"""HYPOSTASES — Causal Policy Evaluator & Cost-Optimal Interventional Planner.

Spec Ref: docs/WAVE_2_FRONT_08/front_08_causal_world_models_spec.md
Synthesizes Zhang & Bareinboim (2019) and Bareinboim et al. (2015).
Evaluates candidate plans using interventional distributions P(g | do(a))
and computes cost-optimal intervention sets X* under resource constraints \rho_{ext}.
"""

from __future__ import annotations

import numpy as np

from hypostases.causal.causal_types import Intervention
from hypostases.causal.structural_causal_model import StructuralCausalModel


class CostOptimalPlanner:
    """Computes minimum-cost interventional target sets X* for goal states (Zhang & Bareinboim 2019)."""

    def __init__(self, scm: StructuralCausalModel) -> None:
        self.scm = scm

    def select_cost_optimal_intervention(
        self,
        goal_variable: str,
        target_goal_value: float,
        candidate_actions: list[str],
        action_costs: dict[str, float] | None = None,
        max_budget: float = np.inf,
    ) -> tuple[str | None, float]:
        """Finds minimum cost intervention x_i* = do(X_i = val) that maximizes P(goal | do(X_i))."""
        action_costs = action_costs or {a: 1.0 for a in candidate_actions}

        best_action: str | None = None
        best_cost = np.inf
        best_utility = -np.inf

        for action_var in candidate_actions:
            cost = action_costs.get(action_var, 1.0)
            if cost > max_budget:
                continue

            # Evaluate interventional response do(action_var = 1.0)
            interv = Intervention(target_values={action_var: 1.0}, cost=cost)
            res_state = self.scm.evaluate_intervention(interv)
            achieved_val = res_state.get(goal_variable, 0.0)

            # Measure proximity to target goal value
            utility = -abs(achieved_val - target_goal_value)

            # Target selection: prioritize higher utility, lower cost
            if utility > best_utility or (abs(utility - best_utility) < 1e-4 and cost < best_cost):
                best_utility = utility
                best_cost = cost
                best_action = action_var

        return best_action, best_cost


class CausalPolicyEvaluator:
    """Evaluates policies and plans using interventional and counterfactual criteria."""

    def __init__(self, scm: StructuralCausalModel) -> None:
        self.scm = scm
        self.planner = CostOptimalPlanner(scm)

    def evaluate_interventional_payoff(
        self, action_var: str, action_val: float, goal_var: str
    ) -> float:
        """Rung 2: Evaluates interventional outcome E(goal | do(action = val))."""
        interv = Intervention(target_values={action_var: action_val})
        state = self.scm.evaluate_intervention(interv)
        return float(state.get(goal_var, 0.0))

    def evaluate_counterfactual_regret(
        self,
        executed_action_var: str,
        executed_action_val: float,
        hypothetical_action_val: float,
        observed_outcome_var: str,
        execution_trace: dict[str, float],
    ) -> float:
        """Rung 3: Evaluates counterfactual regret E(Y_{X=x'} - Y_{X=x} | X=x, Y=y).

        Compares observed outcome against hypothetical outcome if alternative action x' had been taken.
        """
        ett = self.scm.compute_ett(
            treatment_var=executed_action_var,
            treatment_val=hypothetical_action_val,
            baseline_val=executed_action_val,
            outcome_var=observed_outcome_var,
            evidence=execution_trace,
        )
        return ett
