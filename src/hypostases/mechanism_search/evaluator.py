"""Evaluator for Mechanism Search Layer (Wave 4 Front 10).

Computes social welfare aggregation (Productivity x Equality, Rawlsian Max-Min, Pareto Efficiency,
Weighted Linear), Incentive Compatibility (IC) regret, Individual Rationality (IR), and Budget Balance penalties.
"""

from typing import Any

import numpy as np

from hypostases.mechanism_search.mechanism_space import MechanismCandidate


class MechanismEvaluator:
    """Evaluates multi-agent trajectory simulation outputs for mechanism candidate mu."""

    def __init__(
        self,
        aggregator_type: str = "productivity_gini",
        lambda_ic: float = 10.0,
        lambda_ir: float = 10.0,
        lambda_budget: float = 5.0,
    ):
        self.aggregator_type = aggregator_type
        self.lambda_ic = lambda_ic
        self.lambda_ir = lambda_ir
        self.lambda_budget = lambda_budget

    def compute_gini(self, utilities: list[float]) -> float:
        """Compute Gini inequality index over non-negative agent utilities."""
        if not utilities or len(utilities) <= 1:
            return 0.0
        arr = np.sort(np.array(utilities, dtype=np.float64))
        n = len(arr)
        sum_u = np.sum(arr)
        if sum_u <= 0.0:
            return 0.0
        index = np.arange(1, n + 1)
        return float((2.0 * np.sum(index * arr) / (n * sum_u)) - (n + 1.0) / n)

    def compute_social_welfare(
        self, utilities: list[float], pareto_max: float = 100.0
    ) -> dict[str, float]:
        """Compute multi-criteria social welfare based on configured aggregator_type."""
        if not utilities:
            return {"welfare": 0.0, "productivity": 0.0, "equality": 1.0, "gini": 0.0}

        productivity = float(np.sum(utilities))
        gini = self.compute_gini(utilities)
        equality = max(0.0, 1.0 - gini)

        if self.aggregator_type == "productivity_gini":
            welfare = productivity * equality
        elif self.aggregator_type == "rawlsian_maxmin":
            welfare = float(np.min(utilities))
        elif self.aggregator_type == "pareto_efficiency":
            welfare = productivity / max(pareto_max, 1e-6)
        elif self.aggregator_type == "weighted_linear":
            welfare = 0.5 * productivity + 0.5 * (equality * 10.0)
        else:
            # Default fallback: productivity * equality
            welfare = productivity * equality

        return {
            "welfare": float(welfare),
            "productivity": productivity,
            "equality": equality,
            "gini": gini,
        }

    def compute_ic_regret(
        self,
        candidate: MechanismCandidate,
        bids: list[float],
        valuations: list[float],
        state: dict[str, Any],
    ) -> float:
        """Compute Incentive Compatibility (IC) regret violation: max_{b'_i} U_i(b'_i, b_{-i}) - U_i(b_i, b_{-i})."""
        n = len(bids)
        if n == 0:
            return 0.0

        allocs = candidate.allocation_rule.allocate(bids, state)
        payments = candidate.payment_rule.calculate_payments(bids, allocs, state)
        true_utilities = [v * a - p for v, a, p in zip(valuations, allocs, payments, strict=False)]

        regret_sum = 0.0
        # Evaluate grid of potential misreports
        misreport_grid = np.linspace(0.0, max(valuations) * 1.5 + 1.0, 10)

        for i in range(n):
            best_misreport_utility = true_utilities[i]
            for b_prime in misreport_grid:
                test_bids = list(bids)
                test_bids[i] = float(b_prime)
                test_allocs = candidate.allocation_rule.allocate(test_bids, state)
                test_payments = candidate.payment_rule.calculate_payments(
                    test_bids, test_allocs, state
                )
                u_misreport = valuations[i] * test_allocs[i] - test_payments[i]
                if u_misreport > best_misreport_utility:
                    best_misreport_utility = u_misreport

            regret_sum += max(0.0, best_misreport_utility - true_utilities[i])

        return float(regret_sum)

    def compute_ir_violation(
        self,
        candidate: MechanismCandidate,
        bids: list[float],
        valuations: list[float],
        state: dict[str, Any],
        reservation_utilities: list[float] | None = None,
    ) -> float:
        """Compute Individual Rationality (IR) violation penalty: sum max(0, U_reservation - U_i)."""
        n = len(bids)
        if n == 0:
            return 0.0
        if reservation_utilities is None:
            reservation_utilities = [0.0] * n

        allocs = candidate.allocation_rule.allocate(bids, state)
        payments = candidate.payment_rule.calculate_payments(bids, allocs, state)
        utilities = [v * a - p for v, a, p in zip(valuations, allocs, payments, strict=False)]

        ir_sum = 0.0
        for i in range(n):
            ir_sum += max(0.0, reservation_utilities[i] - utilities[i])
        return float(ir_sum)

    def compute_budget_balance_violation(
        self,
        candidate: MechanismCandidate,
        bids: list[float],
        allocations: list[float],
        payments: list[float],
        state: dict[str, Any],
    ) -> float:
        """Compute Budget Balance violation penalty: |sum(P_i) - TotalSubsidies|.

        Only enforced if budget_balance_required is True in state or candidate config.
        For standard single-item auctions, budget balance penalty is 0.0.
        """
        if not state.get("budget_balance_required", False) and candidate.payment_rule.rule_type in (
            "second_highest_price",
            "vickrey",
            "virtual_valuation_max",
            "virtual_threshold",
        ):
            return 0.0

        total_payments = float(sum(payments))
        total_subsidies = float(state.get("total_subsidies", 0.0))
        return float(abs(total_payments - total_subsidies))

    def evaluate_candidate(
        self,
        candidate: MechanismCandidate,
        bids: list[float],
        valuations: list[float],
        state: dict[str, Any],
    ) -> float:
        """Compute complete objective fitness score J(mu) for candidate mechanism mu."""
        allocs = candidate.allocation_rule.allocate(bids, state)
        payments = candidate.payment_rule.calculate_payments(bids, allocs, state)
        agent_utilities = [v * a - p for v, a, p in zip(valuations, allocs, payments, strict=False)]

        welfare_metrics = self.compute_social_welfare(agent_utilities)
        welfare = welfare_metrics["welfare"]

        r_ic = self.compute_ic_regret(candidate, bids, valuations, state)
        r_ir = self.compute_ir_violation(candidate, bids, valuations, state)
        r_budget = self.compute_budget_balance_violation(candidate, bids, allocs, payments, state)

        fitness = (
            welfare - self.lambda_ic * r_ic - self.lambda_ir * r_ir - self.lambda_budget * r_budget
        )
        candidate.fitness_score = float(fitness)
        return float(fitness)
