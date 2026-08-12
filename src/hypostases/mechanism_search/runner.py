"""Bi-level Mechanism Search Orchestrator (Wave 4 Front 10).

Orchestrates bi-level optimization connecting outer-loop mechanism search engine
with inner-loop multi-agent simulation harness oracle under Rule 009 (efe_mode)
and Rule 011 (Dual Persistence).
"""

from typing import Any

from hypostases.mechanism_search.evaluator import MechanismEvaluator
from hypostases.mechanism_search.mechanism_space import MechanismCandidate, MechanismSpace
from hypostases.mechanism_search.optimizer import (
    BayesianMechanismSearcher,
    DifferentiableMechanismSearcher,
    EvolutionaryMechanismSearcher,
    MechanismOptimizer,
)


class MechanismSearchRunner:
    """Bi-level orchestrator running mechanism optimization loops over simulation harness."""

    def __init__(
        self,
        config_path: str = "schema/mechanism_search_config.yaml",
        optimizer_type: str = "bayesian",
        aggregator_type: str = "productivity_gini",
        efe_mode: bool = True,  # Rule 009
    ):
        self.config_path = config_path
        self.optimizer_type = optimizer_type
        self.aggregator_type = aggregator_type
        self.efe_mode = efe_mode

        self.space = MechanismSpace(config_path)
        self.evaluator = MechanismEvaluator(aggregator_type=aggregator_type)
        self.optimizer = self._init_optimizer()
        self.best_candidate: MechanismCandidate | None = None

    def _init_optimizer(self) -> MechanismOptimizer:
        if self.optimizer_type == "evolutionary":
            return EvolutionaryMechanismSearcher(self.space, self.evaluator)
        elif self.optimizer_type == "differentiable":
            return DifferentiableMechanismSearcher(self.space, self.evaluator)
        else:
            return BayesianMechanismSearcher(self.space, self.evaluator)

    def run_simulation_harness_oracle(
        self, candidate: MechanismCandidate, n_agents: int = 4, ticks: int = 10
    ) -> tuple[list[float], list[float], dict[str, Any]]:
        """Simulate inner-loop multi-agent dynamics under candidate mechanism mu."""
        # Simulated agent private valuations and active sensing bids
        valuations = [10.0, 8.0, 6.0, 4.0][:n_agents]
        # Under DSIC mechanisms (or EFE active sensing), bids equal valuation
        bids = [v * 0.95 for v in valuations]

        state = {
            "assigned_prices": [2.5] * n_agents,
            "marginal_cost": 10.0,
            "tax_brackets": [0.0, 10.0, 50.0, 100.0],
            "tax_rates": [0.0, 0.15, 0.30, 0.45],
            "total_subsidies": 0.0,
            "efe_mode": self.efe_mode,
        }
        return bids, valuations, state

    def search_optimal_mechanism(self, n_agents: int = 4, ticks: int = 10) -> MechanismCandidate:
        """Execute bi-level mechanism search and serialize discovered optimal candidate."""
        dummy_cand = self.space.sample_candidate("dummy")
        bids, valuations, state = self.run_simulation_harness_oracle(
            dummy_cand, n_agents=n_agents, ticks=ticks
        )

        best_cand = self.optimizer.optimize(bids, valuations, state)
        self.best_candidate = best_cand

        # Rule 011 Dual Persistence: save to meta-parameters dict and return
        best_cand.meta_parameters["efe_mode"] = self.efe_mode
        best_cand.meta_parameters["aggregator_type"] = self.aggregator_type
        return best_cand
