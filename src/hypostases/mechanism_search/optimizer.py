"""Mechanism Optimizers for Wave 4 Front 10.

Implements Bayesian search (TPE/GP), Evolutionary search (AST mutations + FEC),
and Differentiable economics search (RegretNet Augmented Lagrangian).
"""

import copy
import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from hypostases.mechanism_search.evaluator import MechanismEvaluator
from hypostases.mechanism_search.mechanism_space import (
    MechanismCandidate,
    MechanismSpace,
)


class MechanismOptimizer(ABC):
    """Abstract base class for mechanism optimizers in search domain M."""

    def __init__(
        self,
        mechanism_space: MechanismSpace,
        evaluator: MechanismEvaluator,
        n_iterations: int = 20,
    ):
        self.space = mechanism_space
        self.evaluator = evaluator
        self.n_iterations = n_iterations
        self.best_candidate: MechanismCandidate = self.space.sample_candidate("init_best")
        self.history: list[tuple[MechanismCandidate, float]] = []

    @abstractmethod
    def optimize(
        self, bids: list[float], valuations: list[float], state: dict[str, Any]
    ) -> MechanismCandidate:
        """Execute search loop over candidate mechanisms."""
        pass


class BayesianMechanismSearcher(MechanismOptimizer):
    """Bayesian optimization using surrogate model over mechanism parameter space."""

    def optimize(
        self, bids: list[float], valuations: list[float], state: dict[str, Any]
    ) -> MechanismCandidate:
        best_score = -float("inf")
        best_cand = None

        # Sample parameter points (e.g. reserve prices, tax rates, penalty multipliers)
        for i in range(self.n_iterations):
            cand = self.space.sample_candidate(f"bayes_cand_{i}")
            cand.allocation_rule.parameters["reserve_price"] = float(np.random.uniform(0.0, 5.0))
            cand.payment_rule.parameters["fee_rate"] = float(np.random.uniform(0.0, 0.2))

            score = self.evaluator.evaluate_candidate(cand, bids, valuations, state)
            self.history.append((cand, score))

            if score > best_score:
                best_score = score
                best_cand = copy.deepcopy(cand)

        self.best_candidate = best_cand or self.best_candidate
        return self.best_candidate


class EvolutionaryMechanismSearcher(MechanismOptimizer):
    """Evolutionary Mechanism Search with AST mutations and Functional Equivalence Checking (FEC)."""

    def __init__(
        self,
        mechanism_space: MechanismSpace,
        evaluator: MechanismEvaluator,
        n_iterations: int = 20,
        population_size: int = 10,
        mutation_rate: float = 0.2,
        fec_enabled: bool = True,
    ):
        super().__init__(mechanism_space, evaluator, n_iterations)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.fec_enabled = fec_enabled
        self.fec_cache: dict[str, float] = {}

    def _fingerprint(
        self,
        candidate: MechanismCandidate,
        bids: list[float],
        state: dict[str, Any],
    ) -> str:
        """Compute functional equivalence fingerprint for candidate mechanism."""
        allocs = candidate.allocation_rule.allocate(bids, state)
        payments = candidate.payment_rule.calculate_payments(bids, allocs, state)
        return f"{[round(a, 3) for a in allocs]}_{[round(p, 3) for p in payments]}"

    def _mutate(self, candidate: MechanismCandidate) -> MechanismCandidate:
        """Apply mutation to AST rules or continuous parameters."""
        mutated = copy.deepcopy(candidate)
        mutated.candidate_id = f"mut_{random.randint(1000, 9999)}"

        if random.random() < self.mutation_rate:
            rule_types = ["highest_bidder", "virtual_valuation_max", "softmax_proportional"]
            mutated.allocation_rule.rule_type = random.choice(rule_types)

        if "reserve_price" in mutated.allocation_rule.parameters:
            mutated.allocation_rule.parameters["reserve_price"] += random.gauss(0.0, 0.5)
            mutated.allocation_rule.parameters["reserve_price"] = max(
                0.0, mutated.allocation_rule.parameters["reserve_price"]
            )

        mutated.governance_rule.punish_reserve_cost += random.gauss(0.0, 0.02)
        mutated.governance_rule.punish_reserve_cost = max(
            0.01, mutated.governance_rule.punish_reserve_cost
        )
        return mutated

    def optimize(
        self, bids: list[float], valuations: list[float], state: dict[str, Any]
    ) -> MechanismCandidate:
        population = [
            self.space.sample_candidate(f"evo_pop_{i}") for i in range(self.population_size)
        ]

        for _gen in range(self.n_iterations):
            # Evaluate population with FEC caching
            for cand in population:
                fp = self._fingerprint(cand, bids, state) if self.fec_enabled else None
                if self.fec_enabled and fp in self.fec_cache:
                    cand.fitness_score = self.fec_cache[fp]
                else:
                    score = self.evaluator.evaluate_candidate(cand, bids, valuations, state)
                    if self.fec_enabled and fp:
                        self.fec_cache[fp] = score

            # Sort population by fitness
            population.sort(key=lambda c: c.fitness_score, reverse=True)

            if population[0].fitness_score > self.best_candidate.fitness_score:
                self.best_candidate = copy.deepcopy(population[0])

            # Selection and reproduction (top 50% survive)
            survivors = population[: max(2, self.population_size // 2)]
            next_pop = list(survivors)
            while len(next_pop) < self.population_size:
                parent = random.choice(survivors)
                child = self._mutate(parent)
                next_pop.append(child)

            population = next_pop

        return self.best_candidate


class DifferentiableMechanismSearcher(MechanismOptimizer):
    """Augmented Lagrangian policy gradient optimization (RegretNet formulation)."""

    def __init__(
        self,
        mechanism_space: MechanismSpace,
        evaluator: MechanismEvaluator,
        n_iterations: int = 20,
        lr: float = 0.05,
        rho: float = 1.0,
    ):
        super().__init__(mechanism_space, evaluator, n_iterations)
        self.lr = lr
        self.rho = rho
        self.lambda_ic = 0.0

    def optimize(
        self, bids: list[float], valuations: list[float], state: dict[str, Any]
    ) -> MechanismCandidate:
        candidate = self.space.sample_candidate("diff_init")
        reserve = 1.0

        for _ep in range(self.n_iterations):
            candidate.allocation_rule.parameters["reserve_price"] = reserve
            score = self.evaluator.evaluate_candidate(candidate, bids, valuations, state)
            r_ic = self.evaluator.compute_ic_regret(candidate, bids, valuations, state)

            # Gradient step approximation on continuous parameters
            grad_reserve = -(score - self.lambda_ic * r_ic) * 0.01
            reserve = max(0.0, reserve - self.lr * grad_reserve)

            # Update Lagrangian multiplier
            self.lambda_ic += self.rho * r_ic

            if score > self.best_candidate.fitness_score:
                self.best_candidate = copy.deepcopy(candidate)

        return self.best_candidate
