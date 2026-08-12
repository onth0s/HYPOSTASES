"""Unit & Integration tests for Mechanism Optimizers (Wave 4 Front 10)."""

from hypostases.mechanism_search.evaluator import MechanismEvaluator
from hypostases.mechanism_search.mechanism_space import MechanismSpace
from hypostases.mechanism_search.optimizer import (
    BayesianMechanismSearcher,
    DifferentiableMechanismSearcher,
    EvolutionaryMechanismSearcher,
)


def test_bayesian_mechanism_searcher():
    space = MechanismSpace()
    evaluator = MechanismEvaluator(aggregator_type="productivity_gini")
    searcher = BayesianMechanismSearcher(space, evaluator, n_iterations=5)

    valuations = [10.0, 8.0, 5.0]
    bids = [10.0, 8.0, 5.0]
    state = {}

    best_cand = searcher.optimize(bids, valuations, state)
    assert best_cand is not None
    assert len(searcher.history) == 5


def test_evolutionary_mechanism_searcher_fec():
    space = MechanismSpace()
    evaluator = MechanismEvaluator(aggregator_type="productivity_gini")
    searcher = EvolutionaryMechanismSearcher(
        space, evaluator, n_iterations=5, population_size=4, fec_enabled=True
    )

    valuations = [10.0, 8.0, 5.0]
    bids = [10.0, 8.0, 5.0]
    state = {}

    best_cand = searcher.optimize(bids, valuations, state)
    assert best_cand is not None
    assert len(searcher.fec_cache) > 0


def test_differentiable_mechanism_searcher():
    space = MechanismSpace()
    evaluator = MechanismEvaluator(aggregator_type="productivity_gini")
    searcher = DifferentiableMechanismSearcher(space, evaluator, n_iterations=5)

    valuations = [10.0, 8.0, 5.0]
    bids = [10.0, 8.0, 5.0]
    state = {}

    best_cand = searcher.optimize(bids, valuations, state)
    assert best_cand is not None
    assert searcher.lambda_ic >= 0.0
