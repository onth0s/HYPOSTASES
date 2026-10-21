"""Domain-Agnostic AlphaBetaSearch Tests without domain-specific dependencies.

Spec Ref: Phase 5.1 Architecture Decoupling & Test Modernization.
Verifies that AlphaBetaSearch operates correctly on any abstract Domain Protocol implementation.
"""

from __future__ import annotations

from hypostases.domains import DomainRegistry, NimState, SimpleNimDomain, nim_evaluator
from hypostases.world_model.alphabeta_search import AlphaBetaSearch, SearchConfig


def test_domain_registry_lookup_toy_nim() -> None:
    """Verifies that the reference Nim domain is registered and instantiable via DomainRegistry."""
    domain = DomainRegistry.get("toy_nim", initial_stones=9)
    assert isinstance(domain, SimpleNimDomain)
    state = domain.initial_state()
    assert isinstance(state, NimState)
    assert state.stones == 9
    assert state.turn == 1
    assert domain.valid_actions(state) == [1, 2, 3]


def test_domain_agnostic_alphabeta_search_finds_optimal_move() -> None:
    """Verifies that domain-agnostic AlphaBetaSearch selects winning move without chess."""
    domain = SimpleNimDomain(
        initial_stones=5
    )  # 5 stones: taking 1 leaves 4 (a losing position for opponent)
    searcher = AlphaBetaSearch(
        domain=domain,
        evaluator=nim_evaluator,
        config=SearchConfig(max_depth=4, time_budget_ms=100.0),
    )

    state = domain.initial_state()
    best_move, best_eval, telemetry = searcher.search(state)

    # Taking 1 stone leaves 4 (opponent faces 4 % 4 == 0, which is losing)
    assert best_move == 1
    assert best_eval > 0.0
    assert telemetry.searched_nodes > 0


def test_domain_agnostic_alphabeta_budget_and_tie_breaking() -> None:
    """Verifies determinism and node budget limiting on abstract domain."""
    domain = SimpleNimDomain(initial_stones=8)
    config = SearchConfig(max_depth=3, max_total_nodes=5, time_budget_ms=100.0)
    searcher = AlphaBetaSearch(domain=domain, evaluator=nim_evaluator, config=config)

    state = domain.initial_state()
    move1, eval1, tele1 = searcher.search(state)
    move2, eval2, _tele2 = searcher.search(state)

    assert move1 == move2
    assert eval1 == eval2
    assert tele1.searched_nodes <= 15
