"""Updated Alpha-Beta search unit tests conforming to decoupled Domain architecture.

Spec Ref: scratch/DISSONANCES.md D-008
"""

import chess

from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.world_model.alphabeta_search import AlphaBetaSearch, SearchConfig
from hypostases.world_model.nnue_net import NNUENet, extract_halfkp_features
from hypostases.world_model.nnue_training import bootstrap_eval


def test_alphabeta_tie_breaking_determinism() -> None:
    """T-NNUE-2: Verifies deterministic move selection and tie-breaking invariant."""
    domain = ChessDomain()
    board = domain.initial_state()

    config = SearchConfig(max_depth=2, time_budget_ms=100.0)
    searcher = AlphaBetaSearch(domain=domain, evaluator=bootstrap_eval, config=config)

    move1, eval1, tele1 = searcher.search(board)
    move2, eval2, tele2 = searcher.search(board)

    assert move1 == move2
    assert eval1 == eval2
    assert tele1.searched_nodes > 0


def test_alphabeta_quiescence_and_depth() -> None:
    domain = ChessDomain()
    board = domain.initial_state()

    config = SearchConfig(max_depth=3, time_budget_ms=200.0)
    net = NNUENet(seed=42)

    def nnue_evaluator(state: chess.Board) -> float:
        acc = net.create_accumulator(state)
        _, _, aux = extract_halfkp_features(state)
        return net.forward(acc, aux)

    searcher = AlphaBetaSearch(domain=domain, evaluator=nnue_evaluator, config=config)

    move, score, tele = searcher.search(board)

    assert move in board.legal_moves
    assert tele.searched_nodes >= 1
    assert tele.leaf_evals >= 1
