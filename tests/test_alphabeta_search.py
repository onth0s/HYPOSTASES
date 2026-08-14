"""Updated Alpha-Beta search unit tests conforming to decoupled Domain architecture.

Spec Ref: scratch/DISSONANCES.md D-008
"""

import chess

from hypostases.plugins.domains.chess.chess_agent_adapter import _make_nnue_evaluator
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
    move2, eval2, _tele2 = searcher.search(board)

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

    move, _score, tele = searcher.search(board)

    assert move in board.legal_moves
    assert tele.searched_nodes >= 1
    assert tele.leaf_evals >= 1


def test_search_root_children_mate_in_one() -> None:
    """search_root_children exposes exact child evals; mate-in-1 root move scores +100."""
    domain = ChessDomain()
    board = chess.Board("7k/6pp/8/8/8/8/7K/4R3 w - - 0 1")
    net = NNUENet(seed=42)

    config = SearchConfig(
        max_depth=2, time_budget_ms=2000.0, quiescence_depth=2, max_quiescence_nodes=256
    )
    searcher = AlphaBetaSearch(domain=domain, evaluator=_make_nnue_evaluator(net), config=config)

    children = searcher.search_root_children(board)
    best_move, best_eval = max(children, key=lambda item: item[1])

    assert len(children) == len(list(board.legal_moves))
    assert best_move == chess.Move.from_uci("e1e8")
    assert best_eval == 100.0


def test_search_root_children_determinism() -> None:
    """search_root_children is deterministic: identical children order and evals on rerun."""
    domain = ChessDomain()
    board = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1")

    config = SearchConfig(
        max_depth=2, time_budget_ms=2000.0, quiescence_depth=2, max_quiescence_nodes=64
    )
    searcher = AlphaBetaSearch(domain=domain, evaluator=bootstrap_eval, config=config)

    children1 = [(m.uci(), v) for m, v in searcher.search_root_children(board)]
    children2 = [(m.uci(), v) for m, v in searcher.search_root_children(board)]

    assert children1 == children2


def test_search_root_children_matches_search_best_move() -> None:
    """Root-child argmax equals search() best action: full-window pass is exact vs alpha-beta."""
    domain = ChessDomain()
    board = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1")
    config = SearchConfig(
        max_depth=2, time_budget_ms=2000.0, quiescence_depth=2, max_quiescence_nodes=64
    )

    best_move, _, _ = AlphaBetaSearch(
        domain=domain, evaluator=bootstrap_eval, config=config
    ).search(board)

    children = AlphaBetaSearch(
        domain=domain, evaluator=bootstrap_eval, config=config
    ).search_root_children(board)
    argmax_move = max(children, key=lambda item: item[1])[0]

    assert argmax_move == best_move


def test_search_root_children_qnode_budget_bounds_explosion() -> None:
    """max_quiescence_nodes caps quiescence explosion in tactical middlegames."""
    domain = ChessDomain()
    board = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1")

    def run(cap: int) -> tuple[int, int]:
        config = SearchConfig(
            max_depth=2, time_budget_ms=2000.0, quiescence_depth=4, max_quiescence_nodes=cap
        )
        searcher = AlphaBetaSearch(domain=domain, evaluator=bootstrap_eval, config=config)
        searcher.search_root_children(board)
        tele = searcher.recorder.get_telemetry()
        return tele.quiescence_nodes, tele.leaf_evals

    uncapped_qnodes, uncapped_leaves = run(0)
    capped_qnodes, capped_leaves = run(32)

    assert uncapped_qnodes > 0
    assert capped_qnodes < uncapped_qnodes
    assert capped_leaves < uncapped_leaves
    assert capped_qnodes <= 32 + capped_leaves


def test_search_root_children_total_node_budget_bounds_cost() -> None:
    """max_total_nodes caps expensive deep expansion; every root child still gets an eval.

    Provable invariant: leaf_evals <= max_total_nodes (a leaf eval is only recorded on a
    non-exhausted budget decrement, and there are at most `max_total_nodes` of those).
    Behavioral invariant: a budgeted depth-3 search does strictly less total work than the
    unbounded depth-3 search at the same time limit.
    """
    domain = ChessDomain()
    board = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1")

    def run(budget: int) -> tuple[int, int, int]:
        config = SearchConfig(
            max_depth=3,
            time_budget_ms=5000.0,
            quiescence_depth=2,
            max_quiescence_nodes=256,
            max_total_nodes=budget,
        )
        searcher = AlphaBetaSearch(domain=domain, evaluator=bootstrap_eval, config=config)
        children = searcher.search_root_children(board)
        tele = searcher.recorder.get_telemetry()
        return tele.searched_nodes, tele.leaf_evals, len(children)

    uncapped_nodes, uncapped_leaves, _ = run(0)
    bounded_nodes, bounded_leaves, n_children = run(4096)

    assert n_children == len(list(board.legal_moves))
    assert bounded_leaves <= 4096
    assert bounded_nodes < uncapped_nodes
    assert bounded_leaves < uncapped_leaves
    assert bounded_nodes + bounded_leaves < uncapped_nodes + uncapped_leaves


def test_search_total_node_budget_zero_means_unbounded() -> None:
    """max_total_nodes == 0 disables the budget (default): full-width search is unaffected."""
    domain = ChessDomain()
    board = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1")
    config = SearchConfig(
        max_depth=2, time_budget_ms=2000.0, quiescence_depth=2, max_quiescence_nodes=64
    )

    best_move, _, _ = AlphaBetaSearch(
        domain=domain, evaluator=bootstrap_eval, config=config
    ).search(board)
    children = AlphaBetaSearch(
        domain=domain, evaluator=bootstrap_eval, config=config
    ).search_root_children(board)

    assert max(children, key=lambda item: item[1])[0] == best_move
