"""Unit and Integration Tests for Chess Domain & Dual Testing Grounds."""

from __future__ import annotations

import chess
import pytest

from hypostases.domains.base import Domain
from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.plugins.domains.chess.ground_a_self_play import GroundASelfPlay, PolicySnapshot
from hypostases.plugins.domains.chess.ground_b_stockfish import (
    GroundBStockfish,
)


def test_chess_domain_protocol_compliance() -> None:
    """Verifies that ChessDomain satisfies the Domain Protocol interface."""
    domain = ChessDomain()
    assert isinstance(domain, Domain)


def test_chess_domain_initial_and_valid_actions() -> None:
    """Verifies initial state generation and legal move extraction."""
    domain = ChessDomain()
    board = domain.initial_state()
    assert isinstance(board, chess.Board)

    actions = domain.valid_actions(board)
    assert len(actions) == 20  # Standard starting chess moves count


def test_chess_domain_illegal_move_raises() -> None:
    """Verifies that attempting an illegal move raises ValueError."""
    domain = ChessDomain()
    board = domain.initial_state()
    illegal_move = chess.Move.from_uci("e2e5")  # Illegal opening pawn jump

    with pytest.raises(ValueError):
        domain.step(board, illegal_move)


def test_ground_a_self_play_tournament_execution() -> None:
    """Tests Ground A self-play tournament between dummy random policies."""
    domain = ChessDomain()

    def random_policy(board: chess.Board, legal_moves: list[chess.Move]) -> chess.Move:
        return legal_moves[0]  # Deterministic first move selection

    s0 = PolicySnapshot(generation=0, policy_fn=random_policy)
    s5 = PolicySnapshot(generation=5, policy_fn=random_policy)

    harness = GroundASelfPlay(chess_domain=domain)
    result = harness.run_snapshot_tournament(snapshots=[s0, s5], games_per_pair=2, max_moves=10)

    assert result.snapshot_ids == [0, 5]
    assert (0, 5) in result.games
    assert result.games[(0, 5)] == 2

    elo_ratings = GroundASelfPlay.compute_internal_elo(result, base_elo=1000.0)
    assert 0 in elo_ratings
    assert 5 in elo_ratings


def test_ground_b_stockfish_mock_fallback() -> None:
    """Tests Ground B evaluation using the deterministic MockStockfishEngine fallback."""
    domain = ChessDomain()

    def simple_policy(board: chess.Board, legal_moves: list[chess.Move]) -> chess.Move:
        return legal_moves[0]

    snapshot = PolicySnapshot(generation=0, policy_fn=simple_policy)

    # Force mock engine by passing non-existent path
    harness = GroundBStockfish(
        stockfish_path="non_existent_stockfish_bin", reference_elo=1300.0, chess_domain=domain
    )
    res = harness.evaluate_snapshot(snapshot=snapshot, games_n=2, max_moves=10)

    assert res.generation == 0
    assert res.games_played == 2
    assert res.wins + res.losses + res.draws + res.capped == 2
    assert res.effective_games == 2 - res.capped
    assert res.reference_elo == 1300.0
    assert res.estimated_elo > 0.0
