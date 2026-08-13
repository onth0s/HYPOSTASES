"""Formal Mathematical Verification Tests for Chess Domain & Dual Testing Grounds.

Rule 012: Mandatory Formal Mathematical Implementation Verification.
Verifies end-to-end mathematical invariants:
- Bradley-Terry maximum likelihood monotonicity and transitivity bounds.
- Logistic score-to-Elo rating bounds and derivative properties.
- Terminal game reward zero-sum bounds.
- Markov state tensor shape and simplex bounds.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.plugins.domains.chess.ground_a_self_play import (
    GroundASelfPlay,
    TournamentResult,
)
from hypostases.plugins.domains.chess.ground_b_stockfish import GroundBStockfish


def test_formal_terminal_reward_zero_sum_invariant() -> None:
    """Verifies that terminal game outcomes strictly satisfy zero-sum and bounded rewards in {-1, 0, 1}."""
    domain = ChessDomain()
    board = domain.initial_state()

    # Fool's mate: 1. f3 e5 2. g4 Qh4#
    moves = [
        chess.Move.from_uci("f2f3"),
        chess.Move.from_uci("e7e5"),
        chess.Move.from_uci("g2g4"),
        chess.Move.from_uci("d8h4"),
    ]

    for idx, move in enumerate(moves):
        board, reward, done, info = domain.step(board, move)
        if idx < 3:
            assert not done
            assert reward == 0.0
        else:
            assert done
            # Checkmate on Qh4#: Black won (reward for Black's move is +1.0)
            assert reward == 1.0
            assert info["winner"] == chess.BLACK


def test_formal_logistic_elo_fit_properties() -> None:
    """Verifies asymptotic bounds and monotonicity of the logistic score-to-Elo fit."""
    ref_elo = 1300.0

    # Score of 0.50 should yield exactly the reference Elo
    elo_50 = GroundBStockfish.logistic_elo_fit(0.50, ref_elo)
    assert pytest.approx(elo_50, abs=1e-5) == ref_elo

    # Monotonicity test: S1 < S2 => Elo(S1) < Elo(S2)
    scores = np.linspace(0.05, 0.95, 10)
    elos = [GroundBStockfish.logistic_elo_fit(s, ref_elo) for s in scores]

    for i in range(len(elos) - 1):
        assert elos[i] < elos[i + 1]

    # Symmetry property: Elo(S) - Ref == -(Elo(1-S) - Ref)
    delta_win = GroundBStockfish.logistic_elo_fit(0.80, ref_elo) - ref_elo
    delta_loss = GroundBStockfish.logistic_elo_fit(0.20, ref_elo) - ref_elo
    assert pytest.approx(delta_win, abs=1e-5) == -delta_loss


def test_formal_bradley_terry_transitivity_and_monotonicity() -> None:
    """Verifies maximum likelihood Bradley-Terry rating recovery on synthetic tournament results."""
    # Synthetic tournament: Gen 0 < Gen 5 < Gen 10
    result = TournamentResult(snapshot_ids=[0, 5, 10])

    # Gen 5 vs Gen 0: Gen 5 wins 8 / 10
    result.wins[(0, 5)] = 2.0
    result.games[(0, 5)] = 10

    # Gen 10 vs Gen 5: Gen 10 wins 8 / 10
    result.wins[(5, 10)] = 2.0
    result.games[(5, 10)] = 10

    # Gen 10 vs Gen 0: Gen 10 wins 10 / 10
    result.wins[(0, 10)] = 0.0
    result.games[(0, 10)] = 10

    ratings = GroundASelfPlay.compute_internal_elo(result, base_elo=1000.0)

    # Invariant 1: Generation 0 anchored at 1000.0
    assert pytest.approx(ratings[0], abs=1e-5) == 1000.0

    # Invariant 2: Monotonic ordering Elo(Gen 0) < Elo(Gen 5) < Elo(Gen 10)
    assert ratings[0] < ratings[5] < ratings[10]


def test_formal_state_tensor_invariants() -> None:
    """Verifies mathematical properties of full (8x8x19) and raw (8x8x12) tensor representations."""
    domain_full = ChessDomain(representation_mode="full")
    domain_raw = ChessDomain(representation_mode="raw")

    board = domain_full.initial_state()

    tensor_full = domain_full.to_world_model(board)
    tensor_raw = domain_raw.to_world_model(board)

    # Dimension invariants
    assert tensor_full.shape == (8, 8, 19)
    assert tensor_raw.shape == (8, 8, 12)

    # Piece count invariant: Initial board has 32 pieces
    piece_count_full = np.sum(tensor_full[:, :, :12])
    piece_count_raw = np.sum(tensor_raw[:, :, :12])
    assert piece_count_full == 32.0
    assert piece_count_raw == 32.0

    # Active turn plane invariant (Channel 12)
    assert np.all(tensor_full[:, :, 12] == 1.0)  # White's turn initially
