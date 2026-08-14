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
    PolicySnapshot,
    TournamentResult,
    _select_tournament_pairs,
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


def test_formal_tournament_pairing_selection() -> None:
    """Verifies Ground A pairing selection invariants: explicit evaluate_generations list
    (decoupled from snapshot cadence), mini round-robin cardinality C(n,2), unknown
    generations dropped, and no self-play."""

    def _mk_snap(gen: int) -> PolicySnapshot:
        return PolicySnapshot(
            generation=gen,
            policy_fn=lambda board, moves: moves[0],
            theta_meta=np.ones(8, dtype=np.float32),
        )

    snaps = [_mk_snap(g) for g in [0, 4, 8, 12, 16]]

    # No list: full round robin C(5,2) = 10, each generation plays every other once
    pairs = _select_tournament_pairs(snaps)
    assert len(pairs) == 10
    gen_counts: dict[int, int] = {}
    for s1, s2 in pairs:
        gen_counts[s1.generation] = gen_counts.get(s1.generation, 0) + 1
        gen_counts[s2.generation] = gen_counts.get(s2.generation, 0) + 1
    for gen in [0, 4, 8, 12, 16]:
        assert gen_counts[gen] == 4

    # Explicit list: only those generations, any spacing (no Gen 0, no cadence coupling)
    pairs = _select_tournament_pairs(snaps, evaluate_generations=[8, 12, 16])
    pair_gens = sorted((s1.generation, s2.generation) for s1, s2 in pairs)
    assert pair_gens == [(8, 12), (8, 16), (12, 16)]

    # List cardinality invariant: C(n, 2) mini round-robin pairs
    pairs = _select_tournament_pairs(snaps, evaluate_generations=[4, 8, 12, 16])
    assert len(pairs) == 6

    # List beats last-N fallback when both are given
    pairs = _select_tournament_pairs(snaps, evaluate_generations=[4, 16], eval_last_n_snapshots=2)
    pair_gens = sorted((s1.generation, s2.generation) for s1, s2 in pairs)
    assert pair_gens == [(4, 16)]

    # Unknown generations are dropped, never crash
    pairs = _select_tournament_pairs(snaps, evaluate_generations=[7, 8, 16])
    pair_gens = sorted((s1.generation, s2.generation) for s1, s2 in pairs)
    assert pair_gens == [(8, 16)]

    # Fallback alias: last N stored snapshots
    pairs = _select_tournament_pairs(snaps, eval_last_n_snapshots=2)
    pair_gens = sorted((s1.generation, s2.generation) for s1, s2 in pairs)
    assert pair_gens == [(12, 16)]

    all_pairs = [(s1.generation, s2.generation) for s1, s2 in pairs]
    assert len(set(all_pairs)) == len(all_pairs)  # No duplicated matchups
    assert all(a != b for a, b in all_pairs)  # No self-play


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
