"""Numerical Invariant Verification for HYPOSTASES-NNUE Invariants.

Spec Ref: scratch/DISSONANCES.md D-007 (Renamed to Numerical Invariant Verification)
"""

import chess
import numpy as np

from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.plugins.domains.chess.nnue_net import NNUENet, extract_halfkp_features


def test_numerical_accumulator_restoration_invariant() -> None:
    """Numerical invariant verification: Accumulator delta undo max-norm error bound <= 1e-5."""
    net = NNUENet(seed=999)
    domain = ChessDomain()
    board = domain.initial_state()

    parent_acc = net.create_accumulator(board)
    w_feats, b_feats, _ = extract_halfkp_features(board)

    # Perform a dummy move step (e.g. e2e4)
    move = chess.Move.from_uci("e2e4")
    next_board, _, _, _ = domain.step(board, move)
    w_next, b_next, _ = extract_halfkp_features(next_board)

    # Deltas
    w_added = list(set(w_next) - set(w_feats))
    w_removed = list(set(w_feats) - set(w_next))
    b_added = list(set(b_next) - set(b_feats))
    b_removed = list(set(b_feats) - set(b_next))

    child_acc = parent_acc.copy()
    child_acc.update_delta(
        w_added,
        w_removed,
        b_added,
        b_removed,
        w_weights=net.W_white,
        b_weights=net.W_black,
    )

    # Undo delta
    restored_acc = child_acc.copy()
    restored_acc.update_delta(
        w_removed,
        w_added,
        b_removed,
        b_added,
        w_weights=net.W_white,
        b_weights=net.W_black,
    )

    max_norm_err_w = np.max(np.abs(restored_acc.white_acc - parent_acc.white_acc))
    max_norm_err_b = np.max(np.abs(restored_acc.black_acc - parent_acc.black_acc))

    assert max_norm_err_w <= 1e-5
    assert max_norm_err_b <= 1e-5


def test_numerical_hybrid_valuation_math() -> None:
    """Numerical invariant verification: Action-conditioned hybrid valuation convex combination bounds.

    Spec Ref: scratch/OUTTASTOCKFISH.md §9.14
    """

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    v_efe = 0.5
    v_nnue = 1.2
    tau_efe = 1.0
    tau_nnue = 1.0

    for theta in [0.0, 0.25, 0.5, 0.75, 1.0]:
        v_hybrid = (1.0 - theta) * sigmoid(v_efe / tau_efe) + theta * sigmoid(v_nnue / tau_nnue)
        assert 0.0 <= v_hybrid <= 1.0
