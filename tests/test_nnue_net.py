"""Unit tests for HalfKP feature extraction, accumulators, and NNUENet topology.

Spec Ref: scratch/OUTTASTOCKFISH.md §8 (T-NNUE-1) & §9.10–9.11
"""

import numpy as np

from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.world_model.nnue_net import Accumulator, NNUENet, extract_halfkp_features


def test_halfkp_feature_bounds() -> None:
    domain = ChessDomain()
    board = domain.initial_state()

    w_feats, b_feats, aux = extract_halfkp_features(board)

    # Initial position has 30 non-king pieces (15 each)
    assert len(w_feats) == 30
    assert len(b_feats) == 30
    assert aux.shape == (29,)

    assert all(0 <= idx < 40960 for idx in w_feats)
    assert all(0 <= idx < 40960 for idx in b_feats)


def test_accumulator_refresh_equivalence() -> None:
    """T-NNUE-1: Proves accumulator refresh vs incremental equivalence invariant."""
    net = NNUENet(seed=123)
    domain = ChessDomain()

    board = domain.initial_state()
    acc_refreshed = net.create_accumulator(board)

    # Manual incremental update verification
    w_feats, b_feats, _ = extract_halfkp_features(board)
    acc_incremental = Accumulator()
    acc_incremental.update_delta(
        w_added=w_feats,
        w_removed=[],
        b_added=b_feats,
        b_removed=[],
        w_weights=net.W_white,
        b_weights=net.W_black,
    )

    # Max-norm floating point error assertion (<= 1e-5)
    max_err_white = np.max(np.abs(acc_refreshed.white_acc - acc_incremental.white_acc))
    max_err_black = np.max(np.abs(acc_refreshed.black_acc - acc_incremental.black_acc))

    assert max_err_white <= 1e-5
    assert max_err_black <= 1e-5


def test_nnue_forward_pass() -> None:
    net = NNUENet()
    domain = ChessDomain()
    board = domain.initial_state()

    acc = net.create_accumulator(board)
    _, _, aux = extract_halfkp_features(board)

    score = net.forward(acc, aux)
    assert isinstance(score, float)
    assert not np.isnan(score)
