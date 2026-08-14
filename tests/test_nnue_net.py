"""Unit tests for HalfKP feature extraction, accumulators, and NNUENet topology.

Spec Ref: scratch/OUTTASTOCKFISH.md §8 (T-NNUE-1) & §9.10–9.11
"""

import chess
import numpy as np

from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.world_model.nnue_net import Accumulator, NNUENet, extract_halfkp_features
from hypostases.world_model.nnue_training import train_nnue


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


def _dataset_mse(net: NNUENet, dataset: list[tuple[chess.Board, float]]) -> float:
    """Computes mean squared error of the side-to-move forward pass over a dataset."""
    errors = []
    for board, label in dataset:
        acc = net.create_accumulator(board)
        _, _, aux = extract_halfkp_features(board)
        pred = net.forward(acc, aux)
        errors.append((pred - label) ** 2)
    return float(np.mean(errors))


_STM_BASE_FENS = [
    "r3k2r/pppp1ppp/8/8/8/8/PPPP1PPP/R3K2R w KQkq - 0 1",
    "r3k2r/pppp1ppp/8/8/8/2B5/PPPP1PPP/R3K2R w KQkq - 0 1",
    "r1bqk1nr/pppp1ppp/8/8/8/8/PPPPQPPP/RNB1K1NR w KQkq - 0 1",
    "r1bqk1nr/ppp2ppp/8/3p4/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
]
_STM_WHITE_LABELS = [0.0, 1.0, 1.0, 1.0]


def _mixed_stm_dataset() -> list[tuple[chess.Board, float]]:
    """White- and black-to-move boards labeled from the side-to-move perspective."""
    dataset = []
    for fen, white_label in zip(_STM_BASE_FENS, _STM_WHITE_LABELS, strict=False):
        board_w = chess.Board(fen)
        dataset.append((board_w, white_label))
        board_b = chess.Board(fen)
        board_b.turn = chess.BLACK
        dataset.append((board_b, -white_label))
    return dataset


def test_nnue_training_reduces_loss_on_black_to_move_boards() -> None:
    """Formal convergence (T-NNUE-1 / D-006): train_nnue must reduce MSE on black-to-move
    boards labeled from black's side-to-move perspective. Regression for the bug where the
    sparse accumulator gradient was routed to W_white regardless of turn (frozen learning)."""
    dataset = []
    for fen, white_label in zip(_STM_BASE_FENS, _STM_WHITE_LABELS, strict=False):
        board_b = chess.Board(fen)
        board_b.turn = chess.BLACK
        dataset.append((board_b, -white_label))

    net = NNUENet(seed=7)
    initial_mse = _dataset_mse(net, dataset)
    final_loss = train_nnue(net, dataset, epochs=10, lr=0.02, verbose=False)

    assert final_loss < initial_mse * 0.9


def test_nnue_training_converges_on_mixed_stm_dataset() -> None:
    """Formal convergence: end-to-end train_nnue must reduce MSE on a mixed white/black
    side-to-move-labeled dataset (both perspectives in one training run)."""
    dataset = _mixed_stm_dataset()

    net = NNUENet(seed=7)
    initial_mse = _dataset_mse(net, dataset)
    final_loss = train_nnue(net, dataset, epochs=60, lr=0.1, verbose=False)

    assert final_loss < initial_mse * 0.5
    assert final_loss < 0.4
