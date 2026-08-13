"""HYPOSTASES NNUE Network Topology and Accumulator Implementation.

Implements HalfKP sparse feature encoding, dual accumulators (White/Black perspectives) with incremental updates,
and Option B separated pathway net evaluation.
Spec Ref: scratch/OUTTASTOCKFISH.md §2.1 & §9.10–9.11
"""

from __future__ import annotations

import chess
import numpy as np

HALF_KP_DIM = 40960
AUX_DIM = 29
ACCUMULATOR_DIM = 256
DENSE_HIDDEN_DIM = 32

PIECE_TYPE_MAP = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}


def compute_halfkp_index(king_sq: int, piece_sq: int, piece_type: int, is_black_piece: bool) -> int:
    """Computes HalfKP feature index: Index(k, p, t, c) = k*640 + c*320 + t*64 + p.

    Spec Ref: scratch/OUTTASTOCKFISH.md §9.11
    """
    c = 1 if is_black_piece else 0
    t = PIECE_TYPE_MAP[piece_type]
    return king_sq * 640 + c * 320 + t * 64 + piece_sq


def extract_halfkp_features(state: chess.Board) -> tuple[list[int], list[int], np.ndarray]:
    """Extracts active HalfKP feature indices for White and Black perspectives and auxiliary state vector.

    Returns:
        (white_features, black_features, aux_vector)
    """
    w_king = state.king(chess.WHITE)
    b_king = state.king(chess.BLACK)

    if w_king is None or b_king is None:
        raise ValueError("Board must contain both kings to extract HalfKP features.")

    w_features: list[int] = []
    b_features: list[int] = []

    # Orient black king relative index if needed, or standard square index (0..63)
    for sq, piece in state.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        is_black = piece.color == chess.BLACK
        # White perspective:
        w_idx = compute_halfkp_index(w_king, sq, piece.piece_type, is_black)
        w_features.append(w_idx)

        # Black perspective:
        # Orient black king perspective (flipped vertically for black king perspective or standard indexing)
        b_idx = compute_halfkp_index(b_king, sq, piece.piece_type, is_black)
        b_features.append(b_idx)

    # Auxiliary vector x_aux in R^29
    aux = np.zeros(AUX_DIM, dtype=np.float32)
    aux[0] = 1.0 if state.turn == chess.WHITE else 0.0

    if state.has_kingside_castling_rights(chess.WHITE):
        aux[1] = 1.0
    if state.has_queenside_castling_rights(chess.WHITE):
        aux[2] = 1.0
    if state.has_kingside_castling_rights(chess.BLACK):
        aux[3] = 1.0
    if state.has_queenside_castling_rights(chess.BLACK):
        aux[4] = 1.0

    if state.ep_square is not None:
        file_idx = chess.square_file(state.ep_square)
        aux[5 + file_idx] = 1.0

    # Halfmove clock coarse buckets (16 buckets)
    bucket = min(state.halfmove_clock // 6, 15)
    aux[13 + bucket] = 1.0

    return w_features, b_features, aux


class Accumulator:
    """Perspective dual accumulators (White/Black) supporting incremental delta updates and parent snapshot recovery."""

    def __init__(
        self, white_acc: np.ndarray | None = None, black_acc: np.ndarray | None = None
    ) -> None:
        self.white_acc = (
            white_acc.copy()
            if white_acc is not None
            else np.zeros(ACCUMULATOR_DIM, dtype=np.float32)
        )
        self.black_acc = (
            black_acc.copy()
            if black_acc is not None
            else np.zeros(ACCUMULATOR_DIM, dtype=np.float32)
        )

    def copy(self) -> Accumulator:
        return Accumulator(self.white_acc, self.black_acc)

    def update_delta(
        self,
        w_added: list[int],
        w_removed: list[int],
        b_added: list[int],
        b_removed: list[int],
        w_weights: np.ndarray,
        b_weights: np.ndarray,
    ) -> None:
        """Applies incremental weight updates to dual accumulators."""
        for idx in w_added:
            self.white_acc += w_weights[idx]
        for idx in w_removed:
            self.white_acc -= w_weights[idx]

        for idx in b_added:
            self.black_acc += b_weights[idx]
        for idx in b_removed:
            self.black_acc -= b_weights[idx]

    def refresh(self, state: chess.Board, w_weights: np.ndarray, b_weights: np.ndarray) -> None:
        """Full recomputation of active feature sum."""
        w_feats, b_feats, _ = extract_halfkp_features(state)
        self.white_acc.fill(0.0)
        self.black_acc.fill(0.0)
        for idx in w_feats:
            self.white_acc += w_weights[idx]
        for idx in b_feats:
            self.black_acc += b_weights[idx]


class NNUENet:
    """HYPOSTASES-NNUE Option B Separated Pathway Architecture.

    Topology:
      Sparse Weight Matrices: W_white, W_black (40960 x 256)
      Accumulators: A_white (256), A_black (256)
      Concatenated Input: A_input = [A_white; A_black; x_aux] (541)
      Dense Layers: 541 -> 32 -> 1 with Clipped ReLU
    """

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        # Small random initialization
        self.W_white = rng.normal(0, 0.01, (HALF_KP_DIM, ACCUMULATOR_DIM)).astype(np.float32)
        self.W_black = rng.normal(0, 0.01, (HALF_KP_DIM, ACCUMULATOR_DIM)).astype(np.float32)

        input_dim = ACCUMULATOR_DIM * 2 + AUX_DIM  # 541
        self.W_l1 = rng.normal(0, 0.05, (input_dim, DENSE_HIDDEN_DIM)).astype(np.float32)
        self.b_l1 = np.zeros(DENSE_HIDDEN_DIM, dtype=np.float32)

        self.W_l2 = rng.normal(0, 0.05, (DENSE_HIDDEN_DIM, 1)).astype(np.float32)
        self.b_l2 = np.zeros(1, dtype=np.float32)

    def clipped_relu(self, x: np.ndarray, max_val: float = 127.0) -> np.ndarray:
        return np.minimum(np.maximum(x, 0.0), max_val) / max_val

    def create_accumulator(self, state: chess.Board) -> Accumulator:
        acc = Accumulator()
        acc.refresh(state, self.W_white, self.W_black)
        return acc

    def forward(self, accum: Accumulator, x_aux: np.ndarray) -> float:
        """Evaluates dense pass given accumulator state and aux vector.

        Returns scalar score (in pawns / centipawns normalized).
        """
        w_clipped = self.clipped_relu(accum.white_acc)
        b_clipped = self.clipped_relu(accum.black_acc)

        concat_input = np.concatenate([w_clipped, b_clipped, x_aux], axis=0)
        h1 = self.clipped_relu(np.dot(concat_input, self.W_l1) + self.b_l1)
        out = np.dot(h1, self.W_l2) + self.b_l2
        return float(out[0])
