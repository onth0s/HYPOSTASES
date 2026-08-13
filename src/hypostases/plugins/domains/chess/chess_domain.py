"""Chess Domain Plugin implementing the HYPOSTASES Domain Protocol.

Uses python-chess for move generation, state transitions, and board encoding.
Terminal-only rewards (+1, -1, 0), zero heuristic shaping.
"""

from __future__ import annotations

from typing import Any, Literal

import chess
import numpy as np

from hypostases.domains.base import Domain


class ChessDomain(Domain):
    """Pluggable Chess domain implementation conforming to the Domain Protocol."""

    def __init__(self, representation_mode: Literal["full", "raw"] = "full") -> None:
        """Initialize ChessDomain with specified representation mode.

        Args:
            representation_mode: 'full' for 8x8x19 (Markov MDP planes) or 'raw' for 8x8x12 (spatial occupancy).
        """
        if representation_mode not in ("full", "raw"):
            raise ValueError(
                f"Invalid representation_mode: {representation_mode}. Must be 'full' or 'raw'."
            )
        self.representation_mode = representation_mode

    def initial_state(self) -> chess.Board:
        """Returns standard initial chess board state."""
        return chess.Board()

    def valid_actions(self, state: chess.Board) -> list[chess.Move]:
        """Returns legal moves for the current board state."""
        return list(state.legal_moves)

    def step(
        self, state: chess.Board, action: chess.Move
    ) -> tuple[chess.Board, float, bool, dict[str, Any]]:
        """Executes a move and returns (next_state, reward, done, info).

        Reward is terminal-only:
          +1.0 if the move resulted in a win for the moving player,
          -1.0 if the move resulted in checkmate loss,
           0.0 for draws or non-terminal states.
        """
        if action not in state.legal_moves:
            raise ValueError(f"Illegal move attempted: {action} in board state {state.fen()}")

        player_turn = state.turn
        next_state = state.copy()
        next_state.push(action)
        done = next_state.is_game_over()

        reward = 0.0
        result_str = "*"
        winner = None

        if done:
            outcome = next_state.outcome()
            if outcome is not None:
                result_str = outcome.result()
                winner = outcome.winner
                if outcome.winner == player_turn:
                    reward = 1.0
                elif outcome.winner is not None:
                    reward = -1.0
                else:
                    reward = 0.0

        info = {
            "fen": next_state.fen(),
            "done_reason": next_state.outcome().termination.name
            if done and next_state.outcome()
            else None,
            "result": result_str,
            "winner": winner,
        }

        return next_state, reward, done, info

    def to_world_model(self, state: chess.Board) -> np.ndarray:
        """Encodes chess board into numerical state tensor representation.

        Returns:
            np.ndarray of shape (8, 8, 19) if 'full' mode, or (8, 8, 12) if 'raw' mode.
        """
        tensor_channels = 19 if self.representation_mode == "full" else 12
        tensor = np.zeros((8, 8, tensor_channels), dtype=np.float32)

        # Map 12 piece planes
        piece_type_map = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }

        for square, piece in state.piece_map().items():
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            channel_offset = 0 if piece.color == chess.WHITE else 6
            channel = piece_type_map[piece.piece_type] + channel_offset
            tensor[rank, file, channel] = 1.0

        if self.representation_mode == "full":
            # Channel 12: Active turn (1.0 for White, 0.0 for Black)
            if state.turn == chess.WHITE:
                tensor[:, :, 12] = 1.0

            # Castling rights
            if state.has_kingside_castling_rights(chess.WHITE):
                tensor[:, :, 13] = 1.0
            if state.has_queenside_castling_rights(chess.WHITE):
                tensor[:, :, 14] = 1.0
            if state.has_kingside_castling_rights(chess.BLACK):
                tensor[:, :, 15] = 1.0
            if state.has_queenside_castling_rights(chess.BLACK):
                tensor[:, :, 16] = 1.0

            # En Passant target
            if state.ep_square is not None:
                ep_rank = chess.square_rank(state.ep_square)
                ep_file = chess.square_file(state.ep_square)
                tensor[ep_rank, ep_file, 17] = 1.0

            # Halfmove clock normalized
            tensor[:, :, 18] = min(state.halfmove_clock / 100.0, 1.0)

        return tensor
