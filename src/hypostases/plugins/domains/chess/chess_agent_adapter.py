"""HYPOSTASES Chess Agent Adapter.

Bridges ChessDomain with the core HYPOSTASES agent state σ = (c, w, g, ρ_ext).
Evaluates legal chess moves using Active Perception Expected Free Energy (EFE mode)
and pragmatic-epistemic utility dynamics.
"""

from __future__ import annotations

import chess
import numpy as np

from hypostases.engine.types import Characteristics, GoalCategory, GoalHierarchy
from hypostases.plugins.domains.chess.chess_domain import ChessDomain


class ChessAgentAdapter:
    """Adapts a HYPOSTASES agent to evaluate and select moves in ChessDomain."""

    def __init__(
        self,
        domain: ChessDomain | None = None,
        beta_efe: float = 0.2,
        temperature: float = 0.5,
        theta_meta: np.ndarray | None = None,
    ) -> None:
        """Initialize adapter.

        Args:
            domain: ChessDomain instance.
            beta_efe: Epistemic weight β in U_total = (1-β) U_pragmatic + β U_epistemic.
            temperature: Softmax policy temperature τ.
            theta_meta: Meta-parameter weight vector for value estimation.
        """
        self.domain = domain or ChessDomain(representation_mode="full")
        self.beta_efe = beta_efe
        self.temperature = max(0.01, temperature)

        # Agent state σ = (c, w, g, ρ_ext)
        self.characteristics = Characteristics(skill=0.8, resilience=0.7)
        self.goal_hierarchy = GoalHierarchy()  # Latent utility weights u ∈ ℝ^{n_k}

        # Meta-parameters θ_meta for feature valuation
        # Features: [piece_material, center_control, king_safety, check_opportunity, mobility]
        if theta_meta is None:
            self.theta_meta = np.array([1.0, 0.3, 0.5, 0.4, 0.2], dtype=np.float32)
        else:
            self.theta_meta = np.array(theta_meta, dtype=np.float32)

    def extract_move_features(self, board: chess.Board, move: chess.Move) -> np.ndarray:
        """Extracts numerical feature vector for a candidate legal move."""
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.25,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0,
        }
        center_squares = {
            chess.E4,
            chess.D4,
            chess.E5,
            chess.D5,
            chess.C4,
            chess.C5,
            chess.F4,
            chess.F5,
        }

        # Feature 0: Material capture value
        capture_val = 0.0
        if board.is_capture(move):
            cap = board.piece_at(move.to_square)
            if cap:
                capture_val = piece_values[cap.piece_type]

        # Feature 1: Center control
        center_val = 1.0 if move.to_square in center_squares else 0.0

        # Feature 2: King safety / defense
        king_safety_val = 0.0
        if board.gives_check(move):
            king_safety_val = 1.0

        # Feature 3: Checkmate opportunity
        test_board = board.copy()
        test_board.push(move)
        checkmate_val = 10.0 if test_board.is_checkmate() else 0.0

        # Feature 4: Next-state mobility
        mobility_val = len(list(test_board.legal_moves)) / 20.0

        return np.array(
            [capture_val, center_val, king_safety_val, checkmate_val, mobility_val],
            dtype=np.float32,
        )

    def evaluate_efe_utility(
        self, board: chess.Board, move: chess.Move
    ) -> tuple[float, float, float]:
        """Evaluates pragmatic, epistemic, and total Expected Free Energy utility.

        U_total = (1 - β) * U_pragmatic + β * U_epistemic
        """
        features = self.extract_move_features(board, move)

        # Pragmatic utility: Dot product of θ_meta features with goal weights u
        goal_categories = list(GoalCategory)
        u_survival = self.goal_hierarchy.u[goal_categories.index(GoalCategory.SURVIVAL)]
        u_acquisition = self.goal_hierarchy.u[goal_categories.index(GoalCategory.ACQUISITION)]

        u_pragmatic = float(np.dot(self.theta_meta, features) * (u_survival + u_acquisition))

        # Epistemic utility: Expected information gain (reduction in state entropy / novel exploration)
        # Higher mobility / unexplored branches provide higher epistemic utility
        test_board = board.copy()
        test_board.push(move)
        epistemic_entropy = np.log(len(list(test_board.legal_moves)) + 1.0)
        u_epistemic = float(epistemic_entropy * self.characteristics.skill)

        # Total expected utility
        u_total = (1.0 - self.beta_efe) * u_pragmatic + self.beta_efe * u_epistemic

        return u_total, u_pragmatic, u_epistemic

    def select_move(self, board: chess.Board, legal_moves: list[chess.Move]) -> chess.Move:
        """Selects a legal move using active sensing softmax policy distribution π(a|σ)."""
        if not legal_moves:
            raise ValueError("No legal moves available in state.")

        # Check for immediate checkmates
        for move in legal_moves:
            test_board = board.copy()
            test_board.push(move)
            if test_board.is_checkmate():
                return move

        utilities = []
        for move in legal_moves:
            u_tot, _, _ = self.evaluate_efe_utility(board, move)
            utilities.append(u_tot)

        u_arr = np.array(utilities, dtype=np.float32)

        # Softmax sampling with temperature τ
        max_u = np.max(u_arr)
        exp_u = np.exp((u_arr - max_u) / self.temperature)
        probs = exp_u / np.sum(exp_u)

        chosen_idx = int(np.random.choice(len(legal_moves), p=probs))
        return legal_moves[chosen_idx]
