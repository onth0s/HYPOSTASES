"""HYPOSTASES Chess Agent Adapter.

Bridges ChessDomain with the core HYPOSTASES agent state σ = (c, w, g, ρ_ext).
Evaluates legal chess moves using Active Perception Expected Free Energy (EFE mode)
and pragmatic-epistemic utility dynamics.
"""

from __future__ import annotations

from typing import Any

import chess
import numpy as np

from hypostases.engine.types import Characteristics, GoalCategory, GoalHierarchy
from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.world_model.alphabeta_search import AlphaBetaSearch, SearchConfig
from hypostases.world_model.nnue_net import Accumulator, extract_halfkp_features

# Max-magnitude scale per tactical feature (used to normalize u_linear so the
# handcrafted garnish is not dominated by large-scale features like capture_delta).
FEATURE_SCALES = np.array([9.0, 1.0, 1.0, 10.0, 0.95, 1.0, 10.0, 1.0], dtype=np.float32)

# Search budget for the single-search policy: shallow depth-3 lookahead with a hard
# quiescence node cap AND a global node budget so tactical middlegame positions
# cannot explode. Measured (2026-08-14): depth-3 middlegame is ~2000ms unbounded
# (~11k leaf evals) vs ~250ms bounded at 4096 total nodes; depth-2 is ~220ms.
SEARCH_DEPTH = 3
SEARCH_QUIESCENCE_DEPTH = 2
SEARCH_MAX_QUIESCENCE_NODES = 256
SEARCH_MAX_TOTAL_NODES = 4096
SEARCH_TIME_BUDGET_MS = 2000.0


def _make_nnue_evaluator(nnue_net: Any) -> Any:
    """Builds a leaf evaluator doing a single HalfKP extraction per position.

    Side to move is checkmated => -100.0; any other game-over => 0.0; otherwise a
    single feature extraction feeds the accumulator and dense forward pass.
    """

    def evaluate(s: chess.Board) -> float:
        if s.is_checkmate():
            return -100.0
        if s.is_game_over():
            return 0.0
        w_feats, b_feats, aux = extract_halfkp_features(s)
        acc = Accumulator()
        for idx in w_feats:
            acc.white_acc += nnue_net.W_white[idx]
        for idx in b_feats:
            acc.black_acc += nnue_net.W_black[idx]
        return nnue_net.forward(acc, aux)

    return evaluate


def _capture_ordering(state: chess.Board, actions: list[chess.Move]) -> list[chess.Move]:
    """Capture-first, check-second move ordering for alpha-beta pruning."""
    return sorted(
        actions,
        key=lambda a: (int(state.is_capture(a)), int(state.gives_check(a))),
        reverse=True,
    )


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
        self._beta_efe = max(0.01, min(0.99, beta_efe))
        self._temperature = max(0.05, temperature)

        # Agent state σ = (c, w, g, ρ_ext)
        self.characteristics = Characteristics(skill=0.8, resilience=0.7)
        self.goal_hierarchy = GoalHierarchy()  # Latent utility weights u ∈ ℝ^{n_k}

        # Meta-parameters θ_meta ∈ ℝ^K for feature valuation, temperature, and learned beta_efe
        if isinstance(theta_meta, str) and theta_meta.lower() == "random":
            k_dim = 10
            self.theta_meta = np.random.uniform(0.3, 1.0, size=k_dim).astype(np.float32)
            self.theta_meta[8] = self._temperature
            self.theta_meta[9] = float(np.log(self._beta_efe / (1.0 - self._beta_efe)))
        elif isinstance(theta_meta, str) and theta_meta.lower() == "uniform":
            k_dim = 10
            self.theta_meta = np.ones(k_dim, dtype=np.float32)
            self.theta_meta[8] = self._temperature
            self.theta_meta[9] = float(np.log(self._beta_efe / (1.0 - self._beta_efe)))
        elif isinstance(theta_meta, int):
            self.theta_meta = np.ones(theta_meta, dtype=np.float32)
            self.theta_meta[8] = self._temperature
            if theta_meta >= 10:
                self.theta_meta[9] = float(np.log(self._beta_efe / (1.0 - self._beta_efe)))
        elif theta_meta is None:
            # Default: random uniform initialization (K=10)
            k_dim = 10
            self.theta_meta = np.random.uniform(0.3, 1.0, size=k_dim).astype(np.float32)
            self.theta_meta[8] = self._temperature
            self.theta_meta[9] = float(np.log(self._beta_efe / (1.0 - self._beta_efe)))
        else:
            self.theta_meta = np.array(theta_meta, dtype=np.float32)

        # Persistent search engine cache: one AlphaBetaSearch (with a transposition table
        # surviving across plies) per (nnue_net, depth) combination. Rebuilt whenever a
        # different net instance is supplied (e.g. a fresh net for a new training game).
        self._searcher: AlphaBetaSearch | None = None
        self._search_net: Any | None = None
        self._search_depth: int = 0

    @property
    def beta_efe(self) -> float:
        if len(self.theta_meta) >= 10:
            sig = float(1.0 / (1.0 + np.exp(-self.theta_meta[9])))
            return max(0.01, min(0.99, sig))
        return self._beta_efe

    @beta_efe.setter
    def beta_efe(self, val: float) -> None:
        clamped = max(0.01, min(0.99, val))
        self._beta_efe = clamped
        if len(self.theta_meta) >= 10:
            self.theta_meta[9] = float(np.log(clamped / (1.0 - clamped)))

    @property
    def temperature(self) -> float:
        if len(self.theta_meta) >= 9:
            return float(max(0.05, self.theta_meta[8]))
        return self._temperature

    @temperature.setter
    def temperature(self, val: float) -> None:
        self._temperature = max(0.05, val)
        if len(self.theta_meta) >= 9:
            self.theta_meta[8] = max(0.05, val)

    def extract_move_features(self, board: chess.Board, move: chess.Move) -> np.ndarray:
        """Extracts 8-dimensional numerical feature vector for a candidate legal move."""
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

        # Feature 2: King safety / check threat
        king_safety_val = 1.0 if board.gives_check(move) else 0.0

        # Feature 3: Checkmate opportunity
        test_board = board.copy()
        test_board.push(move)
        checkmate_val = 10.0 if test_board.is_checkmate() else 0.0

        # Feature 4: Next-state mobility
        legal_next = list(test_board.legal_moves)
        mobility_val = len(legal_next) / 20.0

        # Feature 5: Defended friendly pieces (hanging piece prevention)
        defended_val = 0.0
        to_sq = move.to_square
        if test_board.is_attacked_by(board.turn, to_sq):
            defended_val += 1.0
        if test_board.is_attacked_by(not board.turn, to_sq):
            defended_val -= 1.0

        # Feature 6: Capture delta (material gain - potential re-capture loss)
        capture_delta = capture_val
        if test_board.is_attacked_by(not board.turn, to_sq):
            moved_piece = board.piece_at(move.from_square)
            if moved_piece:
                capture_delta -= piece_values[moved_piece.piece_type]
        capture_delta_val = float(max(-10.0, capture_delta))

        # Feature 7: Attacks on enemy king zone
        king_attackers_val = 0.0
        enemy_king_sq = test_board.king(not board.turn)
        if enemy_king_sq is not None:
            king_zone = test_board.attacks(enemy_king_sq)
            if to_sq in king_zone:
                king_attackers_val += 1.0

        return np.array(
            [
                capture_val,
                center_val,
                king_safety_val,
                checkmate_val,
                mobility_val,
                defended_val,
                capture_delta_val,
                king_attackers_val,
            ],
            dtype=np.float32,
        )

    def evaluate_efe_utility(
        self,
        board: chess.Board,
        move: chess.Move,
        depth: int = SEARCH_DEPTH,
        nnue_net: Any | None = None,
    ) -> tuple[float, float, float]:
        """Evaluates pragmatic, epistemic, and total Expected Free Energy utility with search lookahead.

        U_total = (1 - β) * U_pragmatic + β * U_epistemic

        Pragmatic evaluation uses a depth-parameterized alpha-beta negamax search with
        quiescence (capture-first ordering) and the NNUE value net as leaf evaluator.
        The epistemic term is normalized by max legal branching and kept bounded in
        [0, 1] so it only weakly biases play (recalibrated EFE mode, AGENTS.md 009).
        """
        test_board = board.copy()
        test_board.push(move)

        if test_board.is_checkmate():
            return 100.0, 100.0, 0.0

        # Extract linear tactical features weighted by theta_meta (feature-normalized)
        features = self.extract_move_features(board, move)
        goal_categories = list(GoalCategory)
        u_survival = self.goal_hierarchy.u[goal_categories.index(GoalCategory.SURVIVAL)]
        u_acquisition = self.goal_hierarchy.u[goal_categories.index(GoalCategory.ACQUISITION)]
        u_linear = float(
            np.dot(self.theta_meta[: len(features)], features / FEATURE_SCALES[: len(features)])
            * (u_survival + u_acquisition)
        )

        # Pragmatic evaluation via NNUENet + alpha-beta negamax with quiescence
        if nnue_net is not None:
            config = SearchConfig(
                max_depth=max(1, depth),
                time_budget_ms=SEARCH_TIME_BUDGET_MS,
                quiescence_depth=SEARCH_QUIESCENCE_DEPTH,
                max_quiescence_nodes=SEARCH_MAX_QUIESCENCE_NODES,
                max_total_nodes=SEARCH_MAX_TOTAL_NODES,
            )
            searcher = AlphaBetaSearch(
                domain=self.domain,
                evaluator=_make_nnue_evaluator(nnue_net),
                config=config,
                ordering_fn=_capture_ordering,
            )
            # search() returns the value from the perspective of the side to move at
            # test_board (the opponent after `move`), so negate for the agent's view.
            _, opponent_value, _ = searcher.search(test_board)
            u_nnue = float(-opponent_value)
            u_pragmatic = u_nnue + 0.1 * u_linear
        else:
            u_pragmatic = u_linear

        # Bounded epistemic utility: normalized by max legal branching -> [0, 1]
        u_epistemic = self.epistemic_utility(test_board)

        # Total expected utility with active perception beta mixing
        u_total = (1.0 - self.beta_efe) * u_pragmatic + self.beta_efe * u_epistemic

        return u_total, u_pragmatic, u_epistemic

    def epistemic_utility(self, board: chess.Board) -> float:
        """Bounded epistemic utility for a position: normalized by max branching -> [0, 1]."""
        n_legal = len(list(board.legal_moves))
        epistemic_entropy = np.log(n_legal + 1.0) / np.log(31.0)
        return float(epistemic_entropy * self.characteristics.skill)

    def _goal_scale(self) -> float:
        goal_categories = list(GoalCategory)
        u_survival = self.goal_hierarchy.u[goal_categories.index(GoalCategory.SURVIVAL)]
        u_acquisition = self.goal_hierarchy.u[goal_categories.index(GoalCategory.ACQUISITION)]
        return float(u_survival + u_acquisition)

    def _move_utility(
        self,
        board: chess.Board,
        move: chess.Move,
        child_eval: float,
        use_linear_garnish: bool,
    ) -> float:
        """Computes U_total = (1-β)·U_pragmatic + β·U_epistemic for a root move.

        With a value net, U_pragmatic = child_eval + 0.1·U_linear (search dominates); without
        a net, U_pragmatic = U_linear (handcrafted features only).
        """
        features = self.extract_move_features(board, move)
        u_linear = float(
            np.dot(self.theta_meta[: len(features)], features / FEATURE_SCALES[: len(features)])
            * self._goal_scale()
        )
        test_board = board.copy()
        test_board.push(move)
        u_pragmatic = child_eval + (0.1 * u_linear if use_linear_garnish else u_linear)
        u_epistemic = self.epistemic_utility(test_board)
        return (1.0 - self.beta_efe) * u_pragmatic + self.beta_efe * u_epistemic

    def _get_searcher(self, nnue_net: Any, depth: int) -> AlphaBetaSearch:
        """Returns a cached AlphaBetaSearch bound to `nnue_net`, rebuilt when the net changes."""
        if (
            self._searcher is None
            or self._search_net is not nnue_net
            or self._search_depth != depth
        ):
            config = SearchConfig(
                max_depth=max(1, depth),
                time_budget_ms=SEARCH_TIME_BUDGET_MS,
                quiescence_depth=SEARCH_QUIESCENCE_DEPTH,
                max_quiescence_nodes=SEARCH_MAX_QUIESCENCE_NODES,
                max_total_nodes=SEARCH_MAX_TOTAL_NODES,
            )
            self._searcher = AlphaBetaSearch(
                domain=self.domain,
                evaluator=_make_nnue_evaluator(nnue_net),
                config=config,
                ordering_fn=_capture_ordering,
            )
            self._search_net = nnue_net
            self._search_depth = depth
        return self._searcher

    def select_move(
        self,
        board: chess.Board,
        legal_moves: list[chess.Move],
        depth: int = SEARCH_DEPTH,
        nnue_net: Any | None = None,
    ) -> chess.Move:
        """Selects a legal move using active sensing softmax policy distribution with EFE lookahead search.

        With a value net present, a single depth-limited alpha-beta search over the root
        (one pass, no per-move re-searches) yields an exact eval per child; those evals
        are mixed with the tactical/linear and epistemic terms before softmax sampling.
        """
        if not legal_moves:
            raise ValueError("No legal moves available in state.")

        # Check for immediate 1-ply checkmates
        for move in legal_moves:
            test_board = board.copy()
            test_board.push(move)
            if test_board.is_checkmate():
                return move

        if nnue_net is not None:
            searcher = self._get_searcher(nnue_net, depth)
            children = searcher.search_root_children(board)
            child_eval = dict(children)
            utilities = [
                self._move_utility(board, move, child_eval.get(move, 0.0), True)
                for move in legal_moves
            ]
        else:
            utilities = [self._move_utility(board, move, 0.0, False) for move in legal_moves]

        u_arr = np.array(utilities, dtype=np.float32)

        # Softmax sampling with temperature τ
        max_u = np.max(u_arr)
        exp_u = np.exp((u_arr - max_u) / self.temperature)
        probs = exp_u / np.sum(exp_u)

        chosen_idx = int(np.random.choice(len(legal_moves), p=probs))
        return legal_moves[chosen_idx]
