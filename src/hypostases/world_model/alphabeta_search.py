"""Domain-Agnostic Alpha-Beta Negamax Search Engine.

Fully decoupled from specific domain representations (Chess, Tetris, GridWorld).
Ref: DISSONANCES.md D-008 (Priority 1 Domain Decoupling).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from hypostases.domains.base import Domain
from hypostases.world_model.telemetry import SearchTelemetry, TelemetryMode, TelemetryRecorder


@dataclass
class SearchConfig:
    max_depth: int = 6
    time_budget_ms: float = 100.0
    tt_size_mb: int = 16
    quiescence_depth: int = 4
    max_quiescence_nodes: int = 0
    max_total_nodes: int = 0
    telemetry_mode: TelemetryMode = TelemetryMode.FAST


# Type aliases for domain-agnostic callbacks
EvaluatorFn = Callable[[Any], float]
OrderingFn = Callable[[Any, list[Any]], list[Any]]


def default_ordering_fn(state: Any, actions: list[Any]) -> list[Any]:
    """Default string representation sort for deterministic tie-breaking."""
    return sorted(actions, key=lambda a: str(a))


class AlphaBetaSearch:
    """Domain-agnostic Negamax Alpha-Beta search driver."""

    def __init__(
        self,
        domain: Domain,
        evaluator: EvaluatorFn,
        config: SearchConfig | None = None,
        ordering_fn: OrderingFn | None = None,
        quiescence_evaluator: EvaluatorFn | None = None,
    ) -> None:
        self.domain = domain
        self.evaluator = evaluator
        self.config = config or SearchConfig()
        self.ordering_fn = ordering_fn or default_ordering_fn
        self.quiescence_evaluator = quiescence_evaluator or evaluator
        self.tt: dict[str, tuple[int, float, str]] = {}  # state_key -> (depth, eval, flag)
        self.recorder = TelemetryRecorder(mode=self.config.telemetry_mode)
        self._qnode_remaining: int | None = (
            self.config.max_quiescence_nodes if self.config.max_quiescence_nodes > 0 else None
        )
        self._node_remaining: int | None = (
            self.config.max_total_nodes if self.config.max_total_nodes > 0 else None
        )

    def _reset_qnode_budget(self) -> None:
        """Resets the per-search quiescence node budget from the current config."""
        self._qnode_remaining = (
            self.config.max_quiescence_nodes if self.config.max_quiescence_nodes > 0 else None
        )

    def _reset_node_budget(self) -> None:
        """Resets the per-search global node budget from the current config."""
        self._node_remaining = (
            self.config.max_total_nodes if self.config.max_total_nodes > 0 else None
        )

    def _node_budget_exhausted(self) -> bool:
        """Decrements the global node budget; True when the budget is exhausted.

        Once exhausted, all subsequent search/expansion/evaluation work is skipped
        and a neutral 0.0 value is returned, so the expensive deep expansion work is
        capped at roughly `max_total_nodes` budget points. A trailing breadth pass
        over already-scheduled sibling nodes may still visit shallow nodes, so this
        bounds cost but does not impose a hard node-count cap.
        """
        if self._node_remaining is None:
            return False
        self._node_remaining -= 1
        return self._node_remaining < 0

    def search(self, state: Any) -> tuple[Any | None, float, SearchTelemetry]:
        """Performs domain-agnostic iterative deepening negamax search."""
        self.recorder = TelemetryRecorder(mode=self.config.telemetry_mode)
        self._reset_qnode_budget()
        self._reset_node_budget()
        start_time = time.perf_counter() * 1000.0

        valid_actions = self.domain.valid_actions(state)
        if not valid_actions:
            return None, 0.0, self.recorder.get_telemetry()

        # Deterministic tie-breaking initial selection
        valid_actions = self.ordering_fn(state, valid_actions)
        best_action = valid_actions[0]
        best_eval = 0.0

        for depth in range(1, self.config.max_depth + 1):
            elapsed_ms = (time.perf_counter() * 1000.0) - start_time
            if elapsed_ms >= self.config.time_budget_ms or self._node_budget_exhausted():
                break

            self.recorder.telemetry.max_pv_depth = depth
            current_best_action = None
            current_best_eval = -float("inf")
            alpha = -float("inf")
            beta = float("inf")

            ordered_actions = self.ordering_fn(state, valid_actions)

            for action in ordered_actions:
                next_state, _, _done, _ = self.domain.step(state, action)

                val = -self._negamax(
                    next_state,
                    depth - 1,
                    -beta,
                    -alpha,
                    start_time,
                )

                if val > current_best_eval:
                    current_best_eval = val
                    current_best_action = action

                alpha = max(alpha, val)
                if alpha >= beta:
                    self.recorder.record_cutoff()
                    break

            if current_best_action is not None:
                best_action = current_best_action
                best_eval = current_best_eval

        return best_action, best_eval, self.recorder.get_telemetry()

    def search_root_children(self, state: Any) -> list[tuple[Any, float]]:
        """Single depth-limited negamax pass returning (action, eval) for every root child.

        Performs one search of depth `max_depth` (no iterative deepening re-search) and
        returns the value of each root action from the perspective of the side to move at
        `state`. Root children are searched with full (-inf, inf) windows so every returned
        eval is exact — safe to feed into a per-move policy (e.g. softmax mixing) rather
        than only keeping the best move. The transposition table persists across calls.
        """
        self.recorder = TelemetryRecorder(mode=self.config.telemetry_mode)
        self._reset_qnode_budget()
        self._reset_node_budget()
        start_time = time.perf_counter() * 1000.0

        valid_actions = self.domain.valid_actions(state)
        if not valid_actions:
            return []

        children: list[tuple[Any, float]] = []
        ordered_actions = self.ordering_fn(state, valid_actions)
        for action in ordered_actions:
            next_state, _, _done, _ = self.domain.step(state, action)
            val = -self._negamax(
                next_state,
                self.config.max_depth - 1,
                -float("inf"),
                float("inf"),
                start_time,
            )
            children.append((action, float(val)))
        return children

    def _negamax(
        self,
        state: Any,
        depth: int,
        alpha: float,
        beta: float,
        start_time: float,
    ) -> float:
        self.recorder.record_node(is_quiescence=False)

        if self._node_budget_exhausted():
            return 0.0

        elapsed_ms = (time.perf_counter() * 1000.0) - start_time
        if elapsed_ms >= self.config.time_budget_ms:
            return self._evaluate(state)

        valid_actions = self.domain.valid_actions(state)
        if depth <= 0 or not valid_actions:
            return self._quiescence(state, self.config.quiescence_depth, alpha, beta)

        state_key = str(state)
        if state_key in self.tt:
            tt_depth, tt_val, tt_flag = self.tt[state_key]
            if tt_depth >= depth:
                self.recorder.record_tt_lookup(hit=True)
                if tt_flag == "EXACT":
                    return tt_val
                elif tt_flag == "LOWERBOUND":
                    alpha = max(alpha, tt_val)
                elif tt_flag == "UPPERBOUND":
                    beta = min(beta, tt_val)
                if alpha >= beta:
                    return tt_val
        else:
            self.recorder.record_tt_lookup(hit=False)

        ordered_actions = self.ordering_fn(state, valid_actions)
        best_val = -float("inf")
        original_alpha = alpha

        for action in ordered_actions:
            next_state, _, _, _ = self.domain.step(state, action)
            val = -self._negamax(next_state, depth - 1, -beta, -alpha, start_time)

            if val > best_val:
                best_val = val

            alpha = max(alpha, val)
            if alpha >= beta:
                self.recorder.record_cutoff()
                break

        flag = "EXACT"
        if best_val <= original_alpha:
            flag = "UPPERBOUND"
        elif best_val >= beta:
            flag = "LOWERBOUND"
        self.tt[state_key] = (depth, best_val, flag)

        return best_val

    def _quiescence(
        self,
        state: Any,
        q_depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        self.recorder.record_node(is_quiescence=True)
        if self._node_budget_exhausted():
            return 0.0
        if self._qnode_remaining is not None:
            self._qnode_remaining -= 1
            if self._qnode_remaining < 0:
                return self._evaluate(state)
        stand_pat = self._evaluate(state)

        if q_depth <= 0:
            return stand_pat

        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        valid_actions = self.domain.valid_actions(state)
        # Quiescence filtering: only search tactical moves (captures or checks)
        tactical_actions = [a for a in valid_actions if state.is_capture(a) or state.gives_check(a)]

        for action in tactical_actions:
            next_state, _, _, _ = self.domain.step(state, action)
            score = -self._quiescence(next_state, q_depth - 1, -beta, -alpha)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def _evaluate(self, state: Any) -> float:
        if self._node_budget_exhausted():
            return 0.0
        self.recorder.record_leaf_eval()
        return self.recorder.measure("dense_forward_ns", self.evaluator, state)
