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

    def search(self, state: Any) -> tuple[Any | None, float, SearchTelemetry]:
        """Performs domain-agnostic iterative deepening negamax search."""
        self.recorder = TelemetryRecorder(mode=self.config.telemetry_mode)
        start_time = time.process_time() * 1000.0

        valid_actions = self.domain.valid_actions(state)
        if not valid_actions:
            return None, 0.0, self.recorder.get_telemetry()

        # Deterministic tie-breaking initial selection
        valid_actions = self.ordering_fn(state, valid_actions)
        best_action = valid_actions[0]
        best_eval = 0.0

        for depth in range(1, self.config.max_depth + 1):
            elapsed_ms = (time.process_time() * 1000.0) - start_time
            if elapsed_ms >= self.config.time_budget_ms:
                break

            self.recorder.telemetry.max_pv_depth = depth
            current_best_action = None
            current_best_eval = -float("inf")
            alpha = -float("inf")
            beta = float("inf")

            ordered_actions = self.ordering_fn(state, valid_actions)

            for action in ordered_actions:
                next_state, _, done, _ = self.domain.step(state, action)

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

    def _negamax(
        self,
        state: Any,
        depth: int,
        alpha: float,
        beta: float,
        start_time: float,
    ) -> float:
        self.recorder.record_node(is_quiescence=False)

        elapsed_ms = (time.process_time() * 1000.0) - start_time
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
        stand_pat = self._evaluate(state)

        if q_depth <= 0:
            return stand_pat

        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        valid_actions = self.domain.valid_actions(state)
        for action in valid_actions:
            next_state, _, _, _ = self.domain.step(state, action)
            score = -self._quiescence(next_state, q_depth - 1, -beta, -alpha)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def _evaluate(self, state: Any) -> float:
        self.recorder.record_leaf_eval()
        return self.recorder.measure("dense_forward_ns", self.evaluator, state)
