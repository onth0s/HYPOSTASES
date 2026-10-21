"""HYPOSTASES Pluggable Toy Domain — Finite Subtraction Game (Nim).

A zero-dependency reference implementation of the Domain Protocol.
In this 2-player subtraction game:
  - There is a pile of N stones.
  - On each turn, the player to move may take 1, 2, or 3 stones (constrained by remaining stones).
  - The player who takes the last stone wins (+1.0 reward for the winning move).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hypostases.domains.base import Domain
from hypostases.domains.registry import DomainRegistry


@dataclass(frozen=True)
class NimState:
    """State for 1-pile subtraction game: remaining stones and active player (+1 or -1)."""

    stones: int
    turn: int = 1

    def __repr__(self) -> str:
        return f"Nim(stones={self.stones}, turn={self.turn})"


@DomainRegistry.register("toy_nim")
@DomainRegistry.register("nim")
class SimpleNimDomain(Domain):
    """Zero-dependency reference Domain protocol implementation for Nim."""

    def __init__(self, initial_stones: int = 7) -> None:
        self._initial_stones = int(initial_stones)

    def initial_state(self) -> NimState:
        return NimState(stones=self._initial_stones, turn=1)

    def valid_actions(self, state: Any) -> list[int]:
        if not isinstance(state, NimState) or state.stones <= 0:
            return []
        return [take for take in [1, 2, 3] if take <= state.stones]

    def step(self, state: Any, action: Any) -> tuple[NimState, float, bool, dict[str, Any]]:
        assert isinstance(state, NimState), f"Expected NimState, got {type(state)}"
        take = int(action)
        rem = max(0, state.stones - take)
        done = rem == 0
        reward = 1.0 if done else 0.0
        next_state = NimState(stones=rem, turn=-state.turn)
        return next_state, reward, done, {}

    def to_world_model(self, state: Any) -> np.ndarray:
        assert isinstance(state, NimState), f"Expected NimState, got {type(state)}"
        return np.array([float(state.stones), float(state.turn)], dtype=np.float32)


def nim_evaluator(state: NimState) -> float:
    """Heuristic / terminal evaluator from perspective of side to move.

    If stones == 0, the previous player took the last stone and won,
    so the current player loses (-100.0).
    Winning positions in Nim(3) are those where stones % 4 != 0.
    """
    if state.stones == 0:
        return -100.0
    return 10.0 if (state.stones % 4 != 0) else -10.0
