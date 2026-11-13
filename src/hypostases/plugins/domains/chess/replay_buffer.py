"""Experience Replay buffer and position storage for chess self-play training."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chess


class ExperienceReplayBuffer:
    """FIFO replay buffer for positions with discounted outcome value targets."""

    def __init__(self, capacity: int = 6000) -> None:
        self.capacity = capacity
        self.positions: list[tuple[chess.Board, float]] = []

    def extend(self, new_positions: list[tuple[chess.Board, float]]) -> None:
        """Add new labeled positions and trim to capacity using FIFO ordering."""
        self.positions.extend(new_positions)
        if len(self.positions) > self.capacity:
            self.positions = self.positions[-self.capacity :]

    def sample(self, max_samples: int | None = None) -> list[tuple[chess.Board, float]]:
        """Sample from positions uniformly or return full buffer if smaller than requested."""
        if not self.positions:
            return []
        if max_samples is None or len(self.positions) <= max_samples:
            return list(self.positions)
        return random.sample(self.positions, max_samples)

    def __len__(self) -> int:
        return len(self.positions)
