"""HYPOSTASES Pluggable Domain Interface — Base Protocol Specification.

Defines the load-bearing Domain Protocol contract.
Core engine components (engine, planning, meta_learning) depend strictly on this interface.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Domain(Protocol):
    """Abstract Domain Protocol interface for environment pluggability."""

    def initial_state(self) -> Any:
        """Returns the initial state of the domain environment."""
        ...

    def valid_actions(self, state: Any) -> list[Any]:
        """Returns the list of valid, legal actions given the current state."""
        ...

    def step(self, state: Any, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        """Executes action from state and returns (next_state, reward, done, info).

        Rewards must be non-shaped, outcome-based game outcomes (e.g. +1.0, -1.0, 0.0).
        """
        ...

    def to_world_model(self, state: Any) -> np.ndarray:
        """Encodes state into a numerical tensor representation for world model processing."""
        ...
