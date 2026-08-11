"""HYPOSTASES — Agent-based modeling framework (Specification v4)."""

from __future__ import annotations

from hypostases.counterfactual import (
    CounterfactualBranch,
    CounterfactualEngine,
    VirtualEnvironmentSandbox,
)

__version__ = "0.2.0"


__all__ = [
    "CounterfactualBranch",
    "CounterfactualEngine",
    "VirtualEnvironmentSandbox",
]
