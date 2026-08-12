"""HYPOSTASES — Agent-based modeling framework (Specification v4)."""

from __future__ import annotations

from hypostases.counterfactual import (
    CounterfactualBranch,
    CounterfactualEngine,
    VirtualEnvironmentSandbox,
)
from hypostases.mechanism_search import (
    MechanismCandidate,
    MechanismEvaluator,
    MechanismOptimizer,
    MechanismSearchRunner,
    MechanismSpace,
)

__version__ = "0.2.0"


__all__ = [
    "CounterfactualBranch",
    "CounterfactualEngine",
    "MechanismCandidate",
    "MechanismEvaluator",
    "MechanismOptimizer",
    "MechanismSearchRunner",
    "MechanismSpace",
    "VirtualEnvironmentSandbox",
]
