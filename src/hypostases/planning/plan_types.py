"""HYPOSTASES Planning — Plan Types & Data Structures.

Spec Ref: docs/WAVE_2_FRONT_02/front_02_explicit_planning_layer_spec.md
Synthesizes GOAP (Orkin 2003/2006) and HTN (Erol et al. 1994) task network data models.
Core Invariant Compliance: Operates strictly over primitive state tuple σ = (c, w, g, ρ_ext).
Rule 005 Prohibition: Pure game-theoretic state dynamics; zero artificial human cognitive defects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PlanStatus(str, Enum):
    """Execution status of an explicit Plan object."""

    PLANNED = "PLANNED"
    EXECUTING = "EXECUTING"
    PAUSED = "PAUSED"
    INTERRUPTED = "INTERRUPTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class ContingencyBranch:
    """Dynamic contingency branch (LATS Synthesis).

    Triggers alternative plan node when state predicate condition evaluates to True.
    """

    branch_id: str
    condition_predicate: Callable[[dict, dict], bool]  # Evaluates over (w, c)
    target_node_id: str
    description: str = ""


@dataclass
class PlanNode:
    """Single task network node (GOAP/HTN Synthesis).

    n_i = (a_i, φ_pre, φ_post, σ_hat, Δu_expected)
    """

    node_id: str
    action_name: str
    action_params: dict[str, Any] = field(default_factory=dict)
    preconditions: dict[str, Any] = field(default_factory=dict)  # Required state predicates (GOAP)
    effects: dict[str, Any] = field(default_factory=dict)  # Expected postcondition updates (GOAP)
    expected_utility_delta: float = 0.0  # Game-theoretic payoff expectation
    expected_predicted_state: dict[str, Any] | None = None  # Predicted outcome state σ_hat
    contingency_branches: list[ContingencyBranch] = field(default_factory=list)


@dataclass
class Plan:
    """First-class computational Plan object (Π).

    Π = (g, N, E, φ_global_inv, status, breakpoint_k)
    """

    plan_id: str
    goal_name: str
    goal_params: dict[str, Any] = field(default_factory=dict)
    nodes: list[PlanNode] = field(default_factory=list)
    global_invariants: dict[str, Any] = field(default_factory=dict)
    status: PlanStatus = PlanStatus.PLANNED
    breakpoint_k: int = 0  # start_from checkpoint index (AdaPlanner Synthesis)
    metadata: dict[str, Any] = field(default_factory=dict)

    def current_node(self) -> PlanNode | None:
        """Returns the active PlanNode based on current breakpoint_k."""
        if 0 <= self.breakpoint_k < len(self.nodes):
            return self.nodes[self.breakpoint_k]
        return None

    def advance_step(self) -> None:
        """Advances breakpoint_k to next node."""
        self.breakpoint_k += 1
        if self.breakpoint_k >= len(self.nodes):
            self.status = PlanStatus.COMPLETED

    def is_finished(self) -> bool:
        """Returns True if plan has reached COMPLETED or FAILED state."""
        return self.status in (PlanStatus.COMPLETED, PlanStatus.FAILED)
