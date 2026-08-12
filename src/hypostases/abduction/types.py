"""HYPOSTASES Engine — Wave 3 Front 11 Abductive Reasoning Types.

Spec Ref: docs/WAVE_3_FRONT_11/front_11_abductive_reasoning_hypothesis_objects_spec.md
Defines hypothesis categories, evidence records, and metric structures for abductive reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)


class HypothesisCategory(StrEnum):
    ENVIRONMENT = "ENVIRONMENT"
    PEER_INTENT = "PEER_INTENT"
    CAUSAL_STRUCTURE = "CAUSAL_STRUCTURE"


@dataclass
class EvidenceRecord:
    """Tracks empirical observation evidence for or against a hypothesis at timestamp t."""

    timestamp: int
    observation_val: float
    predicted_val: float
    residual: float
    is_supporting: bool


@dataclass
class AbductiveMetrics:
    """Summary metrics tracking agent abductive inference trajectory."""

    total_anomalies_detected: int = 0
    total_hypotheses_generated: int = 0
    active_pool_size: int = 0
    top_hypothesis_id: str | None = None
    top_hypothesis_posterior: float = 0.0
    latest_surprise: float = 0.0
