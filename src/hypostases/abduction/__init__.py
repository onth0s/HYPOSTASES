"""HYPOSTASES Engine — Wave 3 Front 11 Abductive Reasoning & Hypothesis Objects.

Spec Ref: docs/WAVE_3_FRONT_11/front_11_abductive_reasoning_hypothesis_objects_spec.md
Literature Ref: MacKay (2003), De Kleer & Williams (1987), Friston et al. (2017), Pearl (2000).
"""

from hypostases.abduction.abductive_engine import AbductiveEngine
from hypostases.abduction.anomaly_detector import SurpriseDetector
from hypostases.abduction.hypothesis import Hypothesis
from hypostases.abduction.hypothesis_generator import HypothesisGenerator
from hypostases.abduction.types import AbductiveMetrics, EvidenceRecord, HypothesisCategory

__all__ = [
    "AbductiveEngine",
    "AbductiveMetrics",
    "EvidenceRecord",
    "Hypothesis",
    "HypothesisCategory",
    "HypothesisGenerator",
    "SurpriseDetector",
]
