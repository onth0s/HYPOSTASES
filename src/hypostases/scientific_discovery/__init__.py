"""HYPOSTASES Wave 4 Front 12 — Scientific Discovery Loop Package.

Implements the 8-stage iterative scientific discovery cycle:
Observe -> Infer -> Generate Hypotheses -> Rank Explanations -> Design Experiment -> Collect Evidence -> Update Hypotheses -> Act

Under state substrate: \\sigma = (c, w, g, \rho_{ext})
Strictly adhering to Rule 005 (pure game-theoretic rationality),
Rule 006 (data-driven YAML schemas), Rule 009 (Friston EFE mode), and Rule 011 (dual persistence).
"""

from hypostases.scientific_discovery.bayesian_updater import BayesianUpdater
from hypostases.scientific_discovery.experimental_design import (
    AdaptiveContrastiveEstimation,
    BayesianExperimentalDesignEngine,
)
from hypostases.scientific_discovery.hypothesis_manager import HypothesisManager
from hypostases.scientific_discovery.pipeline import ScientificDiscoveryPipeline
from hypostases.scientific_discovery.schemas import (
    Evidence,
    ExperimentalDesign,
    Hypothesis,
    ScientificDiscoveryConfig,
)

__all__ = [
    "AdaptiveContrastiveEstimation",
    "BayesianExperimentalDesignEngine",
    "BayesianUpdater",
    "Evidence",
    "ExperimentalDesign",
    "Hypothesis",
    "HypothesisManager",
    "ScientificDiscoveryConfig",
    "ScientificDiscoveryPipeline",
]
