"""HYPOSTASES Engine — Wave 3 Front 11 Hypothesis Object.

Spec Ref: docs/WAVE_3_FRONT_11/front_11_abductive_reasoning_hypothesis_objects_spec.md
Literature Ref: MacKay (2003) Ch. 28 (Occam's Razor), De Kleer & Williams (1987).

Represents an explicit computational explanation hypothesis H_k with predictive model,
assumptions, evidence tracking, and MacKay Occam complexity penalty scoring.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hypostases.abduction.types import EvidenceRecord, HypothesisCategory


@dataclass
class Hypothesis:
    """Explicit computational hypothesis object H_k.

    Maintains predictive model expectations, structural assumptions, empirical evidence,
    and MacKay Occam complexity regularization.
    """

    identifier: str = field(default_factory=lambda: f"H_{uuid.uuid4().hex[:8]}")
    description: str = ""
    category: HypothesisCategory = HypothesisCategory.ENVIRONMENT
    assumptions: dict[str, Any] = field(default_factory=dict)
    prior: float = 0.5
    likelihood: float = 1.0
    posterior: float = 0.5
    complexity: float = 1.0  # Structural complexity score C(H_k) (MacKay Occam factor / MDL length)
    supporting_evidence: list[EvidenceRecord] = field(default_factory=list)
    contradicting_evidence: list[EvidenceRecord] = field(default_factory=list)
    confidence: float = 0.5
    # Optional parameter function or delta dict for forward simulation
    predictive_params: dict[str, float] = field(default_factory=dict)

    def compute_posterior(self, lambda_mdl: float = 0.2) -> float:
        """Computes Bayesian posterior score with MacKay Occam factor penalty.

        ln P(H_k | D) = ln P(D | H_k) - lambda_mdl * C(H_k) + ln P(H_k)
        """
        eps = 1e-9
        safe_likelihood = max(self.likelihood, eps)
        safe_prior = max(self.prior, eps)

        log_like = np.log(safe_likelihood)
        log_prior = np.log(safe_prior)
        occam_penalty = lambda_mdl * self.complexity

        log_posterior = log_like - occam_penalty + log_prior
        # Normalized posterior probability proxy via sigmoid / exponential mapping
        self.posterior = float(1.0 / (1.0 + np.exp(-log_posterior)))
        self.confidence = float(self.posterior)
        return self.posterior

    def update_evidence(self, record: EvidenceRecord, lambda_mdl: float = 0.2) -> None:
        """Appends evidence record and updates empirical likelihood and posterior."""
        if record.is_supporting:
            self.supporting_evidence.append(record)
            # Increment likelihood under Gaussian error model
            residual_factor = np.exp(-0.5 * (record.residual**2))
            self.likelihood = float(0.9 * self.likelihood + 0.1 * residual_factor)
        else:
            self.contradicting_evidence.append(record)
            penalty_factor = np.exp(-record.residual)
            self.likelihood = float(0.85 * self.likelihood * penalty_factor)

        self.compute_posterior(lambda_mdl=lambda_mdl)

    def predict_value(self, base_value: float) -> float:
        """Generates forward expected value given baseline state and predictive params."""
        shift = self.predictive_params.get("shift", 0.0)
        scale = self.predictive_params.get("scale", 1.0)
        return float(base_value * scale + shift)

    def to_dict(self) -> dict[str, Any]:
        """Serializes Hypothesis object to dict representation."""
        return {
            "identifier": self.identifier,
            "description": self.description,
            "category": str(self.category),
            "assumptions": self.assumptions.copy(),
            "prior": self.prior,
            "likelihood": self.likelihood,
            "posterior": self.posterior,
            "complexity": self.complexity,
            "confidence": self.confidence,
            "predictive_params": self.predictive_params.copy(),
            "supporting_count": len(self.supporting_evidence),
            "contradicting_count": len(self.contradicting_evidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hypothesis:
        """Instantiates Hypothesis from dictionary."""
        category = HypothesisCategory(data.get("category", HypothesisCategory.ENVIRONMENT))
        return cls(
            identifier=data.get("identifier", f"H_{uuid.uuid4().hex[:8]}"),
            description=data.get("description", ""),
            category=category,
            assumptions=data.get("assumptions", {}),
            prior=data.get("prior", 0.5),
            likelihood=data.get("likelihood", 1.0),
            posterior=data.get("posterior", 0.5),
            complexity=data.get("complexity", 1.0),
            confidence=data.get("confidence", 0.5),
            predictive_params=data.get("predictive_params", {}),
        )
