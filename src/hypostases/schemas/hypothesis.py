"""HYPOSTASES Unified Base Hypothesis Abstraction.

Spec Ref: Part III §5, Front 11 (Abductive Reasoning), Front 12 (Scientific Discovery).
"""

from __future__ import annotations


class BaseHypothesis:
    """Unified Base Hypothesis abstract interface for explanatory models.

    Acts as the common base class for Peircean abduction hypotheses (Front 11)
    and Scientific Discovery causal DAG hypotheses (Front 12).
    """

    description: str

    @property
    def id(self) -> str:
        """Normalized unique identifier for the hypothesis."""
        return getattr(self, "hypothesis_id", getattr(self, "identifier", ""))

    @property
    def prior_val(self) -> float:
        """Normalized prior probability."""
        return getattr(self, "prior_probability", getattr(self, "prior", 0.5))

    @property
    def posterior_val(self) -> float:
        """Normalized posterior probability."""
        return getattr(self, "posterior_probability", getattr(self, "posterior", 0.5))

    @property
    def complexity_val(self) -> float:
        """Normalized structural complexity / MDL penalty."""
        return getattr(self, "complexity", getattr(self, "mdl_complexity", 1.0))
