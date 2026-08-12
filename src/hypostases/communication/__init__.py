"""Communication as Bayesian Evidence Module (Wave 3 Front 06).

Provides probabilistic likelihood message evaluation, dual Bayesian belief updates over continuous state
vectors and discrete hypothesis spaces, ReGreT multi-faceted trust tracking, Subjective Logic evidence
operators, uRSA pragmatics, Crawford-Sobel cheap-talk partition noise filtering, and Acemoglu network deduplication.
"""

from hypostases.communication.bayesian_updater import BayesianCommunicationEngine
from hypostases.communication.deception_signaling import DeceptionSignalingFilter
from hypostases.communication.trust_reputation import TrustReputationEngine
from hypostases.communication.types import (
    BayesianBeliefState,
    DiscreteHypothesisPosterior,
    PeerMessage,
    SubjectiveOpinion,
    TrustProfile,
)

__all__ = [
    "BayesianBeliefState",
    "BayesianCommunicationEngine",
    "DeceptionSignalingFilter",
    "DiscreteHypothesisPosterior",
    "PeerMessage",
    "SubjectiveOpinion",
    "TrustProfile",
    "TrustReputationEngine",
]
