"""HYPOSTASES — Agent-based modeling framework (Specification v4)."""

from __future__ import annotations

from hypostases.active_perception import (
    execute_epistemic_action,
    execute_multivariate_epistemic_action,
    load_active_sensing_config,
)
from hypostases.counterfactual import (
    CounterfactualBranch,
    CounterfactualEngine,
    VirtualEnvironmentSandbox,
)
from hypostases.epistemic_utility import (
    compute_epistemic_utility,
    compute_expected_free_energy,
    compute_expected_information_gain,
)
from hypostases.mechanism_search import (
    MechanismCandidate,
    MechanismEvaluator,
    MechanismOptimizer,
    MechanismSearchRunner,
    MechanismSpace,
)
from hypostases.natural_language_compression import (
    CommunicativeLanguageSymbolismRouter,
    NaturalLanguageCompressionEngine,
    NaturalLanguageGovernanceProtocol,
    SymbolicAbductionInterface,
    SymbolicCompressionEngine,
    SymbolicMappingTransferLayer,
    SymbolicMessage,
    SymbolToken,
    VisualEpistemicDualityMapper,
    VisualGistCell,
)

__version__ = "0.4.0"


__all__ = [
    "CommunicativeLanguageSymbolismRouter",
    "CounterfactualBranch",
    "CounterfactualEngine",
    "MechanismCandidate",
    "MechanismEvaluator",
    "MechanismOptimizer",
    "MechanismSearchRunner",
    "MechanismSpace",
    "NaturalLanguageCompressionEngine",
    "NaturalLanguageGovernanceProtocol",
    "SymbolToken",
    "SymbolicAbductionInterface",
    "SymbolicCompressionEngine",
    "SymbolicMappingTransferLayer",
    "SymbolicMessage",
    "VirtualEnvironmentSandbox",
    "VisualEpistemicDualityMapper",
    "VisualGistCell",
    "compute_epistemic_utility",
    "compute_expected_free_energy",
    "compute_expected_information_gain",
    "execute_epistemic_action",
    "execute_multivariate_epistemic_action",
    "load_active_sensing_config",
]
