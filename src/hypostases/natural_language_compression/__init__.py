"""Wave 5 Front 14 — Natural Language as Symbolic Compression & Visual-Epistemic Duality."""

from __future__ import annotations

from hypostases.natural_language_compression.duality import VisualEpistemicDualityMapper
from hypostases.natural_language_compression.engine import SymbolicCompressionEngine
from hypostases.natural_language_compression.interfaces import (
    NaturalLanguageGovernanceProtocol,
    SymbolicAbductionInterface,
)
from hypostases.natural_language_compression.router import CommunicativeLanguageSymbolismRouter
from hypostases.natural_language_compression.transfer import SymbolicMappingTransferLayer
from hypostases.natural_language_compression.types import (
    SymbolicMessage,
    SymbolToken,
    VisualGistCell,
)

# Maintain backward compatibility with previous NaturalLanguageCompressionEngine alias
NaturalLanguageCompressionEngine = SymbolicCompressionEngine

__all__ = [
    "CommunicativeLanguageSymbolismRouter",
    "NaturalLanguageCompressionEngine",
    "NaturalLanguageGovernanceProtocol",
    "SymbolToken",
    "SymbolicAbductionInterface",
    "SymbolicCompressionEngine",
    "SymbolicMappingTransferLayer",
    "SymbolicMessage",
    "VisualEpistemicDualityMapper",
    "VisualGistCell",
]
