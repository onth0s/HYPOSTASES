"""HYPOSTASES — Hierarchical World Models & Conceptual Spaces (Wave 2 Front 01).

Provides multi-level semantic abstraction layers (Levels 1-6), Gärdenfors Conceptual Spaces,
Voronoi tessellation categorization, and TEM entorhinal grid-cell relational factorizations.
"""

from hypostases.world_model.conceptual_spaces import ConceptualSpaceEngine
from hypostases.world_model.hierarchical_types import (
    AbstractionLevel,
    ConceptualRegion,
    HierarchicalState,
    QualityDimension,
    TEMBasis,
)
from hypostases.world_model.hierarchical_world_model import HierarchicalWorldModel
from hypostases.world_model.tem_factorization import TEMFactorizationEngine

__all__ = [
    "AbstractionLevel",
    "ConceptualRegion",
    "ConceptualSpaceEngine",
    "HierarchicalState",
    "HierarchicalWorldModel",
    "QualityDimension",
    "TEMBasis",
    "TEMFactorizationEngine",
]
