"""HYPOSTASES — Hierarchical World Model Data Types & Enums.

Spec Ref: docs/WAVE_2_FRONT_01/front_01_hierarchical_world_models_spec.md
Defines data structures for 6-level abstraction hierarchy, Gärdenfors conceptual spaces,
and TEM entorhinal grid-cell structural basis representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


class AbstractionLevel(IntEnum):
    """6-Level Abstraction Hierarchy Enums."""

    LEVEL_1_ENVIRONMENT = 1
    LEVEL_2_OBJECTS = 2
    LEVEL_3_RELATIONS = 3
    LEVEL_4_CONCEPTUAL_SPACES = 4
    LEVEL_5_INSTITUTIONS = 5
    LEVEL_6_METAMODELS = 6


@dataclass
class QualityDimension:
    r"""Represents a metric axis in Gärdenfors Conceptual Space (Omega \subset R^D)."""

    name: str
    min_val: float
    max_val: float
    metric: str = "euclidean"
    description: str = ""


@dataclass
class ConceptualRegion:
    """Represents a convex Voronoi concept region centered at prototype vector mu_k."""

    id: str
    label: str
    prototype: np.ndarray
    gamma: float = 0.5
    covariance_matrix: np.ndarray = field(default_factory=lambda: np.eye(5, dtype=np.float64))

    def __post_init__(self) -> None:
        """Ensures numpy array types for numerical performance."""
        if not isinstance(self.prototype, np.ndarray):
            self.prototype = np.array(self.prototype, dtype=np.float64)
        if not isinstance(self.covariance_matrix, np.ndarray):
            self.covariance_matrix = np.array(self.covariance_matrix, dtype=np.float64)


@dataclass
class TEMBasis:
    """Tolman-Eichenbaum Machine (TEM) Entorhinal Grid-Cell Basis Factorization.

    Synthesizes Whittington et al. (Cell 2020):
    Structural Basis Matrix G, Sensory Binding Matrix X, and Action Transitions W_a.
    """

    grid_size: int = 4
    num_grid_modules: int = 3
    structural_basis_G: np.ndarray = field(  # noqa: N815
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    sensory_binding_X: np.ndarray = field(  # noqa: N815
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )

    action_transitions: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class HierarchicalState:
    """Complete snapshot of the 6-level semantic abstraction hierarchy at tick t."""

    level_1_environment: dict[str, Any] = field(default_factory=dict)
    level_2_objects: dict[str, Any] = field(default_factory=dict)
    level_3_relations: dict[str, Any] = field(default_factory=dict)
    level_4_conceptual: dict[str, Any] = field(default_factory=dict)
    level_5_institutions: dict[str, Any] = field(default_factory=dict)
    level_6_metamodels: dict[str, Any] = field(default_factory=dict)
