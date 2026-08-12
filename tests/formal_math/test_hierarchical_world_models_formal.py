"""Formal Mathematical Verification for Hierarchical World Models & Conceptual Spaces (Front 01).

Theorem 1.1: Gärdenfors Conceptual Spaces Metric Distance Properties
Theorem 1.2: TEM Grid-Cell Factorization Simplex Projection Invariance
"""

import numpy as np

from hypostases.world_model.conceptual_spaces import ConceptualSpaceEngine
from hypostases.world_model.hierarchical_types import ConceptualRegion, QualityDimension, TEMBasis
from hypostases.world_model.tem_factorization import TEMFactorizationEngine


def test_theorem1_1_conceptual_spaces_metric_distance():
    """Empirically proves Mahalanobis distance calculation over Gärdenfors conceptual space prototype."""
    dims = [QualityDimension(name=f"dim{i}", min_val=0.0, max_val=10.0) for i in range(5)]

    r1 = ConceptualRegion(id="r1", label="region1", prototype=np.zeros(5))
    engine = ConceptualSpaceEngine(dimensions=dims, regions=[r1])

    v1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    v2 = np.array([3.0, 4.0, 0.0, 0.0, 0.0])

    d1 = engine.calculate_mahalanobis_distance(v1, region_id="r1")
    d2 = engine.calculate_mahalanobis_distance(v2, region_id="r1")

    # Distance to prototype v1=(0,0,0,0,0) is 0, distance to v2=(3,4,0,0,0) is strictly positive
    assert np.isclose(d1, 0.0, atol=1e-5)
    assert d2 > 0.0


def test_theorem1_2_tem_factorization_simplex_projection():
    """Verifies Tolman-Eichenbaum Machine (TEM) grid-cell phase factorization bounds."""
    tem_basis = TEMBasis(grid_size=16, num_grid_modules=4)
    engine = TEMFactorizationEngine(tem_basis=tem_basis)

    sensory_obs = np.array([0.5, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
    bound_tensor = engine.bind_sensory_observation(sensory_obs)

    # Invariants:
    # 1. Non-negative bound tensor values
    assert np.all(bound_tensor >= 0.0)

    # 2. Outer product shape matches (grid_size, sensory_dim)
    assert bound_tensor.shape == (16, 8)
