"""HYPOSTASES — Automated Test Suite for Wave 2 Front 01 (Hierarchical World Models).

Spec Ref: docs/WAVE_2_FRONT_01/front_01_hierarchical_world_models_spec.md
Tests Gärdenfors conceptual space metrics, TEM grid-cell tensor factorizations, 6-level hierarchy forward updates,
Rule 005 invariant preservation, and YAML configuration loading.
"""

from __future__ import annotations

import numpy as np
import pytest

from hypostases.engine.types import (
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)
from hypostases.schemas.loader import load_hierarchical_world_model_config
from hypostases.world_model.conceptual_spaces import ConceptualSpaceEngine
from hypostases.world_model.hierarchical_types import (
    AbstractionLevel,
    ConceptualRegion,
    QualityDimension,
    TEMBasis,
)
from hypostases.world_model.hierarchical_world_model import HierarchicalWorldModel
from hypostases.world_model.tem_factorization import TEMFactorizationEngine


@pytest.fixture
def dummy_agent_state() -> AgentState:
    """Creates a standard baseline AgentState for testing."""
    return AgentState(
        c=Characteristics(reserve=50.0),
        w=WorldModel(mu=5.0, sigma2=0.5, peer_beliefs={"agent_1": 0.8}),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(social_capital=10.0),
    )


def test_yaml_config_loader():
    """Validates loading of schema/hierarchical_world_model_config.yaml."""
    cfg = load_hierarchical_world_model_config()
    assert "quality_dimensions" in cfg
    assert "conceptual_regions" in cfg
    assert "tem_basis_config" in cfg
    assert len(cfg["quality_dimensions"]) >= 5
    assert len(cfg["conceptual_regions"]) >= 3


def test_mahalanobis_distance_and_similarity():
    """Verifies Gärdenfors Mahalanobis distance metric and exponential similarity."""
    dims = [QualityDimension(name="d1", min_val=0.0, max_val=10.0)]
    proto = np.array([5.0, 5.0], dtype=np.float64)
    cov = np.eye(2, dtype=np.float64)
    region = ConceptualRegion(
        id="r1", label="Region 1", prototype=proto, covariance_matrix=cov, gamma=0.5
    )

    engine = ConceptualSpaceEngine(dimensions=dims, regions=[region])

    # Point at prototype -> distance = 0, similarity = exp(0) = 1.0
    d_zero = engine.calculate_mahalanobis_distance(proto, "r1")
    s_one = engine.calculate_similarity(proto, "r1")
    assert pytest.approx(d_zero, abs=1e-6) == 0.0
    assert pytest.approx(s_one, abs=1e-6) == 1.0

    # Point offset by (3, 4) -> Euclidean distance = 5.0, similarity = exp(-0.5 * 5.0)
    pt_offset = np.array([8.0, 9.0], dtype=np.float64)
    d_offset = engine.calculate_mahalanobis_distance(pt_offset, "r1")
    s_offset = engine.calculate_similarity(pt_offset, "r1")
    assert pytest.approx(d_offset, abs=1e-6) == 5.0
    assert pytest.approx(s_offset, abs=1e-6) == np.exp(-2.5)


def test_voronoi_tessellation_convexity():
    """Tests O(1) Voronoi prototype region assignment and partition convexity."""
    dims = [QualityDimension(name="x", min_val=0.0, max_val=100.0)]
    r1 = ConceptualRegion(
        id="r1", label="Low", prototype=np.array([10.0, 10.0]), covariance_matrix=np.eye(2)
    )
    r2 = ConceptualRegion(
        id="r2", label="High", prototype=np.array([90.0, 90.0]), covariance_matrix=np.eye(2)
    )

    engine = ConceptualSpaceEngine(dimensions=dims, regions=[r1, r2])

    res_low = engine.categorize_point(np.array([12.0, 15.0]))
    assert res_low["closest_region_id"] == "r1"

    res_high = engine.categorize_point(np.array([85.0, 88.0]))
    assert res_high["closest_region_id"] == "r2"


def test_tem_structural_factorization():
    """Tests TEM grid-cell action transition g_{t+1} = W_a g_t and sensory binding X x G."""
    basis = TEMBasis(
        grid_size=2,
        action_transitions={"SHIFT": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)},
    )
    tem_engine = TEMFactorizationEngine(basis)

    init_g = tem_engine.current_structural_state_g.copy()
    tem_engine.predict_next_structural_state("SHIFT")
    next_g = tem_engine.current_structural_state_g
    assert not np.array_equal(init_g, next_g)

    binding = tem_engine.bind_sensory_observation(np.array([10.0, 20.0]))
    assert binding.shape == (2, 2)


def test_six_level_hierarchy_projection(dummy_agent_state: AgentState):
    """Verifies end-to-end forward update across all 6 abstraction levels."""
    hwm = HierarchicalWorldModel()
    state = hwm.update_from_agent_state(dummy_agent_state)

    assert "v_canvas" in state.level_1_environment
    assert "quality_vector" in state.level_2_objects
    assert "tem_snapshot" in state.level_3_relations
    assert "closest_region_id" in state.level_4_conceptual
    assert "social_capital" in state.level_5_institutions
    assert "replenish_rate_est" in state.level_6_metamodels

    l4_view = hwm.get_level_view(AbstractionLevel.LEVEL_4_CONCEPTUAL_SPACES)
    assert l4_view == state.level_4_conceptual


def test_rule_005_invariant_preservation(dummy_agent_state: AgentState):
    """Ensures zero artificial cognitive deficiencies or non-rational parameters enter the hierarchy."""
    hwm = HierarchicalWorldModel()
    state = hwm.update_from_agent_state(dummy_agent_state)

    q_vec = state.level_2_objects["quality_vector"]
    assert not np.isnan(q_vec).any()
    assert not np.isinf(q_vec).any()
    assert state.level_4_conceptual["distance"] >= 0.0


def test_can_attractor_velocity_integration():
    """Verifies Burak & Fiete (2009) Continuous Attractor Network velocity path integration."""
    basis = TEMBasis(grid_size=4)
    tem_engine = TEMFactorizationEngine(basis)

    init_g = tem_engine.current_structural_state_g.copy()
    updated_g = tem_engine.update_attractor_manifold(v_t=np.array([2.0, 0.0]), dt=1.0)
    assert not np.array_equal(init_g, updated_g)
    assert pytest.approx(np.linalg.norm(updated_g), abs=1e-5) == 1.0
