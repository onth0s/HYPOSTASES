"""HYPOSTASES — Master Hierarchical World Model & Abstraction Hierarchy.

Spec Ref: docs/WAVE_2_FRONT_01/front_01_hierarchical_world_models_spec.md
Integrates 6 abstraction levels over persistent primitive state sigma = (c, w, g, rho_ext).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hypostases.engine.types import AgentState
from hypostases.schemas.loader import load_hierarchical_world_model_config
from hypostases.world_model.conceptual_spaces import ConceptualSpaceEngine
from hypostases.world_model.hierarchical_types import (
    AbstractionLevel,
    ConceptualRegion,
    HierarchicalState,
    QualityDimension,
    TEMBasis,
)
from hypostases.world_model.tem_factorization import TEMFactorizationEngine


class HierarchicalWorldModel:
    """Master engine maintaining the 6-level semantic abstraction hierarchy in state component w."""

    def __init__(self, config_dict: dict[str, Any] | None = None) -> None:
        if config_dict is None:
            config_dict = load_hierarchical_world_model_config()

        self.config = config_dict
        self._init_conceptual_spaces()
        self._init_tem_factorization()
        self.state = HierarchicalState()

    def _init_conceptual_spaces(self) -> None:
        """Initializes Level 4 Gärdenfors conceptual space engine from YAML config."""
        dim_configs = self.config.get("quality_dimensions", [])
        dims = [
            QualityDimension(
                name=d["name"],
                min_val=float(d["min_val"]),
                max_val=float(d["max_val"]),
                metric=d.get("metric", "euclidean"),
                description=d.get("description", ""),
            )
            for d in dim_configs
        ]

        region_configs = self.config.get("conceptual_regions", [])
        regions = [
            ConceptualRegion(
                id=r["id"],
                label=r["label"],
                prototype=np.array(r["prototype"], dtype=np.float64),
                gamma=float(r.get("gamma", 0.5)),
                covariance_matrix=np.array(
                    r.get("covariance_matrix", np.eye(len(r["prototype"]))),
                    dtype=np.float64,
                ),
            )
            for r in region_configs
        ]

        self.conceptual_engine = ConceptualSpaceEngine(dimensions=dims, regions=regions)

    def _init_tem_factorization(self) -> None:
        """Initializes Level 3 Tolman-Eichenbaum Machine (TEM) relational engine from YAML config."""
        tem_cfg = self.config.get("tem_basis_config", {})
        grid_size = int(tem_cfg.get("grid_size", 4))
        num_modules = int(tem_cfg.get("num_grid_modules", 3))

        transitions_cfg = tem_cfg.get("action_transitions", {})
        action_transitions = {
            act: np.array(mat, dtype=np.float64) for act, mat in transitions_cfg.items()
        }

        tem_basis = TEMBasis(
            grid_size=grid_size,
            num_grid_modules=num_modules,
            structural_basis_G=np.eye(grid_size, dtype=np.float64),
            sensory_binding_X=np.eye(grid_size, dtype=np.float64),
            action_transitions=action_transitions,
        )

        self.tem_engine = TEMFactorizationEngine(tem_basis)

    def update_from_agent_state(
        self, agent: AgentState, raw_sensory: dict[str, Any] | None = None
    ) -> HierarchicalState:
        """Executes multi-level forward update across all 6 abstraction levels.

        Args:
            agent: Primitive agent state tuple sigma = (c, w, g, rho_ext).
            raw_sensory: Optional raw environment observation array V_canvas.
        """
        # Level 1: Environment Raw Sensory Array
        env_data = raw_sensory if raw_sensory is not None else {}
        self.state.level_1_environment = {
            "v_canvas": env_data.get("v_canvas", np.zeros((4, 4))),
            "timestamp": env_data.get("timestamp", 0),
        }

        # Level 2: Objects & Disentangled Quality Vector x \in R^D
        # Extracted from c.reserve, w.pool_density, peer_trust, kappa, w.sigma2
        quality_vector = np.array(
            [
                float(agent.c.reserve),
                float(getattr(agent.w, "pool_density", 0.5)),
                float(
                    np.mean(list(agent.w.peer_beliefs.values())) if agent.w.peer_beliefs else 0.5
                ),
                float(getattr(agent.w, "scarcity_index", 0.1)),
                float(agent.w.sigma2),
            ],
            dtype=np.float64,
        )
        self.state.level_2_objects = {
            "quality_vector": quality_vector,
            "dimensions": list(self.conceptual_engine.dimensions.keys()),
        }

        # Level 3: Relations (TEM Grid-Cell Tensor Factorization)
        sensory_binding = self.tem_engine.bind_sensory_observation(quality_vector[:4])
        self.state.level_3_relations = {
            "tem_snapshot": self.tem_engine.get_relational_snapshot(),
            "sensory_binding_tensor": sensory_binding,
        }

        # Level 4: Conceptual Spaces (Gärdenfors Voronoi Tessellation & Categorization)
        cat_result = self.conceptual_engine.categorize_point(quality_vector)
        self.state.level_4_conceptual = cat_result

        # Level 5: Institutions & Norms
        self.state.level_5_institutions = {
            "social_capital": float(agent.rho_ext.social_capital),
            "normative_alignment": float(agent.c.sociality),
        }

        # Level 6: Meta-models (SCM Belief Priors)
        self.state.level_6_metamodels = {
            "replenish_rate_est": float(agent.w.replenish_rate_est),
            "epistemic_variance": float(agent.w.sigma2),
        }

        return self.state

    def predict_action_structural_step(self, action_key: str) -> np.ndarray:
        """Executes TEM path integration step for level 3 prediction."""
        return self.tem_engine.predict_next_structural_state(action_key)

    def get_level_view(self, level: AbstractionLevel) -> dict[str, Any]:
        """Returns the specific state view for requested hierarchy level (Levels 1-6)."""
        mapping = {
            AbstractionLevel.LEVEL_1_ENVIRONMENT: self.state.level_1_environment,
            AbstractionLevel.LEVEL_2_OBJECTS: self.state.level_2_objects,
            AbstractionLevel.LEVEL_3_RELATIONS: self.state.level_3_relations,
            AbstractionLevel.LEVEL_4_CONCEPTUAL_SPACES: self.state.level_4_conceptual,
            AbstractionLevel.LEVEL_5_INSTITUTIONS: self.state.level_5_institutions,
            AbstractionLevel.LEVEL_6_METAMODELS: self.state.level_6_metamodels,
        }
        return mapping[level]
