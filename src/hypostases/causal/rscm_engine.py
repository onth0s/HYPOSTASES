"""HYPOSTASES — Relational Structural Causal Model (RSCM) Engine.

Spec Ref: docs/WAVE_2_FRONT_08/front_08_causal_world_models_spec.md
Synthesizes Ejaz & Bareinboim (2026).
Extends SCMs to object-relational domains with dynamic entity counts,
enabling zero-shot interventional and counterfactual transfer across unseen graph skeletons.
"""

from __future__ import annotations

from hypostases.causal.causal_types import RSCMSchema
from hypostases.causal.structural_causal_model import StructuralCausalModel


class RelationalSCMEngine:
    """Relational Structural Causal Model engine managing entity-relation causal templates."""

    def __init__(self, schema: RSCMSchema) -> None:
        self.schema = schema

    def instantiate_ground_scm(
        self,
        entity_instances: dict[str, list[str]],
        relation_instances: list[tuple[str, str, str]],
    ) -> StructuralCausalModel:
        """Instantiates a Ground SCM graph M_\rho for a specific skeleton configuration \rho."""
        scm = StructuralCausalModel(name="ground_rscm_instance")

        # Add ground attribute nodes for each entity instance
        for entity_type, instance_ids in entity_instances.items():
            attributes = self.schema.attribute_templates.get(entity_type, ["state", "utility"])
            for inst_id in instance_ids:
                for attr in attributes:
                    node_name = f"{inst_id}.{attr}"
                    scm.add_node(node_name)

        # Add ground causal edges based on relational links
        for src_id, rel_name, tgt_id in relation_instances:
            # Look up template edges or equations
            src_node = f"{src_id}.state"
            tgt_node = f"{tgt_id}.state"
            if src_node in scm.nodes and tgt_node in scm.nodes:
                scm.add_edge(src_node, tgt_node, weight=1.0, label=rel_name)

        return scm

    def transfer_mechanisms(
        self, source_scm: StructuralCausalModel, target_scm: StructuralCausalModel
    ) -> StructuralCausalModel:
        """Transfers shared mechanism equations across different ground skeletons (Ejaz & Bareinboim 2026)."""
        for node_name, eq in source_scm.equations.items():
            # Match template attribute type
            attr_type = node_name.split(".")[-1] if "." in node_name else node_name
            for tgt_node in target_scm.nodes:
                if tgt_node.endswith(f".{attr_type}") and tgt_node not in target_scm.equations:
                    target_scm.set_equation(tgt_node, eq)

        return target_scm
