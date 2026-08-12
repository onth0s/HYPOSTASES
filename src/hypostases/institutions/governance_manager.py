"""GovernanceManager orchestrating multi-institution interactions across the swarm."""

from typing import Any

from hypostases.institutions.institution_agent import InstitutionAgent
from hypostases.institutions.types import (
    ADICORule,
    DeonticModality,
    InstitutionArchetype,
)


class GovernanceManager:
    """Manages active institutional entities, rule registries, and swarm-level governance."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.institutions: dict[str, InstitutionAgent] = {}
        self.punish_reserve_cost = self.config.get("institution_layer", {}).get(
            "punish_reserve_cost", 0.333
        )

    def create_institution(
        self,
        institution_id: str,
        name: str,
        archetype: InstitutionArchetype,
        initial_resources: float = 100.0,
    ) -> InstitutionAgent:
        """Charter a new institutional entity within the simulation environment."""
        inst = InstitutionAgent(
            institution_id=institution_id,
            name=name,
            archetype=archetype,
            initial_resources=initial_resources,
            punish_reserve_cost=self.punish_reserve_cost,
        )
        self.institutions[institution_id] = inst
        return inst

    def get_institution(self, institution_id: str) -> InstitutionAgent | None:
        """Retrieve an active institution by ID."""
        return self.institutions.get(institution_id)

    def evaluate_swarm_action(
        self,
        agent_id: str,
        action_attempted: str,
        state_context: dict[str, Any],
        timestamp: int = 0,
    ) -> dict[str, list]:
        """Evaluate an agent's action across all institutions it holds membership in."""
        results = {}
        for inst_id, inst in self.institutions.items():
            if agent_id in inst.state.members:
                sanctions = inst.process_action(
                    agent_id, action_attempted, state_context, timestamp
                )
                if sanctions:
                    results[inst_id] = sanctions
        return results

    def load_presets_from_config(self) -> None:
        """Load institutional presets from YAML configuration if provided."""
        archetypes_config = self.config.get("archetypes", {})

        if "government" in archetypes_config:
            gov_cfg = archetypes_config["government"]
            gov = self.create_institution(
                "gov_primary", "Primary Government", InstitutionArchetype.GOVERNMENT
            )
            for r in gov_cfg.get("adico_rules", []):
                gov.add_rule(
                    ADICORule(
                        rule_id=r["rule_id"],
                        attribute=r["attribute"],
                        deontic=DeonticModality(r["deontic"]),
                        aim=r["aim"],
                        condition=r["condition"],
                        or_else_penalty=r["or_else_penalty"],
                    )
                )

        if "market" in archetypes_config:
            mkt_cfg = archetypes_config["market"]
            mkt = self.create_institution(
                "mkt_primary", "Primary Market", InstitutionArchetype.MARKET
            )
            for r in mkt_cfg.get("adico_rules", []):
                mkt.add_rule(
                    ADICORule(
                        rule_id=r["rule_id"],
                        attribute=r["attribute"],
                        deontic=DeonticModality(r["deontic"]),
                        aim=r["aim"],
                        condition=r["condition"],
                        or_else_penalty=r["or_else_penalty"],
                    )
                )
