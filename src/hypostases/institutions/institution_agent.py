"""First-class InstitutionAgent class managing resources, governance, and authority."""

from typing import Any

from src.hypostases.institutions.adico_engine import ADICOEngine
from src.hypostases.institutions.types import (
    ADICORule,
    DisputeCase,
    InstitutionalRole,
    InstitutionArchetype,
    InstitutionState,
    SanctionRecord,
)


class InstitutionAgent:
    """First-class institutional entity operating as an agent with authority & governance rules."""

    def __init__(
        self,
        institution_id: str,
        name: str,
        archetype: InstitutionArchetype,
        initial_resources: float = 100.0,
        punish_reserve_cost: float = 0.333,
    ):
        self.state = InstitutionState(
            institution_id=institution_id,
            name=name,
            archetype=archetype,
            resources=initial_resources,
        )
        self.adico_engine = ADICOEngine(punish_reserve_cost=punish_reserve_cost)

    def register_member(
        self,
        agent_id: str,
        role: InstitutionalRole = InstitutionalRole.MEMBER,
        initial_authority: float = 0.5,
    ) -> None:
        """Register an agent as a member of this institution with a specific role."""
        self.state.members.add(agent_id)
        self.state.roles[agent_id] = role
        self.state.authority_matrix[agent_id] = initial_authority

    def add_rule(self, rule: ADICORule) -> None:
        """Add an ADICO rule to the institution's rulebook."""
        self.state.rules.append(rule)

    def process_action(
        self,
        agent_id: str,
        action_attempted: str,
        state_context: dict[str, Any],
        timestamp: int = 0,
    ) -> list[SanctionRecord]:
        """Process an agent action against institutional rules and issue sanctions if violated."""
        if agent_id not in self.state.members:
            return []

        agent_role = self.state.roles.get(agent_id, InstitutionalRole.MEMBER)
        sanctions_issued = []

        for rule in self.state.rules:
            compliant = self.adico_engine.evaluate_rule(
                rule, agent_role.value, action_attempted, state_context
            )
            if not compliant:
                sanction = self.adico_engine.execute_sanction(
                    rule,
                    target_agent_id=agent_id,
                    enforcer_id=self.state.institution_id,
                    timestamp=timestamp,
                )
                self.state.sanctions_history.append(sanction)
                self.state.resources -= sanction.cost_to_enforcer
                sanctions_issued.append(sanction)

        return sanctions_issued

    def collect_tax_or_dues(self, agent_id: str, amount: float) -> float:
        """Collect taxes or membership dues from an agent."""
        if agent_id in self.state.members:
            self.state.resources += amount
            return amount
        return 0.0

    def allocate_public_goods(
        self, recipient_ids: list[str], total_amount: float
    ) -> dict[str, float]:
        """Allocate resources from public capital pool to member agents."""
        if self.state.resources < total_amount or not recipient_ids:
            return {}

        per_agent = total_amount / len(recipient_ids)
        self.state.resources -= total_amount
        return {agent_id: per_agent for agent_id in recipient_ids}

    def resolve_dispute(
        self,
        complainant_id: str,
        respondent_id: str,
        claim_amount: float,
        ruling_in_favor_of_complainant: bool,
    ) -> DisputeCase:
        """Arbitrate a dispute between two members (Court archetype)."""
        case_id = f"case_{complainant_id}_{respondent_id}_{len(self.state.disputes)}"
        ruling = claim_amount if ruling_in_favor_of_complainant else 0.0
        dispute = DisputeCase(
            case_id=case_id,
            complainant_id=complainant_id,
            respondent_id=respondent_id,
            claim_amount=claim_amount,
            status="RESOLVED",
            ruling_amount=ruling,
        )
        self.state.disputes.append(dispute)
        return dispute
