"""Ostrom ADICO Rule Engine and Sanctioning Logic."""

from typing import Any

from hypostases.institutions.types import (
    ADICORule,
    DeonticModality,
    SanctionRecord,
)


class ADICOEngine:
    """Evaluates ADICO rules and executes game-theoretic sanctions (Fehr & Gächter 2002)."""

    def __init__(self, punish_reserve_cost: float = 0.333, sanction_multiplier: float = 3.0):
        """Initialize the ADICO engine.

        Args:
            punish_reserve_cost: Ratio of enforcer cost to target penalty (1:3 -> 0.333).
            sanction_multiplier: Impact multiplier on target per unit of enforcer cost.
        """
        self.punish_reserve_cost = punish_reserve_cost
        self.sanction_multiplier = sanction_multiplier

    def evaluate_rule(
        self,
        rule: ADICORule,
        agent_role: str,
        action_attempted: str,
        state_context: dict[str, Any],
    ) -> bool:
        """Evaluate if an attempted action complies with a given ADICO rule.

        Returns:
            True if compliant, False if rule is violated.
        """
        if not rule.active:
            return True

        # Check if attribute applies to agent role
        if rule.attribute != "ALL" and rule.attribute != agent_role:
            return True

        # Evaluate condition predicate in context
        condition_met = state_context.get(rule.condition, True)
        if not condition_met:
            return True

        # Evaluate deontic operator against attempted action
        aim_matches = action_attempted == rule.aim

        if rule.deontic == DeonticModality.MUST:
            return aim_matches
        elif rule.deontic == DeonticModality.MUST_NOT:
            return not aim_matches
        elif rule.deontic == DeonticModality.MAY:
            return True

        return True

    def execute_sanction(
        self,
        rule: ADICORule,
        target_agent_id: str,
        enforcer_id: str,
        timestamp: int = 0,
    ) -> SanctionRecord:
        """Execute a sanction for a rule violation using Fehr & Gächter 1:3 cost ratio.

        Returns:
            SanctionRecord detailing enforcer cost and target penalty.
        """
        penalty = rule.or_else_penalty
        enforcer_cost = penalty * self.punish_reserve_cost

        return SanctionRecord(
            sanction_id=f"sanc_{rule.rule_id}_{target_agent_id}_{timestamp}",
            rule_id=rule.rule_id,
            target_agent_id=target_agent_id,
            enforcer_id=enforcer_id,
            cost_to_enforcer=enforcer_cost,
            penalty_to_target=penalty,
            timestamp=timestamp,
        )
