"""Mechanism space and candidate representations for Wave 4 Front 10.

Manages allocation rules, payment rules, governance parameters, and candidate mechanism serialization
under Rule 006 (YAML-driven) and Rule 011 (Dual persistence).
"""

import math
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class AllocationRule:
    """Representation of an institutional allocation rule X(b, s)."""

    rule_type: str = (
        "parametric"  # "highest_bidder", "virtual_valuation_max", "samuelson_efficient", "ast_tree"
    )
    parameters: dict[str, float] = field(default_factory=dict)
    ast_expression: str | None = None

    def allocate(self, bids: list[float], state: dict[str, Any]) -> list[float]:
        """Compute allocation probabilities/shares on the 1-simplex sum(X_i) <= 1."""
        n = len(bids)
        if n == 0:
            return []

        if self.rule_type == "highest_bidder" or self.rule_type == "vickrey":
            max_bid = max(bids) if bids else 0.0
            if max_bid <= 0.0:
                return [0.0] * n
            # Allocate to highest bidder (tie-breaking uniformly)
            winners = [i for i, b in enumerate(bids) if b == max_bid]
            alloc = [0.0] * n
            for w in winners:
                alloc[w] = 1.0 / len(winners)
            return alloc

        elif self.rule_type == "virtual_valuation_max":
            reserve = self.parameters.get("reserve_price", 5.0)
            valid_bids = [(i, b) for i, b in enumerate(bids) if b >= reserve]
            if not valid_bids:
                return [0.0] * n
            max_bid = max(b for _, b in valid_bids)
            winners = [i for i, b in valid_bids if b == max_bid]
            alloc = [0.0] * n
            for w in winners:
                alloc[w] = 1.0 / len(winners)
            return alloc

        elif self.rule_type == "samuelson_efficient":
            # Public goods allocation: 1 if sum(bids) >= cost else 0
            cost = self.parameters.get("marginal_cost", 10.0)
            provision = 1.0 if sum(bids) >= cost else 0.0
            return [provision] * n

        elif self.rule_type == "softmax_proportional":
            temp = max(self.parameters.get("temperature", 1.0), 1e-6)
            exp_bids = [math.exp(b / temp) for b in bids]
            sum_exp = sum(exp_bids)
            return [eb / sum_exp for eb in exp_bids] if sum_exp > 0 else [1.0 / n] * n

        else:
            # Default fallback: uniform allocation
            return [1.0 / n] * n


@dataclass
class PaymentRule:
    """Representation of an institutional payment/tax rule P(b, s)."""

    rule_type: str = "second_highest_price"  # "second_highest_price", "externality_tax", "progressive_tax", "virtual_threshold"
    parameters: dict[str, float] = field(default_factory=dict)

    def calculate_payments(
        self, bids: list[float], allocations: list[float], state: dict[str, Any]
    ) -> list[float]:
        """Compute payment transfers P_i given bids, allocations, and institutional state."""
        n = len(bids)
        if n == 0:
            return []

        if self.rule_type in (
            "second_highest_price",
            "vickrey",
            "virtual_valuation_max",
            "virtual_threshold",
        ):
            reserve = self.parameters.get("reserve_price", 0.0)
            payments = [0.0] * n
            for i in range(n):
                if allocations[i] > 0.0:
                    other_bids = [bids[j] for j in range(n) if j != i]
                    second_price = max(other_bids) if other_bids else 0.0
                    threshold_payment = max(reserve, second_price)
                    payments[i] = threshold_payment * allocations[i]
            return payments

        elif self.rule_type == "externality_tax":
            # Clarke pivot tax: payment_i = externality imposed on others
            payments = [0.0] * n
            cost = self.parameters.get("marginal_cost", 10.0)
            assigned_prices = state.get("assigned_prices", [cost / max(n, 1)] * n)
            for i in range(n):
                others_demand = sum(bids[j] for j in range(n) if j != i)
                # Pivot tax applies if participant i changed the outcome for others
                if others_demand < cost and (others_demand + bids[i]) >= cost:
                    payments[i] = max(0.0, cost - others_demand)
                else:
                    payments[i] = assigned_prices[i] if allocations[i] > 0.0 else 0.0
            return payments

        elif self.rule_type == "progressive_tax":
            # AI Economist progressive tax brackets
            brackets = state.get("tax_brackets", [0.0, 10.0, 50.0, 100.0])
            rates = state.get("tax_rates", [0.0, 0.15, 0.30, 0.45])
            payments = []
            for y in bids:  # income y
                tax = 0.0
                for k in range(len(brackets)):
                    b_low = brackets[k]
                    b_high = brackets[k + 1] if k + 1 < len(brackets) else float("inf")
                    if y > b_low:
                        taxable = min(y, b_high) - b_low
                        tax += taxable * rates[k]
                payments.append(tax)
            return payments

        else:
            # Pay bid
            return [b * a for b, a in zip(bids, allocations, strict=False)]


@dataclass
class GovernanceRule:
    """Representation of institutional governance parameters rho_ext."""

    punish_reserve_cost: float = 0.1
    voting_threshold: float = 0.5
    authority_penalty_multiplier: float = 1.5
    institutional_reserve_req: float = 1.0


@dataclass
class MechanismCandidate:
    """Complete candidate mechanism mu in Mechanism Space M."""

    candidate_id: str
    name: str
    allocation_rule: AllocationRule
    payment_rule: PaymentRule
    governance_rule: GovernanceRule = field(default_factory=GovernanceRule)
    fitness_score: float = 0.0
    meta_parameters: dict[str, Any] = field(default_factory=dict)

    def to_yaml(self) -> str:
        """Rule 011 Compliance: Human-readable YAML snapshot format."""
        data = {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "fitness_score": self.fitness_score,
            "allocation_rule": {
                "rule_type": self.allocation_rule.rule_type,
                "parameters": self.allocation_rule.parameters,
                "ast_expression": self.allocation_rule.ast_expression,
            },
            "payment_rule": {
                "rule_type": self.payment_rule.rule_type,
                "parameters": self.payment_rule.parameters,
            },
            "governance_rule": {
                "punish_reserve_cost": self.governance_rule.punish_reserve_cost,
                "voting_threshold": self.governance_rule.voting_threshold,
                "authority_penalty_multiplier": self.governance_rule.authority_penalty_multiplier,
                "institutional_reserve_req": self.governance_rule.institutional_reserve_req,
            },
            "meta_parameters": self.meta_parameters,
        }
        return yaml.dump(data, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "MechanismCandidate":
        """Deserialize from YAML snapshot."""
        data = yaml.safe_load(yaml_str)
        alloc_data = data.get("allocation_rule", {})
        pay_data = data.get("payment_rule", {})
        gov_data = data.get("governance_rule", {})

        return cls(
            candidate_id=data.get("candidate_id", "mu_0"),
            name=data.get("name", "Unnamed Mechanism"),
            allocation_rule=AllocationRule(
                rule_type=alloc_data.get("rule_type", "parametric"),
                parameters=alloc_data.get("parameters", {}),
                ast_expression=alloc_data.get("ast_expression"),
            ),
            payment_rule=PaymentRule(
                rule_type=pay_data.get("rule_type", "second_highest_price"),
                parameters=pay_data.get("parameters", {}),
            ),
            governance_rule=GovernanceRule(
                punish_reserve_cost=gov_data.get("punish_reserve_cost", 0.1),
                voting_threshold=gov_data.get("voting_threshold", 0.5),
                authority_penalty_multiplier=gov_data.get("authority_penalty_multiplier", 1.5),
                institutional_reserve_req=gov_data.get("institutional_reserve_req", 1.0),
            ),
            fitness_score=data.get("fitness_score", 0.0),
            meta_parameters=data.get("meta_parameters", {}),
        )


class MechanismSpace:
    """Defines and samples candidate mechanisms within search domain M."""

    def __init__(self, config_path: str = "schema/mechanism_search_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        try:
            with open(self.config_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception:
            return {}

    def sample_candidate(self, candidate_id: str = "mu_sample") -> MechanismCandidate:
        """Sample a candidate mechanism from the search space domain."""
        return MechanismCandidate(
            candidate_id=candidate_id,
            name=f"Sampled Mechanism {candidate_id}",
            allocation_rule=AllocationRule(
                rule_type="highest_bidder", parameters={"reserve_price": 0.5}
            ),
            payment_rule=PaymentRule(
                rule_type="second_highest_price", parameters={"fee_rate": 0.0}
            ),
            governance_rule=GovernanceRule(punish_reserve_cost=0.1),
        )
