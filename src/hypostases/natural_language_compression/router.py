"""Communicative Language Symbolism Router (CLSR, Pei et al. ICML 2026) for Front 14."""

from __future__ import annotations

import math
from typing import Any


class CommunicativeLanguageSymbolismRouter:
    """Pei et al. (ICML 2026) Communicative Language Symbolism Router (CLSR).

    Optimizes active token allocation under Theorem 3.2 lower bound:
    E[|T|] >= max(0, I_req) / kappa_theta.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        router_cfg = config.get("router_config", {})
        self.kappa_theta: float = router_cfg.get("kappa_theta", 1.2)
        self.default_budget: int = router_cfg.get("token_budget_default", 128)
        self.default_target_acc: float = router_cfg.get("target_accuracy_default", 0.95)

    def compute_required_information(self, uncertainty_entropy: float, target_acc: float) -> float:
        """Computes I_req(x, delta) = H(Y|X=x) - h2(delta) - delta * log2(|Y_x| - 1)."""
        delta = 1.0 - target_acc
        delta = max(1e-5, min(0.499, delta))

        h2_delta = -delta * math.log2(delta) - (1.0 - delta) * math.log2(1.0 - delta)
        cardinality_term = delta * math.log2(max(2, 10 - 1))  # Default 10 classes
        i_req = uncertainty_entropy - h2_delta - cardinality_term
        return max(0.0, i_req)

    def compute_min_token_bound(self, uncertainty_entropy: float, target_acc: float) -> float:
        """Computes Theorem 3.2 lower bound for generated tokens."""
        i_req = self.compute_required_information(uncertainty_entropy, target_acc)
        return i_req / max(self.kappa_theta, 1e-6)

    def route_token_allocation(
        self, query_uncertainty: float, token_budget: int | None = None
    ) -> tuple[int, str]:
        """Routes task to Direct (T=1), Multi-LSF, or Extended Reasoning protocol."""
        budget = token_budget if token_budget is not None else self.default_budget
        min_tokens = self.compute_min_token_bound(query_uncertainty, self.default_target_acc)

        if min_tokens <= 1.0:
            return 1, "DIRECT_SYMBOLIC_EXECUTION"
        elif min_tokens <= budget / 2:
            allocated = math.ceil(min_tokens * 1.2)
            return min(allocated, budget), "MULTI_LSF_AGGREGATION"
        else:
            return budget, "EXTENDED_REASONING_PROTOCOL"
