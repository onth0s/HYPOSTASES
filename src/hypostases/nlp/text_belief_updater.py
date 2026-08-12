"""Adversarial Threat Model, Multi-Criteria Sanitizer, & Belief Updater for Front 14.

Spec Ref: docs/State-Vectors-to-Natural-Language/state_vectors_to_natural_language_plan.md
Step 3: Multi-Criteria Promotion Acceptance Rule & Endogenous Goal Hierarchy g.u Evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hypostases.schemas.loader import load_nlp_decoder_config


class TextBeliefUpdater:
    """Sanitizes incoming natural language messages from peer agents and updates internal belief.

    Protects agent world model w and goals g against prompt injection and deceptive state manipulation.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_nlp_decoder_config()
        san_cfg = self.config.get("sanitizer_config", {})
        self.tau_causal: float = san_cfg.get("tau_causal", 0.25)
        self.tau_promotion: float = san_cfg.get("tau_promotion", 0.10)
        self.scm_audit_enabled: bool = san_cfg.get("scm_audit_enabled", True)

    def audit_format(self, text_message: str) -> bool:
        """Format audit: verifies syntactic structure and checks for prompt injection keywords."""
        if not text_message or not isinstance(text_message, str):
            return False
        # Check against basic injection attacks or empty strings
        injection_keywords = ["IGNORE PREVIOUS INSTRUCTIONS", "DROP TABLE", "MALICIOUS_OVERRIDE"]
        upper_text = text_message.upper()
        return all(kw not in upper_text for kw in injection_keywords)

    def audit_scm(self, w_current: np.ndarray, w_sandbox: np.ndarray) -> float:
        """Structural Causal Model (SCM) audit: computes divergence distance between w and w_sandbox."""
        w_curr = np.asarray(w_current, dtype=np.float64)
        w_sand = np.asarray(w_sandbox, dtype=np.float64)
        distance = float(np.linalg.norm(w_curr - w_sand))
        return distance

    def compute_endogenous_utility(self, state: dict[str, Any]) -> float:
        """Evaluates receiving agent's endogenous Goal Hierarchy g.u over state sigma."""
        g_vec = np.asarray(state.get("g", [0.8] * 4), dtype=np.float64)
        w_vec = np.asarray(state.get("w", [0.2] * 4), dtype=np.float64)
        c_vec = np.asarray(state.get("c", [0.5] * 4), dtype=np.float64)
        # Utility function g.u(sigma)
        u_val = float(
            np.dot(g_vec[: min(len(g_vec), len(w_vec))], w_vec[: min(len(g_vec), len(w_vec))])
        ) + 0.1 * float(np.mean(c_vec))
        return u_val

    def evaluate_promotion(
        self,
        current_state: dict[str, Any],
        sandbox_state: dict[str, Any],
        text_message: str,
        peer_trust: float = 1.0,
    ) -> bool:
        """Multi-Criteria Promotion Acceptance Rule:

        Promote(sigma_sandbox -> sigma) <==>
            Audit_format and
            (Audit_SCM(w, w_sandbox) <= tau_causal) and
            (Trust(peer) * Delta U_expected >= tau_promotion)
        """
        # 1. Format Audit
        if not self.audit_format(text_message):
            return False

        # 2. SCM Audit
        w_curr = np.asarray(current_state.get("w", [0.2] * 4), dtype=np.float64)
        w_sand = np.asarray(sandbox_state.get("w", [0.2] * 4), dtype=np.float64)
        scm_dist = self.audit_scm(w_curr, w_sand)
        if self.scm_audit_enabled and scm_dist > self.tau_causal:
            return False

        # 3. Trust-Discounted Expected Utility Gain
        u_curr = self.compute_endogenous_utility(current_state)
        u_sand = self.compute_endogenous_utility(sandbox_state)
        delta_u = u_sand - u_curr
        discounted_gain = peer_trust * delta_u

        return discounted_gain >= self.tau_promotion

    def update_belief(
        self,
        current_state: dict[str, Any],
        proposed_state: dict[str, Any],
        text_message: str,
        peer_trust: float = 1.0,
    ) -> tuple[dict[str, Any], bool]:
        """Sanitizes text message and updates belief state if promotion criteria are met."""
        promoted = self.evaluate_promotion(
            current_state, proposed_state, text_message, peer_trust=peer_trust
        )

        if promoted:
            # Update state with proposed state
            updated_state = dict(current_state)
            updated_state["w"] = list(proposed_state.get("w", current_state.get("w")))
            updated_state["g"] = list(proposed_state.get("g", current_state.get("g")))
            return updated_state, True

        return current_state, False
