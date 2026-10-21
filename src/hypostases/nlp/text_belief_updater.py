"""Adversarial Threat Model, Multi-Criteria Sanitizer, & Belief Updater for Front 14.

Spec Ref: docs/State-Vectors-to-Natural-Language/state_vectors_to_natural_language_plan.md
Step 3: Multi-Criteria Promotion Acceptance Rule & Endogenous Goal Hierarchy g.u Evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hypostases.engine.types import AgentState
from hypostases.schemas.loader import load_nlp_decoder_config


class TextBeliefUpdater:
    """Sanitizes incoming natural language messages from peer agents and updates internal belief.

    Protects agent world model w and goals g against prompt injection and deceptive state manipulation.
    Supports both native AgentState instances and state dictionaries.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_nlp_decoder_config()
        san_cfg = self.config.get("sanitizer_config", {})
        self.tau_causal: float = san_cfg.get("tau_causal", 0.25)
        self.tau_promotion: float = san_cfg.get("tau_promotion", 0.10)
        self.scm_audit_enabled: bool = san_cfg.get("scm_audit_enabled", True)

    def _extract_vectors(
        self, state: AgentState | dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts (c_vec, w_vec, g_vec) representation from AgentState or dict."""
        if isinstance(state, AgentState):
            w_vec = np.array(
                [state.w.mu, state.w.sigma2, state.w.replenish_rate_est, state.w.last_surprise],
                dtype=np.float64,
            )
            g_vec = state.g.u.astype(np.float64)
            c_vec = np.array(
                [state.c.reserve, state.c.resilience, state.c.skill, state.c.sociality],
                dtype=np.float64,
            )
            return c_vec, w_vec, g_vec
        g_vec = np.asarray(state.get("g", [0.8] * 4), dtype=np.float64)
        w_vec = np.asarray(state.get("w", [0.2] * 4), dtype=np.float64)
        c_vec = np.asarray(state.get("c", [0.5] * 4), dtype=np.float64)
        return c_vec, w_vec, g_vec

    def audit_format(self, text_message: str) -> bool:
        """Format audit: verifies syntactic structure and checks for prompt injection keywords."""
        if not text_message or not isinstance(text_message, str):
            return False
        injection_keywords = ["IGNORE PREVIOUS INSTRUCTIONS", "DROP TABLE", "MALICIOUS_OVERRIDE"]
        upper_text = text_message.upper()
        return all(kw not in upper_text for kw in injection_keywords)

    def audit_scm(self, w_current: np.ndarray, w_sandbox: np.ndarray) -> float:
        """Structural Causal Model (SCM) audit: computes divergence distance between w and w_sandbox."""
        w_curr = np.asarray(w_current, dtype=np.float64)
        w_sand = np.asarray(w_sandbox, dtype=np.float64)
        distance = float(np.linalg.norm(w_curr - w_sand))
        return distance

    def compute_endogenous_utility(self, state: AgentState | dict[str, Any]) -> float:
        """Evaluates receiving agent's endogenous Goal Hierarchy g.u over state sigma."""
        c_vec, w_vec, g_vec = self._extract_vectors(state)
        min_len = min(len(g_vec), len(w_vec))
        u_val = float(np.dot(g_vec[:min_len], w_vec[:min_len])) + 0.1 * float(np.mean(c_vec))
        return u_val

    def evaluate_promotion(
        self,
        current_state: AgentState | dict[str, Any],
        sandbox_state: AgentState | dict[str, Any],
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
        _, w_curr, _ = self._extract_vectors(current_state)
        _, w_sand, _ = self._extract_vectors(sandbox_state)
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
        current_state: Any,
        proposed_state: Any,
        text_message: str,
        peer_trust: float = 1.0,
    ) -> tuple[Any, bool]:
        """Sanitizes text message and updates belief state if promotion criteria are met."""
        promoted = self.evaluate_promotion(
            current_state, proposed_state, text_message, peer_trust=peer_trust
        )

        if promoted:
            if isinstance(current_state, AgentState) and isinstance(proposed_state, AgentState):
                new_agent = current_state.clone()
                new_agent.w = proposed_state.w.clone()
                new_agent.g = proposed_state.g.clone()
                return new_agent, True
            if isinstance(current_state, dict):
                updated_state = dict(current_state)
                w_val = proposed_state.get("w", current_state.get("w"))
                g_val = proposed_state.get("g", current_state.get("g"))
                if w_val is not None:
                    updated_state["w"] = list(w_val)
                if g_val is not None:
                    updated_state["g"] = list(g_val)
                return updated_state, True

        return current_state, False
