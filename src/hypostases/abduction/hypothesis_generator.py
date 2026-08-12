"""HYPOSTASES Engine — Wave 3 Front 11 Multi-Modal Hypothesis Generator.

Spec Ref: docs/WAVE_3_FRONT_11/front_11_abductive_reasoning_hypothesis_objects_spec.md
Literature Ref: De Kleer & Williams (1987), Pearl (2000) Ch. 7/9, Tenenbaum et al. (2011).

Generates candidate hypothesis objects across Environment, Peer Intent, and Causal Graph modalities.
"""

from __future__ import annotations

import uuid
from typing import Any

from hypostases.abduction.hypothesis import Hypothesis
from hypostases.abduction.types import HypothesisCategory


class HypothesisGenerator:
    """Generates candidate hypothesis objects to account for detected environment and peer anomalies."""

    def __init__(
        self,
        enable_env: bool = True,
        enable_peer: bool = True,
        enable_causal: bool = True,
    ) -> None:
        self.enable_env = enable_env
        self.enable_peer = enable_peer
        self.enable_causal = enable_causal

    def generate_candidates(
        self,
        observed_val: float,
        predicted_mean: float,
        timestamp: int,
        context: dict[str, Any] | None = None,
    ) -> list[Hypothesis]:
        """Generates ensemble of candidate hypotheses explaining the anomaly (observed_val - predicted_mean)."""
        candidates: list[Hypothesis] = []
        residual = observed_val - predicted_mean
        ctx = context or {}

        # 1. Environment Anomaly Hypotheses (De Kleer & Williams 1987 + Tenenbaum 2011)
        if self.enable_env:
            # H_env_decay: Environmental degradation / decay rate shift
            decay_scale = float(observed_val / max(predicted_mean, 1e-6))
            candidates.append(
                Hypothesis(
                    identifier=f"H_env_decay_{uuid.uuid4().hex[:6]}",
                    description=f"Environment pool decay rate shifted (residual: {residual:.2f})",
                    category=HypothesisCategory.ENVIRONMENT,
                    assumptions={"type": "resource_decay", "observed_residual": residual},
                    prior=0.4,
                    likelihood=0.7,
                    complexity=1.2,  # 1 parameter shift
                    predictive_params={"scale": decay_scale, "shift": 0.0},
                )
            )
            # H_env_shock: External exogenous shock
            candidates.append(
                Hypothesis(
                    identifier=f"H_env_shock_{uuid.uuid4().hex[:6]}",
                    description=f"Exogenous environmental shock at step {timestamp}",
                    category=HypothesisCategory.ENVIRONMENT,
                    assumptions={"type": "exogenous_shock", "timestamp": timestamp},
                    prior=0.3,
                    likelihood=0.6,
                    complexity=1.5,
                    predictive_params={"scale": 1.0, "shift": residual},
                )
            )

        # 2. Peer Intent / Alignment Hypotheses (Front 06 uRSA / Goodman & Frank 2016)
        if self.enable_peer:
            target_peer = ctx.get("peer_id", "peer_agent")
            candidates.append(
                Hypothesis(
                    identifier=f"H_peer_intent_{uuid.uuid4().hex[:6]}",
                    description=f"Peer agent '{target_peer}' goal utility shift or withdrawal increase",
                    category=HypothesisCategory.PEER_INTENT,
                    assumptions={"type": "utility_shift", "peer_id": target_peer},
                    prior=0.35,
                    likelihood=0.65,
                    complexity=1.8,  # Peer parameter expansion
                    predictive_params={"scale": 0.8, "shift": residual * 0.5},
                )
            )

        # 3. Causal Graph Structural Mutation Hypotheses (Pearl 2000 Ch. 7 SCM Abduction)
        if self.enable_causal:
            candidates.append(
                Hypothesis(
                    identifier=f"H_causal_edge_{uuid.uuid4().hex[:6]}",
                    description="Structural causal graph edge mutation (exogenous variable U shift)",
                    category=HypothesisCategory.CAUSAL_STRUCTURE,
                    assumptions={"type": "scm_edge_mutation", "Pearl_PNS_bound": 0.8},
                    prior=0.25,
                    likelihood=0.8,
                    complexity=2.2,  # Higher structural complexity penalty (MacKay Occam factor)
                    predictive_params={
                        "scale": decay_scale if self.enable_env else 1.0,
                        "shift": residual * 0.8,
                    },
                )
            )

        # Normalize priors across candidate ensemble
        total_prior = sum(h.prior for h in candidates)
        if total_prior > 0:
            for h in candidates:
                h.prior /= total_prior

        return candidates
