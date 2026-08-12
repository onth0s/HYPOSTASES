"""HYPOSTASES Engine — Wave 3 Front 11 Abductive Engine.

Spec Ref: docs/WAVE_3_FRONT_11/front_11_abductive_reasoning_hypothesis_objects_spec.md
Literature Ref: MacKay (2003) Ch. 28, De Kleer & Williams (1987), Friston et al. (2017), Pearl (2000).

Primary manager for abductive inference, managing hypothesis candidate ensembles, anomaly detection,
MacKay Occam posterior evaluation, pruning, and memory consolidation.
"""

from __future__ import annotations

from typing import Any

from hypostases.abduction.anomaly_detector import SurpriseDetector
from hypostases.abduction.hypothesis import Hypothesis
from hypostases.abduction.hypothesis_generator import HypothesisGenerator
from hypostases.abduction.types import AbductiveMetrics, EvidenceRecord, HypothesisCategory


class AbductiveEngine:
    """Manages the computational hypothesis ensemble H = {H_1, ..., H_K} over state trajectories."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        surprise_threshold: float = 1.5,
        complexity_penalty_lambda: float = 0.2,
        pruning_threshold: float = 0.05,
        max_pool_size: int = 16,
        consolidation_threshold: float = 0.85,
    ) -> None:
        cfg = (config or {}).get("abductive_reasoning", {})
        self.surprise_threshold = cfg.get("surprise_threshold", surprise_threshold)
        self.lambda_mdl = cfg.get("complexity_penalty_lambda", complexity_penalty_lambda)
        self.pruning_threshold = cfg.get("pruning_threshold", pruning_threshold)
        self.max_pool_size = cfg.get("max_hypothesis_pool_size", max_pool_size)
        self.consolidation_threshold = cfg.get(
            "consolidation_confidence_threshold", consolidation_threshold
        )

        modalities = cfg.get("generation_modalities", {})
        enable_env = modalities.get("environment_anomalies", True)
        enable_peer = modalities.get("peer_intent_anomalies", True)
        enable_causal = modalities.get("causal_structural_mutations", True)

        self.detector = SurpriseDetector(surprise_threshold=self.surprise_threshold)
        self.generator = HypothesisGenerator(
            enable_env=enable_env,
            enable_peer=enable_peer,
            enable_causal=enable_causal,
        )

        self.hypotheses: dict[str, Hypothesis] = {}
        self.metrics = AbductiveMetrics()

    def process_observation(
        self,
        observed_val: float,
        predicted_mean: float,
        predicted_var: float = 2.0,
        timestamp: int = 0,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, list[Hypothesis]]:
        """Processes empirical observation through the abductive lifecycle.

        1. Evaluate surprise / Free Energy bound F(q, o)
        2. If anomaly, generate candidate hypotheses
        3. Update evidence & re-evaluate MacKay Occam posteriors
        4. Prune low-posterior hypotheses and cap pool size
        """
        is_anomaly, fe_value = self.detector.check_anomaly(
            observed_val=observed_val,
            predicted_mean=predicted_mean,
            predicted_var=predicted_var,
        )
        self.metrics.latest_surprise = fe_value

        newly_generated: list[Hypothesis] = []
        if is_anomaly:
            self.metrics.total_anomalies_detected += 1
            newly_generated = self.generator.generate_candidates(
                observed_val=observed_val,
                predicted_mean=predicted_mean,
                timestamp=timestamp,
                context=context,
            )
            for h in newly_generated:
                if len(self.hypotheses) < self.max_pool_size:
                    self.hypotheses[h.identifier] = h
                    self.metrics.total_hypotheses_generated += 1

        # Evaluate evidence & update posteriors for active pool
        residual = observed_val - predicted_mean
        for h in list(self.hypotheses.values()):
            pred_h = h.predict_value(predicted_mean)
            h_residual = float(abs(observed_val - pred_h))
            is_supporting = h_residual < abs(residual)

            record = EvidenceRecord(
                timestamp=timestamp,
                observation_val=observed_val,
                predicted_val=pred_h,
                residual=h_residual,
                is_supporting=is_supporting,
            )
            h.update_evidence(record, lambda_mdl=self.lambda_mdl)

        # Normalize posteriors across active hypotheses pool
        self._normalize_posteriors()

        # Prune hypotheses with posterior < pruning_threshold
        self._prune_hypotheses()

        # Update metrics
        self.metrics.active_pool_size = len(self.hypotheses)
        top = self.get_top_hypothesis()
        if top:
            self.metrics.top_hypothesis_id = top.identifier
            self.metrics.top_hypothesis_posterior = top.posterior

        return is_anomaly, newly_generated

    def update_from_peer_message(
        self, message_content: str, sender_id: str, trust_score: float = 0.8
    ) -> None:
        """Updates hypothesis posteriors given incoming peer signal (Front 06 integration)."""
        for h in self.hypotheses.values():
            if h.category == HypothesisCategory.PEER_INTENT:
                # Peer message confirming intent hypothesis increases likelihood
                h.likelihood = float(min(1.0, h.likelihood * (1.0 + 0.2 * trust_score)))
                h.compute_posterior(lambda_mdl=self.lambda_mdl)
        self._normalize_posteriors()

    def get_top_hypothesis(self) -> Hypothesis | None:
        """Returns highest posterior hypothesis from active pool."""
        if not self.hypotheses:
            return None
        return max(self.hypotheses.values(), key=lambda h: h.posterior)

    def consolidate_semantic_hypotheses(self) -> list[Hypothesis]:
        """Identifies high-confidence hypotheses exceeding consolidation threshold."""
        consolidated: list[Hypothesis] = []
        for h in self.hypotheses.values():
            if h.posterior >= self.consolidation_threshold:
                consolidated.append(h)
        return consolidated

    def _normalize_posteriors(self) -> None:
        """Normalizes posterior probabilities across active hypothesis pool."""
        if not self.hypotheses:
            return
        total_post = sum(h.posterior for h in self.hypotheses.values())
        if total_post > 0:
            for h in self.hypotheses.values():
                h.posterior = float(h.posterior / total_post)
                h.confidence = h.posterior

    def _prune_hypotheses(self) -> None:
        """Prunes hypotheses whose posterior falls below pruning_threshold."""
        prune_keys = [k for k, h in self.hypotheses.items() if h.posterior < self.pruning_threshold]
        for k in prune_keys:
            del self.hypotheses[k]
