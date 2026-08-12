"""Hypothesis Manager for Scientific Discovery Loop (Wave 4 Front 12).

Manages hypothesis pool H_k, abductive triggers, Occam penalties (MDL),
and Gottweis (2025) Elo evolutionary debate tournaments for ranking explanations.
"""

import math

import numpy as np

from hypostases.scientific_discovery.schemas import Hypothesis, ScientificDiscoveryConfig


class HypothesisManager:
    """Manages candidate hypothesis generation, priors, and evolutionary Elo tournaments.

    Adheres strictly to Rule 005 (zero artificial human cognitive biases or emotional hacks).
    """

    def __init__(self, config: ScientificDiscoveryConfig | None = None):
        self.config = config or ScientificDiscoveryConfig()
        self.hypotheses: list[Hypothesis] = []

    def check_anomaly_trigger(
        self,
        predicted_dist: np.ndarray,
        observed_dist: np.ndarray,
    ) -> tuple[bool, float]:
        """Check whether empirical observation anomaly exceeds KL divergence threshold eta.

        A_t = D_KL(P_obs || P_pred)
        """
        eps = 1e-12
        p = np.clip(observed_dist, eps, 1.0)
        q = np.clip(predicted_dist, eps, 1.0)
        p = p / np.sum(p)
        q = q / np.sum(q)

        kl_div = float(np.sum(p * np.log(p / q)))
        is_anomaly = kl_div > self.config.anomaly_threshold_eta
        return is_anomaly, kl_div

    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Add hypothesis to pool and prune if capacity max_hypotheses_k is exceeded."""
        self.hypotheses.append(hypothesis)
        if len(self.hypotheses) > self.config.max_hypotheses_k:
            self._prune_excess_hypotheses()

    def update_priors_via_mdl(self, num_nodes: int = 4) -> None:
        """Compute prior probabilities P_0(H_k) using Minimum Description Length (MDL) Occam penalties.

        P_0(H_k) proportional to exp(-beta_MDL * C(H_k))
        """
        if not self.hypotheses:
            return

        beta = self.config.mdl_complexity_penalty_beta
        unnormalized_priors = []

        for h in self.hypotheses:
            mdl = h.compute_mdl(num_nodes=num_nodes)
            weight = math.exp(-beta * mdl)
            unnormalized_priors.append(weight)

        total_weight = sum(unnormalized_priors)
        if total_weight <= 0:
            total_weight = 1.0

        for h, weight in zip(self.hypotheses, unnormalized_priors, strict=False):
            h.prior_probability = weight / total_weight
            h.posterior_probability = h.prior_probability

    def run_elo_tournament(self, evidence_batch: list[dict]) -> list[Hypothesis]:
        """Run Gottweis (2025) Elo-based evolutionary debate tournament over candidate hypotheses.

        Evaluates pairwise likelihood performance of hypotheses against evidence batch.
        Updates Elo ratings: E_A = 1 / (1 + 10^((R_B - R_A)/400)), R_A' = R_A + K * (S_A - E_A).
        """
        if len(self.hypotheses) < 2:
            return self.hypotheses

        k_factor = self.config.elo_k_factor

        for i in range(len(self.hypotheses)):
            for j in range(i + 1, len(self.hypotheses)):
                h_a = self.hypotheses[i]
                h_b = self.hypotheses[j]

                # Evaluate empirical likelihood scores for H_a and H_b
                score_a = self._evaluate_hypothesis_score(h_a, evidence_batch)
                score_b = self._evaluate_hypothesis_score(h_b, evidence_batch)

                if score_a > score_b:
                    s_a, s_b = 1.0, 0.0
                    h_a.supporting_evidence_count += 1
                    h_b.contradicting_evidence_count += 1
                elif score_b > score_a:
                    s_a, s_b = 0.0, 1.0
                    h_b.supporting_evidence_count += 1
                    h_a.contradicting_evidence_count += 1
                else:
                    s_a, s_b = 0.5, 0.5

                # Elo expected outcome calculation
                e_a = 1.0 / (1.0 + 10.0 ** ((h_b.elo_rating - h_a.elo_rating) / 400.0))
                e_b = 1.0 / (1.0 + 10.0 ** ((h_a.elo_rating - h_b.elo_rating) / 400.0))

                # Update Elo ratings
                h_a.elo_rating += k_factor * (s_a - e_a)
                h_b.elo_rating += k_factor * (s_b - e_b)

        # Sort hypotheses by Elo rating descending
        self.hypotheses.sort(key=lambda h: h.elo_rating, reverse=True)
        return self.hypotheses

    def _evaluate_hypothesis_score(
        self, hypothesis: Hypothesis, evidence_batch: list[dict]
    ) -> float:
        """Compute empirical likelihood score for a single hypothesis against evidence."""
        if not evidence_batch:
            return 0.0

        total_err = 0.0
        for ev in evidence_batch:
            for var, val in ev.items():
                if var in hypothesis.parameters:
                    pred_val = hypothesis.parameters[var]
                    total_err += (pred_val - val) ** 2

        mse = total_err / max(len(evidence_batch), 1)
        # Higher score for lower mean squared error
        return float(np.exp(-mse))

    def _prune_excess_hypotheses(self) -> None:
        """Prune worst hypothesis based on posterior probability and Elo rating."""
        if not self.hypotheses:
            return

        # Sort by posterior probability ascending
        self.hypotheses.sort(key=lambda h: (h.posterior_probability, h.elo_rating))

        # Evict lowest probability item
        self.hypotheses.pop(0)

        # Normalize remaining posteriors
        total_post = sum(h.posterior_probability for h in self.hypotheses)
        if total_post > 0:
            for h in self.hypotheses:
                h.posterior_probability /= total_post
