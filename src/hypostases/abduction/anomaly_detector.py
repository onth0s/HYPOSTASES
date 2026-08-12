"""HYPOSTASES Engine — Wave 3 Front 11 Anomaly & Surprise Detector.

Spec Ref: docs/WAVE_3_FRONT_11/front_11_abductive_reasoning_hypothesis_objects_spec.md
Literature Ref: Friston et al. (2017) Active Inference, Curiosity and Insight.

Computes Variational Free Energy F(q, o) and surprise metrics over observation trajectories.
Triggers abductive hypothesis generation when surprise exceeds threshold tau_surprise.
"""

from __future__ import annotations

import numpy as np


class SurpriseDetector:
    """Evaluates observation surprise against active world model predictions under Friston Free Energy bounds."""

    def __init__(self, surprise_threshold: float = 1.5) -> None:
        self.surprise_threshold = surprise_threshold
        self.last_surprise: float = 0.0

    def compute_free_energy(
        self,
        observed_val: float,
        predicted_mean: float,
        predicted_var: float = 2.0,
        prior_mean: float = 10.0,
        prior_var: float = 5.0,
    ) -> float:
        """Computes Variational Free Energy F(q, o) = Complexity - Accuracy.

        Complexity = D_KL(N(μ_pred, σ²_pred) || N(μ_prior, σ²_prior))
        Accuracy = - E_q[ln P(o | s)] = (o - μ_pred)² / (2 σ²_pred) + 0.5 ln(2 π σ²_pred)
        """
        eps = 1e-6
        var_p = max(predicted_var, eps)
        var_0 = max(prior_var, eps)

        # 1. KL Divergence (Complexity)
        kl_complexity = 0.5 * (
            np.log(var_0 / var_p) + (var_p + (predicted_mean - prior_mean) ** 2) / var_0 - 1.0
        )

        # 2. Log Accuracy error
        accuracy_err = 0.5 * ((observed_val - predicted_mean) ** 2) / var_p + 0.5 * np.log(
            2.0 * np.pi * var_p
        )

        free_energy = float(kl_complexity + accuracy_err)
        self.last_surprise = free_energy
        return free_energy

    def check_anomaly(
        self,
        observed_val: float,
        predicted_mean: float,
        predicted_var: float = 2.0,
        prior_mean: float = 10.0,
        prior_var: float = 5.0,
    ) -> tuple[bool, float]:
        """Checks whether observed value constitutes an anomaly exceeding tau_surprise."""
        fe = self.compute_free_energy(
            observed_val=observed_val,
            predicted_mean=predicted_mean,
            predicted_var=predicted_var,
            prior_mean=prior_mean,
            prior_var=prior_var,
        )
        is_anomaly = fe > self.surprise_threshold
        return is_anomaly, fe
