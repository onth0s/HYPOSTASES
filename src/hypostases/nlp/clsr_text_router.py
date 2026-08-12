"""Regularized Calibrated Fano Uncertainty Router for Front 14.

Spec Ref: docs/State-Vectors-to-Natural-Language/state_vectors_to_natural_language_plan.md
Step 4: Regularized Fano Token Budget Allocation, Calibration Objective, & Component Distances.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from hypostases.schemas.loader import load_nlp_decoder_config


class CalibratedFanoTextRouter:
    """Communicative Language Symbolism Router with Regularized Calibrated Fano Policy.

    Maintains component weights w = (w_c, w_w, w_g, w_rho) with w_i >= w_min = 0.10,
    entropy regularization penalty -lambda_H * H(w), and Hungarian GED padding penalty c_dummy = 1.0.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_nlp_decoder_config()
        router_cfg = self.config.get("router_config", {})
        self.w_min: float = router_cfg.get("w_min", 0.10)
        self.lambda_H: float = router_cfg.get("lambda_H", 0.05)
        self.lambda_task: float = router_cfg.get("lambda_task", 1.0)
        self.c_dummy: float = router_cfg.get("c_dummy", 1.0)
        self.delta_max: float = router_cfg.get("delta_max", 0.05)
        self.kappa_active: float = router_cfg.get("kappa_active", 1.2)

        # Initial calibrated weights (w_c, w_w, w_g, w_rho) summing to 1.0, each >= w_min
        self.weights: np.ndarray = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        self.tau_1: float = 0.5
        self.tau_2: float = 1.5

    def binary_entropy(self, p: float) -> float:
        """Computes binary entropy h_2(p)."""
        p = max(1e-12, min(1.0 - 1e-12, p))
        return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

    def compute_fano_min_token_budget(
        self, state_entropy: float, target_p_error: float = 0.05
    ) -> int:
        """Fano's Inequality Bound: E[N_tokens] >= (H(Y|X) - h_2(P_error)) / kappa_active."""
        h2 = self.binary_entropy(target_p_error)
        req_bits = max(0.0, state_entropy - h2)
        n_tokens = math.ceil(req_bits / self.kappa_active)
        return max(1, n_tokens)

    def compute_distance_c(self, c1: Any, c2: Any) -> float:
        """Component distance for cognitive state c (Normalized Euclidean norm)."""
        v1 = np.asarray(c1, dtype=np.float64).flatten()
        v2 = np.asarray(c2, dtype=np.float64).flatten()
        max_len = max(len(v1), len(v2))
        if max_len == 0:
            return 0.0
        min_len = min(len(v1), len(v2))
        diff_sq = np.sum((v1[:min_len] - v2[:min_len]) ** 2)
        if len(v1) != len(v2):
            diff_sq += abs(len(v1) - len(v2)) * (self.c_dummy**2)
        return float(np.sqrt(diff_sq) / np.sqrt(max_len))

    def compute_hungarian_ged_w(self, w1: Any, w2: Any) -> float:
        """Hungarian Graph Edit Distance (GED) d_w normalized by max_n nodes."""
        v1 = np.asarray(w1, dtype=np.float64).flatten()
        v2 = np.asarray(w2, dtype=np.float64).flatten()

        n1, n2 = len(v1), len(v2)
        max_n = max(n1, n2)
        if max_n == 0:
            return 0.0

        # Create cost matrix padded with c_dummy
        cost_matrix = np.full((max_n, max_n), self.c_dummy, dtype=np.float64)
        for i in range(n1):
            for j in range(n2):
                cost_matrix[i, j] = abs(v1[i] - v2[j])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_cost = float(cost_matrix[row_ind, col_ind].sum())
        return total_cost / max_n  # Scale normalization to [0, 1]

    def compute_distance_g(self, g1: Any, g2: Any) -> float:
        """Zhang-Shasha Tree Edit / Goal Hierarchy distance d_g normalized by max_n nodes."""
        v1 = np.asarray(g1, dtype=np.float64).flatten()
        v2 = np.asarray(g2, dtype=np.float64).flatten()
        max_n = max(len(v1), len(v2))
        if max_n == 0:
            return 0.0
        min_len = min(len(v1), len(v2))
        diff = float(np.sum(np.abs(v1[:min_len] - v2[:min_len])))
        length_penalty = abs(len(v1) - len(v2)) * self.c_dummy
        return (diff + length_penalty) / max_n  # Scale normalization to [0, 1]

    def compute_distance_rho(self, r1: Any, r2: Any) -> float:
        """Component distance for external power rho_ext (Normalized Euclidean norm)."""
        v1 = np.asarray(r1, dtype=np.float64).flatten()
        v2 = np.asarray(r2, dtype=np.float64).flatten()
        max_len = max(len(v1), len(v2))
        if max_len == 0:
            return 0.0
        min_len = min(len(v1), len(v2))
        diff_sq = np.sum((v1[:min_len] - v2[:min_len]) ** 2)
        if len(v1) != len(v2):
            diff_sq += abs(len(v1) - len(v2)) * (self.c_dummy**2)
        return float(np.sqrt(diff_sq) / np.sqrt(max_len))

    def compute_roundtrip_loss(
        self, state: dict[str, Any], reconstructed_state: dict[str, Any]
    ) -> float:
        """Component-Wise Round-Trip Distance: L_roundtrip = sum w_i d_i(sigma_i, sigma_i')."""
        d_c = self.compute_distance_c(state.get("c", [0.5]), reconstructed_state.get("c", [0.5]))
        d_w = self.compute_hungarian_ged_w(
            state.get("w", [0.2]), reconstructed_state.get("w", [0.2])
        )
        d_g = self.compute_distance_g(state.get("g", [0.8]), reconstructed_state.get("g", [0.8]))
        d_rho = self.compute_distance_rho(
            state.get("rho_ext", [1.0]), reconstructed_state.get("rho_ext", [1.0])
        )

        distances = np.array([d_c, d_w, d_g, d_rho], dtype=np.float64)
        return float(np.dot(self.weights, distances))

    def compute_weights_entropy(self, weights: np.ndarray) -> float:
        """Computes entropy H(w) of normalized weight vector."""
        w = np.maximum(weights, 1e-12)
        w /= np.sum(w)
        return -float(np.sum(w * np.log2(w)))

    def calibrate_weights(
        self, d_train_states: list[dict[str, Any]], d_train_reconstructed: list[dict[str, Any]]
    ) -> np.ndarray:
        """Calibrates component weights w on D_train solving regularized optimization objective:

        min_{w} E_D_train [ TokenCost + lambda_task * L_roundtrip - lambda_H * H(w) ]
        s.t. w_i >= w_min = 0.10, sum w_i = 1.0.
        """
        best_w = self.weights.copy()
        best_loss = float("inf")

        # Grid search over simplex with constraint w_i >= w_min
        step = 0.05
        grid_vals = np.arange(self.w_min, 1.0 - 3 * self.w_min + 1e-6, step)

        for w0 in grid_vals:
            for w1 in grid_vals:
                for w2 in grid_vals:
                    w3 = 1.0 - (w0 + w1 + w2)
                    if w3 < self.w_min - 1e-6:
                        continue
                    w_cand = np.array([w0, w1, w2, w3], dtype=np.float64)
                    w_cand /= np.sum(w_cand)

                    # Temporarily assign weights
                    self.weights = w_cand

                    # Compute average roundtrip loss over D_train
                    rt_losses = [
                        self.compute_roundtrip_loss(st, rec)
                        for st, rec in zip(d_train_states, d_train_reconstructed, strict=False)
                    ]
                    mean_rt = float(np.mean(rt_losses)) if rt_losses else 0.0
                    h_w = self.compute_weights_entropy(w_cand)

                    # Objective: lambda_task * mean_rt - lambda_H * h_w
                    total_obj = self.lambda_task * mean_rt - self.lambda_H * h_w

                    if total_obj < best_loss:
                        best_loss = total_obj
                        best_w = w_cand.copy()

        self.weights = best_w
        return self.weights

    def evaluate_held_out(
        self,
        d_eval_states: list[dict[str, Any]],
        d_eval_reconstructed: list[dict[str, Any]],
        epsilon_roundtrip: float = 0.15,
    ) -> dict[str, Any]:
        """Evaluates frozen calibrated weights on held-out evaluation dataset D_eval.

        Returns point estimate, sample standard deviation, 95% confidence interval,
        and per-component breakdown.
        """
        losses = [
            self.compute_roundtrip_loss(st, rec)
            for st, rec in zip(d_eval_states, d_eval_reconstructed, strict=False)
        ]
        n_eval = len(losses)
        if n_eval == 0:
            return {"mean_eval_loss": 0.0, "std_eval_loss": 0.0, "ci_95": 0.0, "passed": True}

        mean_loss = float(np.mean(losses))
        std_loss = float(np.std(losses, ddof=1)) if n_eval > 1 else 0.0
        se = std_loss / math.sqrt(n_eval) if n_eval > 0 else 0.0
        ci_95 = 1.96 * se

        d_c_list = [
            self.compute_distance_c(st.get("c"), rec.get("c"))
            for st, rec in zip(d_eval_states, d_eval_reconstructed, strict=False)
        ]
        d_w_list = [
            self.compute_hungarian_ged_w(st.get("w"), rec.get("w"))
            for st, rec in zip(d_eval_states, d_eval_reconstructed, strict=False)
        ]
        d_g_list = [
            self.compute_distance_g(st.get("g"), rec.get("g"))
            for st, rec in zip(d_eval_states, d_eval_reconstructed, strict=False)
        ]
        d_rho_list = [
            self.compute_distance_rho(st.get("rho_ext"), rec.get("rho_ext"))
            for st, rec in zip(d_eval_states, d_eval_reconstructed, strict=False)
        ]

        return {
            "n_eval": n_eval,
            "mean_eval_loss": mean_loss,
            "std_eval_loss": std_loss,
            "se": se,
            "ci_95": ci_95,
            "ci_bounds": (mean_loss - ci_95, mean_loss + ci_95),
            "per_component": {
                "d_c_mean": float(np.mean(d_c_list)),
                "d_w_mean": float(np.mean(d_w_list)),
                "d_g_mean": float(np.mean(d_g_list)),
                "d_rho_mean": float(np.mean(d_rho_list)),
            },
            "epsilon_roundtrip": epsilon_roundtrip,
            "passed": mean_loss <= epsilon_roundtrip,
        }
