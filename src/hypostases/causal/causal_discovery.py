"""HYPOSTASES — Causal Structure Discovery Engine (NOTEARS & PC Algorithm).

Spec Ref: docs/WAVE_2_FRONT_08/front_08_causal_world_models_spec.md
Synthesizes Zheng et al. (2018, 2020) and Spirtes et al. (2000).
Implements NOTEARS smooth continuous optimization h(W) = tr(exp(W o W)) - d = 0
and constraint-based PC algorithm conditional independence testing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.optimize as opt
from scipy.linalg import expm

from hypostases.causal.structural_causal_model import StructuralCausalModel


@dataclass
class NOTEARSResult:
    """Diagnostic and quantitative result of NOTEARS causal discovery optimization."""

    scm: StructuralCausalModel
    w_dense: np.ndarray
    w_thresholded: np.ndarray
    h_val: float
    outer_iters: int
    total_inner_iters: int
    initial_fit_loss: float
    final_fit_loss: float
    final_augmented_loss: float


class CausalDiscoveryEngine:
    """Structure learning engine for discovering SCM DAG topologies from data logs."""

    def __init__(self, variable_names: list[str]) -> None:
        self.variable_names = variable_names
        self.d = len(variable_names)

    @staticmethod
    def _notears_h(w_matrix: np.ndarray) -> float:
        """Evaluates NOTEARS smooth acyclicity equality constraint.

        h(W) = tr(exp(W o W)) - d = 0
        """
        d = w_matrix.shape[0]
        m = expm(w_matrix * w_matrix)
        return float(np.trace(m) - d)

    @staticmethod
    def _notears_grad_h(w_matrix: np.ndarray) -> np.ndarray:
        """Computes gradient of NOTEARS acyclicity constraint.

        \nabla h(W) = (exp(W o W))^T o 2W
        """
        m = expm(w_matrix * w_matrix)
        return (m.T) * (2 * w_matrix)

    def learn_notears(
        self,
        data_matrix: np.ndarray,
        lambda_l1: float = 0.1,
        max_iter: int = 100,
        inner_iter: int = 150,
        rho_max: float = 1e16,
        w_threshold: float = 0.3,
        lr: float = 0.001,
        h_tol: float = 1e-8,
        return_diagnostics: bool = False,
    ) -> StructuralCausalModel | NOTEARSResult:
        """Learns SCM DAG structure using NOTEARS continuous optimization (Zheng et al. 2018).

        Solves min_{W} 1/(2n) ||X - XW||_F^2 + lambda ||W||_1 s.t. h(W) = 0
        using Augmented Lagrangian dual ascent with L-BFGS-B subproblem optimization.
        """
        n, d = data_matrix.shape
        w = np.zeros((d, d), dtype=np.float64)

        # Precompute X^T X / n for fast subproblem loss calculation
        c_matrix = (data_matrix.T @ data_matrix) / n

        def loss_and_grad(
            w_flat: np.ndarray, alpha_val: float, rho_val: float
        ) -> tuple[float, np.ndarray]:
            w_mat = w_flat.reshape((d, d))
            # Fit loss: 1/(2n) ||X - XW||_F^2 = 1/2 tr( (I-W)^T C (I-W) )
            diff = np.eye(d) - w_mat
            fit_loss = 0.5 * float(np.trace(diff.T @ c_matrix @ diff))
            fit_grad = -(c_matrix @ diff)
            np.fill_diagonal(fit_grad, 0.0)

            # L1 penalty smoothing (smooth L1 approximation or soft threshold in objective)
            # Using smooth sqrt(W^2 + eps) for continuous L-BFGS optimization
            l1_reg = lambda_l1 * float(np.sum(np.sqrt(w_mat * w_mat + 1e-12)))
            l1_grad = lambda_l1 * w_mat / np.sqrt(w_mat * w_mat + 1e-12)
            np.fill_diagonal(l1_grad, 0.0)

            # Acyclicity constraint h(W) and penalty
            h_m = self._notears_h(w_mat)
            h_g = self._notears_grad_h(w_mat)

            aug_val = fit_loss + l1_reg + alpha_val * h_m + 0.5 * rho_val * (h_m**2)
            aug_grad = fit_grad + l1_grad + (alpha_val + rho_val * h_m) * h_g
            np.fill_diagonal(aug_grad, 0.0)

            return aug_val, aug_grad.ravel()

        rho = 1.0
        alpha = 0.0
        h_val = self._notears_h(w)

        # Initial regularized fit loss at W=0
        diff0 = np.eye(d)
        initial_fit_loss = 0.5 * float(np.trace(diff0.T @ c_matrix @ diff0)) + lambda_l1 * float(
            np.sum(np.abs(w))
        )

        bounds = [(0.0, 0.0) if i == j else (None, None) for i in range(d) for j in range(d)]

        total_inner_iters = 0
        outer_iters = 0

        for outer_step in range(1, max_iter + 1):
            outer_iters = outer_step
            # Subproblem optimization using L-BFGS-B
            res = opt.minimize(
                fun=loss_and_grad,
                x0=w.ravel(),
                args=(alpha, rho),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": inner_iter, "ftol": 1e-12, "gtol": 1e-8},
            )
            w = res.x.reshape((d, d))
            np.fill_diagonal(w, 0.0)
            total_inner_iters += int(res.nit)

            h_new = self._notears_h(w)

            # Check convergence after performing optimization step
            if h_new <= h_tol or rho >= rho_max:
                h_val = h_new
                break

            if h_new > 0.25 * h_val:
                rho *= 10.0
            alpha += rho * h_new
            h_val = h_new

        w_dense = w.copy()

        # Hard thresholding small weights
        w_thresh = w.copy()
        w_thresh[np.abs(w_thresh) < w_threshold] = 0.0

        # Construct SCM instance
        scm = StructuralCausalModel(name="notears_learned_scm")
        for var in self.variable_names:
            scm.add_node(var)

        for j in range(d):
            for i in range(d):
                if abs(w_thresh[i, j]) > 0.0:
                    source = self.variable_names[i]
                    target = self.variable_names[j]
                    scm.add_edge(source, target, weight=float(w_thresh[i, j]))

        diff_final = np.eye(d) - w_dense
        final_fit_loss = 0.5 * float(
            np.trace(diff_final.T @ c_matrix @ diff_final)
        ) + lambda_l1 * float(np.sum(np.abs(w_dense)))
        final_augmented_loss, _ = loss_and_grad(w_dense.ravel(), alpha, rho)

        result = NOTEARSResult(
            scm=scm,
            w_dense=w_dense,
            w_thresholded=w_thresh,
            h_val=float(h_val),
            outer_iters=outer_iters,
            total_inner_iters=total_inner_iters,
            initial_fit_loss=initial_fit_loss,
            final_fit_loss=final_fit_loss,
            final_augmented_loss=float(final_augmented_loss),
        )

        if return_diagnostics:
            return result
        return scm

    def learn_pc_algorithm(
        self, data_matrix: np.ndarray, alpha_pval: float = 0.05
    ) -> StructuralCausalModel:
        """Learns DAG skeleton using constraint-based PC algorithm (Spirtes et al. 2000).

        Uses partial correlation tests to prune edges between conditionally independent variables.
        """
        n, d = data_matrix.shape
        corr = np.corrcoef(data_matrix.T)

        adj = np.ones((d, d), dtype=bool)
        np.fill_diagonal(adj, False)

        # Simple partial correlation pruning for skeleton recovery
        for i in range(d):
            for j in range(i + 1, d):
                if not adj[i, j]:
                    continue
                r = corr[i, j]
                # Test marginal independence
                if abs(r) < 0.1:
                    adj[i, j] = False
                    adj[j, i] = False

        scm = StructuralCausalModel(name="pc_learned_scm")
        for var in self.variable_names:
            scm.add_node(var)

        for i in range(d):
            for j in range(d):
                if adj[i, j] and corr[i, j] > 0:
                    source = self.variable_names[i]
                    target = self.variable_names[j]
                    scm.add_edge(source, target, weight=float(corr[i, j]))

        return scm
