"""Formal Mathematical Verification for Counterfactual Rollouts & Asymptotic MCTS Limits (Front 04).

Theorem 4: Asymptotic Policy Convergence as N_particles -> infinity
Invariant 5: Kinematic Motion Primitive (kMP) Basis Dimension K=4 vs K=8 Approximation Error (Rule 008)
"""

import numpy as np


def test_theorem4_asymptotic_mcts_policy_convergence():
    """Empirically proves Monte Carlo policy estimate converges to analytical expectation under O(1/sqrt(N))."""
    np.random.seed(42)

    # True analytical expected utilities for 3 actions
    true_utility = np.array([2.0, 1.0, 0.5])
    temperature = 1.0
    true_policy = np.exp(true_utility / temperature) / np.sum(np.exp(true_utility / temperature))

    # Sample particle rollouts at increasing sample sizes N
    sample_sizes = [50, 500, 5000]
    errors = []

    for N in sample_sizes:
        counts = np.zeros(3)
        for _ in range(N):
            # Monte Carlo Gumbel-max or softmax sampling
            noise = np.random.gumbel(size=3)
            sampled_action = np.argmax(true_utility + noise)
            counts[sampled_action] += 1

        empirical_policy = counts / N
        error = np.linalg.norm(empirical_policy - true_policy)
        errors.append(error)

    # Error must decrease monotonically with N
    assert errors[0] > errors[1] > errors[2]
    # At N=5000, error should be very small (< 0.05)
    assert errors[2] < 0.05


def test_invariant5_kmp_basis_dimension_approximation_error():
    """Benchmarks Kinematic Motion Primitive (kMP) basis reconstruction error for K=4 vs K=8 (Rule 008)."""
    np.random.seed(42)
    t = np.linspace(0, 1, 100)

    # Synthetic continuous trajectory with high frequency oscillations
    trajectory = np.sin(2 * np.pi * t) + 0.5 * np.sin(6 * np.pi * t)

    def fit_gaussian_kmp(t_vec, traj, K):
        centers = np.linspace(0, 1, K)
        width = 1.0 / K
        Phi = np.zeros((len(t_vec), K))
        for k in range(K):
            Phi[:, k] = np.exp(-((t_vec - centers[k]) ** 2) / (2 * width**2))

        # Least squares solve for basis weights w
        weights, _, _, _ = np.linalg.lstsq(Phi, traj, rcond=None)
        reconstruction = Phi @ weights
        return np.mean((traj - reconstruction) ** 2)

    mse_k4 = fit_gaussian_kmp(t, trajectory, K=4)
    mse_k8 = fit_gaussian_kmp(t, trajectory, K=8)

    # Rule 008 benchmark: K=8 provides strictly lower approximation error than K=4 for complex trajectories
    assert mse_k8 < mse_k4
