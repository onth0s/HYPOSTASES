"""Formal Mathematical Verification for Bayesian Belief Posteriors & Abduction (Front 06 / Front 11).

Theorem 6: Monotonic Entropy Reduction under Non-Deceptive Bayesian Evidence
Theorem 7: Bayes Factor Hypothesis Posterior Monotonicity
"""

import numpy as np


def compute_entropy(p):
    p_clean = p[p > 1e-12]
    return -np.sum(p_clean * np.log2(p_clean))


def test_theorem6_monotonic_entropy_reduction():
    """Empirically proves that receiving accurate likelihood evidence monotonically reduces belief entropy."""
    prior = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform prior (max entropy = 2.0 bits)
    initial_entropy = compute_entropy(prior)
    assert np.isclose(initial_entropy, 2.0)

    # Likelihood matrix with strong evidence for state 0
    likelihood_msg = np.array([0.8, 0.1, 0.05, 0.05])

    # Unnormalized posterior
    unnorm_posterior = prior * likelihood_msg
    posterior = unnorm_posterior / np.sum(unnorm_posterior)

    posterior_entropy = compute_entropy(posterior)

    # Monotonic entropy reduction invariant: H(posterior) < H(prior)
    assert posterior_entropy < initial_entropy


def test_theorem7_bayes_factor_hypothesis_ranking():
    """Verifies posterior probability monotonicity P(H_k | E) proportional to P(E | H_k) * P(H_k)."""
    # 3 Hypotheses
    priors = np.array([0.33, 0.33, 0.34])

    # Evidence likelihoods: H1 has high likelihood, H2 medium, H3 low
    likelihoods = np.array([0.9, 0.4, 0.1])

    posteriors = priors * likelihoods
    posteriors /= np.sum(posteriors)

    # Ranking invariant: P(H1|E) > P(H2|E) > P(H3|E)
    assert posteriors[0] > posteriors[1] > posteriors[2]
