"""Bayesian Posterior Updater & Ensemble Pruner (Wave 4 Front 12).

Implements exact Bayesian update dynamics over hypothesis space H_k,
posterior normalization, and threshold-based ensemble pruning.
"""

import numpy as np

from hypostases.scientific_discovery.schemas import Evidence, Hypothesis, ScientificDiscoveryConfig


class BayesianUpdater:
    """Performs exact Bayesian updating and hypothesis ensemble pruning.

    Adheres strictly to Rule 005 (zero artificial human cognitive biases/emotional hacks).
    """

    def __init__(self, config: ScientificDiscoveryConfig | None = None):
        self.config = config or ScientificDiscoveryConfig()

    def update_posteriors(
        self,
        hypotheses: list[Hypothesis],
        evidence: Evidence,
    ) -> list[Hypothesis]:
        """Compute exact Bayesian posterior probabilities P(H_k | E_t).

        P(H_k | E_t) = P(E_t | H_k, d*) * P(H_k) / sum_j [ P(E_t | H_j, d*) * P(H_j) ]
        """
        if not hypotheses:
            return []

        unnormalized_posteriors = []

        for h in hypotheses:
            likelihood = self.compute_likelihood(h, evidence)
            prior = h.posterior_probability
            unnormalized_posteriors.append(likelihood * prior)

        total_post = sum(unnormalized_posteriors)

        if total_post <= 1e-12:
            # Uniform fallback if all likelihoods collapse
            num_h = len(hypotheses)
            for h in hypotheses:
                h.posterior_probability = 1.0 / num_h
        else:
            for h, unnorm in zip(hypotheses, unnormalized_posteriors, strict=False):
                h.posterior_probability = float(unnorm / total_post)

        # Prune unlikely hypotheses falling below epsilon threshold
        return self.prune_hypotheses(hypotheses)

    def compute_likelihood(self, hypothesis: Hypothesis, evidence: Evidence) -> float:
        """Compute Gaussian empirical likelihood P(E_t | H_k, d*)."""
        if not evidence.observations:
            return 1.0

        total_sq_err = 0.0
        count = 0

        for var_name, obs_val in evidence.observations.items():
            pred_val = hypothesis.parameters.get(var_name, 0.0)
            total_sq_err += (obs_val - pred_val) ** 2
            count += 1

        if count == 0:
            return 1.0

        mse = total_sq_err / count
        # Gaussian likelihood formula with unit variance
        likelihood = float(np.exp(-0.5 * mse))
        return max(likelihood, 1e-12)

    def prune_hypotheses(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Prune hypotheses whose posterior probability is below pruning_threshold_epsilon."""
        eps = self.config.pruning_threshold_epsilon
        retained = [h for h in hypotheses if h.posterior_probability >= eps]

        if not retained:
            # Always retain at least the single best hypothesis
            best_h = max(hypotheses, key=lambda h: h.posterior_probability)
            retained = [best_h]

        # Renormalize posteriors over retained hypotheses
        total_post = sum(h.posterior_probability for h in retained)
        if total_post > 0:
            for h in retained:
                h.posterior_probability /= total_post

        return retained
