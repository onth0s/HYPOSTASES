"""Bayesian Optimal Experimental Design (BED) & Active Sensing Engine (Wave 4 Front 12).

Implements Lindley (1956) EIG Shannon entropy reduction, Foster (2020) ACE lower bounds,
King (2004) ASE cost-weighted design, and Friston Expected Free Energy (EFE) action selection (Rule 009).
"""

import math

import numpy as np

from hypostases.scientific_discovery.schemas import (
    ExperimentalDesign,
    Hypothesis,
    ScientificDiscoveryConfig,
)


class AdaptiveContrastiveEstimation:
    """Implements Foster et al. (2020) Adaptive Contrastive Estimation (ACE) lower bound for BOED.

    I_ACE(xi, phi, L) = E [ log( p(y | theta_0, xi) / ( (1/(L+1)) * sum_{l=0}^L p(y | theta_l, xi) ) ) ]
    """

    def __init__(self, num_samples_l: int = 64):
        self.num_samples_l = max(num_samples_l, 1)

    def compute_ace_bound(
        self,
        design: ExperimentalDesign,
        hypotheses: list[Hypothesis],
    ) -> float:
        """Compute the ACE variational lower bound on Expected Information Gain."""
        if not hypotheses:
            return 0.0

        # Sample theta_0 (primary hypothesis) according to posterior
        posteriors = np.array([h.posterior_probability for h in hypotheses])
        posteriors = np.maximum(posteriors, 1e-12)
        posteriors /= np.sum(posteriors)

        primary_idx = int(np.random.choice(len(hypotheses), p=posteriors))
        primary_h = hypotheses[primary_idx]

        # Generate simulated observation y under primary hypothesis
        y_sim = primary_h.parameters.get(design.target_variable, design.intervention_value)

        # Compute likelihood p(y_sim | theta_0)
        p_theta_0 = self._likelihood(y_sim, primary_h, design)

        # Draw L contrastive contrast samples theta_l
        contrast_indices = np.random.choice(
            len(hypotheses), size=self.num_samples_l, replace=True, p=posteriors
        )

        sum_contrast_p = p_theta_0  # l=0 term
        for c_idx in contrast_indices:
            c_h = hypotheses[c_idx]
            sum_contrast_p += self._likelihood(y_sim, c_h, design)

        avg_contrast_p = sum_contrast_p / (self.num_samples_l + 1.0)
        avg_contrast_p = max(avg_contrast_p, 1e-12)

        ace_score = math.log(max(p_theta_0, 1e-12) / avg_contrast_p)
        # Ensure non-negative lower bound
        return max(ace_score, 0.0)

    def _likelihood(
        self, y_obs: float, hypothesis: Hypothesis, design: ExperimentalDesign
    ) -> float:
        """Compute likelihood p(y_obs | hypothesis, design) under Gaussian noise assumption."""
        pred = hypothesis.parameters.get(design.target_variable, 0.0)
        var = 1.0
        diff = y_obs - pred
        return float(np.exp(-0.5 * (diff**2) / var) / math.sqrt(2 * math.pi * var))


class BayesianExperimentalDesignEngine:
    """Engine computing Expected Information Gain (EIG), ASE cost-normalized design, and Friston EFE selection."""

    def __init__(self, config: ScientificDiscoveryConfig | None = None):
        self.config = config or ScientificDiscoveryConfig()
        self.ace_estimator = AdaptiveContrastiveEstimation(
            num_samples_l=self.config.ace_num_contrastive_samples_l
        )

    def compute_shannon_entropy(self, hypotheses: list[Hypothesis]) -> float:
        """Compute Shannon entropy H(H) over hypothesis ensemble.

        H(H) = - sum_k P(H_k) * log2(P(H_k))
        """
        if not hypotheses:
            return 0.0

        entropy = 0.0
        for h in hypotheses:
            p = h.posterior_probability
            if p > 1e-12:
                entropy -= p * math.log2(p)

        return max(entropy, 0.0)

    def compute_expected_information_gain(
        self,
        design: ExperimentalDesign,
        hypotheses: list[Hypothesis],
    ) -> float:
        """Compute Expected Information Gain (EIG) Shannon entropy reduction.

        EIG(d) = H(H) - E_{P(E|d)} [ H(H | E, d) ]
        Guaranteed EIG(d) >= 0.
        """
        if not hypotheses:
            return 0.0

        current_entropy = self.compute_shannon_entropy(hypotheses)
        if current_entropy <= 1e-12:
            design.expected_information_gain = 0.0
            return 0.0

        # Monte Carlo estimation over potential experimental outcome evidence
        expected_conditional_entropy = 0.0
        mc_samples = min(self.config.eig_monte_carlo_samples, 100)

        for _ in range(mc_samples):
            # Sample candidate hypothesis
            posteriors = np.array([h.posterior_probability for h in hypotheses])
            posteriors = np.maximum(posteriors, 1e-12)
            posteriors /= np.sum(posteriors)

            sample_idx = int(np.random.choice(len(hypotheses), p=posteriors))
            sampled_h = hypotheses[sample_idx]

            # Simulated observation
            sim_obs = sampled_h.parameters.get(design.target_variable, 0.0) + np.random.normal(
                0, 0.1
            )

            # Compute posterior under sim_obs
            temp_posteriors = []
            for h in hypotheses:
                pred = h.parameters.get(design.target_variable, 0.0)
                lik = float(np.exp(-0.5 * ((sim_obs - pred) ** 2)))
                temp_posteriors.append(h.posterior_probability * lik)

            total_temp = sum(temp_posteriors)
            if total_temp > 0:
                temp_posteriors = [p / total_temp for p in temp_posteriors]
            else:
                temp_posteriors = [1.0 / len(hypotheses)] * len(hypotheses)

            cond_ent = 0.0
            for p in temp_posteriors:
                if p > 1e-12:
                    cond_ent -= p * math.log2(p)

            expected_conditional_entropy += cond_ent / mc_samples

        eig = max(current_entropy - expected_conditional_entropy, 0.0)
        design.expected_information_gain = float(eig)
        return design.expected_information_gain

    def select_optimal_experimental_design(
        self,
        candidate_designs: list[ExperimentalDesign],
        hypotheses: list[Hypothesis],
    ) -> ExperimentalDesign:
        """Select optimal experimental design d* prioritizing EIG, ASE cost weight, and Friston EFE (Rule 009).

        d* = argmax_{d} [ EIG(d) / (C(d) + gamma) ]
        """
        if not candidate_designs:
            raise ValueError("Candidate designs list cannot be empty.")

        gamma = self.config.ase_cost_weight_gamma

        for design in candidate_designs:
            # 1. Compute EIG Shannon entropy reduction
            eig = self.compute_expected_information_gain(design, hypotheses)

            # 2. Compute Foster (2020) ACE lower bound
            ace = self.ace_estimator.compute_ace_bound(design, hypotheses)
            design.ace_lower_bound = ace

            # 3. Friston Expected Free Energy mode selection (Rule 009)
            if self.config.efe_mode:
                # Under pure epistemic sensing, EFE utility equals cost-normalized EIG / ACE bound
                cost_norm = max(design.execution_cost + gamma, 1e-6)
                design.friston_efe_utility = (0.5 * eig + 0.5 * ace) / cost_norm
            else:
                design.friston_efe_utility = eig / max(design.execution_cost, 1e-6)

        # Select design maximizing Friston EFE utility
        best_design = max(candidate_designs, key=lambda d: d.friston_efe_utility)
        return best_design
