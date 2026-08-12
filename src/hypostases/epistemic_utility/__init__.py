"""HYPOSTASES Engine — Epistemic Utility & Information Gain Dynamics."""

from __future__ import annotations

from hypostases.epistemic_utility.utility import (
    compute_epistemic_utility,
    compute_expected_free_energy,
    compute_expected_information_gain,
    compute_kl_divergence_gaussian,
    compute_learning_progress,
    compute_multivariate_information_gain,
    compute_multivariate_shannon_entropy,
    compute_shannon_entropy,
    compute_variational_free_energy,
)

__all__ = [
    "compute_epistemic_utility",
    "compute_expected_free_energy",
    "compute_expected_information_gain",
    "compute_kl_divergence_gaussian",
    "compute_learning_progress",
    "compute_multivariate_information_gain",
    "compute_multivariate_shannon_entropy",
    "compute_shannon_entropy",
    "compute_variational_free_energy",
]
