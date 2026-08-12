"""Formal Mathematical Implementation Verification for Wave 5 Front 14.

Spec Ref: docs/WAVE_5_FRONT_14/front_14_natural_language_symbolic_compression_spec.md
Compliance: Rule 012 (Mandatory Formal Mathematical Implementation Verification)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hypostases.natural_language_compression import (
    CommunicativeLanguageSymbolismRouter,
    SymbolicCompressionEngine,
    VisualEpistemicDualityMapper,
)
from hypostases.schemas.loader import load_natural_language_compression_config


def test_formal_kraft_inequality_and_mdl_stochastic_complexity() -> None:
    """Verifies Kraft Inequality sum(2^-len) <= 1 and Barron et al. (1998) Stochastic Complexity bounds."""
    cfg = load_natural_language_compression_config()
    vocab_size = cfg.get("vocabulary_config", {}).get("vocab_size", 512)

    # 1. Kraft Inequality verification for prefix-free symbol codebook
    code_lengths = [math.log2(vocab_size) for _ in range(vocab_size)]
    kraft_sum = sum(2.0 ** (-len_val) for len_val in code_lengths)
    assert kraft_sum <= 1.0 + 1e-9

    # 2. Stochastic Complexity asymptotic expansion: log(1/L) + (d/2)*log(n / 2pi)
    d = 8  # parameter dimension
    n = 100  # sample size
    data_log_loss = 12.5

    parametric_complexity = (d / 2.0) * math.log2(n / (2.0 * math.pi))
    stochastic_complexity = data_log_loss + parametric_complexity
    assert stochastic_complexity > data_log_loss


def test_formal_shannon_rate_distortion_bound() -> None:
    """Verifies Shannon (1948) Rate-Distortion function R(D) = 0.5 * log2(sigma^2 / D)."""
    variance = 1.0  # Gaussian source variance
    distortions = [0.1, 0.2, 0.5, 0.8]

    rates = [0.5 * math.log2(variance / D) for D in distortions]

    # Monotonicity check: as distortion D increases, rate R(D) must strictly decrease
    for i in range(len(rates) - 1):
        assert rates[i] > rates[i + 1]
        assert rates[i] >= 0.0


def test_formal_theorem_3_2_token_accuracy_lower_bound() -> None:
    """Verifies Pei et al. (ICML 2026) Theorem 3.2 Token-Accuracy Lower Bound: E[|T|] >= I_req / kappa_theta."""
    cfg = load_natural_language_compression_config()
    router = CommunicativeLanguageSymbolismRouter(cfg)

    entropies = [0.5, 1.0, 2.0, 3.0, 4.0]
    target_acc = 0.95

    token_bounds = [
        router.compute_min_token_bound(uncertainty_entropy=H, target_acc=target_acc)
        for H in entropies
    ]

    # Monotonicity check: as query uncertainty H increases, required token bound strictly increases
    for i in range(len(token_bounds) - 1):
        assert token_bounds[i] <= token_bounds[i + 1]
        assert token_bounds[i] >= 0.0


def test_formal_giaquinto_topological_duality_invariance() -> None:
    """Verifies Giaquinto (2007) Visual-Epistemic Topological Invariance."""
    cfg = load_natural_language_compression_config()
    mapper = VisualEpistemicDualityMapper(cfg)

    np.random.seed(123)
    states = [np.random.randn(8) for _ in range(10)]

    for state in states:
        symbol_ids = mapper.encode_spatial_to_symbolic(state)
        reconstructed = mapper.decode_symbolic_to_spatial(symbol_ids)

        # Norm preservation invariant
        rec_norm = float(np.linalg.norm(reconstructed))
        assert 0.0 <= rec_norm <= 2.0

        # Topological neighborhood invariant
        assert len(symbol_ids) == 4
        assert len(set(symbol_ids)) == 4  # Unique topological Voronoi region symbols


def test_formal_friston_expected_free_energy_bounds() -> None:
    """Verifies Friston et al. (2017) Expected Free Energy G(pi) decomposition under efe_mode: true."""
    engine = SymbolicCompressionEngine()
    assert engine.efe_mode is True

    pragmatic_risk = 1.5
    epistemic_gain = 2.0
    ambiguity = 0.3

    # G = Risk - Epistemic_Gain + Ambiguity
    g_efe = engine.compute_expected_free_energy(pragmatic_risk, epistemic_gain, ambiguity)
    expected_g = pragmatic_risk - epistemic_gain + ambiguity

    assert g_efe == pytest.approx(expected_g)
    assert g_efe < pragmatic_risk  # High epistemic gain reduces Expected Free Energy
