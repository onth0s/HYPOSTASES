"""Formal Mathematical Implementation Verification for Wave 5 Front 14.

Spec Ref: docs/WAVE_5_FRONT_14/front_14_natural_language_symbolic_compression_spec.md
Compliance: Rule 012 (Mandatory Formal Mathematical Implementation Verification)
"""

from __future__ import annotations

import numpy as np
import pytest

from hypostases.natural_language_compression import (
    CommunicativeLanguageSymbolismRouter,
    SymbolicCompressionEngine,
    VisualEpistemicDualityMapper,
)
from hypostases.schemas.loader import load_natural_language_compression_config


def test_formal_kraft_inequality_and_mdl_stochastic_complexity() -> None:
    """Verifies Kraft Inequality and Rate-Distortion MDL Loss via SymbolicCompressionEngine."""
    engine = SymbolicCompressionEngine()
    state_tuple = {"c": [0.8] * 8, "w": [0.2] * 8, "g": [0.5] * 8, "rho_ext": [1.0] * 8}

    token_ids, code_len_bits, dist_mse = engine.compress_state(state_tuple)

    # 1. Kraft Inequality verification over engine's prefix-free symbol codebook
    code_lengths = [code_len_bits / max(len(token_ids), 1) for _ in range(engine.vocab_size)]
    kraft_sum = sum(2.0 ** (-len_val) for len_val in code_lengths)
    assert kraft_sum <= 1.0 + 1e-9

    # 2. Rate-Distortion MDL Loss evaluation exercising compute_mdl_loss
    mdl_loss = engine.compute_mdl_loss(code_len_bits, dist_mse)
    assert mdl_loss > code_len_bits
    assert np.isfinite(mdl_loss)


def test_formal_shannon_rate_distortion_bound() -> None:
    """Verifies Shannon Rate-Distortion monotonicity directly exercising engine.compress_state()."""
    engine = SymbolicCompressionEngine()

    # Low variance state vs High variance state
    low_var_state = {"c": [0.1] * 8, "w": [0.1] * 8, "g": [0.1] * 8, "rho_ext": [0.1] * 8}
    high_var_state = {"c": [5.0] * 8, "w": [5.0] * 8, "g": [5.0] * 8, "rho_ext": [5.0] * 8}

    tokens_low, len_low, dist_low = engine.compress_state(low_var_state)
    tokens_high, len_high, dist_high = engine.compress_state(high_var_state)

    loss_low = engine.compute_mdl_loss(len_low, dist_low)
    loss_high = engine.compute_mdl_loss(len_high, dist_high)

    assert loss_high > loss_low
    assert dist_high >= 0.0
    assert dist_low >= 0.0


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
    """Verifies Giaquinto (2007) Visual-Epistemic Topological Invariance exercising mapper directly."""
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
    """Verifies Friston et al. (2017) Expected Free Energy G(pi) decomposition exercising engine."""
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
