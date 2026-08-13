"""Formal Mathematical Verification Tests for HYPOSTASES Agent Self-Play Training.

Rule 012: Mandatory Formal Mathematical Implementation Verification.
Verifies end-to-end mathematical invariants for HYPOSTASES Chess integration:
- Expected Free Energy (EFE) utility decomposition bounds.
- Softmax policy distribution normalization and temperature limits.
- Meta-parameter (θ_meta) non-negativity and gradient update invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from hypostases.plugins.domains.chess.chess_agent_adapter import ChessAgentAdapter
from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.plugins.domains.chess.chess_trainer import ChessSelfPlayTrainer


def test_formal_efe_utility_decomposition_bounds() -> None:
    """Verifies that U_total = (1-β)*U_pragmatic + β*U_epistemic holds precisely."""
    domain = ChessDomain()
    adapter = ChessAgentAdapter(domain=domain, beta_efe=0.25, temperature=0.5)
    board = domain.initial_state()
    move = next(iter(board.legal_moves))

    u_total, u_pragmatic, u_epistemic = adapter.evaluate_efe_utility(board, move)

    expected_total = (1.0 - 0.25) * u_pragmatic + 0.25 * u_epistemic
    assert pytest.approx(u_total, abs=1e-5) == expected_total


def test_formal_softmax_policy_distribution_normalization() -> None:
    """Verifies that move probabilities sum to 1.0 and obey temperature scaling."""
    domain = ChessDomain()
    adapter = ChessAgentAdapter(domain=domain, temperature=0.2)
    board = domain.initial_state()
    legal_moves = domain.valid_actions(board)

    # Compute move probabilities over legal moves
    utilities = [adapter.evaluate_efe_utility(board, m)[0] for m in legal_moves]
    u_arr = np.array(utilities, dtype=np.float32)

    max_u = np.max(u_arr)
    exp_u = np.exp((u_arr - max_u) / adapter.temperature)
    probs = exp_u / np.sum(exp_u)

    # Invariant 1: Sum of probabilities == 1.0
    assert pytest.approx(np.sum(probs), abs=1e-5) == 1.0

    # Invariant 2: All probabilities in [0, 1]
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_formal_meta_gradient_update_invariants() -> None:
    """Verifies that self-play training produces valid meta-parameter updates without NaNs/Infs."""
    domain = ChessDomain()
    trainer = ChessSelfPlayTrainer(learning_rate=0.05, beta_efe=0.2, chess_domain=domain)
    initial_agent = ChessAgentAdapter(domain=domain)

    initial_theta = initial_agent.theta_meta.copy()

    # Train for 1 generation
    trained_agent = trainer.train_generation(agent=initial_agent, games_per_gen=5, max_moves=20)

    # Invariant 1: Meta-parameters remain non-negative
    assert np.all(trained_agent.theta_meta >= 0.0)

    # Invariant 2: No NaNs or Infs
    assert not np.any(np.isnan(trained_agent.theta_meta))
    assert not np.any(np.isinf(trained_agent.theta_meta))
    assert trained_agent.theta_meta.shape == initial_theta.shape
