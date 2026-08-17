"""Formal Mathematical Verification: β (θ_meta[9]) is meta-learned, never pinned.

Rule 012: Mandatory Formal Mathematical Implementation Verification.
Rule 015 / AGENTS.md 015: β is a LEARNED meta-parameter. Its staticity is an EMERGENT
property of zero reward-feature covariance — never a manual re-assignment inside the
training loop. Config values only seed the Gen-0 logit prior.

These tests drive the estimator's batch finalization deterministically and verify:
- nonzero reward-feature covariance moves the β-logit (β is learned, not pinned),
- the β(1-β)-inverse compensation exactly matches its closed-form prediction,
- zero covariance leaves the logit untouched (emergent staticity),
- β always remains a valid probability and the logit stays within ±BETA_LOGIT_MAX.
"""

from __future__ import annotations

import numpy as np
import pytest

from hypostases.plugins.domains.chess.chess_agent_adapter import ChessAgentAdapter
from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.plugins.domains.chess.chess_trainer import (
    BETA_COMPENSATION_CAP,
    BETA_LOGIT_GRADIENT_BOOST,
    BETA_LOGIT_MAX,
    ChessSelfPlayTrainer,
)

BETA_PRIOR = 0.05
# Cast through float32 so the reference matches what a float32 θ_meta array stores.
LOGIT_PRIOR = float(np.float32(np.log(BETA_PRIOR / (1.0 - BETA_PRIOR))))


def _make_agent() -> ChessAgentAdapter:
    """Agent with a fully-specified θ_meta: uniform tactical features, τ prior, β prior."""
    agent = ChessAgentAdapter(
        domain=ChessDomain(),
        beta_efe=BETA_PRIOR,
        temperature=0.8,
        theta_meta=np.ones(10, dtype=np.float32),
    )
    agent.theta_meta[8] = 0.8
    agent.theta_meta[9] = LOGIT_PRIOR
    return agent


def _feature_trace(feat9: float) -> np.ndarray:
    """A unit-norm-aligned feature trace; only the β-logit slot is non-zero."""
    trace = np.zeros(10, dtype=np.float32)
    trace[9] = feat9
    return trace


def test_formal_beta_logit_learns_with_positive_covariance() -> None:
    """Positive reward-feature covariance must strictly increase the β-logit (learned)."""
    trainer = ChessSelfPlayTrainer(learning_rate=0.05, beta_efe=BETA_PRIOR)
    agent = _make_agent()

    # Two games: win correlates with a HIGH epistemic feature, loss with a LOW one.
    trainer._grad_batches["k"] = (
        [_feature_trace(+1.0), _feature_trace(-1.0)],
        [+1.0, -1.0],
    )
    trainer._finalize_generation_gradient("k", agent)

    assert agent.theta_meta[9] > LOGIT_PRIOR
    assert agent.beta_efe > BETA_PRIOR
    assert agent.beta_efe <= 0.99
    assert trainer._grad_batches == {}


def test_formal_beta_compensation_closed_form() -> None:
    """The β(1-β)-inverse compensation matches its exact analytic prediction.

    grad[9] = η · Σ_i adv_i · feat9_i, then × min(CAP, 1/(β(1-β))) × BOOST, then clip.
    At β=0.05: 1/(0.05·0.95) ≈ 21.05 < CAP=25, so the compensation is NOT clipped here.
    """
    trainer = ChessSelfPlayTrainer(learning_rate=0.05, beta_efe=BETA_PRIOR)
    agent = _make_agent()

    # adv = [+1, -1] (centered, unit variance); grad[9] = η·(1·0.01 + (-1)·(-0.01)) = 0.001
    feat = 0.01
    trainer._grad_batches["k"] = (
        [_feature_trace(+feat), _feature_trace(-feat)],
        [+1.0, -1.0],
    )
    trainer._finalize_generation_gradient("k", agent)

    raw = trainer.learning_rate * 2.0 * feat
    comp = min(BETA_COMPENSATION_CAP, 1.0 / max(0.05, BETA_PRIOR * (1.0 - BETA_PRIOR)))
    expected = min(0.5, raw * comp * BETA_LOGIT_GRADIENT_BOOST)

    assert expected > 0.05  # the compensation materially amplifies the step
    assert pytest.approx(agent.theta_meta[9] - LOGIT_PRIOR, abs=1e-6) == expected
    assert np.abs(agent.theta_meta[9]) <= BETA_LOGIT_MAX


def test_formal_beta_staticity_from_zero_covariance() -> None:
    """Zero reward-feature covariance must leave the β-logit EXACTLY unchanged.

    This is the documented emergent-staticity property: β converges to a constant only
    when the reward signal carries no covariance with the epistemic feature — never via
    a manual pin. (A constant feature with equal rewards triggers the zero-variance
    early-return; a constant feature with differing rewards yields a zero gradient.)
    """
    trainer = ChessSelfPlayTrainer(learning_rate=0.05, beta_efe=BETA_PRIOR)

    # Constant feature across both games: covariance is exactly zero.
    agent = _make_agent()
    trainer._grad_batches["a"] = (
        [_feature_trace(+1.0), _feature_trace(+1.0)],
        [+1.0, -1.0],
    )
    trainer._finalize_generation_gradient("a", agent)
    assert agent.theta_meta[9] == pytest.approx(LOGIT_PRIOR, abs=1e-9)
    assert agent.beta_efe == pytest.approx(BETA_PRIOR, abs=1e-6)

    # All-equal rewards: zero reward variance triggers the early-return (no update).
    agent2 = _make_agent()
    trainer._grad_batches["b"] = (
        [_feature_trace(+1.0), _feature_trace(-1.0)],
        [+0.5, +0.5],
    )
    trainer._finalize_generation_gradient("b", agent2)
    assert agent2.theta_meta[9] == pytest.approx(LOGIT_PRIOR, abs=1e-9)


def test_formal_beta_logit_clamped_within_bounds() -> None:
    """A pathological signal must leave the logit clamped to ±BETA_LOGIT_MAX and β valid."""
    trainer = ChessSelfPlayTrainer(learning_rate=0.5, beta_efe=BETA_PRIOR)
    agent = _make_agent()

    # Massive repeated positive covariance: the clipped step (≤0.5) caps accumulation,
    # and the final logit clamp at ±BETA_LOGIT_MAX bounds the worst case.
    for _ in range(5):
        trainer._grad_batches["c"] = (
            [_feature_trace(+100.0), _feature_trace(-100.0)],
            [+1.0, -1.0],
        )
        trainer._finalize_generation_gradient("c", agent)

    assert np.abs(agent.theta_meta[9]) <= BETA_LOGIT_MAX
    assert 0.01 <= agent.beta_efe <= 0.99
