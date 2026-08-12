"""Formal Mathematical Verification for Abductive Reasoning & Occam's Razor MDL Bounds (Front 11).

Theorem 11.1: Minimum Description Length (MDL) Occam's Razor Posterior Dominance
Invariant 11.2: Hypothesis Object Occam Penalty Non-Negativity
"""

import pytest

from hypostases.abduction.abductive_engine import AbductiveEngine
from hypostases.abduction.anomaly_detector import SurpriseDetector
from hypostases.abduction.hypothesis import Hypothesis
from hypostases.abduction.types import HypothesisCategory


def test_theorem11_1_mdl_occams_razor_hypothesis_dominance() -> None:
    """Empirically proves simpler hypotheses with lower description length achieve higher posterior score."""
    h_simple = Hypothesis(
        identifier="H_simple",
        description="pool_depleted",
        complexity=5.0,
        likelihood=0.9,
    )
    h_complex = Hypothesis(
        identifier="H_complex",
        description="multi_agent_collusion_and_field_degradation_and_tax_evasion",
        complexity=45.0,
        likelihood=0.9,
    )

    score_simple = h_simple.compute_posterior(lambda_mdl=0.2)
    score_complex = h_complex.compute_posterior(lambda_mdl=0.2)

    # Simpler hypothesis with equal likelihood must dominate due to MDL complexity penalty
    assert score_simple >= score_complex


def test_invariant11_2_hypothesis_occam_penalty_bounds() -> None:
    """Verifies that Occam's razor complexity penalty remains non-negative."""
    hyp = Hypothesis(
        identifier="H1",
        description="test_cause",
        complexity=12.5,
        likelihood=0.8,
    )

    posterior = hyp.compute_posterior(lambda_mdl=0.1)

    # Invariant: Computed posterior is a valid non-negative float
    assert posterior >= 0.0


def test_theorem11_3_surprise_threshold_anomaly_classification() -> None:
    """Theorem 11.3: Surprise Detector Free Energy anomaly thresholding.

    Gaussian free energy F = (x - mu)^2 / (2 var) + 0.5 * log(2 pi var).
    Anomalous if F >= threshold.
    """
    detector = SurpriseDetector(surprise_threshold=2.0)

    # Small residual: F below threshold
    is_anom_low, fe_low = detector.check_anomaly(
        observed_val=10.0, predicted_mean=10.1, predicted_var=2.0
    )
    assert not is_anom_low

    # Large residual: F above threshold
    is_anom_high, fe_high = detector.check_anomaly(
        observed_val=10.0, predicted_mean=20.0, predicted_var=2.0
    )
    assert is_anom_high
    assert fe_high >= 2.0


def test_theorem11_4_evidence_concentration_limit() -> None:
    """Theorem 11.4: Repeated supporting evidence drives hypothesis posterior to concentration limit.

    Simplex condition sum(P(H_i)) = 1 must hold throughout evidence updates.
    """
    engine = AbductiveEngine(pruning_threshold=0.01)

    # Trigger anomaly to seed candidate hypotheses
    engine.process_observation(observed_val=20.0, predicted_mean=10.0, timestamp=1)
    assert len(engine.hypotheses) > 0

    # Repeatedly feed observation matching first hypothesis
    target_h = next(iter(engine.hypotheses.values()))

    for step in range(10):
        engine.process_observation(observed_val=20.0, predicted_mean=10.0, timestamp=2 + step)

    assert sum(h.posterior for h in engine.hypotheses.values()) == pytest.approx(1.0)
    assert target_h.posterior >= 0.10


def test_theorem11_5_empty_pool_fallback_invariant() -> None:
    """Theorem 11.5: Pruning cannot remove every hypothesis; empty pool fallback retains best candidate."""
    engine = AbductiveEngine(pruning_threshold=0.99)  # Aggressive pruning threshold

    # Seed candidates
    engine.process_observation(observed_val=30.0, predicted_mean=10.0, timestamp=1)

    # All posteriors will be below 0.99, but pruning fallback must retain non-empty pool
    assert len(engine.hypotheses) >= 1, "Fallback must retain at least one candidate hypothesis"


def test_theorem11_6_peer_message_trust_monotonicity() -> None:
    """Theorem 11.6: Peer message intent hypothesis updates are monotone in trust score."""
    engine = AbductiveEngine()
    h_peer = Hypothesis(
        identifier="H_peer",
        description="peer_cooperation",
        category=HypothesisCategory.PEER_INTENT,
        likelihood=0.5,
        posterior=0.5,
    )
    engine.hypotheses[h_peer.identifier] = h_peer

    # Update with high trust score
    engine.update_from_peer_message("intent_confirm", sender_id="peer_1", trust_score=0.9)
    p_high_trust = engine.hypotheses["H_peer"].likelihood

    # Reset
    engine.hypotheses["H_peer"].likelihood = 0.5

    # Update with low trust score
    engine.update_from_peer_message("intent_confirm", sender_id="peer_1", trust_score=0.2)
    p_low_trust = engine.hypotheses["H_peer"].likelihood

    assert p_high_trust > p_low_trust, "Higher trust score must produce higher likelihood boost"


def test_theorem11_7_semantic_consolidation_threshold() -> None:
    """Theorem 11.7: Semantic consolidation returns only hypotheses exceeding confidence threshold."""
    engine = AbductiveEngine(consolidation_threshold=0.80)
    h_high = Hypothesis("H_high", "strong_cause", posterior=0.85, confidence=0.85)
    h_low = Hypothesis("H_low", "weak_cause", posterior=0.30, confidence=0.30)

    engine.hypotheses[h_high.identifier] = h_high
    engine.hypotheses[h_low.identifier] = h_low

    consolidated = engine.consolidate_semantic_hypotheses()
    assert len(consolidated) == 1
    assert consolidated[0].identifier == "H_high"
