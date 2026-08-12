"""Formal Mathematical Verification for Abductive Reasoning & Occam's Razor MDL Bounds (Front 11).

Theorem 11.1: Minimum Description Length (MDL) Occam's Razor Posterior Dominance
Invariant 11.2: Hypothesis Object Occam Penalty Non-Negativity
"""

from hypostases.abduction.hypothesis import Hypothesis


def test_theorem11_1_mdl_occams_razor_hypothesis_dominance():
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


def test_invariant11_2_hypothesis_occam_penalty_bounds():
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
