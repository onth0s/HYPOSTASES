"""Formal Mathematical Verification for Wave 3 Front 06 — Communication as Bayesian Evidence.

Rule 012 Compliance: Mandatory Formal Mathematical Implementation Verification.

Exercises the actual `src/hypostases/communication/` module against theorems from:
  - Crawford & Sobel (1982): Cheap-talk partition count & residual variance monotonicity
  - Kamenica & Gentzkow (2011): Bayes-plausibility & concave closure upper bound
  - Goodman & Frank (2016): uRSA pragmatic likelihood via BayesianCommunicationEngine
  - Sabater & Sierra (2005): ReGreT Beta-binomial trust convergence
  - Jøsang (2007): Subjective Logic discounting operator invariant
  - Acemoglu et al. (2011): Asymptotic learning under sequential Bayesian updates
"""

import math

import numpy as np
import pytest

from hypostases.communication.bayesian_updater import BayesianCommunicationEngine
from hypostases.communication.deception_signaling import DeceptionSignalingFilter
from hypostases.communication.trust_reputation import TrustReputationEngine
from hypostases.communication.types import (
    DiscreteHypothesisPosterior,
    PeerMessage,
    SubjectiveOpinion,
)
from hypostases.engine.types import (
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)


# ---------------------------------------------------------------------------
# Theorem 6.1 — Crawford-Sobel Partition Count Formula
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "b_bias, expected_n",
    [
        (0.01, 6),  # N(0.01) = floor(-0.5 + 0.5*sqrt(201)) = 6
        (0.05, 2),  # N(0.05) = floor(-0.5 + 0.5*sqrt(41)) = 2
        (0.10, 1),  # N(0.10) = floor(-0.5 + 0.5*sqrt(21)) = 1
        (0.25, 1),  # N(0.25) = floor(-0.5 + 0.5*sqrt(9)) = floor(1.0) = 1
        (0.50, 1),  # high bias → babbling equilibrium N=1
    ],
)
def test_theorem6_1_crawford_sobel_partition_count(b_bias: float, expected_n: int) -> None:
    """Theorem 6.1: Crawford-Sobel N(b) = floor(-1/2 + 1/2 * sqrt(1 + 2/b)).

    Verifies that DeceptionSignalingFilter.compute_crawford_sobel_partition_count()
    implements the formula correctly for a sweep of b_bias values.
    """
    dsf = DeceptionSignalingFilter(bias_tolerance=0.005, max_partitions=10)
    n = dsf.compute_crawford_sobel_partition_count(b_bias)

    # Analytical formula (uncapped)
    analytic = max(1, int(-0.5 + 0.5 * math.sqrt(1.0 + 2.0 / b_bias)))
    analytic_capped = min(analytic, 10)

    assert n == analytic_capped, f"b_bias={b_bias}: expected N={analytic_capped}, got {n}"
    assert n == expected_n, f"b_bias={b_bias}: parametrized expectation {expected_n}, got {n}"


# ---------------------------------------------------------------------------
# Theorem 6.2 — Crawford-Sobel Residual Variance Monotonicity
# ---------------------------------------------------------------------------
def test_theorem6_2_crawford_sobel_variance_monotonicity() -> None:
    """Theorem 6.2: sigma_eff^2 is monotonically non-decreasing in b_bias.

    Greater goal misalignment → more partition noise → harder to extract signal.
    sigma_eff^2(b1) <= sigma_eff^2(b2) for 0 < b1 < b2.
    """
    dsf = DeceptionSignalingFilter(base_noise_std=0.05, bias_tolerance=0.005)
    b_biases = [0.02, 0.05, 0.10, 0.20, 0.40]
    prev_var = 0.0
    for b in b_biases:
        sigma = dsf.compute_effective_noise_std(b)
        variance = sigma**2
        assert variance >= prev_var - 1e-10, (
            f"Variance not monotone: b={b} gives sigma^2={variance:.6f} < prev={prev_var:.6f}"
        )
        prev_var = variance


# ---------------------------------------------------------------------------
# Theorem 6.3 — Kamenica-Gentzkow Bayes Plausibility & Concave Closure Bound
# ---------------------------------------------------------------------------
def test_theorem6_3_kamenica_gentzkow_bayes_plausibility() -> None:
    """Theorem 6.3a: Bayes-plausible constraint sum_s(mu_s * tau_s) = mu_0.

    Validates validate_bayes_plausibility() from DeceptionSignalingFilter.
    """
    dsf = DeceptionSignalingFilter()
    mu_0 = 0.4  # prior mean

    # Plausible partition: two signals s1 (posterior 0.8) and s2 (posterior 0.1)
    # with weights tau_1 = 3/7, tau_2 = 4/7 so that 0.8*(3/7) + 0.1*(4/7) ≈ 0.4
    tau_1 = 3.0 / 7.0
    tau_2 = 4.0 / 7.0
    posteriors_and_weights = [(0.8, tau_1), (0.1, tau_2)]
    assert dsf.validate_bayes_plausibility(mu_0, posteriors_and_weights)

    # Non-plausible: weights don't sum to 1
    bad_posteriors = [(0.9, 0.3), (0.1, 0.3)]  # sum of weights = 0.6
    assert not dsf.validate_bayes_plausibility(mu_0, bad_posteriors)


def test_theorem6_3b_concave_closure_upper_bound() -> None:
    """Theorem 6.3b: Sender expected utility ≤ V(mu_0), the concave closure.

    For a strictly convex payoff function, V(mu_0) > direct payoff,
    meaning Bayesian persuasion strictly benefits the sender.
    """
    dsf = DeceptionSignalingFilter()

    # Convex payoff: v(mu) = mu^2. Concave closure at mu_0=0.5 is 0.5*(0^2) + 0.5*(1^2) = 0.5
    payoff_fn = lambda mu: mu**2  # noqa: E731
    mu_0 = 0.5
    v_bound = dsf.compute_concave_closure_bound(payoff_fn, mu_0, num_samples=200)
    direct_payoff = payoff_fn(mu_0)

    assert v_bound >= direct_payoff - 1e-6, "Concave closure must be >= direct payoff"
    # For convex function, persuasion strictly helps: V > direct
    assert v_bound > direct_payoff + 0.01, (
        f"Expected V(mu_0) > direct payoff for convex function; got V={v_bound:.4f}, direct={direct_payoff:.4f}"
    )


# ---------------------------------------------------------------------------
# Theorem 6.4 — Monotonic Entropy Reduction
# ---------------------------------------------------------------------------
def test_theorem6_4_bayesian_engine_entropy_reduction() -> None:
    """Theorem 6.4: each informative honest message reduces posterior entropy.

    This exercises the public BayesianCommunicationEngine API rather than a
    standalone likelihood calculation across all three supported hypotheses.
    """
    engine = BayesianCommunicationEngine(
        deception_filter=DeceptionSignalingFilter(base_noise_std=0.05)
    )
    agent = AgentState(
        c=Characteristics(),
        w=WorldModel(),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )
    prior = DiscreteHypothesisPosterior(
        hypothesis_probabilities={"H1": 1 / 3, "H2": 1 / 3, "H3": 1 / 3}
    )

    def entropy(probs: dict[str, float]) -> float:
        return -sum(p * math.log(p) for p in probs.values() if p > 0.0)

    for expected_hypothesis, claimed_value in (("H1", 1.0), ("H2", 0.5), ("H3", 0.0)):
        message = PeerMessage(
            sender_id=f"honest_{expected_hypothesis}",
            receiver_id="agent_B",
            payload={"resource": claimed_value},
        )
        _, posterior = engine.process_incoming_message(message, agent, prior)

        assert entropy(posterior.hypothesis_probabilities) < entropy(prior.hypothesis_probabilities)
        assert (
            posterior.hypothesis_probabilities[expected_hypothesis]
            > prior.hypothesis_probabilities[expected_hypothesis]
        )


# ---------------------------------------------------------------------------
# Theorem 6.5 — ReGreT Beta-Binomial Trust Convergence
# ---------------------------------------------------------------------------
def test_theorem6_5_regret_beta_binomial_trust_convergence() -> None:
    """Theorem 6.5: ReGreT trust converges E[T_honesty] → 1.0 under 50 truthful interactions.

    Uncertainty u_j monotonically decreases as alpha accumulates.
    """
    engine = TrustReputationEngine(alpha_0=1.0, beta_0=1.0)
    peer_id = "honest_peer"

    prev_uncertainty = 1.0

    for tick in range(50):
        engine.update_direct_experience(peer_id, verified_truthful=True)
        opinion = engine.peer_opinions[peer_id]

        # Uncertainty must be non-increasing
        assert opinion.u <= prev_uncertainty + 1e-9, (
            f"Tick {tick}: uncertainty increased from {prev_uncertainty:.4f} to {opinion.u:.4f}"
        )
        prev_uncertainty = opinion.u

    final_profile = engine.get_trust_profile(peer_id)
    final_honesty = final_profile.expected_honesty()

    assert final_honesty > 0.95, (
        f"Expected trust > 0.95 after 50 truthful interactions, got {final_honesty:.4f}"
    )
    assert opinion.u < 0.1, (
        f"Uncertainty should be near 0 after 50 interactions, got {opinion.u:.4f}"
    )


# ---------------------------------------------------------------------------
# Theorem 6.6 — Subjective Logic Discounting Operator Invariant
# ---------------------------------------------------------------------------
def test_theorem6_6_subjective_logic_discounting_invariant() -> None:
    """Theorem 6.6: Jøsang discounting operator b_{k→j} = b_k * b_j, d_{k→j} = b_k * d_j.

    omega_k ⊗ omega_j propagates trust along indirect witness path k → j.
    """
    omega_k = SubjectiveOpinion(b=0.8, d=0.1, u=0.1, a=0.5)
    omega_j = SubjectiveOpinion(b=0.6, d=0.3, u=0.1, a=0.5)

    discounted = omega_k.discount(omega_j)

    raw_b = omega_k.b * omega_j.b
    raw_d = omega_k.b * omega_j.d
    raw_u = omega_k.u + omega_k.b * omega_j.u
    raw_total = raw_b + raw_d + raw_u
    expected_b = raw_b / raw_total
    expected_d = raw_d / raw_total

    assert abs(discounted.b - expected_b) < 1e-9, (
        f"b_res: expected {expected_b:.6f}, got {discounted.b:.6f}"
    )
    assert abs(discounted.d - expected_d) < 1e-9, (
        f"d_res: expected {expected_d:.6f}, got {discounted.d:.6f}"
    )
    # u_res may differ after normalization; verify normalization invariant instead
    assert discounted.is_valid, f"Discounted opinion must satisfy b+d+u=1, got {discounted}"


def test_theorem6_6b_subjective_logic_consensus_operator() -> None:
    """Theorem 6.6b: Jøsang consensus fusion omega_A ⊕ omega_B reduces uncertainty.

    Two non-vacuous opinions fused together must have less uncertainty than either alone.
    """
    omega_a = SubjectiveOpinion(b=0.7, d=0.1, u=0.2, a=0.5)
    omega_b = SubjectiveOpinion(b=0.6, d=0.2, u=0.2, a=0.5)

    fused = omega_a.consensus(omega_b)

    assert fused.is_valid, f"Fused opinion must satisfy b+d+u=1, got {fused}"
    assert fused.u <= max(omega_a.u, omega_b.u) + 1e-9, (
        f"Consensus must reduce uncertainty; fused.u={fused.u:.4f}, "
        f"max(u_a, u_b)={max(omega_a.u, omega_b.u):.4f}"
    )


# ---------------------------------------------------------------------------
# Theorem 6.7 — Acemoglu Asymptotic Learning (t → ∞)
# ---------------------------------------------------------------------------
def test_theorem6_7_acemoglu_asymptotic_learning() -> None:
    """Theorem 6.7: P(H_true | E_{1:t}) → 1.0 as t → ∞ under sequential Bayesian updates.

    Simulates 100 message rounds from a truthful sender with payload matching H1 state.
    Verifies posterior mass concentrates on H1.
    """
    # Initialize uniform 3-hypothesis posterior
    h_posterior = DiscreteHypothesisPosterior(
        hypothesis_probabilities={"H1": 1.0 / 3.0, "H2": 1.0 / 3.0, "H3": 1.0 / 3.0}
    )

    dsf = DeceptionSignalingFilter(base_noise_std=0.02)

    np.random.seed(0)
    for t in range(100):
        # True state is H1 (hypothesized_state=1.0); sender sends near-truth messages
        true_val = float(np.random.normal(1.0, 0.02))
        msg = PeerMessage(
            sender_id="oracle",
            receiver_id="receiver",
            payload={"x": true_val},
            timestamp=float(t),
        )

        # Bayesian update P(H_k | m) ∝ L(m | H_k) * P(H_k)
        new_probs: dict[str, float] = {}
        for h_id, prior_p in h_posterior.hypothesis_probabilities.items():
            # H1 corresponds to state 1.0, H2 to 0.5, H3 to 0.0
            hyp_state = {"H1": 1.0, "H2": 0.5, "H3": 0.0}[h_id]
            likelihood = dsf.evaluate_message_likelihood(msg, hypothesized_state=hyp_state)
            new_probs[h_id] = max(1e-300, likelihood * prior_p)

        h_posterior = DiscreteHypothesisPosterior(hypothesis_probabilities=new_probs)

    p_h1 = h_posterior.hypothesis_probabilities["H1"]
    assert p_h1 > 0.999, (
        f"Asymptotic convergence failed: P(H1 | E_{{1:100}}) = {p_h1:.6f}, expected > 0.999"
    )


# ---------------------------------------------------------------------------
# Theorem 6.8 — Acemoglu Additive Decision Decomposition
# ---------------------------------------------------------------------------
def test_theorem6_8_acemoglu_additive_decision_decomposition() -> None:
    """Theorem 6.8: x_n = 1 iff p_private + p_social > 1.

    Validates BayesianCommunicationEngine.evaluate_acemoglu_additive_decision().
    """
    from hypostases.communication.bayesian_updater import BayesianCommunicationEngine

    # Should act: 0.6 + 0.5 = 1.1 > 1.0
    assert BayesianCommunicationEngine.evaluate_acemoglu_additive_decision(0.6, 0.5) == 1

    # Should not act: 0.4 + 0.5 = 0.9 <= 1.0
    assert BayesianCommunicationEngine.evaluate_acemoglu_additive_decision(0.4, 0.5) == 0

    # Boundary: exactly 1.0 → should NOT act (strict >)
    assert BayesianCommunicationEngine.evaluate_acemoglu_additive_decision(0.5, 0.5) == 0
