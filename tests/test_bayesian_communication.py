"""Pytest Test Suite for Communication as Bayesian Evidence (Wave 3 Front 06)."""

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
from hypostases.schemas.loader import load_bayesian_communication_config
from hypostases.schemas.validators import assert_invariants, validate_subjective_opinion


def _make_test_agent() -> AgentState:
    return AgentState(
        c=Characteristics(skill=0.5, reserve=10.0, mood=0.2),
        w=WorldModel(sigma2=1.0),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(social_capital=1.0),
    )


def test_types_and_opinions() -> None:
    """Tests Subjective Logic opinion normalization, consensus, and discounting."""
    op1 = SubjectiveOpinion(b=0.6, d=0.2, u=0.2, a=0.5)
    assert op1.is_valid
    assert pytest.approx(op1.expected_probability()) == 0.7

    op2 = SubjectiveOpinion(b=0.4, d=0.4, u=0.2, a=0.5)
    assert op2.is_valid

    # Test Consensus Operator (op1 (+) op2)
    consensus_op = op1.consensus(op2)
    assert consensus_op.is_valid
    assert consensus_op.b > op1.b or consensus_op.b > op2.b

    # Test Discounting Operator (op1 (x) op2)
    discount_op = op1.discount(op2)
    assert discount_op.is_valid
    assert discount_op.b < op2.b  # Discounting reduces direct belief strength


def test_trust_profile_and_reputation_engine() -> None:
    """Tests Sabater-Sierra ReGreT Beta-binomial trust tracking and Acemoglu deduplication."""
    engine = TrustReputationEngine(alpha_0=2.0, beta_0=1.0)
    profile = engine.get_trust_profile("peer_alpha")

    assert profile.expected_honesty() == 2.0 / 3.0

    # Update with truthful interaction
    engine.update_direct_experience("peer_alpha", verified_truthful=True)
    assert profile.alpha_honesty == 3.0
    assert profile.expected_honesty() == 3.0 / 4.0

    # Test witness reputation aggregation
    witness_ops = [
        ("peer_alpha", SubjectiveOpinion(b=0.8, d=0.1, u=0.1)),
        ("peer_beta", SubjectiveOpinion(b=0.7, d=0.2, u=0.1)),
    ]
    aggregated = engine.aggregate_witness_reputation("target_peer", witness_ops)
    assert aggregated.is_valid

    # Test Acemoglu message deduplication
    msg = PeerMessage(
        sender_id="peer_alpha",
        receiver_id="agent_1",
        payload={"reserve": 150.0},
        provenance_chain=["peer_beta", "peer_alpha"],
    )

    assert not engine.is_duplicate_message(msg)
    assert engine.is_duplicate_message(msg)  # Second occurrence is identified as duplicate


def test_deception_and_cheap_talk_filter() -> None:
    """Tests Crawford-Sobel partition scaling, uRSA pragmatics, and Kamenica-Gentzkow bounds."""
    filter_engine = DeceptionSignalingFilter(base_noise_std=0.05, bias_tolerance=0.05)

    # Low bias -> High partition count N(b)
    n_low_bias = filter_engine.compute_crawford_sobel_partition_count(0.02)
    assert n_low_bias == filter_engine.max_partitions

    # High bias -> Low partition count N(b)
    n_high_bias = filter_engine.compute_crawford_sobel_partition_count(0.3)
    assert n_high_bias < n_low_bias

    # Noise std increases with bias
    std_low = filter_engine.compute_effective_noise_std(0.02)
    std_high = filter_engine.compute_effective_noise_std(0.3)
    assert std_high > std_low

    # Kamenica-Gentzkow Bayes Plausibility Validation
    prior_mu = 0.3
    valid_posteriors = [(0.0, 0.4), (0.5, 0.6)]  # 0.0 * 0.4 + 0.5 * 0.6 = 0.3
    assert DeceptionSignalingFilter.validate_bayes_plausibility(prior_mu, valid_posteriors)

    invalid_posteriors = [(0.1, 0.5), (0.8, 0.5)]
    assert not DeceptionSignalingFilter.validate_bayes_plausibility(prior_mu, invalid_posteriors)

    # Concave closure bound
    def linear_payoff(mu: float) -> float:
        return 2.0 * mu

    bound = DeceptionSignalingFilter.compute_concave_closure_bound(linear_payoff, 0.3)
    assert pytest.approx(bound) == 0.6


def test_bayesian_communication_engine_dual_updates() -> None:
    """Tests dual Bayesian posterior updating over continuous states & discrete hypothesis spaces."""
    agent = _make_test_agent()
    agent.w.peer_beliefs["reserve"] = 0.0

    comm_engine = BayesianCommunicationEngine()

    msg = PeerMessage(
        sender_id="peer_alpha",
        receiver_id="agent_self",
        payload={"reserve": 1.0},
        timestamp=1.0,
        declared_variance=0.05,
    )

    hyp_posterior = DiscreteHypothesisPosterior(
        hypothesis_probabilities={"H1": 0.33, "H2": 0.33, "H3": 0.34}
    )

    belief_state, updated_hyp = comm_engine.process_incoming_message(
        message=msg,
        agent=agent,
        hypothesis_posterior=hyp_posterior,
        goal_misalignment=0.02,
    )

    # Verify continuous state posterior update
    assert "reserve" in belief_state.state_means
    assert agent.w.peer_beliefs["reserve"] > 0.0

    # Verify discrete hypothesis space posterior update P(H_k | m)
    assert pytest.approx(sum(updated_hyp.hypothesis_probabilities.values())) == 1.0
    assert (
        updated_hyp.hypothesis_probabilities["H1"] != hyp_posterior.hypothesis_probabilities["H1"]
    )

    # Verify Acemoglu additive decision decomposition
    dec = BayesianCommunicationEngine.evaluate_acemoglu_additive_decision(0.6, 0.5)
    assert dec == 1

    dec_low = BayesianCommunicationEngine.evaluate_acemoglu_additive_decision(0.3, 0.4)
    assert dec_low == 0


def test_yaml_config_loading_and_invariants() -> None:
    """Tests data-driven YAML configuration loading and invariant validation."""
    cfg = load_bayesian_communication_config()
    assert "trust_defaults" in cfg
    assert "cheap_talk_config" in cfg
    assert "deduplication_config" in cfg

    # Validate opinion invariant checker
    assert not validate_subjective_opinion(0.6, 0.2, 0.2)
    violations = validate_subjective_opinion(0.6, 0.2, 0.5)
    assert len(violations) > 0

    # Validate AgentState hard invariants
    agent = _make_test_agent()
    assert_invariants(agent)
