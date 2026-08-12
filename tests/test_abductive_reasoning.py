"""Pytest suite for Wave 3 Front 11: Abductive Reasoning & Hypothesis Objects.

Spec Ref: docs/WAVE_3_FRONT_11/front_11_abductive_reasoning_hypothesis_objects_spec.md
Literature Ref: MacKay (2003), De Kleer & Williams (1987), Friston et al. (2017), Pearl (2000).
"""

from pathlib import Path

import yaml

from hypostases.abduction.abductive_engine import AbductiveEngine
from hypostases.abduction.anomaly_detector import SurpriseDetector
from hypostases.abduction.hypothesis import Hypothesis
from hypostases.abduction.hypothesis_generator import HypothesisGenerator
from hypostases.abduction.types import (
    HypothesisCategory,
)
from hypostases.communication.bayesian_updater import BayesianCommunicationEngine
from hypostases.communication.types import PeerMessage
from hypostases.engine.types import (
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)


def test_hypothesis_serialization_and_occam_penalty():
    """Tests Hypothesis creation, MacKay Occam factor posterior calculation, and dict serialization."""
    h = Hypothesis(
        description="Hidden resource decay anomaly",
        category=HypothesisCategory.ENVIRONMENT,
        prior=0.4,
        likelihood=0.8,
        complexity=1.5,
        predictive_params={"scale": 0.8, "shift": -1.0},
    )

    # Calculate posterior score with MacKay Occam penalty
    post = h.compute_posterior(lambda_mdl=0.2)
    assert 0.0 <= post <= 1.0
    assert h.confidence == post

    # Test serialization
    h_dict = h.to_dict()
    assert h_dict["category"] == "ENVIRONMENT"
    assert h_dict["complexity"] == 1.5

    # Test deserialization
    h_restored = Hypothesis.from_dict(h_dict)
    assert h_restored.category == HypothesisCategory.ENVIRONMENT
    assert h_restored.complexity == 1.5


def test_surprise_detector_free_energy():
    """Tests Friston Free Energy computation and surprise threshold anomaly detection."""
    detector = SurpriseDetector(surprise_threshold=1.5)

    # 1. Normal observation matching expectation -> Low surprise
    is_anomaly, fe1 = detector.check_anomaly(
        observed_val=10.0,
        predicted_mean=10.0,
        predicted_var=2.0,
    )
    assert not is_anomaly
    assert fe1 < 1.5

    # 2. Anomaly observation deviating significantly -> High surprise
    is_anomaly_2, fe2 = detector.check_anomaly(
        observed_val=2.0,
        predicted_mean=10.0,
        predicted_var=1.0,
    )
    assert is_anomaly_2
    assert fe2 > 1.5


def test_hypothesis_generator_modalities():
    """Tests multi-modal candidate hypothesis generation."""
    generator = HypothesisGenerator(enable_env=True, enable_peer=True, enable_causal=True)

    candidates = generator.generate_candidates(
        observed_val=4.0,
        predicted_mean=10.0,
        timestamp=5,
        context={"peer_id": "agent_beta"},
    )

    assert len(candidates) >= 3
    categories = {h.category for h in candidates}
    assert HypothesisCategory.ENVIRONMENT in categories
    assert HypothesisCategory.PEER_INTENT in categories
    assert HypothesisCategory.CAUSAL_STRUCTURE in categories

    # Verify prior normalization across ensemble
    total_prior = sum(h.prior for h in candidates)
    assert abs(total_prior - 1.0) < 1e-5


def test_abductive_engine_full_lifecycle():
    """Tests AbductiveEngine anomaly detection, candidate generation, Occam evaluation, and pruning."""
    engine = AbductiveEngine(
        surprise_threshold=1.5,
        complexity_penalty_lambda=0.2,
        pruning_threshold=0.01,
        max_pool_size=10,
    )

    # Step 1: Normal observation -> No anomaly
    is_anomaly_1, new_h_1 = engine.process_observation(
        observed_val=10.0,
        predicted_mean=10.0,
        timestamp=1,
    )
    assert not is_anomaly_1
    assert len(new_h_1) == 0

    # Step 2: Anomaly observation -> Triggers candidate generation & Occam posterior update
    is_anomaly_2, new_h_2 = engine.process_observation(
        observed_val=3.0,
        predicted_mean=10.0,
        timestamp=2,
    )
    assert is_anomaly_2
    assert len(new_h_2) > 0
    assert engine.metrics.total_anomalies_detected == 1

    top_h = engine.get_top_hypothesis()
    assert top_h is not None
    assert top_h.posterior > 0.0

    # Step 3: Continued supporting evidence for top hypothesis
    for t in range(3, 8):
        engine.process_observation(
            observed_val=3.1,
            predicted_mean=10.0,
            timestamp=t,
        )

    assert engine.metrics.active_pool_size > 0


def test_abductive_reasoning_yaml_config_loading():
    """Tests data-driven YAML configuration loading (Rule 006)."""
    yaml_path = Path("schema/abductive_reasoning_config.yaml")
    assert yaml_path.exists()

    with open(yaml_path, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    assert "abductive_reasoning" in config_data
    cfg = config_data["abductive_reasoning"]
    assert cfg["enabled"] is True
    assert cfg["surprise_threshold"] == 1.5

    engine = AbductiveEngine(config=config_data)
    assert engine.surprise_threshold == 1.5
    assert engine.lambda_mdl == 0.2


def test_bayesian_communication_abductive_integration():
    """Tests integration of Front 06 Bayesian Communication with Front 11 AbductiveEngine."""
    abduction = AbductiveEngine()
    # Populate pool with a peer intent hypothesis
    h_peer = Hypothesis(
        identifier="H_peer_intent_test",
        category=HypothesisCategory.PEER_INTENT,
        prior=0.5,
        likelihood=0.5,
    )
    abduction.hypotheses[h_peer.identifier] = h_peer

    # Construct AgentState with abductive_engine attached to WorldModel
    wm = WorldModel()
    wm.abductive_engine = abduction
    state = AgentState(
        c=Characteristics(),
        w=wm,
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )

    comm_engine = BayesianCommunicationEngine()
    msg = PeerMessage(
        sender_id="peer_alpha",
        receiver_id="agent_self",
        payload={"reserve": 12.0},
    )

    # Process message through comm engine
    comm_engine.process_incoming_message(msg, state)

    # Verify that abductive engine peer hypothesis posterior was updated
    assert h_peer.likelihood > 0.5
