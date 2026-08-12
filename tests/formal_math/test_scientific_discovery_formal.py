"""Formal Mathematical Implementation Verification for Wave 4 Front 12 (Rule 012).

Tests:
1. EIG Non-Negativity Invariant: EIG(d) >= 0 for all candidate experimental designs.
2. Asymptotic Bayesian Convergence Theorem: P(H* | E_1:t) -> 1.0 as t -> inf for true SCM H*.
3. Foster (2020) ACE Lower Bound Monotonicity & Non-Negativity Invariant.
4. Gottweis (2025) Elo Tournament Evolutionary Ranking Convergence.
5. Minimum Description Length (MDL) Occam's Razor Invariant.
6. Rule 005 Pure Rationality Invariant (Zero artificial human cognitive defects).
"""

import numpy as np
import pytest

from hypostases.scientific_discovery.bayesian_updater import BayesianUpdater
from hypostases.scientific_discovery.experimental_design import (
    AdaptiveContrastiveEstimation,
    BayesianExperimentalDesignEngine,
)
from hypostases.scientific_discovery.hypothesis_manager import HypothesisManager
from hypostases.scientific_discovery.pipeline import ScientificDiscoveryPipeline
from hypostases.scientific_discovery.schemas import (
    Evidence,
    ExperimentalDesign,
    Hypothesis,
    ScientificDiscoveryConfig,
)


def test_eig_non_negativity_invariant():
    """Verify that Expected Information Gain (EIG) Shannon entropy reduction is strictly non-negative.

    EIG(d) = H(H) - E_{P(E|d)} [ H(H | E, d) ] >= 0
    """
    config = ScientificDiscoveryConfig(eig_monte_carlo_samples=100)
    bed_engine = BayesianExperimentalDesignEngine(config=config)

    hypotheses = [
        Hypothesis(
            hypothesis_id="H1",
            description="Model 1",
            parameters={"x": 5.0},
            posterior_probability=0.6,
        ),
        Hypothesis(
            hypothesis_id="H2",
            description="Model 2",
            parameters={"x": 10.0},
            posterior_probability=0.4,
        ),
    ]

    design = ExperimentalDesign(
        design_id="d1", target_variable="x", intervention_value=5.0, execution_cost=1.0
    )
    eig = bed_engine.compute_expected_information_gain(design, hypotheses)

    assert eig >= 0.0, f"EIG must be non-negative, got {eig}"


def test_asymptotic_bayesian_posterior_convergence_theorem():
    """Verify asymptotic Bayesian posterior convergence theorem: P(H* | E_1:t) -> 1.0 as t -> inf.

    Simulates t=1..50 evidence collection steps where true data-generating process is H*.
    """
    config = ScientificDiscoveryConfig(pruning_threshold_epsilon=1e-6)
    updater = BayesianUpdater(config=config)

    true_h = Hypothesis(
        hypothesis_id="H_true",
        description="True SCM",
        parameters={"param_a": 4.2},
        posterior_probability=0.33,
    )
    false_h1 = Hypothesis(
        hypothesis_id="H_false_1",
        description="False SCM 1",
        parameters={"param_a": 1.0},
        posterior_probability=0.33,
    )
    false_h2 = Hypothesis(
        hypothesis_id="H_false_2",
        description="False SCM 2",
        parameters={"param_a": 9.0},
        posterior_probability=0.34,
    )

    hypotheses = [true_h, false_h1, false_h2]

    # Normalize initial posteriors
    total = sum(h.posterior_probability for h in hypotheses)
    for h in hypotheses:
        h.posterior_probability /= total

    np.random.seed(42)
    # Simulate sequential observation ticks
    for t in range(1, 30):
        # Generate empirical observation centered around true parameter 4.2
        obs_val = float(np.random.normal(4.2, 0.1))
        evidence = Evidence(
            evidence_id=f"ev_{t}", design_id="d1", timestamp=t, observations={"param_a": obs_val}
        )

        hypotheses = updater.update_posteriors(hypotheses, evidence)

    # Find posterior probability of true hypothesis
    h_true_retained = next((h for h in hypotheses if h.hypothesis_id == "H_true"), None)

    assert h_true_retained is not None, "True hypothesis should not be pruned"
    assert h_true_retained.posterior_probability == pytest.approx(1.0, abs=1e-2), (
        f"Expected true hypothesis posterior ~ 1.0, got {h_true_retained.posterior_probability}"
    )


def test_foster_ace_bound_non_negativity():
    """Verify Foster et al. (2020) Adaptive Contrastive Estimation (ACE) lower bound non-negativity."""
    ace_estimator = AdaptiveContrastiveEstimation(num_samples_l=32)

    hypotheses = [
        Hypothesis(
            hypothesis_id="H1", description="M1", parameters={"y": 1.0}, posterior_probability=0.5
        ),
        Hypothesis(
            hypothesis_id="H2", description="M2", parameters={"y": 5.0}, posterior_probability=0.5
        ),
    ]
    design = ExperimentalDesign(design_id="d1", target_variable="y", intervention_value=1.0)

    ace_score = ace_estimator.compute_ace_bound(design, hypotheses)
    assert ace_score >= 0.0, f"ACE lower bound must be non-negative, got {ace_score}"


def test_gottweis_elo_tournament_ranking_convergence():
    """Verify Gottweis et al. (2025) Elo evolutionary debate tournament ranking convergence."""
    config = ScientificDiscoveryConfig(elo_k_factor=32.0)
    manager = HypothesisManager(config=config)

    h_accurate = Hypothesis(
        hypothesis_id="H_acc", description="Accurate", parameters={"val": 10.0}, elo_rating=1000.0
    )
    h_inaccurate = Hypothesis(
        hypothesis_id="H_inacc",
        description="Inaccurate",
        parameters={"val": 0.0},
        elo_rating=1000.0,
    )

    manager.add_hypothesis(h_accurate)
    manager.add_hypothesis(h_inaccurate)

    # Evidence batch matching accurate hypothesis
    evidence_batch = [{"val": 10.1}, {"val": 9.9}, {"val": 10.0}]

    ranked = manager.run_elo_tournament(evidence_batch)

    assert ranked[0].hypothesis_id == "H_acc"
    assert ranked[0].elo_rating > ranked[1].elo_rating
    assert ranked[0].supporting_evidence_count > 0


def test_mdl_occam_razor_invariant():
    """Verify Minimum Description Length (MDL) Occam's razor prior penalization invariant.

    Simpler structural graph H_simple (fewer edges) must receive lower MDL complexity
    and higher prior probability than complex graph H_complex under equal performance.
    """
    config = ScientificDiscoveryConfig(mdl_complexity_penalty_beta=0.5)
    manager = HypothesisManager(config=config)

    h_simple = Hypothesis(
        hypothesis_id="H_simple",
        description="Simple DAG",
        causal_edges=[("A", "B")],
        parameters={"w1": 1.0},
    )

    h_complex = Hypothesis(
        hypothesis_id="H_complex",
        description="Complex DAG",
        causal_edges=[("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")],
        parameters={"w1": 1.0, "w2": 2.0, "w3": 3.0, "w4": 4.0, "w5": 5.0},
    )

    manager.add_hypothesis(h_simple)
    manager.add_hypothesis(h_complex)

    manager.update_priors_via_mdl(num_nodes=4)

    assert h_simple.mdl_complexity < h_complex.mdl_complexity
    assert h_simple.prior_probability > h_complex.prior_probability


def test_rule_005_pure_rationality_pipeline_integration():
    """Verify end-to-end pipeline execution under Rule 005 pure rationality invariant."""
    config = ScientificDiscoveryConfig(efe_mode=True, enabled=True)
    pipeline = ScientificDiscoveryPipeline(config=config)

    obs = {"temperature": 25.0, "pressure": 101.3}

    best_h, selected_d, evidence = pipeline.step(
        observation=obs,
        candidate_designs=None,
    )

    assert best_h is not None
    assert selected_d is not None
    assert evidence is not None
    assert pipeline.current_tick == 1
    assert len(pipeline.discovery_logs) == 1
    assert selected_d.friston_efe_utility >= 0.0

    # Test Rule 011 dual persistence YAML snapshot export
    snapshot_yaml = pipeline.export_snapshot_yaml()
    assert "scientific_discovery_snapshot" in snapshot_yaml
    assert "hypotheses" in snapshot_yaml
