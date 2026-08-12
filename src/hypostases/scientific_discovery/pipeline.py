"""Scientific Discovery Pipeline Manager (Wave 4 Front 12).

Coordinates the full 8-stage scientific discovery loop:
Observe -> Infer -> Generate Hypotheses -> Rank Explanations -> Design Experiment -> Collect Evidence -> Update Hypotheses -> Act

Operates directly over the core state substrate: \\sigma = (c, w, g, \rho_{ext}).
Dual persistence enabled under Rule 011 (YAML snapshots).
"""

import numpy as np

from hypostases.scientific_discovery.bayesian_updater import BayesianUpdater
from hypostases.scientific_discovery.experimental_design import BayesianExperimentalDesignEngine
from hypostases.scientific_discovery.hypothesis_manager import HypothesisManager
from hypostases.scientific_discovery.schemas import (
    Evidence,
    ExperimentalDesign,
    Hypothesis,
    ScientificDiscoveryConfig,
)


class ScientificDiscoveryPipeline:
    """Master manager executing the 8-stage scientific discovery cycle."""

    def __init__(self, config: ScientificDiscoveryConfig | None = None):
        self.config = config or ScientificDiscoveryConfig()
        self.hypothesis_manager = HypothesisManager(config=self.config)
        self.bed_engine = BayesianExperimentalDesignEngine(config=self.config)
        self.bayesian_updater = BayesianUpdater(config=self.config)

        self.current_tick: int = 0
        self.evidence_history: list[Evidence] = []
        self.discovery_logs: list[dict] = []

    def step(
        self,
        observation: dict[str, float],
        candidate_designs: list[ExperimentalDesign],
        predicted_dist: np.ndarray | None = None,
        observed_dist: np.ndarray | None = None,
    ) -> tuple[Hypothesis, ExperimentalDesign, Evidence]:
        """Execute one full tick of the 8-stage Scientific Discovery Loop.

        Returns:
            Tuple of (Best Hypothesis H*, Selected Design d*, Collected Evidence E_t)
        """
        self.current_tick += 1

        # Stage 1 & 2: Observe & Infer Anomaly Detection
        if predicted_dist is not None and observed_dist is not None:
            is_anomaly, _ = self.hypothesis_manager.check_anomaly_trigger(
                predicted_dist, observed_dist
            )
        else:
            is_anomaly = False

        # Stage 3: Generate Hypotheses (if anomaly triggered or pool empty)
        if is_anomaly or not self.hypothesis_manager.hypotheses:
            self._generate_abductive_hypotheses(observation)

        # Stage 4: Rank Explanations (MDL Occam Priors & Gottweis 2025 Elo Tournaments)
        self.hypothesis_manager.update_priors_via_mdl(num_nodes=max(len(observation), 1))
        recent_evidence_dicts = [e.observations for e in self.evidence_history[-10:]]
        ranked_hypotheses = self.hypothesis_manager.run_elo_tournament(recent_evidence_dicts)
        best_hypothesis = ranked_hypotheses[0]

        # Stage 5: Design Experiment (BED, EIG, ACE Bound & Friston EFE Selection)
        if not candidate_designs:
            # Generate default candidate interventional design if none provided
            candidate_designs = [
                ExperimentalDesign(
                    design_id=f"design_{var}",
                    target_variable=var,
                    intervention_value=val + 1.0,
                    execution_cost=1.0,
                )
                for var, val in observation.items()
            ] or [
                ExperimentalDesign(
                    design_id="design_default",
                    target_variable="state_var",
                    intervention_value=1.0,
                    execution_cost=1.0,
                )
            ]

        selected_design = self.bed_engine.select_optimal_experimental_design(
            candidate_designs=candidate_designs,
            hypotheses=ranked_hypotheses,
        )

        # Stage 6: Collect Evidence (Execute do(d*) intervention in rho_ext)
        evidence = Evidence(
            evidence_id=f"ev_tick_{self.current_tick}",
            design_id=selected_design.design_id,
            timestamp=self.current_tick,
            observations=observation,
        )
        self.evidence_history.append(evidence)

        # Stage 7: Update Hypotheses (Exact Bayesian Update & Ensemble Pruning)
        updated_hypotheses = self.bayesian_updater.update_posteriors(
            hypotheses=ranked_hypotheses,
            evidence=evidence,
        )
        self.hypothesis_manager.hypotheses = updated_hypotheses

        # Stage 8: Act & Log Record
        log_entry = {
            "tick": self.current_tick,
            "best_hypothesis_id": best_hypothesis.hypothesis_id,
            "best_hypothesis_posterior": best_hypothesis.posterior_probability,
            "best_hypothesis_elo": best_hypothesis.elo_rating,
            "selected_design_id": selected_design.design_id,
            "eig": selected_design.expected_information_gain,
            "ace_bound": selected_design.ace_lower_bound,
            "efe_utility": selected_design.friston_efe_utility,
        }
        self.discovery_logs.append(log_entry)

        return best_hypothesis, selected_design, evidence

    def _generate_abductive_hypotheses(self, observation: dict[str, float]) -> None:
        """Helper to instantiate candidate SCM structural hypotheses."""
        var_names = list(observation.keys()) or ["var_0"]

        h1 = Hypothesis(
            hypothesis_id=f"H1_direct_{self.current_tick}",
            description="Direct identity structural causal model",
            causal_edges=[(var_names[0], var_names[0])],
            parameters={v: val for v, val in observation.items()},
        )

        h2 = Hypothesis(
            hypothesis_id=f"H2_decay_{self.current_tick}",
            description="Linear decay structural causal model",
            causal_edges=[(var_names[0], var_names[0])],
            parameters={v: val * 0.9 for v, val in observation.items()},
        )

        self.hypothesis_manager.add_hypothesis(h1)
        self.hypothesis_manager.add_hypothesis(h2)

    def export_snapshot_yaml(self) -> str:
        """Export persistent human-readable YAML snapshot of discovery state (Rule 011)."""
        import yaml

        snapshot_data = {
            "scientific_discovery_snapshot": {
                "tick": self.current_tick,
                "hypothesis_count": len(self.hypothesis_manager.hypotheses),
                "hypotheses": [
                    {
                        "id": h.hypothesis_id,
                        "description": h.description,
                        "prior": h.prior_probability,
                        "posterior": h.posterior_probability,
                        "mdl_complexity": h.mdl_complexity,
                        "elo_rating": h.elo_rating,
                    }
                    for h in self.hypothesis_manager.hypotheses
                ],
                "evidence_history_length": len(self.evidence_history),
                "recent_logs": self.discovery_logs[-5:],
            }
        }
        return yaml.dump(snapshot_data, sort_keys=False)
