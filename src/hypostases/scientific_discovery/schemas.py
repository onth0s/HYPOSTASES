"""State schemas and data models for Scientific Discovery Loop (Wave 4 Front 12)."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Hypothesis:
    """Hypothesis object H_k representing candidate explanatory structural model.

    Attributes:
        hypothesis_id: Unique string identifier.
        description: Natural language / symbolic summary.
        causal_edges: List of directed edges (u, v) in structural causal model DAG.
        parameters: Structural equation parameter dictionary.
        prior_probability: Prior probability P_0(H_k).
        posterior_probability: Current Bayesian posterior P(H_k | E_1:t).
        mdl_complexity: Minimum Description Length (MDL) complexity score C(H_k).
        elo_rating: Gottweis (2025) evolutionary debate tournament Elo rating.
        supporting_evidence_count: Number of empirical observations aligned with H_k.
        contradicting_evidence_count: Number of empirical observations disproving H_k.
    """

    hypothesis_id: str
    description: str
    causal_edges: list[tuple[str, str]] = field(default_factory=list)
    parameters: dict[str, float] = field(default_factory=dict)
    prior_probability: float = 0.1
    posterior_probability: float = 0.1
    mdl_complexity: float = 1.0
    elo_rating: float = 1000.0
    supporting_evidence_count: int = 0
    contradicting_evidence_count: int = 0

    def compute_mdl(self, num_nodes: int, num_observations: int = 100) -> float:
        """Compute Minimum Description Length (MDL) Occam complexity score.

        C(H_k) = |E_k| * log2(|V|) + (|Theta_k| / 2) * log2(N_obs)
        """
        edge_cost = len(self.causal_edges) * (np.log2(max(num_nodes, 1)) if num_nodes > 1 else 1.0)
        param_cost = (len(self.parameters) / 2.0) * np.log2(max(num_observations, 1))
        self.mdl_complexity = float(edge_cost + param_cost)
        return self.mdl_complexity


@dataclass
class ExperimentalDesign:
    """Experimental Design object d* representing interventional design do(X_i = x_i).

    Attributes:
        design_id: Unique identifier for experimental action.
        target_variable: Name of state variable undergoing intervention X_i.
        intervention_value: Interventional value x_i assigned to X_i.
        expected_information_gain: EIG(d) Shannon entropy reduction value.
        execution_cost: Resource / energy cost C(d) required to perform experiment in rho_ext.
        ace_lower_bound: Foster (2020) Adaptive Contrastive Estimation bound score.
        friston_efe_utility: Combined Expected Free Energy score (Rule 009).
    """

    design_id: str
    target_variable: str
    intervention_value: float
    expected_information_gain: float = 0.0
    execution_cost: float = 1.0
    ace_lower_bound: float = 0.0
    friston_efe_utility: float = 0.0


@dataclass
class Evidence:
    """Evidence object E_t representing empirical observational outcome from rho_ext.

    Attributes:
        evidence_id: Unique identifier for empirical evidence item.
        design_id: Identifier of the experimental design d* that produced E_t.
        timestamp: Simulation step / tick timestamp.
        observations: Mapping of observed variable names to empirical scalar values.
        raw_vector: Optional dense vector observation.
    """

    evidence_id: str
    design_id: str
    timestamp: int
    observations: dict[str, float] = field(default_factory=dict)
    raw_vector: np.ndarray | None = None


@dataclass
class ScientificDiscoveryConfig:
    """Data-driven ground truth YAML configuration schema for Scientific Discovery."""

    enabled: bool = True
    efe_mode: bool = True
    anomaly_threshold_eta: float = 0.15
    max_hypotheses_k: int = 16
    mdl_complexity_penalty_beta: float = 0.5
    eig_monte_carlo_samples: int = 1000
    ace_num_contrastive_samples_l: int = 64
    elo_initial_rating: float = 1000.0
    elo_k_factor: float = 32.0
    pruning_threshold_epsilon: float = 1e-4
    ase_cost_weight_gamma: float = 1.0
    dual_persistence_enabled: bool = True
    snapshot_interval_ticks: int = 10
