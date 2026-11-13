"""State schemas and data models for Scientific Discovery Loop (Wave 4 Front 12)."""

from dataclasses import dataclass, field

import numpy as np

from hypostases.schemas.hypothesis import BaseHypothesis


@dataclass
class Hypothesis(BaseHypothesis):
    """Hypothesis object H_k representing candidate explanatory structural model.

    Inherits from unified BaseHypothesis.
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

    @property
    def identifier(self) -> str:
        return self.hypothesis_id

    @identifier.setter
    def identifier(self, val: str) -> None:
        self.hypothesis_id = val

    @property
    def prior(self) -> float:
        return self.prior_probability

    @prior.setter
    def prior(self, val: float) -> None:
        self.prior_probability = val

    @property
    def posterior(self) -> float:
        return self.posterior_probability

    @posterior.setter
    def posterior(self, val: float) -> None:
        self.posterior_probability = val

    @property
    def complexity(self) -> float:
        return self.mdl_complexity

    @complexity.setter
    def complexity(self, val: float) -> None:
        self.mdl_complexity = val

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
