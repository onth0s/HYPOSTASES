"""MetaLearner implementing outer-loop bi-level adaptation for HYPOSTASES agents."""

import math
from typing import Any

from hypostases.meta_learning.meta_evaluator import MetaEvaluator
from hypostases.meta_learning.meta_state import MetaParameterVector


class MetaLearner:
    """Outer-loop meta-optimizer applying ALFA learning rate scaling, MetaPEFT modulators,

    MetaClaw skill distillation, and Expected Free Energy (EFE) preference compatibility.

    Strictly complies with Rule 005 (zero artificial human cognitive bias) and Rule 011 (dual persistence).
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        initial_params: MetaParameterVector | None = None,
    ) -> None:
        self.config = config or {}
        self.meta_params = initial_params or MetaParameterVector()
        self.evaluator = MetaEvaluator()
        self.meta_learning_rate = self.config.get("outer_loop", {}).get("meta_learning_rate", 0.01)
        self.step_count = 0

    def enforce_preference_compatibility(
        self, state_prefs: list[float], likelihood_matrix: list[list[float]]
    ) -> list[float]:
        """Enforces observation preference compatibility C_o = A * C_s (Champion et al. 2024)."""
        num_obs = len(likelihood_matrix)
        num_states = len(state_prefs)
        if num_states != len(likelihood_matrix[0]):
            raise ValueError("Dimension mismatch between state preferences and likelihood matrix.")

        obs_prefs = [0.0] * num_obs
        for r in range(num_obs):
            obs_prefs[r] = sum(likelihood_matrix[r][c] * state_prefs[c] for c in range(num_states))

        # Normalize to probability simplex
        total = sum(obs_prefs)
        if total > 0:
            obs_prefs = [p / total for p in obs_prefs]
        return obs_prefs

    def adapt_step(
        self,
        utility_gain: float,
        belief_var_reduction: float,
        compute_cost: float,
        failure_observed: bool = False,
    ) -> MetaParameterVector:
        """Executes a fast adaptation step, updating theta_meta based on empirical meta-reward."""
        self.step_count += 1
        meta_reward = self.evaluator.record_step(utility_gain, belief_var_reduction, compute_cost)

        # Gradient ascent update on meta-parameters (ALFA & MetaPEFT style)
        grad_signal = math.tanh(meta_reward)
        self.meta_params.learning_rate = max(
            0.0001,
            min(
                0.5,
                self.meta_params.learning_rate + self.meta_learning_rate * grad_signal * 0.005,
            ),
        )
        self.meta_params.efe_beta = max(
            0.0,
            min(
                1.0,
                self.meta_params.efe_beta + self.meta_learning_rate * grad_signal * 0.02,
            ),
        )

        # Dynamic adjustment of rollout particle count & depth based on variance reduction requirement
        if belief_var_reduction < 0.05:
            # Need more exploration / inference depth
            self.meta_params.particle_count = min(64, self.meta_params.particle_count + 1)
        elif compute_cost > 10.0:
            # Scale back to save compute
            self.meta_params.particle_count = max(4, self.meta_params.particle_count - 1)

        # MetaClaw skill generation increment on failure
        if failure_observed:
            self.meta_params.version += 1

        return self.meta_params

    def sync_dual_persistence(self, yaml_snapshot_path: str) -> None:
        """Saves current theta_meta state to YAML snapshot (Rule 011)."""
        self.meta_params.save_yaml(yaml_snapshot_path)
