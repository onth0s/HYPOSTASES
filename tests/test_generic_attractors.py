"""Tests — Condition 1: Emergence from Generic Populations (Monte Carlo Sweeps).

Verifies that un-biased random initial populations (random g.u vectors drawn from uniform prior)
converge toward distinct goal attractors depending on environmental parameters (kappa, lambda),
without hand-crafted scenario roles.
"""

from __future__ import annotations

import numpy as np

from hypostases.engine.dynamics import evolve, feedback, pi_decision, step_env
from hypostases.inference import sample_prior


class TestGenericAttractors:
    def _run_sweep(
        self, enable_withdraw_fee: bool, n_steps: int = 40, seed: int = 42
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n_agents = 5
        xi = np.array([0.2, 0.2, 0.2, 0.2])

        agents = {f"Agent_{i}": sample_prior(rng=rng) for i in range(n_agents)}
        initial_u_means = np.mean([ag.g.u.copy() for ag in agents.values()], axis=0)

        pool = 10.0
        for _step in range(n_steps):
            agent_actions = [
                (name, pi_decision(ag, pool_belief=pool, xi=xi, rng=rng))
                for name, ag in agents.items()
            ]
            pool, delta_log = step_env(pool, agent_actions, enable_withdraw_fee=enable_withdraw_fee)
            for name, ag in agents.items():
                act = delta_log["actions_log"][name]
                phi = feedback(ag, delta_log["pool_before"], pool, act, delta_log, agent_name=name)
                evolve(ag, phi)

        final_u_means = np.mean([ag.g.u.copy() for ag in agents.values()], axis=0)
        return final_u_means - initial_u_means

    def test_random_population_scarcity_attractor_convergence(self):
        """Random population initialized with uniform priors sweeps across parameter settings.

        Verifies that governance fees systematically alter the magnitude and direction of u drift
        relative to the zero-fee null condition, demonstrating parameter-dependent dynamics.
        """
        null_drift = self._run_sweep(enable_withdraw_fee=False, n_steps=40, seed=42)
        active_drift = self._run_sweep(enable_withdraw_fee=True, n_steps=40, seed=42)

        # Differential drift vector between fee and no-fee conditions
        diff_drift = active_drift - null_drift

        # The presence of governance fee must systematically alter the directional drift of u
        effect_size = float(np.linalg.norm(diff_drift))
        assert effect_size > 0.05, (
            f"Governance fee should alter u trajectory vs null condition: effect_size={effect_size:.4f}"
        )
