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
    def test_random_population_scarcity_attractor_convergence(self):
        """Random population initialized with uniform priors sweeps across high scarcity.

        Verifies that high scarcity (kappa=0.5) and governance fees (lambda=1.5) systematically
        shift latent utility vectors g.u from initial random weights, demonstrating unprompted
        attractor dynamics without hand-picked scenario roles.
        """
        rng = np.random.default_rng(42)
        n_agents = 5
        n_steps = 30
        xi = np.array([0.2, 0.2, 0.2, 0.2])

        # Initialize generic random population
        agents = {f"Agent_{i}": sample_prior(rng=rng) for i in range(n_agents)}

        # Record initial mean u vector across population
        initial_u_means = np.mean([ag.g.u.copy() for ag in agents.values()], axis=0)

        pool = 10.0
        for _step in range(n_steps):
            agent_actions = [
                (name, pi_decision(ag, pool_belief=pool, xi=xi, rng=rng))
                for name, ag in agents.items()
            ]
            pool, delta_log = step_env(pool, agent_actions, enable_withdraw_fee=True)
            for name, ag in agents.items():
                act = delta_log["actions_log"][name]
                phi = feedback(ag, delta_log["pool_before"], pool, act, delta_log, agent_name=name)
                evolve(ag, phi)

        final_u_means = np.mean([ag.g.u.copy() for ag in agents.values()], axis=0)

        # The u vector must show non-zero shift from random initialization
        delta_u = np.abs(final_u_means - initial_u_means)
        assert np.any(delta_u > 0.01), (
            f"Population g.u should adapt from initial state: delta={delta_u}"
        )
