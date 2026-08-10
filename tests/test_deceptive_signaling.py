"""Tests — Deceptive Signaling & Asymmetric Information.

Evaluates how SMC particle filters handle deceptive state-masking (a agent
with high reserve issuing low REQUEST actions to claim depletion).
"""

from __future__ import annotations

import numpy as np

from hypostases.engine.types import (
    Action,
    ActionType,
)
from hypostases.inference import infer
from hypostases.simulation.scenarios import create_scenario_agents


class TestDeceptiveSignaling:
    def test_scenario_deceptive_loading(self):
        agents = create_scenario_agents("deceptive")
        assert "DeceptiveAgent" in agents
        assert "Observer" in agents

    def test_infer_particle_filter_tracks_deceptive_observation_sequence(self):
        # Deceptive agent emits low REQUESTs (e.g. amount=1.0)
        action_trace = [Action(ActionType.REQUEST, amount=1.0) for _ in range(5)]

        pool_trace = [10.0] * 5
        xi = np.array([0.2, 0.2, 0.2, 0.2])

        particles = infer(
            observed_actions=action_trace,
            observed_pool_trace=pool_trace,
            xi=xi,
            n_particles=100,
            rng=np.random.default_rng(42),
        )

        assert len(particles) == 100
        weights = np.array([p.weight for p in particles])
        assert abs(weights.sum() - 1.0) < 1e-5
        # Particles should remain valid without weight collapse
        mean_reserve = float(np.mean([p.sigma.c.reserve for p in particles]))
        assert mean_reserve > 0.0
