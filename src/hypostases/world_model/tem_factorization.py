"""HYPOSTASES — Tolman-Eichenbaum Machine (TEM) Relational Factorization Engine.

Spec Ref: Whittington et al. (Cell 2020), Behrens et al. (Neuron 2018).
Factorizes environments into invariant entorhinal grid-cell structural basis G
and sensory-specific binding matrices X, executing path integration via W_a transitions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hypostases.world_model.hierarchical_types import TEMBasis


class TEMFactorizationEngine:
    """Engine executing TEM entorhinal grid-cell structural basis factorization."""

    def __init__(self, tem_basis: TEMBasis) -> None:
        self.basis = tem_basis
        self.current_structural_state_g: np.ndarray = np.ones(
            (self.basis.grid_size,), dtype=np.float64
        ) / float(self.basis.grid_size)

    def bind_sensory_observation(self, sensory_vector_x: np.ndarray) -> np.ndarray:
        r"""Computes factorized state tensor r_t = g_t \otimes x_t.

        Returns outer product matrix representing sensory observation grounded in grid space.
        """

        g = self.current_structural_state_g
        x = np.atleast_1d(sensory_vector_x).astype(np.float64)
        return np.outer(g, x)

    def predict_next_structural_state(self, action_key: str) -> np.ndarray:
        """Executes TEM path integration g_{t+1} = W_a g_t for given action.

        If action_key is unknown, defaults to identity transition.
        """
        g = self.current_structural_state_g
        if action_key in self.basis.action_transitions:
            w_a = self.basis.action_transitions[action_key]
            g_next = np.dot(w_a, g)

            # Normalize to preserve probability / unit norm basis
            norm = float(np.linalg.norm(g_next))
            if norm > 0:
                g_next = g_next / norm
            self.current_structural_state_g = g_next
        return self.current_structural_state_g

    def update_attractor_manifold(self, v_t: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Executes Continuous Attractor Network (CAN) velocity path integration (Burak & Fiete 2009).

        Shifts structural basis g_t along velocity vector v_t = [v_x, v_y, ...] over time dt.
        """
        v = np.atleast_1d(v_t).astype(np.float64)
        v_speed = float(np.linalg.norm(v))
        if v_speed == 0.0:
            return self.current_structural_state_g

        g = self.current_structural_state_g
        grid_size = len(g)
        shift = int(np.round(v_speed * dt)) % grid_size

        if shift != 0:
            g_next = np.roll(g, shift)
            # Add Gaussian smoothing to simulate bump dynamics on continuous manifold
            kernel = np.array([0.15, 0.70, 0.15], dtype=np.float64)
            g_smoothed = np.convolve(g_next, kernel, mode="same")
            norm = float(np.linalg.norm(g_smoothed))
            if norm > 0:
                g_smoothed /= norm
            self.current_structural_state_g = g_smoothed

        return self.current_structural_state_g

    def get_relational_snapshot(self) -> dict[str, Any]:
        """Returns snapshot of current structural basis state and action transition rules."""
        return {
            "grid_size": self.basis.grid_size,
            "current_g": self.current_structural_state_g.copy(),
            "num_actions": len(self.basis.action_transitions),
        }
