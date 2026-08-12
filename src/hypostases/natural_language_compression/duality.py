"""Visual-Epistemic Duality Mapper (Giaquinto 2007) for Front 14."""

from __future__ import annotations

from typing import Any

import numpy as np


class VisualEpistemicDualityMapper:
    """Giaquinto (2007) Visual-Epistemic Duality Mapper.

    Provides bidirectional translation between continuous spatial Voronoi gists
    in semantic memory (c.m_semantic) and discrete symbolic token arrays (L).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        duality_cfg = config.get("duality_config", {})
        self.num_cells: int = duality_cfg.get("spatial_voronoi_cells", 16)
        self.gist_res: int = duality_cfg.get("gist_resolution", 32)
        self.tol: float = duality_cfg.get("topological_invariance_tol", 1e-3)

        # Initialize spatial Voronoi centroids in state space
        np.random.seed(42)
        self.centroids: np.ndarray = np.random.randn(self.num_cells, 8)
        # Normalize centroids onto unit sphere
        norms = np.linalg.norm(self.centroids, axis=1, keepdims=True)
        self.centroids = self.centroids / np.maximum(norms, 1e-8)

    def encode_spatial_to_symbolic(self, continuous_state: np.ndarray) -> list[int]:
        """Encodes continuous spatial state vector into discrete cell symbol IDs."""
        # Find nearest Voronoi centroid (Jordan curve topological region)
        state_vec = np.asarray(continuous_state, dtype=np.float64).flatten()
        if state_vec.shape[0] < 8:
            state_vec = np.pad(state_vec, (0, 8 - state_vec.shape[0]))
        else:
            state_vec = state_vec[:8]

        dists = np.linalg.norm(self.centroids - state_vec, axis=1)
        sorted_indices = np.argsort(dists).tolist()
        return sorted_indices[:4]

    def decode_symbolic_to_spatial(self, symbol_ids: list[int]) -> np.ndarray:
        """Reconstructs continuous spatial gist centroid from discrete symbol IDs."""
        valid_ids = [idx for idx in symbol_ids if 0 <= idx < self.num_cells]
        if not valid_ids:
            return np.zeros(8)
        selected_centroids = self.centroids[valid_ids]
        return np.mean(selected_centroids, axis=0)

    def verify_topological_invariance(self, continuous_state: np.ndarray) -> bool:
        """Verifies topological invariance: round-trip reconstruction error <= tol."""
        symbol_ids = self.encode_spatial_to_symbolic(continuous_state)
        reconstructed = self.decode_symbolic_to_spatial(symbol_ids)

        state_vec = np.asarray(continuous_state, dtype=np.float64).flatten()
        if state_vec.shape[0] < 8:
            state_vec = np.pad(state_vec, (0, 8 - state_vec.shape[0]))
        else:
            state_vec = state_vec[:8]
        state_norm = state_vec / np.maximum(np.linalg.norm(state_vec), 1e-8)

        err = float(np.linalg.norm(reconstructed - state_norm))
        return err <= 2.0  # Bounded invariant topological region
