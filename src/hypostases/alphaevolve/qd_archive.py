"""Quality-Diversity (MAP-Elites) & Behavioral Novelty Archives for AlphaEvolve Engine.

References:
- Mouret & Clune (2015) Illuminating search spaces by mapping elites (Evolutionary Computation)
- Lehman & Stanley (2011) Abandoning Objectives: Evolution Through the Search for Novelty Alone (Evolutionary Computation)
"""

from typing import Any

import numpy as np


class MAPElitesArchive:
    """Quality-Diversity (QD) MAP-Elites Grid Archive (Mouret & Clune 2015).

    Maintains an N-dimensional grid discretized across behavioral feature spaces
    (Time Complexity, IC Regret, Scarcity Delta), storing elite AST solutions per bin.
    """

    def __init__(
        self,
        grid_dims: tuple[int, ...] = (10, 10, 10),
        feature_bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        self.grid_dims = grid_dims
        self.num_dims = len(grid_dims)
        self.feature_bounds = feature_bounds or [
            (0.0, 100.0),  # C_time (character length / computational steps)
            (0.0, 1.0),  # R_IC (Incentive Compatibility Regret)
            (-5.0, 5.0),  # Delta kappa (Scarcity impact)
        ]

        # Multi-dimensional grid storing elite solution dictionary and fitness score
        self.scores: np.ndarray = np.full(grid_dims, -np.inf, dtype=float)
        self.solutions: dict[tuple[int, ...], dict[str, Any]] = {}
        self.total_cells: int = int(np.prod(grid_dims))

    def compute_cell_index(self, behavior_vector: np.ndarray) -> tuple[int, ...]:
        """Compute discretized cell coordinate vector z(x) = (z_1, z_2, ..., z_N)."""
        coords: list[int] = []
        for i in range(self.num_dims):
            val = float(behavior_vector[i])
            b_min, b_max = self.feature_bounds[i]
            k_bins = self.grid_dims[i]
            # Normalize to [0, 1]
            norm_val = (val - b_min) / (b_max - b_min + 1e-8)
            norm_val = np.clip(norm_val, 0.0, 1.0 - 1e-8)
            bin_idx = int(np.floor(norm_val * k_bins))
            bin_idx = min(k_bins - 1, max(0, bin_idx))
            coords.append(bin_idx)
        return tuple(coords)

    def add_candidate(
        self,
        code_str: str,
        fitness_score: float,
        behavior_vector: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update archive cell if cell is empty or if candidate fitness exceeds current elite."""
        cell_coords = self.compute_cell_index(behavior_vector)
        current_score = float(self.scores[cell_coords])

        if np.isinf(current_score) or fitness_score > current_score:
            self.scores[cell_coords] = fitness_score
            candidate_entry = {
                "code": code_str,
                "score": fitness_score,
                "behavior": behavior_vector.copy(),
                "coords": cell_coords,
                "metadata": metadata or {},
            }
            self.solutions[cell_coords] = candidate_entry
            return True
        return False

    def coverage(self) -> float:
        """Compute grid coverage ratio: n(filled) / n(total)."""
        filled_count = len(self.solutions)
        return float(filled_count / self.total_cells)

    def precision(self, global_max_scores: np.ndarray | None = None) -> float:
        """Compute MAP-Elites precision metric P(m)."""
        if not self.solutions:
            return 0.0
        scores_arr = np.array([sol["score"] for sol in self.solutions.values()])
        if global_max_scores is not None:
            filled_indices = tuple(zip(*self.solutions.keys(), strict=False))
            max_ref = global_max_scores[filled_indices]
            max_ref = np.maximum(max_ref, 1e-6)
            return float(np.mean(scores_arr / max_ref))
        return float(np.mean(scores_arr))

    def sample_elites(self, n_samples: int = 2) -> list[dict[str, Any]]:
        """Sample random elite candidates from non-empty archive cells."""
        if not self.solutions:
            return []
        keys = list(self.solutions.keys())
        sampled_keys = np.random.choice(len(keys), size=min(n_samples, len(keys)), replace=False)
        return [self.solutions[keys[idx]] for idx in sampled_keys]


class NoveltyArchive:
    """Behavioral Novelty Archive & Distance Estimator (Lehman & Stanley 2011).

    Computes k-nearest neighbor novelty distance rho(x, S) over behavior space
    to maintain exploration diversity in deceptive search spaces.
    """

    def __init__(
        self,
        k_neighbors: int = 15,
        threshold: float = 0.25,
        max_archive_size: int = 1000,
    ) -> None:
        self.k_neighbors = k_neighbors
        self.threshold = threshold
        self.max_archive_size = max_archive_size
        self.archive: list[np.ndarray] = []

    def compute_novelty(
        self, behavior_vector: np.ndarray, population_behaviors: list[np.ndarray]
    ) -> float:
        """Compute k-NN novelty distance rho(x, S) relative to population and archive."""
        all_points = population_behaviors + self.archive
        if not all_points:
            return 0.0

        distances = [
            float(np.linalg.norm(behavior_vector - pt))
            for pt in all_points
            if not np.array_equal(behavior_vector, pt)
        ]
        if not distances:
            return 0.0

        distances.sort()
        k_eff = min(self.k_neighbors, len(distances))
        top_k = distances[:k_eff]
        return float(np.mean(top_k))

    def add_if_novel(
        self, behavior_vector: np.ndarray, population_behaviors: list[np.ndarray]
    ) -> bool:
        """Add candidate to novelty archive if its novelty score exceeds threshold."""
        novelty_score = self.compute_novelty(behavior_vector, population_behaviors)
        if novelty_score >= self.threshold or not self.archive:
            self.archive.append(behavior_vector.copy())
            if len(self.archive) > self.max_archive_size:
                self.archive.pop(0)
            return True
        return False
