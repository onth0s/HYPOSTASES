"""HYPOSTASES — Gärdenfors Conceptual Spaces & Metric Semantics Engine.

Spec Ref: Bechberger & Kühnberger (2017), Gärdenfors (2000, 2014).
Provides metric quality dimensions, Mahalanobis distance calculation, exponential similarity,
and Voronoi convex region categorization.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hypostases.world_model.hierarchical_types import ConceptualRegion, QualityDimension


class ConceptualSpaceEngine:
    r"""Engine executing geometric semantics over quality domain Omega \subset R^D."""

    def __init__(
        self,
        dimensions: list[QualityDimension],
        regions: list[ConceptualRegion],
    ) -> None:
        self.dimensions = {d.name: d for d in dimensions}
        self.regions = {r.id: r for r in regions}
        self._precisions: dict[str, np.ndarray] = {}
        self._precompute_precision_matrices()

    def _precompute_precision_matrices(self) -> None:
        """Precomputes inverse covariance (precision) matrices for fast Mahalanobis lookup."""
        for r_id, region in self.regions.items():
            cov = region.covariance_matrix
            # Ensure pseudo-inverse for stability if singular
            try:
                prec = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                prec = np.linalg.pinv(cov)
            self._precisions[r_id] = prec

    def calculate_mahalanobis_distance(self, x: np.ndarray, region_id: str) -> float:
        """Computes Mahalanobis distance d_M(x, mu_k) = sqrt((x - mu_k)^T M^-1 (x - mu_k))."""
        if region_id not in self.regions:
            raise KeyError(f"Conceptual region '{region_id}' not found.")

        region = self.regions[region_id]
        diff = x - region.prototype
        prec = self._precisions[region_id]
        dist_sq = float(np.dot(diff.T, np.dot(prec, diff)))
        return float(np.sqrt(max(0.0, dist_sq)))

    def calculate_similarity(self, x: np.ndarray, region_id: str) -> float:
        """Computes exponential similarity Sim(x, mu_k) = exp(-gamma * d_M(x, mu_k))."""
        region = self.regions[region_id]
        dist = self.calculate_mahalanobis_distance(x, region_id)
        return float(np.exp(-region.gamma * dist))

    def categorize_point(self, x: np.ndarray) -> dict[str, Any]:
        """Categorizes quality vector x into nearest convex Voronoi region C_k in O(1) time.

        Returns:
            dict containing closest_region_id, label, distance, similarity, and all_similarities.
        """
        best_region_id: str | None = None
        min_distance = float("inf")
        similarities: dict[str, float] = {}

        for r_id in self.regions:
            dist = self.calculate_mahalanobis_distance(x, r_id)
            sim = self.calculate_similarity(x, r_id)
            similarities[r_id] = sim

            if dist < min_distance:
                min_distance = dist
                best_region_id = r_id

        if best_region_id is None:
            raise ValueError("No conceptual regions registered in ConceptualSpaceEngine.")

        best_region = self.regions[best_region_id]
        return {
            "closest_region_id": best_region_id,
            "label": best_region.label,
            "distance": min_distance,
            "similarity": similarities[best_region_id],
            "all_similarities": similarities,
        }
