"""Vocabulary Bootstrapping & Data-Derived Lexicon Mapper for Front 14.

Spec Ref: docs/State-Vectors-to-Natural-Language/state_vectors_to_natural_language_plan.md
Step 1: Stage 0 Seed Corpus & Stage 1+ Outcome-Correlated Clustering Lexicon.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hypostases.schemas.loader import load_nlp_decoder_config


class DataDerivedLexiconMapper:
    """Stage 0 (Seed) -> Stage 1+ (Outcome-Correlated) VQ Lexicon Mapper.

    Maps discrete VQ codebook slots k in K_codebook to semantic tokens
    without hardcoded human cognitive deficiency hacks.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_nlp_decoder_config()
        lex_cfg = self.config.get("lexicon_config", {})
        self.seed_vocab_size: int = lex_cfg.get("seed_vocab_size", 256)
        self.codebook_size: int = lex_cfg.get("codebook_size", 128)
        self.correlation_threshold: float = lex_cfg.get("outcome_correlation_threshold", 0.15)

        # Stage 0 Seed Vocabulary
        self.seed_vocab: list[str] = [f"token_{i}" for i in range(self.seed_vocab_size)]
        # Assign meaningful semantic prefixes to initial seed tokens for readability
        domain_prefixes = [
            "cognition",
            "world_model",
            "goal_utility",
            "power_ext",
            "epistemic_gain",
            "pragmatic_risk",
            "causal_link",
            "boundary",
        ]
        for i, prefix in enumerate(domain_prefixes):
            if i < self.seed_vocab_size:
                self.seed_vocab[i] = prefix

        # Stage 1+ Outcome Correlation Matrix: shape (codebook_size, seed_vocab_size)
        # Seeded deterministically for reproduciblity
        rng = np.random.default_rng(42)
        self.outcome_correlation_matrix: np.ndarray = rng.uniform(
            0.0, 1.0, size=(self.codebook_size, self.seed_vocab_size)
        )
        # Normalize rows to represent conditional distributions P(Delta Outcome | Cluster k)
        row_sums = self.outcome_correlation_matrix.sum(axis=1, keepdims=True)
        self.outcome_correlation_matrix /= np.maximum(row_sums, 1e-12)

    def map_cluster_to_token(self, cluster_idx: int) -> str:
        """LexiconToken(k) = argmax_{t in V} P(Delta Outcome | sigma in Cluster(k))."""
        cluster_k = cluster_idx % self.codebook_size
        token_idx = int(np.argmax(self.outcome_correlation_matrix[cluster_k]))
        return self.seed_vocab[token_idx]

    def map_state_vector_to_clusters(self, vector: np.ndarray) -> list[int]:
        """Maps a continuous state vector to discrete codebook cluster indices."""
        vec = np.asarray(vector, dtype=np.float64).flatten()
        # Quantize vector into codebook slots
        slots = []
        chunk_size = max(1, len(vec) // 4)
        for i in range(0, len(vec), chunk_size):
            chunk = vec[i : i + chunk_size]
            val = float(np.mean(chunk))
            idx = int(abs(val * 100)) % self.codebook_size
            slots.append(idx)
        return slots or [0]

    def update_outcome_correlation(
        self, cluster_idx: int, token_idx: int, delta_outcome: float
    ) -> None:
        """Updates outcome correlation table online using trace logs."""
        k = cluster_idx % self.codebook_size
        t = token_idx % self.seed_vocab_size
        lr = 0.05
        self.outcome_correlation_matrix[k, t] += lr * (
            delta_outcome - self.outcome_correlation_matrix[k, t]
        )
        # Re-normalize row
        self.outcome_correlation_matrix[k] = np.maximum(0.0, self.outcome_correlation_matrix[k])
        s = self.outcome_correlation_matrix[k].sum()
        if s > 0:
            self.outcome_correlation_matrix[k] /= s


class ConceptCompositionEngine:
    """Assembles discrete codebook tokens into compositional semantic feature tuples."""

    def __init__(self, mapper: DataDerivedLexiconMapper | None = None) -> None:
        self.mapper = mapper or DataDerivedLexiconMapper()

    def compose_state_concepts(self, state: dict[str, Any]) -> dict[str, Any]:
        """Extracts compositional semantic feature tuples from sigma = (c, w, g, rho_ext)."""
        c_vec = np.asarray(state.get("c", [0.5] * 4), dtype=np.float64)
        w_vec = np.asarray(state.get("w", [0.2] * 4), dtype=np.float64)
        g_vec = np.asarray(state.get("g", [0.8] * 4), dtype=np.float64)
        rho_vec = np.asarray(state.get("rho_ext", [1.0] * 4), dtype=np.float64)

        c_clusters = self.mapper.map_state_vector_to_clusters(c_vec)
        w_clusters = self.mapper.map_state_vector_to_clusters(w_vec)
        g_clusters = self.mapper.map_state_vector_to_clusters(g_vec)
        rho_clusters = self.mapper.map_state_vector_to_clusters(rho_vec)

        subject = self.mapper.map_cluster_to_token(c_clusters[0])
        predicate = self.mapper.map_cluster_to_token(w_clusters[0])
        goal = self.mapper.map_cluster_to_token(g_clusters[0])
        power = self.mapper.map_cluster_to_token(rho_clusters[0])

        return {
            "compositional_tuple": (subject, predicate, goal, power),
            "clusters": {
                "c": c_clusters,
                "w": w_clusters,
                "g": g_clusters,
                "rho_ext": rho_clusters,
            },
            "tokens": [subject, predicate, goal, power],
        }
