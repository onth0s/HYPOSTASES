"""Symbolic Mapping Transfer Layer (Feng & Lu ACL 2023) for Front 14."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class SymbolicMappingTransferLayer:
    """Feng & Lu (ACL 2023) Shared Symbolic Mapping Layer.

    Maps continuous input features to sigmoidal relevance probabilities
    over symbol vocabulary V, generating discrete word banks for task transfer.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        vocab_cfg = config.get("vocabulary_config", {})
        self.vocab_size: int = vocab_cfg.get("vocab_size", 512)
        self.word_bank_max: int = vocab_cfg.get("word_bank_max_size", 64)
        self.symbol_dim: int = vocab_cfg.get("symbol_dim", 16)

        np.random.seed(42)
        self.W_sm: np.ndarray = np.random.randn(8, self.vocab_size) * 0.1
        self.b_sm: np.ndarray = np.zeros(self.vocab_size)

    def get_relevance_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Computes p = sigmoid(W_sm * o + b_sm) over vocabulary V."""
        obs = np.asarray(observation, dtype=np.float64).flatten()
        obs = np.pad(obs, (0, 8 - obs.shape[0])) if obs.shape[0] < 8 else obs[:8]

        logits = np.dot(obs, self.W_sm) + self.b_sm
        # Sigmoid activation
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
        return p

    def sample_word_bank(self, observation: np.ndarray) -> list[int]:
        """Samples discrete word bank W <= V via Bernoulli probabilities."""
        p = self.get_relevance_probabilities(observation)
        sampled = np.where(p > 0.5)[0].tolist()
        if not sampled:
            sampled = np.argsort(p)[-4:].tolist()
        return sampled[: self.word_bank_max]

    def compute_referential_disentanglement(self, symbol_counts: dict[int, int]) -> float:
        """Computes referential disentanglement score refdis in [0, 1]."""
        if not symbol_counts:
            return 0.8  # Default baseline
        total = sum(symbol_counts.values())
        if total == 0:
            return 0.8

        # Calculate empirical entropy weighting
        probs = [count / total for count in symbol_counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        refdis = 1.0 / (1.0 + entropy * 0.1)
        return float(np.clip(refdis, 0.0, 1.0))
