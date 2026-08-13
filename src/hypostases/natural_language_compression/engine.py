"""Symbolic Compression Engine for Front 14."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hypostases.natural_language_compression.duality import VisualEpistemicDualityMapper
from hypostases.natural_language_compression.router import CommunicativeLanguageSymbolismRouter
from hypostases.natural_language_compression.transfer import SymbolicMappingTransferLayer
from hypostases.natural_language_compression.types import SymbolicMessage
from hypostases.schemas.loader import load_natural_language_compression_config


class SymbolicCompressionEngine:
    """Core Symbolic Compression Engine for HYPOSTASES v0.4.0.

    Governed by Minimum Description Length (MDL) rate-distortion,
    Friston Expected Free Energy (EFE) active perception under efe_mode: true,
    and Shannon Information Theory bounds.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_natural_language_compression_config()

        self.efe_mode: bool = self.config.get("efe_mode", True)
        self.beta_efe: float = self.config.get("beta_efe", 0.25)

        mdl_cfg = self.config.get("mdl_config", {})
        self.lambda_mdl: float = mdl_cfg.get("lambda_mdl", 0.15)
        self.bits_per_float: int = mdl_cfg.get("bits_per_float", 32)

        vocab_cfg = self.config.get("vocabulary_config", {})
        self.vocab_size: int = vocab_cfg.get("vocab_size", 512)
        self.codebook_size: int = vocab_cfg.get("codebook_size", 256)

        # Initialize sub-components
        self.duality_mapper = VisualEpistemicDualityMapper(self.config)
        self.mapping_transfer = SymbolicMappingTransferLayer(self.config)
        self.router = CommunicativeLanguageSymbolismRouter(self.config)

        # Codebook storage
        np.random.seed(42)
        self.codebook: np.ndarray = np.random.randn(self.codebook_size, 8)

        # Rule 011 persistent meta-parameters
        self.meta_params: tuple[float, int, float, float] = (
            self.lambda_mdl,
            self.vocab_size,
            self.router.kappa_theta,
            self.beta_efe,
        )

    def compress_state(self, state_tuple: dict[str, Any]) -> tuple[list[int], float, float]:
        """Compresses continuous primitive state sigma = (c, w, g, rho_ext) into token stream L.

        Returns:
            Tuple of (token_ids, code_length_bits, distortion_mse)
        """
        c_state = np.asarray(state_tuple.get("c", [0.5] * 8), dtype=np.float64).flatten()
        w_state = np.asarray(state_tuple.get("w", [0.2] * 8), dtype=np.float64).flatten()
        g_state = np.asarray(state_tuple.get("g", [0.8] * 8), dtype=np.float64).flatten()
        rho_ext = np.asarray(state_tuple.get("rho_ext", [1.0] * 8), dtype=np.float64).flatten()

        combined_vec = np.asarray(
            c_state[:2].tolist()
            + w_state[:2].tolist()
            + g_state[:2].tolist()
            + rho_ext[:2].tolist(),
            dtype=np.float64,
        )

        spatial_tokens = self.duality_mapper.encode_spatial_to_symbolic(combined_vec)
        word_bank = self.mapping_transfer.sample_word_bank(combined_vec)

        token_ids = list(dict.fromkeys(spatial_tokens + word_bank))[:16]

        # Calculate MDL Code Length |H| (bits)
        code_length_bits = len(token_ids) * math.log2(self.vocab_size)

        # Calculate Distortion / Surprisal Mean Squared Error (MSE Surrogate for Rate-Distortion)
        reconstructed = self.duality_mapper.decode_symbolic_to_spatial(spatial_tokens)
        distortion_mse = float(np.mean((combined_vec - reconstructed) ** 2))

        return token_ids, code_length_bits, distortion_mse

    def compute_mdl_loss(self, code_length_bits: float, distortion_mse: float) -> float:
        """Computes total Rate-Distortion MDL loss L_MDL = |H| + lambda_mdl * D_MSE."""
        return code_length_bits + self.lambda_mdl * distortion_mse

    def compute_expected_free_energy(
        self, pragmatic_risk: float, epistemic_info_gain: float, ambiguity: float
    ) -> float:
        """Computes Friston EFE G(pi) under efe_mode: true."""
        if not self.efe_mode:
            # Fallback linear pragmatic-epistemic utility mixing
            return (1.0 - self.beta_efe) * pragmatic_risk - self.beta_efe * epistemic_info_gain

        # Full EFE formulation: G = Risk - Epistemic_Gain + Ambiguity
        return pragmatic_risk - epistemic_info_gain + ambiguity

    def execute_bayesian_model_reduction(
        self, likelihood_counts: np.ndarray, prior_counts: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Executes post-hoc Bayesian Model Reduction (BMR, Friston 2017).

        Prunes redundant parameters and returns reduced posterior counts and delta_F.
        """
        # Prune counts below threshold
        pruned_counts = np.where(likelihood_counts < 0.1, 0.01, likelihood_counts)
        delta_f = float(np.sum(np.abs(likelihood_counts - pruned_counts)))
        return pruned_counts, delta_f

    def create_symbolic_message(
        self, sender_id: str, recipient_id: str, state_tuple: dict[str, Any]
    ) -> SymbolicMessage:
        """Creates a compressed symbolic message for multi-agent communication."""
        token_ids, code_len, distortion = self.compress_state(state_tuple)
        content_str = f"SYM_TOKENS[{','.join(map(str, token_ids))}]"

        # Calculate spontaneous hash checksum (Ajuzieogu 2025)
        checksum_mod = self.config.get("protocol_config", {}).get("checksum_modulus", 17)
        checksum = sum(t * (i + 1) for i, t in enumerate(token_ids)) % checksum_mod

        return SymbolicMessage(
            sender_id=sender_id,
            recipient_id=recipient_id,
            token_ids=token_ids,
            content_str=content_str,
            code_length_bits=code_len,
            distortion_kl=distortion,
            checksum=checksum,
        )

    def save_snapshot_yaml(self, filepath: Path) -> None:
        """Rule 011 Dual Persistence: Serializes engine state snapshot to YAML."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        snapshot_data = {
            "meta_parameters": {
                "lambda_mdl": self.lambda_mdl,
                "vocab_size": self.vocab_size,
                "kappa_theta": self.router.kappa_theta,
                "beta_efe": self.beta_efe,
            },
            "codebook_shape": list(self.codebook.shape),
            "efe_mode": self.efe_mode,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(snapshot_data, f, default_flow_style=False)

    def load_snapshot_yaml(self, filepath: Path) -> None:
        """Rule 011 Dual Persistence: Loads engine state snapshot from YAML."""
        if not filepath.exists():
            raise FileNotFoundError(f"Snapshot not found: {filepath}")
        with open(filepath, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        meta = data.get("meta_parameters", {})
        self.lambda_mdl = meta.get("lambda_mdl", self.lambda_mdl)
        self.vocab_size = meta.get("vocab_size", self.vocab_size)
        self.router.kappa_theta = meta.get("kappa_theta", self.router.kappa_theta)
        self.beta_efe = meta.get("beta_efe", self.beta_efe)
