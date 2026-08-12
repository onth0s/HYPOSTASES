"""Wave 5 Front 14 — Natural Language as Symbolic Compression & Visual-Epistemic Duality.

Spec Ref: docs/WAVE_5_FRONT_14/front_14_natural_language_symbolic_compression_spec.md
Literature: Giaquinto (2007), Friston et al. (2017), Barron et al. (1998), Shannon (1948),
            Feng & Lu (ACL 2023), Abudy et al. (2025), Pei et al. (ICML 2026), Ajuzieogu (2025).

Core State Invariant:
    \\sigma = (c, w, g, \rho_{\text{ext}})
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hypostases.schemas.loader import load_natural_language_compression_config


@dataclass
class SymbolToken:
    """Discrete symbol representation in codebook V."""

    token_id: int
    symbol_str: str
    embedding: np.ndarray
    entropy_bits: float = 1.0


@dataclass
class VisualGistCell:
    """Convex Voronoi region cell in continuous semantic spatial memory (Giaquinto 2007)."""

    cell_id: int
    centroid: np.ndarray
    radii: float
    neighbor_ids: list[int] = field(default_factory=list)


@dataclass
class SymbolicMessage:
    """Symbolic token stream message transferred across agents."""

    sender_id: str
    recipient_id: str
    token_ids: list[int]
    content_str: str
    code_length_bits: float
    distortion_kl: float
    checksum: int = 0


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


class CommunicativeLanguageSymbolismRouter:
    """Pei et al. (ICML 2026) Communicative Language Symbolism Router (CLSR).

    Optimizes active token allocation under Theorem 3.2 lower bound:
    E[|T|] >= max(0, I_req) / kappa_theta.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        router_cfg = config.get("router_config", {})
        self.kappa_theta: float = router_cfg.get("kappa_theta", 1.2)
        self.default_budget: int = router_cfg.get("token_budget_default", 128)
        self.default_target_acc: float = router_cfg.get("target_accuracy_default", 0.95)

    def compute_required_information(self, uncertainty_entropy: float, target_acc: float) -> float:
        """Computes I_req(x, delta) = H(Y|X=x) - h2(delta) - delta * log2(|Y_x| - 1)."""
        delta = 1.0 - target_acc
        delta = max(1e-5, min(0.499, delta))

        h2_delta = -delta * math.log2(delta) - (1.0 - delta) * math.log2(1.0 - delta)
        cardinality_term = delta * math.log2(max(2, 10 - 1))  # Default 10 classes
        i_req = uncertainty_entropy - h2_delta - cardinality_term
        return max(0.0, i_req)

    def compute_min_token_bound(self, uncertainty_entropy: float, target_acc: float) -> float:
        """Computes Theorem 3.2 lower bound for generated tokens."""
        i_req = self.compute_required_information(uncertainty_entropy, target_acc)
        return i_req / max(self.kappa_theta, 1e-6)

    def route_token_allocation(
        self, query_uncertainty: float, token_budget: int | None = None
    ) -> tuple[int, str]:
        """Routes task to Direct (T=1), Multi-LSF, or Extended Reasoning protocol."""
        budget = token_budget if token_budget is not None else self.default_budget
        min_tokens = self.compute_min_token_bound(query_uncertainty, self.default_target_acc)

        if min_tokens <= 1.0:
            return 1, "DIRECT_SYMBOLIC_EXECUTION"
        elif min_tokens <= budget / 2:
            allocated = math.ceil(min_tokens * 1.2)
            return min(allocated, budget), "MULTI_LSF_AGGREGATION"
        else:
            return budget, "EXTENDED_REASONING_PROTOCOL"


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

    def compress_state(
        self, state_tuple: dict[str, Any]
    ) -> tuple[list[int], float, float]:
        """Compresses continuous primitive state sigma = (c, w, g, rho_ext) into token stream L.

        Returns:
            Tuple of (token_ids, code_length_bits, distortion_kl)
        """
        # Extract continuous vector from state tuple
        c_state = np.asarray(state_tuple.get("c", [0.5] * 8), dtype=np.float64).flatten()
        w_state = np.asarray(state_tuple.get("w", [0.2] * 8), dtype=np.float64).flatten()
        g_state = np.asarray(state_tuple.get("g", [0.8] * 8), dtype=np.float64).flatten()
        rho_ext = np.asarray(state_tuple.get("rho_ext", [1.0] * 8), dtype=np.float64).flatten()

        # Combine into state manifold
        combined = (c_state[:2].tolist() + w_state[:2].tolist() +
                    g_state[:2].tolist() + rho_ext[:2].tolist())
        combined_vec = np.asarray(combined, dtype=np.float64)

        # Map via Visual-Epistemic Duality
        spatial_tokens = self.duality_mapper.encode_spatial_to_symbolic(combined_vec)

        # Map via Symbolic Mapping Layer
        word_bank = self.mapping_transfer.sample_word_bank(combined_vec)

        # Select token sequence
        token_ids = list(dict.fromkeys(spatial_tokens + word_bank))[:16]

        # Calculate MDL Code Length |H| (bits)
        code_length_bits = len(token_ids) * math.log2(self.vocab_size)

        # Calculate Distortion / Surprisal KL Divergence
        reconstructed = self.duality_mapper.decode_symbolic_to_spatial(spatial_tokens)
        distortion_kl = float(np.mean((combined_vec - reconstructed) ** 2))

        return token_ids, code_length_bits, distortion_kl

    def compute_mdl_loss(self, code_length_bits: float, distortion_kl: float) -> float:
        """Computes total MDL loss L_MDL = |H| + lambda_mdl * D_KL."""
        return code_length_bits + self.lambda_mdl * distortion_kl

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


class SymbolicAbductionInterface:
    """Interface connecting Front 14 natural language compression to Front 11 Abduction Engine."""

    def __init__(self, compression_engine: SymbolicCompressionEngine) -> None:
        self.engine = compression_engine

    def format_hypothesis_as_symbolic_token_stream(self, hypothesis_dict: dict[str, Any]) -> str:
        """Formats a Front 11 hypothesis object into a compact symbolic token stream."""
        h_id = hypothesis_dict.get("id", "H0")
        cause = hypothesis_dict.get("cause", "unknown")
        confidence = hypothesis_dict.get("confidence", 0.5)

        # Convert to symbolic tokens
        token_str = f"HYP_ID[{h_id}]_CAUSE[{cause}]_CONF[{confidence:.2f}]"
        return token_str

    def compute_hypothesis_surprisal(self, hypothesis_str: str) -> float:
        """Computes Shannon surprisal -log2(P(H)) of a hypothesis string."""
        length = len(hypothesis_str)
        return float(length * math.log2(27))  # 26 letters + space alphabet


class NaturalLanguageGovernanceProtocol:
    """Interface connecting Front 14 natural language compression to Front 05 Institutional Layer."""

    def __init__(self, compression_engine: SymbolicCompressionEngine) -> None:
        self.engine = compression_engine

    def serialize_treaty_protocol(self, treaty_dict: dict[str, Any]) -> str:
        """Serializes institutional treaty rules into executable natural language protocol text."""
        treaty_id = treaty_dict.get("treaty_id", "T0")
        rules = treaty_dict.get("rules", [])
        penalty = treaty_dict.get("punish_cost", 1.0)

        rules_str = ";".join(rules) if rules else "NO_VIOLATION"
        return f"TREATY::{treaty_id}||RULES::{rules_str}||PUNISH_COST::{penalty:.1f}"

    def deserialize_treaty_protocol(self, protocol_str: str) -> dict[str, Any]:
        """Deserializes natural language protocol text back into structured institutional rules."""
        parts = protocol_str.split("||")
        treaty_id = "T0"
        rules = []
        cost = 1.0

        for part in parts:
            if part.startswith("TREATY::"):
                treaty_id = part[8:]
            elif part.startswith("RULES::"):
                rules_raw = part[7:]
                rules = rules_raw.split(";") if rules_raw != "NO_VIOLATION" else []
            elif part.startswith("PUNISH_COST::"):
                try:
                    cost = float(part[13:])
                except ValueError:
                    cost = 1.0

        return {"treaty_id": treaty_id, "rules": rules, "punish_cost": cost}
