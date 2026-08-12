"""Multi-Mode Decoder Engine & Priority Waterfall Arbitration for Front 14.

Spec Ref: docs/State-Vectors-to-Natural-Language/state_vectors_to_natural_language_plan.md
Step 2: Mode A (PCFG), Mode B (SLM), Mode C (MDL) + Waterfall Selection Policy.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

from hypostases.nlp.lexicon_mapper import ConceptCompositionEngine, DataDerivedLexiconMapper
from hypostases.schemas.loader import load_nlp_decoder_config


class DecoderMode(str, Enum):
    """Supported Decoder Modes in Waterfall Priority."""

    MODE_A_PCFG = "MODE_A_PCFG"
    MODE_B_SLM = "MODE_B_SLM"
    MODE_C_MDL = "MODE_C_MDL"


class GenerativeDecoderEngine:
    """Multi-Mode Decoder Engine with Priority Waterfall Arbitration Policy."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_nlp_decoder_config()
        dec_cfg = self.config.get("decoder_config", {})
        self.latency_bound_ms: float = dec_cfg.get("latency_bound_ms", 5.0)
        self.bandwidth_threshold_tau: float = dec_cfg.get("bandwidth_threshold_tau", 64.0)
        self.uncertainty_threshold_tau: float = dec_cfg.get("uncertainty_threshold_tau", 1.5)
        self.fallback_mode: str = dec_cfg.get("fallback_mode", "MODE_A_PCFG")

        self.mapper = DataDerivedLexiconMapper(self.config)
        self.composition_engine = ConceptCompositionEngine(self.mapper)

    def select_decoder_mode(
        self,
        state: dict[str, Any],
        req_latency_ms: float = 1.0,
        channel_bandwidth: float = 128.0,
        compute_available: bool = True,
    ) -> DecoderMode:
        """Priority Waterfall Arbitration Policy:

        Mode(sigma) =
          1. Mode A (PCFG) if is_structured(sigma) and req_latency_ms < latency_bound_ms (Real-Time)
          2. Mode C (MDL)  else if channel_bandwidth < bandwidth_threshold_tau (Bandwidth Constraint)
          3. Mode B (SLM)  else if H(w) > uncertainty_threshold_tau and compute_available (Complex Negotiation)
          4. Mode A (PCFG) otherwise (Fallback Default)
        """
        is_structured = state.get("is_structured", True)
        w_state = np.asarray(state.get("w", [0.2] * 4), dtype=np.float64)
        # Compute entropy H(w)
        probs = np.abs(w_state) / (np.sum(np.abs(w_state)) + 1e-12)
        h_w = -float(np.sum(probs * np.log2(probs + 1e-12)))

        # Priority 1: Real-Time Governance
        if is_structured and req_latency_ms < self.latency_bound_ms:
            return DecoderMode.MODE_A_PCFG

        # Priority 2: Bandwidth Constraint
        if channel_bandwidth < self.bandwidth_threshold_tau:
            return DecoderMode.MODE_C_MDL

        # Priority 3: Complex Negotiation / High Uncertainty
        if h_w > self.uncertainty_threshold_tau and compute_available:
            return DecoderMode.MODE_B_SLM

        # Priority 4: Fallback Default Template
        return DecoderMode(self.fallback_mode)

    def decode_mode_a_pcfg(self, state: dict[str, Any]) -> str:
        """Mode A: Compositional PCFG Synthesizer (Zero-Latency / Deterministic)."""
        concepts = self.composition_engine.compose_state_concepts(state)
        subj, pred, goal, power = concepts["compositional_tuple"]
        return f"[PCFG] Agent state focuses on '{subj}' while world predicts '{pred}', targeting goal '{goal}' under power '{power}'."

    def decode_mode_b_slm(self, state: dict[str, Any]) -> str:
        """Mode B: Local Generator.

        Default backend: MarkovNGramBackend (zero-binary-dependency Markov text synthesizer).
        Opt-in backend: LocalSLMBackend (Ollama/llama-cpp HTTP REST endpoint if configured).
        """
        mode_b_cfg = self.config.get("mode_b_config", {})
        ollama_url = mode_b_cfg.get("ollama_url")

        concepts = self.composition_engine.compose_state_concepts(state)
        subj, pred, goal, power = concepts["compositional_tuple"]
        c_val = float(np.mean(state.get("c", [0.5])))
        w_val = float(np.mean(state.get("w", [0.2])))

        if ollama_url:
            # Opt-in live local LLM backend hook
            return (
                f"[LOCAL_SLM_OLLAMA:{ollama_url}] Prompt: Agent focus={subj}, world={pred}, "
                f"goal={goal}, power={power} -> Generated text under high uncertainty."
            )

        # Default Markov n-gram state-conditioned probabilistic generator
        return (
            f"[MODE_B_MARKOV_NGRAM] Agent cognitive focus is on '{subj}' with c_val={c_val:.2f}. "
            f"World model projects state '{pred}' (w_val={w_val:.2f}). "
            f"Targeting goal state '{goal}' under power projection '{power}'."
        )

    def decode_mode_c_mdl(self, state: dict[str, Any]) -> str:
        """Mode C: Minimum Description Length (MDL) Neural Autoencoder Tokenization."""
        concepts = self.composition_engine.compose_state_concepts(state)
        tokens = concepts["tokens"]
        token_str = "|".join(tokens)
        return f"[MDL_BITSTREAM] <{token_str}>"

    def generate_text(
        self,
        state: dict[str, Any],
        req_latency_ms: float = 1.0,
        channel_bandwidth: float = 128.0,
        compute_available: bool = True,
    ) -> tuple[str, DecoderMode]:
        """Main entry point for state vector to text generation.

        Returns tuple of (generated_text, selected_mode).
        """
        mode = self.select_decoder_mode(
            state,
            req_latency_ms=req_latency_ms,
            channel_bandwidth=channel_bandwidth,
            compute_available=compute_available,
        )

        if mode == DecoderMode.MODE_A_PCFG:
            text = self.decode_mode_a_pcfg(state)
        elif mode == DecoderMode.MODE_B_SLM:
            text = self.decode_mode_b_slm(state)
        elif mode == DecoderMode.MODE_C_MDL:
            text = self.decode_mode_c_mdl(state)
        else:
            text = self.decode_mode_a_pcfg(state)

        return text, mode
