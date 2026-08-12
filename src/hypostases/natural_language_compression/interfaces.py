"""Interface adapters connecting Front 14 to Front 11 Abduction and Front 05 Governance."""

from __future__ import annotations

import math
from typing import Any

from hypostases.natural_language_compression.engine import SymbolicCompressionEngine


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

        return {
            "treaty_id": treaty_id,
            "rules": rules,
            "punish_cost": cost,
        }
