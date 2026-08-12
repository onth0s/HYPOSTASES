"""Dataclasses and symbolic token definitions for Natural Language Compression (Front 14)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
