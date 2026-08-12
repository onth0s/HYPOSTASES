"""Wave 5 Front 13 — Evolutionary Algorithm Discovery (AlphaEvolve Engine).

Integrates evolutionary AST program search (AlphaEvolve / FunSearch), Quality-Diversity (MAP-Elites),
Behavioral Novelty Search, Regularized Aging Evolution, and Morphological Reservoir Co-Evolution
into the HYPOSTASES state substrate sigma = (c, w, g, rho_ext).
"""

from hypostases.alphaevolve.engine import AlphaEvolveEngine
from hypostases.alphaevolve.evaluator import GameTheoreticOracleEvaluator
from hypostases.alphaevolve.mutator import ASTMutator, FECEvaluator
from hypostases.alphaevolve.qd_archive import MAPElitesArchive, NoveltyArchive
from hypostases.alphaevolve.reservoir import MorphologicalReservoir

__all__ = [
    "ASTMutator",
    "AlphaEvolveEngine",
    "FECEvaluator",
    "GameTheoreticOracleEvaluator",
    "MAPElitesArchive",
    "MorphologicalReservoir",
    "NoveltyArchive",
]
