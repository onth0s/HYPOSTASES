"""Chess domain plugin for HYPOSTASES."""

from hypostases.plugins.domains.chess.chess_agent_adapter import ChessAgentAdapter
from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.plugins.domains.chess.chess_trainer import ChessSelfPlayTrainer
from hypostases.plugins.domains.chess.ground_a_self_play import GroundASelfPlay, PolicySnapshot
from hypostases.plugins.domains.chess.ground_b_stockfish import GroundBStockfish

__all__ = [
    "ChessAgentAdapter",
    "ChessDomain",
    "ChessSelfPlayTrainer",
    "GroundASelfPlay",
    "GroundBStockfish",
    "PolicySnapshot",
]
