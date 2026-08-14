"""Chess domain plugin for HYPOSTASES."""

from hypostases.plugins.domains.chess.chess_agent_adapter import ChessAgentAdapter
from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.plugins.domains.chess.chess_trainer import ChessSelfPlayTrainer
from hypostases.plugins.domains.chess.ground_a_self_play import GroundASelfPlay, PolicySnapshot
from hypostases.plugins.domains.chess.ground_b_stockfish import GroundBStockfish
from hypostases.plugins.domains.chess.nnue_net import Accumulator, NNUENet, extract_halfkp_features
from hypostases.plugins.domains.chess.nnue_training import train_nnue

__all__ = [
    "Accumulator",
    "ChessAgentAdapter",
    "ChessDomain",
    "ChessSelfPlayTrainer",
    "GroundASelfPlay",
    "GroundBStockfish",
    "NNUENet",
    "PolicySnapshot",
    "extract_halfkp_features",
    "train_nnue",
]
