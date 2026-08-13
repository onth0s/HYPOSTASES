"""HYPOSTASES-NNUE Supervised Training and Expanded Dataset Audit (D-006).

Spec Ref: scratch/DISSONANCES.md D-006 (Expanded Dataset Audit with Game Phase & Material Distributions)
"""

from __future__ import annotations

from pathlib import Path
import chess
import numpy as np

from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.world_model.nnue_net import NNUENet, extract_halfkp_features

DATASET_REPORT_PATH = Path("scratch/HYPOSTASES_NNUE_CONVERGENCE_AUDIT.md")

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


# Classical Piece-Square Tables (PST) for positional evaluation
PST_PAWN = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        10,
        10,
        20,
        30,
        30,
        20,
        10,
        10,
        5,
        5,
        10,
        25,
        25,
        10,
        5,
        5,
        0,
        0,
        0,
        20,
        20,
        0,
        0,
        0,
        5,
        -5,
        -10,
        0,
        0,
        -10,
        -5,
        5,
        5,
        10,
        10,
        -20,
        -20,
        10,
        10,
        5,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    dtype=np.float32,
)

PST_KNIGHT = np.array(
    [
        -50,
        -40,
        -30,
        -30,
        -30,
        -30,
        -40,
        -50,
        -40,
        -20,
        0,
        0,
        0,
        0,
        -20,
        -40,
        -30,
        0,
        10,
        15,
        15,
        10,
        0,
        -30,
        -30,
        5,
        15,
        20,
        20,
        15,
        5,
        -30,
        -30,
        0,
        15,
        20,
        20,
        15,
        0,
        -30,
        -30,
        5,
        10,
        15,
        15,
        10,
        5,
        -30,
        -40,
        -20,
        0,
        5,
        5,
        0,
        -20,
        -40,
        -50,
        -40,
        -30,
        -30,
        -30,
        -30,
        -40,
        -50,
    ],
    dtype=np.float32,
)

PST_BISHOP = np.array(
    [
        -20,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -20,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -10,
        0,
        5,
        10,
        10,
        5,
        0,
        -10,
        -10,
        5,
        5,
        10,
        10,
        5,
        5,
        -10,
        -10,
        0,
        10,
        10,
        10,
        10,
        0,
        -10,
        -10,
        10,
        10,
        10,
        10,
        10,
        10,
        -10,
        -10,
        5,
        0,
        0,
        0,
        0,
        5,
        -10,
        -20,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -20,
    ],
    dtype=np.float32,
)

PST_MAP = {
    chess.PAWN: PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
}


def bootstrap_eval(state: chess.Board) -> float:
    """Stage 0 bootstrap static evaluation (material + classical PST positional terms)."""
    if state.is_checkmate():
        return -100.0 if state.turn == chess.WHITE else 100.0
    if state.is_game_over():
        return 0.0

    score = 0.0
    for sq, piece in state.piece_map().items():
        val = PIECE_VALUES[piece.piece_type]
        pst_bonus = PST_MAP[piece.piece_type][sq] if piece.piece_type in PST_MAP else 0.0

        if piece.color == chess.WHITE:
            score += val + pst_bonus
        else:
            score -= val + pst_bonus

    perspective_score = score if state.turn == chess.WHITE else -score
    return float(perspective_score) / 100.0


def classify_game_phase(board: chess.Board) -> str:
    """Classifies game phase based on non-pawn material count."""
    non_pawn_material = 0
    for _sq, piece in board.piece_map().items():
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            non_pawn_material += 3
        elif piece.piece_type == chess.ROOK:
            non_pawn_material += 5
        elif piece.piece_type == chess.QUEEN:
            non_pawn_material += 9

    if non_pawn_material > 26:
        return "Opening"
    elif non_pawn_material > 12:
        return "Middlegame"
    else:
        return "Endgame"


def generate_and_audit_dataset(
    num_positions: int = 1000, seed: int = 42
) -> list[tuple[chess.Board, float]]:
    """Generates random legal board positions and outputs scratch/DATASET_REPORT.md artifact."""
    domain = ChessDomain()
    rng = np.random.default_rng(seed)
    dataset: list[tuple[chess.Board, float]] = []

    fens: set[str] = set()
    duplicate_count = 0
    white_turn_count = 0

    phase_counts = {"Opening": 0, "Middlegame": 0, "Endgame": 0}
    material_histogram = {"[0-10]": 0, "[11-20]": 0, "[21-30]": 0, "[31-40]": 0, "[41+]": 0}
    castling_counts = {"White_K": 0, "White_Q": 0, "Black_K": 0, "Black_Q": 0}
    ep_count = 0

    ply_list: list[int] = []
    check_positions = 0
    checkmate_positions = 0
    stalemate_positions = 0
    promotion_positions = 0

    board = domain.initial_state()
    current_ply = 0

    for _ in range(num_positions):
        if board.is_game_over() or len(dataset) >= num_positions:
            board = domain.initial_state()
            current_ply = 0

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            board = domain.initial_state()
            current_ply = 0
            continue

        move = rng.choice(legal_moves)
        board, _, done, _ = domain.step(board, move)
        current_ply += 1
        ply_list.append(current_ply)

        fen = board.fen()
        if fen in fens:
            duplicate_count += 1
        else:
            fens.add(fen)

        if board.turn == chess.WHITE:
            white_turn_count += 1

        phase = classify_game_phase(board)
        phase_counts[phase] += 1

        if board.is_check():
            check_positions += 1
        if board.is_checkmate():
            checkmate_positions += 1
        if board.is_stalemate():
            stalemate_positions += 1
        if move.promotion:
            promotion_positions += 1

        mat_pts = (
            sum(
                PIECE_VALUES[p.piece_type]
                for p in board.piece_map().values()
                if p.piece_type != chess.KING
            )
            // 100
        )
        if mat_pts <= 10:
            material_histogram["[0-10]"] += 1
        elif mat_pts <= 20:
            material_histogram["[11-20]"] += 1
        elif mat_pts <= 30:
            material_histogram["[21-30]"] += 1
        elif mat_pts <= 40:
            material_histogram["[31-40]"] += 1
        else:
            material_histogram["[41+]"] += 1

        if board.has_kingside_castling_rights(chess.WHITE):
            castling_counts["White_K"] += 1
        if board.has_queenside_castling_rights(chess.WHITE):
            castling_counts["White_Q"] += 1
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_counts["Black_K"] += 1
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_counts["Black_Q"] += 1

        if board.ep_square is not None:
            ep_count += 1

        label = bootstrap_eval(board)
        dataset.append((board.copy(), label))

    dup_pct = (duplicate_count / max(len(dataset), 1)) * 100.0
    white_pct = (white_turn_count / max(len(dataset), 1)) * 100.0

    mean_ply = float(np.mean(ply_list)) if ply_list else 0.0
    median_ply = float(np.median(ply_list)) if ply_list else 0.0
    std_ply = float(np.std(ply_list)) if ply_list else 0.0

    report_md = f"""# DATASET_REPORT.md

> **Development Dataset Audit** (Current Sample: 1,000 positions | Production Target: 500,000 positions)
> Spec Ref: DISSONANCES.md D-006

## Dataset Overview
- **Audit Mode**: Development Dataset Audit
- **Sampled Positions**: {len(dataset)}
- **Production Spec Target**: 500,000 positions
- **Unique Positions**: {len(fens)}
- **Duplicate Rate**: {dup_pct:.2f}%
- **Side-to-Move Balance**: {white_pct:.1f}% White / {100.0 - white_pct:.1f}% Black
- **En Passant Available Frequencies**: {ep_count} positions ({ep_count / len(dataset) * 100:.2f}%)

## Game Depth (Ply Metrics)
- **Mean Ply**: {mean_ply:.2f}
- **Median Ply**: {median_ply:.1f}
- **Std Dev Ply**: {std_ply:.2f}

## Game Phase Distribution (Material-Based Classification)
- **Opening** (>26 non-pawn pts): {phase_counts["Opening"]} positions ({phase_counts["Opening"] / len(dataset) * 100:.1f}%)
- **Middlegame** (13-26 pts): {phase_counts["Middlegame"]} positions ({phase_counts["Middlegame"] / len(dataset) * 100:.1f}%)
- **Endgame** (<=12 pts): {phase_counts["Endgame"]} positions ({phase_counts["Endgame"] / len(dataset) * 100:.1f}%)

## Tactical & Special Position Counts
- **In Check Positions**: {check_positions} ({check_positions / len(dataset) * 100:.2f}%)
- **Checkmate Positions**: {checkmate_positions}
- **Stalemate Positions**: {stalemate_positions}
- **Promotion Moves**: {promotion_positions}

## Total Material Point Histogram (Pawn-Equivalent Units)
- **[0-10 pts]**: {material_histogram["[0-10]"]}
- **[11-20 pts]**: {material_histogram["[11-20]"]}
- **[21-30 pts]**: {material_histogram["[21-30]"]}
- **[31-40 pts]**: {material_histogram["[31-40]"]}
- **[41+ pts]**: {material_histogram["[41+]"]}

## Castling Rights Distribution
- White Kingside (WK): {castling_counts["White_K"]}
- White Queenside (WQ): {castling_counts["White_Q"]}
- Black Kingside (BK): {castling_counts["Black_K"]}
- Black Queenside (BQ): {castling_counts["Black_Q"]}
"""
    DATASET_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATASET_REPORT_PATH.write_text(report_md, encoding="utf-8")

    return dataset


def train_nnue(
    net: NNUENet,
    dataset: list[tuple[chess.Board, float]],
    epochs: int = 5,
    lr: float = 0.01,
    verbose: bool = False,
) -> float:
    """Trains NNUENet end-to-end dense and sparse accumulator weights using SGD."""
    from rich.console import Console

    console = Console()

    final_loss = 0.0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for board, label in dataset:
            acc = net.create_accumulator(board)
            w_feats, b_feats, aux = extract_halfkp_features(board)
            is_white_turn = aux[0] == 1.0
            us_acc = net.clipped_relu(acc.white_acc if is_white_turn else acc.black_acc)
            them_acc = net.clipped_relu(acc.black_acc if is_white_turn else acc.white_acc)
            concat_input = np.concatenate([us_acc, them_acc, aux], axis=0)

            # ReLU activations
            z1 = np.dot(concat_input, net.W_l1) + net.b_l1
            h1 = net.clipped_relu(z1)
            pred = float(np.dot(h1, net.W_l2) + net.b_l2)

            err = pred - label
            total_loss += err**2

            # Gradients
            dL_dout = 2.0 * err
            dL_dW_l2 = dL_dout * h1[:, None]
            dL_db_l2 = np.array([dL_dout], dtype=np.float32)

            dL_dh1 = dL_dout * net.W_l2.flatten() * (h1 > 0)
            concat_input = np.concatenate([acc.white_acc, acc.black_acc, aux])
            dL_dW_l1 = np.outer(concat_input, dL_dh1)
            dL_db_l1 = dL_dh1

            # Backprop into accumulators
            dL_dconcat = np.dot(net.W_l1, dL_dh1)
            dL_dw_acc = dL_dconcat[:256]
            dL_db_acc = dL_dconcat[256:512]

            # Parameter updates with gradient clipping
            grad_clip = 1.0
            dL_dW_l2 = np.clip(dL_dW_l2, -grad_clip, grad_clip)
            dL_db_l2 = np.clip(dL_db_l2, -grad_clip, grad_clip)
            dL_dW_l1 = np.clip(dL_dW_l1, -grad_clip, grad_clip)
            dL_db_l1 = np.clip(dL_db_l1, -grad_clip, grad_clip)

            net.W_l2 -= lr * dL_dW_l2
            net.b_l2 -= lr * dL_db_l2
            net.W_l1 -= lr * dL_dW_l1
            net.b_l1 -= lr * dL_db_l1

            # Feature matrix updates with clipping
            dL_dw_acc = np.clip(dL_dw_acc, -grad_clip, grad_clip)
            dL_db_acc = np.clip(dL_db_acc, -grad_clip, grad_clip)
            for idx in w_feats:
                net.W_white[idx] -= lr * dL_dw_acc * 0.01
            for idx in b_feats:
                net.W_black[idx] -= lr * dL_db_acc * 0.01

        epoch_loss = total_loss / max(len(dataset), 1)
        final_loss = epoch_loss
        if verbose:
            console.print(
                f"  [cyan]Epoch {epoch:02d}/{epochs:02d}[/cyan] — Mean MSE Loss: [bold yellow]{epoch_loss:.4f}[/bold yellow]"
            )

    return final_loss
