"""Light utility script to train NNUENet and execute a Stockfish 1400 Elo match tournament.

Rule 013 Ref: Color user-facing terminal output using `rich`.
Rule 014 Ref: Light script under scripts/light/.
"""

from __future__ import annotations

import argparse
import sys
import chess

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.plugins.domains.chess.ground_a_self_play import PolicySnapshot
from hypostases.plugins.domains.chess.ground_b_stockfish import GroundBStockfish
from hypostases.world_model.alphabeta_search import AlphaBetaSearch, SearchConfig
from hypostases.world_model.nnue_net import NNUENet, extract_halfkp_features
from hypostases.world_model.nnue_training import generate_and_audit_dataset, train_nnue

console = Console()


def run_training_and_match(
    positions: int = 5000,
    epochs: int = 5,
    match_games: int = 20,
    target_elo: float = 1400.0,
    search_depth: int = 4,
    time_budget_ms: float = 200.0,
) -> None:
    console.print(
        Panel(
            f"[bold cyan]HYPOSTASES-NNUE Training & Stockfish {int(target_elo)} Match Session[/bold cyan]"
        )
    )

    # Phase 1: Train NNUENet
    console.print(
        f"[yellow]Phase 1/2: Generating {positions} positions and training NNUENet for {epochs} epochs...[/yellow]"
    )
    dataset = generate_and_audit_dataset(num_positions=positions)
    net = NNUENet(seed=42)
    final_loss = train_nnue(net, dataset, epochs=epochs, lr=0.01)
    console.print(
        f"[bold green][OK] Training Complete! Final MSE Loss: {final_loss:.6f}[/bold green]\n"
    )

    # Phase 2: Prepare Policy Snapshot for Search
    def agent_policy(state: chess.Board, legal_moves: list[chess.Move]) -> chess.Move:
        def nnue_evaluator(s: chess.Board) -> float:
            acc = net.create_accumulator(s)
            _, _, aux = extract_halfkp_features(s)
            return net.forward(acc, aux)

        cfg = SearchConfig(max_depth=search_depth, time_budget_ms=time_budget_ms)
        searcher = AlphaBetaSearch(domain=ChessDomain(), evaluator=nnue_evaluator, config=cfg)
        move, _, _ = searcher.search(state)
        return move if move in legal_moves else legal_moves[0]

    snapshot = PolicySnapshot(generation=1, policy_fn=agent_policy)

    # Phase 3: Match against Stockfish (Utilizing 16 parallel CPU workers)
    console.print(
        f"[yellow]Phase 2/2: Executing {match_games}-game match vs Stockfish (Target Elo: {target_elo}) across 16 parallel CPU workers...[/yellow]"
    )
    harness = GroundBStockfish(
        reference_elo=target_elo, time_control=0.1, max_workers=16, stockfish_threads=1
    )
    result = harness.evaluate_snapshot(snapshot=snapshot, games_n=match_games, verbose=True)

    # Render Summary Results Table
    table = Table(title=f"Match Results vs Stockfish {int(target_elo)} Elo")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Games Played", str(result.games_played))
    table.add_row(
        "Wins / Losses / Draws", f"{result.wins} W / {result.losses} L / {result.draws} D"
    )
    table.add_row("Win Rate Score", f"{result.score * 100.0:.1f}%")
    table.add_row("Estimated Agent Elo", f"{result.estimated_elo:.1f}")
    table.add_row("Stockfish Target Elo", f"{result.reference_elo:.1f}")

    console.print("\n")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train HYPOSTASES-NNUE and run Stockfish match tournament"
    )
    parser.add_argument(
        "--positions", type=int, default=2000, help="Number of bootstrap positions to generate"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument(
        "--games", type=int, default=20, help="Number of match games against Stockfish"
    )
    parser.add_argument("--elo", type=float, default=1400.0, help="Stockfish target Elo rating")
    args = parser.parse_args()

    run_training_and_match(
        positions=args.positions,
        epochs=args.epochs,
        match_games=args.games,
        target_elo=args.elo,
    )


if __name__ == "__main__":
    main()
