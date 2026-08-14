"""CLI script to run supervised dataset generation, NNUENet training, and interactive/play mode against the trained agent.

Rule 013 Ref: Color user-facing terminal output using `rich`.
Rule 014 Ref: Light script under scripts/light/.
"""

from __future__ import annotations

import argparse

import chess
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.world_model.alphabeta_search import AlphaBetaSearch, SearchConfig
from hypostases.world_model.nnue_net import NNUENet, extract_halfkp_features
from hypostases.world_model.nnue_training import generate_and_audit_dataset, train_nnue

console = Console()


def train_session(num_positions: int = 5000, epochs: int = 5) -> NNUENet:
    console.print(
        Panel(
            f"[bold cyan]Starting NNUE Learning Session ({num_positions} positions, {epochs} epochs)[/bold cyan]"
        )
    )

    console.print("[yellow]1/2 Generating Stage-0 bootstrap training positions...[/yellow]")
    dataset = generate_and_audit_dataset(num_positions=num_positions)
    console.print(
        f"[green][OK] Dataset generated ({len(dataset)} labeled positions). Audit saved to scratch/HYPOSTASES_NNUE_CONVERGENCE_AUDIT.md.[/green]"
    )

    net = NNUENet(seed=42)
    console.print("[yellow]2/2 Training NNUENet evaluation weights via MSE loss...[/yellow]")
    final_loss = train_nnue(net, dataset, epochs=epochs, lr=0.01)
    console.print(
        f"[bold green][OK] Training complete! Final MSE Loss: {final_loss:.6f}[/bold green]\n"
    )
    return net


def play_against_agent(net: NNUENet, depth: int = 4, time_budget_ms: float = 500.0) -> None:
    domain = ChessDomain()
    board = domain.initial_state()

    def nnue_evaluator(state: chess.Board) -> float:
        acc = net.create_accumulator(state)
        _, _, aux = extract_halfkp_features(state)
        return net.forward(acc, aux)

    config = SearchConfig(max_depth=depth, time_budget_ms=time_budget_ms)
    agent_searcher = AlphaBetaSearch(domain=domain, evaluator=nnue_evaluator, config=config)

    console.print(
        Panel(
            "[bold green]Interactive Chess Match: User (White) vs HYPOSTASES-NNUE Agent (Black)[/bold green]"
        )
    )

    while not board.is_game_over():
        console.print(f"\n[bold white]Current Board (SAN):[/bold white]\n{board}\n")

        if board.turn == chess.WHITE:
            legal_moves = [m.uci() for m in board.legal_moves]
            move_str = Prompt.ask(
                "[bold yellow]Enter your move (UCI format, e.g. e2e4)[/bold yellow]",
                choices=legal_moves,
            )
            move = chess.Move.from_uci(move_str)
            board.push(move)
        else:
            console.print("[cyan]Agent is thinking...[/cyan]")
            move, score, tele = agent_searcher.search(board)
            if move is None:
                break
            console.print(
                f"[bold magenta]Agent selected move: {move.uci()} (Eval score: {score:.3f}, Searched nodes: {tele.searched_nodes})[/bold magenta]"
            )
            board.push(move)

    console.print(Panel(f"[bold yellow]Game Over! Outcome: {board.outcome()}[/bold yellow]"))


def main() -> None:
    parser = argparse.ArgumentParser(description="HYPOSTASES NNUE Learning & Game Session")
    parser.add_argument(
        "--positions",
        type=int,
        default=2000,
        help="Number of bootstrap positions to sample and train on",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument(
        "--play", action="store_true", help="Launch interactive game session after training"
    )
    args = parser.parse_args()

    net = train_session(num_positions=args.positions, epochs=args.epochs)
    if args.play:
        play_against_agent(net)


if __name__ == "__main__":
    main()
