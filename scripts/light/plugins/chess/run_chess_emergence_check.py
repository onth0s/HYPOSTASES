"""Light diagnostic script to verify emergent HYPOSTASES core engine state evolution with NNUE World Model.

Spec Ref: Rule 013 (Color output with rich), Rule 014 (Light script <10s runtime)
"""

from __future__ import annotations

import time

import chess
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hypostases.plugins.domains.chess.chess_agent_adapter import ChessAgentAdapter
from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.plugins.domains.chess.nnue_net import NNUENet

console = Console()


def run_emergence_check() -> None:
    console.print(
        Panel("[bold cyan]HYPOSTASES Core Engine — Emergent Chess State Verification[/bold cyan]")
    )

    domain = ChessDomain()
    board = domain.initial_state()
    net = NNUENet(seed=42)

    adapter = ChessAgentAdapter(domain=domain, beta_efe=0.2, temperature=0.5)

    table = Table(title="Core Engine State Evolution Cycle (10 Steps)")
    table.add_column("Step", style="cyan")
    table.add_column("Turn", style="yellow")
    table.add_column("Selected Move", style="magenta")
    table.add_column("U_total (EFE)", style="green")
    table.add_column("Mood Decay / Reserve", style="blue")

    from rich.live import Live
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
    )
    task_id = progress.add_task("Simulating State Evolution...", total=10)

    from rich.console import Group

    t0 = time.time()
    with Live(Group(progress, table), console=console, refresh_per_second=4):
        for step_i in range(1, 11):
            if board.is_game_over():
                break

            legal_moves = list(board.legal_moves)
            turn_str = "White" if board.turn == chess.WHITE else "Black"

            move = adapter.select_move(board, legal_moves, depth=1, nnue_net=net)
            u_tot, u_prag, u_epis = adapter.evaluate_efe_utility(board, move, depth=1, nnue_net=net)

            board.push(move)

            # Update core agent state characteristics (Mood decay per Rule 004)
            adapter.characteristics.mood *= 0.9  # 10% MOOD_DECAY_RATE
            adapter.characteristics.reserve = max(0.0, adapter.characteristics.reserve - 0.1)

            table.add_row(
                f"{step_i:02d}",
                turn_str,
                move.uci(),
                f"{u_tot:.3f} (P:{u_prag:.2f}, E:{u_epis:.2f})",
                f"Mood:{adapter.characteristics.mood:.2f} | Res:{adapter.characteristics.reserve:.1f}",
            )
            progress.advance(task_id)

    elapsed = time.time() - t0
    console.print(
        f"\n[bold green][OK] Core engine state evolution check completed in {elapsed:.3f} seconds.[/bold green]"
    )


if __name__ == "__main__":
    run_emergence_check()
