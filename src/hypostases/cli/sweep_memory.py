"""HYPOSTASES CLI — Memory Decay Stability Sweep.

Runs simulations over multiple steps to check how memory decay models impact w.sigma2 stability.
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hypostases.engine import Action, ActionType, evolve, feedback
from hypostases.engine.constants import SIGMA2_MAX, SIGMA2_MIN
from hypostases.simulation.harness import build_agent

console = Console()


def add_subparser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "sweep-memory", help="Run memory decay stability sweep (Phase 2)"
    )
    parser.add_argument(
        "--steps", type=int, default=30, help="Number of simulation steps (default: 30)"
    )
    parser.set_defaults(func=main_sweep_memory)


def run_memory_sweep(steps: int, decay_mode: str, memory_decay: float) -> list[float]:
    agent = build_agent(name="agent", memory_decay=memory_decay, reserve=10.0)
    agent.sigma.decay_mode = decay_mode

    sigma2_history = [agent.sigma.w.sigma2]

    for _ in range(steps):
        pool_before = 10.0
        pool_after = 10.0
        action = Action(ActionType.REQUEST, amount=1.0)
        delta_log = {
            "pool_before": pool_before,
            "pool_after_shares": pool_before,
            "pool_after": pool_after,
            "shares_total": 0.0,
            "requests_total": 1.0,
            "granted": {"agent": 1.0},
        }

        phi = feedback(
            agent.sigma,
            pool_before=pool_before,
            pool_after=pool_after,
            action=action,
            delta_log=delta_log,
            agent_name="agent",
        )
        evolve(agent.sigma, phi)
        sigma2_history.append(agent.sigma.w.sigma2)

    return sigma2_history


def main_sweep_memory(args: argparse.Namespace) -> None:
    decays = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
    modes = ["variance", "precision"]

    console.print()
    console.print(
        Panel(
            f"[bold]Steps:[/bold] {args.steps}  "
            f"[bold]SIGMA2_MAX:[/bold] [yellow]{SIGMA2_MAX:.1f}[/yellow]  "
            f"[bold]SIGMA2_MIN:[/bold] [yellow]{SIGMA2_MIN}[/yellow]",
            title="[bold cyan]HYPOSTASES Memory Decay Stability Sweep[/bold cyan]  [dim](Phase 2)[/dim]",
            border_style="cyan",
        )
    )

    mode_colors = {"variance": "magenta", "precision": "blue"}

    for mode in modes:
        color = mode_colors.get(mode, "white")
        console.print(f"\n[bold {color}]Mode: {mode.upper()}[/bold {color}]")

        table = Table(
            show_header=True,
            header_style=f"bold {color}",
            border_style="dim",
            row_styles=["", "dim"],
        )
        table.add_column("Decay", justify="right", style="cyan", width=8)
        table.add_column("Initial σ²", justify="right", style="yellow", width=12)
        table.add_column("Midpoint σ²", justify="right", width=13)
        table.add_column("Final σ²", justify="right", width=12)
        table.add_column("Stable?", justify="center", width=10)

        for decay in decays:
            history = run_memory_sweep(args.steps, mode, decay)
            mid = history[len(history) // 2]
            final = history[-1]
            stable = SIGMA2_MIN <= final <= SIGMA2_MAX
            stable_str = "[green]✓[/green]" if stable else "[red]✗[/red]"

            # Colour the final value by stability
            final_style = "green" if stable else "red"
            table.add_row(
                f"{decay:.2f}",
                f"{history[0]:.4f}",
                f"{mid:.4f}",
                f"[{final_style}]{final:.4f}[/{final_style}]",
                stable_str,
            )

        console.print(table)
