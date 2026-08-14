"""HYPOSTASES CLI — Forward Simulation Trace Runner.

Spec Ref: Part V, Part VI §8.
Runs forward simulation of social agents sharing a pool.
"""

from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hypostases.simulation import run_simulation_trace

console = Console()


def add_subparser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser("trace", help="Run forward simulation trace (Part VI §8)")
    parser.add_argument(
        "--steps", type=int, default=12, help="Number of Tier-1 steps (default: 12)"
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed (default: 7)")
    parser.add_argument(
        "--export-json", type=str, default=None, help="File path to save JSON trace output"
    )
    parser.set_defaults(func=main_trace)


def main_trace(args: argparse.Namespace) -> None:
    res = run_simulation_trace(steps=args.steps, seed=args.seed)

    console.print()
    console.print(
        Panel(
            f"[bold]Steps:[/bold] {res['steps']}  [bold]Seed:[/bold] {res['seed']}",
            title="[bold cyan]HYPOSTASES Forward Simulation Trace[/bold cyan]  [dim](Part VI §8)[/dim]",
            border_style="cyan",
        )
    )

    # Trace table
    trace_table = Table(
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        row_styles=["", "dim"],
    )
    trace_table.add_column("Step", justify="right", style="cyan", width=6)
    trace_table.add_column("Pool After", justify="right", style="yellow", width=10)
    trace_table.add_column("Agent A Action", justify="center", style="green", width=20)
    trace_table.add_column("Agent B Action", justify="center", style="blue", width=20)

    for s in res["trace"]:
        trace_table.add_row(
            str(s["step"]),
            f"{s['pool_after']:.2f}",
            s["actions"].get("Agent_A", "—"),
            s["actions"].get("Agent_B", "—"),
        )

    console.print(trace_table)

    # Final agent states table
    console.print()
    console.print("[bold]Final Agent States[/bold]", style="bold white")
    state_table = Table(
        show_header=True,
        header_style="bold white",
        border_style="dim",
    )
    state_table.add_column("Agent", style="cyan", width=12)
    state_table.add_column("Reserve", justify="right", style="yellow", width=10)
    state_table.add_column("Social Capital", justify="right", style="green", width=16)
    state_table.add_column("Mood", justify="right", style="magenta", width=8)
    state_table.add_column("Utilities u", style="dim", width=30)

    for name, st in res["agents"].items():
        state_table.add_row(
            name,
            str(st["reserve"]),
            str(st["social_capital"]),
            str(st["mood"]),
            str(st["u"]),
        )

    console.print(state_table)

    if args.export_json:
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        console.print(
            f"\n[bold green]✓[/bold green] Saved trace output to [cyan]{args.export_json}[/cyan]"
        )
