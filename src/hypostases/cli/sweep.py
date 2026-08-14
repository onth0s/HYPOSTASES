"""HYPOSTASES CLI — Diagnostic and Formal Sweep Runner.

Spec Ref: Part VII §12.7.
Formally tests whether the WITHDRAW identifiability gap is closed.
Z(theta_low, theta_high, n, N, seed) = (mu_high - mu_low) / sqrt((var_high + var_low)/2)

Gap closed at (n, N) iff over S >= 5 seeds:
  1. Directional correctness: Z > 0 in >= ceil(0.8*S) seeds
  2. Statistical separation: median(Z) > 1.0
  3. Non-degeneracy: var in [0.5, 20] in all but <= 1 seed
"""

from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hypostases.simulation import run_sweep_benchmark

console = Console()


def run_sweep(
    steps_list: list[int],
    n_particles: int,
    seeds: list[int],
) -> list[dict]:
    return run_sweep_benchmark(
        steps_list=steps_list,
        n_particles=n_particles,
        seeds=seeds,
    )


def format_sweep_output(eval_results: list[dict], output_format: str = "table") -> None:
    if output_format == "json":
        console.print_json(json.dumps(eval_results))
        return

    console.print()
    console.print(
        Panel(
            f"[bold]Conditions tested:[/bold] {len(eval_results)}  "
            f"[bold]Seeds:[/bold] {eval_results[0]['n_seeds'] if eval_results else '—'}",
            title="[bold cyan]HYPOSTASES Phase A: Diagnostic Sweep[/bold cyan]  [dim](Specification §12.7)[/dim]",
            border_style="cyan",
        )
    )

    table = Table(
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        row_styles=["", "dim"],
    )
    table.add_column("n_steps", justify="right", style="cyan", width=9)
    table.add_column("n_particles", justify="right", style="yellow", width=13)
    table.add_column("median Z", justify="right", width=11)
    table.add_column("dir_ok", justify="center", width=10)
    table.add_column("passed", justify="center", width=10)

    for r in eval_results:
        passed = r["passed"]
        median_z = r["cond2_median_z"]
        z_style = "bold green" if median_z > 1.0 else "bold red"
        passed_str = "[bold green]✓ PASS[/bold green]" if passed else "[bold red]✗ FAIL[/bold red]"
        table.add_row(
            str(r["n_steps"]),
            str(r["n_particles"]),
            f"[{z_style}]{median_z:.3f}[/{z_style}]",
            f"{r['cond1_count']}/{r['n_seeds']}",
            passed_str,
        )

    console.print(table)

    n_passed = sum(1 for r in eval_results if r["passed"])
    n_total = len(eval_results)
    all_passed = n_passed == n_total
    summary_style = "bold green" if all_passed else "bold yellow"
    console.print(
        f"\n[{summary_style}]{'✓ All conditions passed' if all_passed else f'{n_passed}/{n_total} conditions passed'}[/{summary_style}]"
    )


def add_subparser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "sweep", help="Run diagnostic / formal 3-condition sweep (Part VII §12.7)"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[10, 50, 200],
        help="List of step counts (default: 10 50 200)",
    )
    parser.add_argument(
        "--particles", type=int, default=300, help="Particle count N (default: 300)"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Random seeds (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--output-format", choices=["table", "json"], default="table", help="Output format"
    )
    parser.set_defaults(func=main_sweep)


def main_sweep(args: argparse.Namespace) -> None:
    res = run_sweep_benchmark(
        steps_list=args.steps,
        n_particles=args.particles,
        seeds=args.seeds,
    )
    format_sweep_output(res, output_format=args.output_format)
