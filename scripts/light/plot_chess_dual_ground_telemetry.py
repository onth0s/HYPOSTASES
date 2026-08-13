#!/usr/bin/env python3
"""Telemetry Plotter & Report Generator for Chess Dual Ground Benchmark.

[LIGHTWEIGHT DIAGNOSTIC / REPORTING SCRIPT]
Permitted for execution by AI agents (<5s execution time).

Parses exported JSON telemetry from exports/chess_experiment_results.json,
renders Rich dual-curve trajectory tables, and displays statistical summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure src is in sys.path for standalone script execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def load_telemetry(json_path: str = "exports/chess_experiment_results.json") -> dict[str, Any]:
    """Loads telemetry results from JSON file."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Telemetry file not found at {json_path}. Run experiment script first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def render_rich_telemetry_table(
    generations: list[int], internal_elo: list[float], external_elo: list[float]
) -> None:
    """Renders formatted rich table comparing Ground A vs Ground B ratings."""
    table = Table(title="DUAL GROUND ELO OVERLAY TELEMETRY", header_style="bold blue")
    table.add_column("Gen", justify="center", style="bold cyan")
    table.add_column("Ground A (Self-Play Elo)", justify="left", style="green")
    table.add_column("Ground B (Stockfish 18 Elo)", justify="left", style="yellow")

    for g, r_a, r_b in zip(generations, internal_elo, external_elo, strict=False):
        bar_a = "█" * int(max(0, (r_a - 900) / 40))
        bar_b = "█" * int(max(0, (r_b - 400) / 40))
        table.add_row(f"G{g}", f"{r_a:6.1f} {bar_a}", f"{r_b:6.1f} {bar_b}")

    console.print(table)


def generate_report(telemetry: dict[str, Any]) -> None:
    """Prints comprehensive statistical report and final verdict using rich."""
    metrics = telemetry["metrics"]
    verdict = telemetry["verdict"]

    render_rich_telemetry_table(
        generations=telemetry["generations"],
        internal_elo=telemetry["internal_elo_ground_a"],
        external_elo=telemetry["external_elo_ground_b"],
    )

    table_stat = Table(title="STATISTICAL RATIFICATION SUMMARY", header_style="bold magenta")
    table_stat.add_column("Metric", style="cyan")
    table_stat.add_column("Value", style="bold white")

    table_stat.add_row("Experiment ID", str(telemetry["experiment_id"]))
    table_stat.add_row(
        "Ground A Monotonicity Rate", f"{metrics['ground_a_monotonicity_ratio'] * 100:.1f}%"
    )
    table_stat.add_row(
        "Ground B Monotonicity Rate", f"{metrics['ground_b_monotonicity_ratio'] * 100:.1f}%"
    )
    table_stat.add_row("Pearson Correlation (r)", f"{metrics['pearson_correlation_r']:.3f}")
    table_stat.add_row("Pre-Registered Threshold", f"{metrics['pass_threshold'] * 100:.1f}%")
    table_stat.add_row("Final Verdict", verdict)

    console.print(table_stat)


def main() -> None:
    """Entry point for telemetry display script."""
    console.print(
        Panel.fit(
            "[bold blue]HYPOSTASES Dual Ground Experiment Telemetry[/bold blue]",
            border_style="blue",
        )
    )
    telemetry = load_telemetry()
    generate_report(telemetry)


if __name__ == "__main__":
    main()
