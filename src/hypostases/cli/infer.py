"""HYPOSTASES CLI — Inverse Inference Runner.

Spec Ref: Part VII §10, §11, §12.
Runs SMC particle filter inference on observed action & pool traces.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hypostases.engine.constants import DEFAULT_XI
from hypostases.inference import (
    goal_posterior,
    infer,
    infer_hierarchical,
    summarize_kalman,
    summarize_map,
)
from hypostases.simulation import generate_sample_trace

console = Console()


def run_cli_infer(
    n_particles: int = 300,
    seed: int = 42,
    agent_name: str = "Agent_A",
    n_steps: int = 12,
    lag_window: int | None = None,
    use_hierarchical: bool = False,
    use_rao_blackwell: bool = False,
) -> dict:
    xi = DEFAULT_XI
    actions, pools = generate_sample_trace(n_steps=n_steps, seed=seed)

    rng = np.random.default_rng(seed)
    if use_hierarchical:
        particles = infer_hierarchical(
            observed_actions=actions,
            observed_pool_trace=pools,
            xi=xi,
            n_particles=n_particles,
            agent_name=agent_name,
            lag_window=lag_window,
            rng=rng,
        )
    else:
        particles = infer(
            observed_actions=actions,
            observed_pool_trace=pools,
            xi=xi,
            n_particles=n_particles,
            agent_name=agent_name,
            lag_window=lag_window,
            use_rao_blackwell=use_rao_blackwell,
            rng=rng,
        )

    map_state = summarize_map(particles)
    kalman = summarize_kalman(particles)
    g_post = goal_posterior(particles)

    return {
        "n_particles": n_particles,
        "seed": seed,
        "agent_name": agent_name,
        "lag_window": lag_window,
        "hierarchical": use_hierarchical,
        "rao_blackwell": use_rao_blackwell,
        "map_estimate": {
            "reserve": round(map_state.c.reserve, 2),
            "mood": round(map_state.c.mood, 2),
            "sociality": round(map_state.c.sociality, 2),
            "u": [round(val, 2) for val in map_state.g.u],
        },
        "kalman_summary": {
            "reserve_mean": round(kalman["reserve_mean"], 2),
            "reserve_var": round(kalman["reserve_var"], 2),
            "mood_mean": round(kalman["mood_mean"], 2),
        },
        "goal_posterior": {k: round(v, 4) for k, v in g_post.items()},
    }


def format_inference_output(res: dict, output_format: str = "table") -> None:
    if output_format == "json":
        console.print_json(json.dumps(res))
        return

    console.print()
    console.print(
        Panel(
            f"[bold]Agent:[/bold] [cyan]{res['agent_name']}[/cyan]  "
            f"[bold]Particles:[/bold] [yellow]{res['n_particles']}[/yellow]  "
            f"[bold]Seed:[/bold] {res['seed']}\n"
            f"[bold]Lag window:[/bold] {res['lag_window']}  "
            f"[bold]Hierarchical:[/bold] {res['hierarchical']}  "
            f"[bold]Rao-Blackwell:[/bold] {res['rao_blackwell']}",
            title="[bold cyan]HYPOSTASES Inverse Inference Report[/bold cyan]  [dim](Part VII §10)[/dim]",
            border_style="cyan",
        )
    )

    # MAP estimate table
    map_table = Table(
        title="[bold]MAP Estimate[/bold]",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        title_style="bold white",
    )
    map_table.add_column("Parameter", style="cyan", width=16)
    map_table.add_column("Value", justify="right", style="yellow", width=12)

    m = res["map_estimate"]
    map_table.add_row("reserve", str(m["reserve"]))
    map_table.add_row("mood", str(m["mood"]))
    map_table.add_row("sociality", str(m["sociality"]))
    map_table.add_row("latent utilities u", str(m["u"]))

    console.print(map_table)
    console.print()

    # Kalman summary table
    kalman_table = Table(
        title="[bold]Kalman Summary[/bold]",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        title_style="bold white",
    )
    kalman_table.add_column("Statistic", style="cyan", width=20)
    kalman_table.add_column("Value", justify="right", style="yellow", width=12)

    k = res["kalman_summary"]
    kalman_table.add_row("reserve mean", str(k["reserve_mean"]))
    kalman_table.add_row("reserve variance", str(k["reserve_var"]))
    kalman_table.add_row("mood mean", str(k["mood_mean"]))

    console.print(kalman_table)
    console.print()

    # Goal posterior table
    goal_table = Table(
        title="[bold]Goal Posterior[/bold]",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        title_style="bold white",
    )
    goal_table.add_column("Goal", style="cyan", width=16)
    goal_table.add_column("P(%)", justify="right", width=10)
    goal_table.add_column("Bar", width=30)

    sorted_goals = sorted(res["goal_posterior"].items(), key=lambda x: -x[1])
    for goal, prob in sorted_goals:
        pct = prob * 100
        bar_len = int(pct / 100 * 28)
        bar = "[green]" + "█" * bar_len + "[/green]" + "░" * (28 - bar_len)
        style = "bold green" if prob == sorted_goals[0][1] else ""
        goal_table.add_row(f"[{style}]{goal}[/{style}]" if style else goal, f"{pct:5.1f}%", bar)

    console.print(goal_table)


def add_subparser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "infer", help="Run inverse inference particle filter (Part VII §10)"
    )
    parser.add_argument(
        "--particles", type=int, default=300, help="Particle count N (default: 300)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--agent-name", type=str, default="Agent_A", help="Target agent name")
    parser.add_argument("--steps", type=int, default=12, help="Trace length steps (default: 12)")
    parser.add_argument("--lag-window", type=int, default=None, help="Bounded history lag window")
    parser.add_argument(
        "--hierarchical", action="store_true", help="Use two-pass hierarchical particle filter"
    )
    parser.add_argument(
        "--use-rao-blackwell", action="store_true", help="Use Kalman world model updates"
    )
    parser.add_argument(
        "--output-format", choices=["table", "json"], default="table", help="Output format"
    )
    parser.set_defaults(func=main_infer)


def main_infer(args: argparse.Namespace) -> None:
    res = run_cli_infer(
        n_particles=args.particles,
        seed=args.seed,
        agent_name=args.agent_name,
        n_steps=args.steps,
        lag_window=args.lag_window,
        use_hierarchical=args.hierarchical,
        use_rao_blackwell=args.use_rao_blackwell,
    )
    format_inference_output(res, output_format=args.output_format)
