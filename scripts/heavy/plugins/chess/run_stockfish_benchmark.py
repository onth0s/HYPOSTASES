"""Load a completed HYPOSTASES training run and benchmark its final agent against Stockfish.

[HEAVY COMPUTATIONAL COST SCRIPT]
Reserved exclusively for manual execution by the User. AI agents must NEVER execute this script.

Loads the final snapshot (theta_meta + NNUE weights) from an exports/runs/ directory
and runs N games against a calibrated Stockfish reference via GroundBStockfish.

Spec Ref: Rule 006 (Data-driven YAML Configuration), Rule 013 (rich output), Rule 014 (heavy script).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.plugins.domains.chess.ground_a_self_play import PolicySnapshot
from hypostases.plugins.domains.chess.ground_b_stockfish import GroundBStockfish

console = Console()

DEFAULT_RUN_DIR = Path("exports/runs/run_20260815_032220")
EXPERIMENT_CONFIG = Path(
    "src/hypostases/plugins/domains/chess/chess_experiment_config.yaml"
)


def load_run_snapshot(run_dir: Path) -> PolicySnapshot:
    """Loads the final checkpoint from a completed training run into a PolicySnapshot."""
    import chess

    from hypostases.plugins.domains.chess.chess_agent_adapter import ChessAgentAdapter
    from hypostases.plugins.domains.chess.nnue_net import NNUENet

    meta_path = run_dir / "run_metadata.json"
    checkpoint_path = run_dir / "checkpoint_state.yaml"
    nnue_path = run_dir / "nnue" / "nnue_latest.npz"

    if not meta_path.exists():
        console.print(f"[bold red]ERROR:[/bold red] No run_metadata.json in {run_dir}")
        sys.exit(1)
    if not checkpoint_path.exists():
        console.print(f"[bold red]ERROR:[/bold red] No checkpoint_state.yaml in {run_dir}")
        sys.exit(1)
    if not nnue_path.exists():
        console.print(f"[bold red]ERROR:[/bold red] No nnue_latest.npz in {run_dir}")
        sys.exit(1)

    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    checkpoint = yaml.safe_load(checkpoint_path.read_text(encoding="utf-8"))
    nnue_data = dict(np.load(nnue_path))

    gen = checkpoint.get("last_saved_gen", meta.get("last_completed_gen", 0))
    agent_state = checkpoint.get("agent", {})
    theta_meta = np.array(agent_state["theta_meta"], dtype=np.float32)
    temperature = float(agent_state["temperature"])
    beta_efe = float(agent_state.get("beta_efe", 0.2))

    console.print(
        f"  [dim]Run: {meta.get('status', '?')} | "
        f"Gen {gen}/{meta.get('target_generations', '?')} | "
        f"{meta.get('total_games_played', '?')} games played | "
        f"Started {meta.get('started_at', '?')}[/dim]"
    )
    console.print(
        f"  [dim]theta_meta[0:8] = [{', '.join(f'{v:.3f}' for v in theta_meta[:8])}][/dim]"
    )
    console.print(
        f"  [dim]tau={temperature:.3f}  beta_logit={theta_meta[9]:.2f}  "
        f"beta={beta_efe:.4f}  nnue_keys={list(nnue_data.keys())}[/dim]"
    )

    def _make_policy_fn(
        _theta: np.ndarray,
        _temp: float,
        _nnue: dict[str, np.ndarray] | None,
    ) -> callable:
        def policy(board: chess.Board, legal_moves: list[chess.Move]) -> chess.Move:
            domain = ChessDomain()
            adapter = ChessAgentAdapter(
                domain=domain,
                beta_efe=0.2,
                temperature=_temp,
                theta_meta=_theta,
            )
            net = NNUENet() if _nnue is not None else None
            if net and _nnue:
                net.W_white = _nnue["W_white"]
                net.W_black = _nnue["W_black"]
                net.W_l1 = _nnue["W_l1"]
                net.b_l1 = _nnue["b_l1"]
                net.W_l2 = _nnue["W_l2"]
                net.b_l2 = _nnue["b_l2"]
            return adapter.select_move(board, legal_moves, depth=3, nnue_net=net)

        return policy

    return PolicySnapshot(
        generation=gen,
        policy_fn=_make_policy_fn(theta_meta, temperature, nnue_data),
        theta_meta=theta_meta,
        temperature=temperature,
        nnue_weights=nnue_data,
    )


def load_ground_b_config(config_path: Path) -> dict:
    """Extracts ground_b parameters from the experiment config."""
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("ground_b", {})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a completed HYPOSTASES training run against Stockfish"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(DEFAULT_RUN_DIR),
        help="Path to the training run directory",
    )
    parser.add_argument("--games", type=int, default=32, help="Number of benchmark games")
    parser.add_argument(
        "--stockfish-elo", type=float, default=800.0, help="Stockfish target Elo"
    )
    parser.add_argument(
        "--time-control", type=float, default=0.02, help="Seconds per engine move"
    )
    parser.add_argument("--workers", type=int, default=20, help="Parallel engine workers")
    parser.add_argument(
        "--stockfish-threads", type=int, default=2, help="CPU threads per Stockfish process"
    )
    parser.add_argument(
        "--search-depth", type=int, default=3, help="Agent alpha-beta search depth"
    )
    parser.add_argument(
        "--eval-temperature",
        type=float,
        default=0.05,
        help="Greedy eval temperature (overrides snapshot tau)",
    )
    parser.add_argument(
        "--max-moves", type=int, default=360, help="Max moves per game before draw adjudication"
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default=None,
        help="Explicit path to Stockfish binary (overrides auto-detect)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        console.print(f"[bold red]ERROR:[/bold red] Run directory not found: {run_dir}")
        sys.exit(1)

    console.print(
        Panel.fit(
            "[bold cyan]HYPOSTASES Stockfish Benchmark[/bold cyan]",
            subtitle=f"Run: {run_dir.name}",
            border_style="cyan",
        )
    )

    snapshot = load_run_snapshot(run_dir)
    console.print(
        f"\n  [bold]Loaded snapshot:[/bold] Gen {snapshot.generation} "
        f"(tau={snapshot.temperature:.3f}, nnue={'yes' if snapshot.nnue_weights else 'no'})"
    )

    harness = GroundBStockfish(
        stockfish_path=args.stockfish_path,
        reference_elo=args.stockfish_elo,
        time_control=args.time_control,
        stockfish_threads=args.stockfish_threads,
        max_workers=args.workers,
        eval_temperature=args.eval_temperature,
        search_depth=args.search_depth,
        chess_domain=ChessDomain(),
    )

    console.print(
        f"\n  [yellow]Starting {args.games}-game match vs Stockfish "
        f"(Elo {args.stockfish_elo:.0f}, {args.time_control*1000:.0f}ms/move, "
        f"{args.workers} workers, {args.stockfish_threads} threads/engine)...[/yellow]"
    )

    result = harness.evaluate_snapshot(
        snapshot=snapshot,
        games_n=args.games,
        max_moves=args.max_moves,
        verbose=True,
        export_pgn_dir="exports/pgn/ground_b",
        timestamp_runs=True,
    )

    table = Table(
        title=f"Stockfish {int(args.stockfish_elo)} Elo Benchmark — Gen {snapshot.generation}",
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Games Played", str(result.games_played))
    table.add_row(
        "Wins / Losses / Draws",
        f"[bold green]{result.wins}W[/bold green] / "
        f"[bold red]{result.losses}L[/bold red] / "
        f"[yellow]{result.draws}D[/yellow]",
    )
    table.add_row("Capped (adj.)", str(result.capped))
    table.add_row("Effective Games", str(result.effective_games))
    table.add_row("Score Ratio", f"{result.score * 100.0:.1f}%")
    table.add_row("Estimated Agent Elo", f"[bold cyan]{result.estimated_elo:.1f}[/bold cyan]")
    table.add_row("Stockfish Reference Elo", f"{result.reference_elo:.1f}")
    table.add_row("Avg Game Length", f"{result.avg_game_length:.1f} moves")

    if result.avg_sf_eval_agent is not None:
        table.add_row("Avg SF Eval (agent)", f"{result.avg_sf_eval_agent:+.0f} cp")
    if result.last_sf_eval_agent is not None:
        table.add_row("Last SF Eval (agent)", f"{result.last_sf_eval_agent:+.0f} cp")
    if result.avg_material_gap_agent is not None:
        table.add_row("Avg Material Gap", f"{result.avg_material_gap_agent:+.1f}")

    console.print()
    console.print(table)
    console.print()

    # Persist benchmark results as timestamped JSON inside the run directory
    benchmarks_dir = run_dir / "stockfish_benchmarks"
    benchmarks_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    benchmark_file = benchmarks_dir / f"benchmark_{ts}.json"

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "benchmark_config": {
            "games": args.games,
            "stockfish_elo": args.stockfish_elo,
            "time_control": args.time_control,
            "workers": args.workers,
            "stockfish_threads": args.stockfish_threads,
            "search_depth": args.search_depth,
            "eval_temperature": args.eval_temperature,
            "max_moves": args.max_moves,
            "stockfish_path": args.stockfish_path,
        },
        "result": result.to_dict(),
    }
    benchmark_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(
        f"  [bold green]Benchmark saved:[/bold green] {benchmark_file}"
    )


if __name__ == "__main__":
    main()
