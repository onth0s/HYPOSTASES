#!/usr/bin/env python3
"""Run Chess Dual Ground Experiment: Ground A (Self-Play) vs Ground B (Stockfish 18).

[HEAVY COMPUTATIONAL COST SCRIPT]
Reserved exclusively for manual execution by the User. AI agents must NEVER execute this script.

Pre-registered experiment runner for evaluating whether self-play learning translates
into genuine chess competence against calibrated Stockfish 18 reference.
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import signal
import sys
from pathlib import Path
from typing import Any

# Ensure src is in sys.path for standalone script execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.plugins.domains.chess.chess_trainer import ChessSelfPlayTrainer
from hypostases.plugins.domains.chess.ground_a_self_play import GroundASelfPlay, PolicySnapshot
from hypostases.plugins.domains.chess.ground_b_stockfish import GroundBStockfish


# Suppress noisy unhandled KeyboardInterrupt tracebacks across main and worker processes
def _quiet_excepthook(kind: type, value: BaseException, tb: Any) -> None:
    if issubclass(kind, KeyboardInterrupt):
        return
    sys.__excepthook__(kind, value, tb)


sys.excepthook = _quiet_excepthook


def _kill_worker_children() -> None:
    """Terminates all live multiprocessing child processes (pool workers).

    Called immediately before os._exit(130) so Ctrl+C never orphans the spawned
    worker processes: they ignore SIGINT and would otherwise outlive the parent.
    """
    with contextlib.suppress(Exception):
        for p in multiprocessing.active_children():
            p.terminate()


def _handle_sigint(signum: int, frame: Any) -> None:
    """Instant termination on Ctrl+C without thread join hanging."""
    console.print(
        "\n[bold red][Experiment Interrupted by User (Ctrl+C)][/bold red] Instantly terminating process."
    )
    _kill_worker_children()
    os._exit(130)


signal.signal(signal.SIGINT, _handle_sigint)

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure src is in sys.path for standalone script execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


console = Console()


def load_experiment_config(
    config_path: str = "src/hypostases/plugins/domains/chess/chess_experiment_config.yaml",
) -> dict[str, Any]:
    """Loads pre-registered experiment parameters from YAML configuration."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_dual_grounds(
    config: dict[str, Any],
    custom_snapshots: list[PolicySnapshot] | None = None,
) -> dict[str, Any]:
    """Executes Ground A and Ground B evaluations on authentic HYPOSTASES trained agent snapshots."""
    g_a_cfg = config["ground_a"]
    g_b_cfg = config["ground_b"]
    pre_reg = config["pre_registration"]

    chess_domain = ChessDomain(representation_mode="full")

    # Generate snapshots via real HYPOSTASES agent self-play training if none provided
    if custom_snapshots is None:
        t_cfg = config.get("training", {})
        trainer = ChessSelfPlayTrainer(
            learning_rate=float(t_cfg.get("learning_rate", 0.01)),
            beta_efe=float(t_cfg.get("efe_beta", 0.05)),
            initial_temperature=float(t_cfg.get("policy_temperature", 0.8)),
            value_gamma=float(t_cfg.get("value_gamma", 0.97)),
            chess_domain=chess_domain,
        )

        total_gens = int(t_cfg.get("generations", 24))
        k_interval = int(g_a_cfg["snapshot_interval_k"])
        games_per_gen = int(t_cfg.get("games_per_generation", 32))
        max_moves_train = int(t_cfg.get("max_moves_training", 400))
        early_adj_mat = float(t_cfg.get("early_adjudication_material", 15.0))
        initial_priors = t_cfg.get("initial_priors", "random")

        console.print(
            Panel.fit(
                "[bold cyan]TRAINING HYPOSTASES AGENTS (EFE ACTIVE SENSING & META-RL)[/bold cyan]",
                border_style="cyan",
            )
        )
        snapshots = trainer.execute_self_play_training_run(
            total_generations=total_gens,
            snapshot_interval_k=k_interval,
            games_per_generation=games_per_gen,
            max_moves_training=max_moves_train,
            early_adjudication_material=early_adj_mat,
            initial_priors=initial_priors,
            value_gamma=float(t_cfg.get("value_gamma", 0.97)),
            curriculum_probability=float(t_cfg.get("curriculum_probability", 0.25)),
            resign_value_threshold=float(t_cfg.get("resign_value_threshold", -3.0)),
            resign_confirm_moves=int(t_cfg.get("resign_confirm_moves", 6)),
            nnue_epochs=int(t_cfg.get("nnue_epochs", 30)),
            nnue_learning_rate=float(t_cfg.get("nnue_learning_rate", 0.003)),
            replay_capacity=int(t_cfg.get("replay_capacity", 6000)),
            search_depth=int(t_cfg.get("search_depth", 3)),
            adjudicate_bare_king_requires_mate=bool(
                t_cfg.get("adjudicate_bare_king_requires_mate", True)
            ),
            log_game_details=bool(t_cfg.get("log_game_details", False)),
            verbose=True,
        )
        train_telemetry = trainer.telemetry
    else:
        snapshots = custom_snapshots
        train_telemetry = None

    generations = [s.generation for s in snapshots]

    # --- Ground A Execution ---
    console.print(
        Panel.fit(
            "[bold green]GROUND A: SELF-PLAY SNAPSHOT TOURNAMENT[/bold green]",
            border_style="green",
        )
    )
    ground_a = GroundASelfPlay(
        chess_domain=chess_domain,
        max_workers=int(g_a_cfg.get("parallel_workers", 20)),
    )
    g_a_last_n = int(g_a_cfg.get("eval_last_n_snapshots", 0))
    g_a_eval_gens = g_a_cfg.get("evaluate_generations")
    tournament_result = ground_a.run_snapshot_tournament(
        snapshots=snapshots,
        games_per_pair=g_a_cfg["games_per_pair"],
        max_moves=g_a_cfg.get("max_moves_eval", 120),
        eval_last_n_snapshots=g_a_last_n,
        evaluate_generations=g_a_eval_gens,
        eval_temperature=float(g_a_cfg.get("eval_temperature", 0.05)),
        search_depth=int(g_a_cfg.get("search_depth", 3)),
        verbose=True,
    )
    internal_elo_dict = GroundASelfPlay.compute_internal_elo(
        result=tournament_result,
        base_elo=float(g_a_cfg["base_elo_anchor"]),
    )
    active_generations = sorted({g for pair in tournament_result.games for g in pair})
    if not active_generations:
        active_generations = generations

    internal_elos = [
        internal_elo_dict.get(g, float(g_a_cfg["base_elo_anchor"])) for g in active_generations
    ]

    table_a = Table(
        title="Ground A Comprehensive Internal Elo & Performance Summary",
        header_style="bold green",
    )
    table_a.add_column("Generation", justify="center", style="cyan")
    table_a.add_column("Internal Elo", justify="right", style="bold yellow")
    table_a.add_column("Record (W-L-D)", justify="center", style="bold white")
    table_a.add_column("Score (%)", justify="right", style="bold green")
    table_a.add_column("Win Rate (%)", justify="right", style="green")
    table_a.add_column("Draw Rate (%)", justify="right", style="magenta")

    for g, elo in zip(active_generations, internal_elos, strict=False):
        # Aggregate games for generation g across all matchups
        total_w = 0.0
        total_l = 0.0
        total_d = 0.0
        total_g = 0

        for (p1, p2), g_count in tournament_result.games.items():
            draws = tournament_result.draw_counts.get((p1, p2), 0)
            p1_score = tournament_result.wins.get((p1, p2), 0.0)
            p1_wins = int(p1_score - 0.5 * draws)
            p2_wins = g_count - p1_wins - draws

            if p1 == g:
                total_w += p1_wins
                total_l += p2_wins
                total_d += draws
                total_g += g_count
            elif p2 == g:
                total_w += p2_wins
                total_l += p1_wins
                total_d += draws
                total_g += g_count

        if total_g > 0:
            score_pct = ((total_w + 0.5 * total_d) / total_g) * 100.0
            win_pct = (total_w / total_g) * 100.0
            draw_pct = (total_d / total_g) * 100.0
        else:
            score_pct = 50.0
            win_pct = 0.0
            draw_pct = 100.0

        table_a.add_row(
            f"Gen {g:0{len(str(max(active_generations)))}d}",
            f"{elo:.1f}",
            f"{int(total_w)}W - {int(total_l)}L - {int(total_d)}D",
            f"{score_pct:.1f}%",
            f"{win_pct:.1f}%",
            f"{draw_pct:.1f}%",
        )
    console.print(table_a)

    # --- Ground B Execution ---
    console.print(
        Panel.fit(
            "[bold yellow]GROUND B: STOCKFISH 18 EXTERNAL BENCHMARK[/bold yellow]",
            border_style="yellow",
        )
    )
    stockfish_path = g_b_cfg.get("stockfish_path")
    max_workers = int(g_b_cfg.get("parallel_workers", 10))
    stockfish_threads = int(g_b_cfg.get("stockfish_threads", 2))

    ground_b = GroundBStockfish(
        stockfish_path=stockfish_path,
        reference_elo=float(g_b_cfg["stockfish_fixed_level"]),
        time_control=float(g_b_cfg["time_control_per_move"]),
        stockfish_threads=stockfish_threads,
        max_workers=max_workers,
        stockfish_multipv=int(g_b_cfg.get("stockfish_multipv", 1)),
        eval_temperature=float(g_b_cfg.get("eval_temperature", 0.05)),
        search_depth=int(g_b_cfg.get("search_depth", 3)),
        chess_domain=chess_domain,
    )

    # Select Ground B snapshot contenders aligned with the Ground A evaluated generations
    available_by_gen = {s.generation: s for s in snapshots}
    b_eval_gens = g_b_cfg.get("evaluate_generations")
    if b_eval_gens:
        b_snapshots = [available_by_gen[g] for g in b_eval_gens if g in available_by_gen]
    else:
        eval_last_n = g_b_cfg.get("eval_last_n_snapshots")
        if eval_last_n is not None and isinstance(eval_last_n, int) and eval_last_n > 0:
            b_snapshots = snapshots[-eval_last_n:]
        else:
            b_snapshots = snapshots
    if not b_snapshots:
        b_snapshots = snapshots

    external_elos = []
    scores_ratio = []
    win_loss_draw_stats = []
    b_results = []

    for snapshot in b_snapshots:
        res = ground_b.evaluate_snapshot(
            snapshot=snapshot,
            games_n=g_b_cfg["games_per_elo_estimate_n"],
            max_moves=g_b_cfg.get("max_moves_eval", 120),
            verbose=True,
        )
        b_results.append(res)
        external_elos.append(res.estimated_elo)
        scores_ratio.append(res.score)
        win_loss_draw_stats.append(
            {
                "wins": res.wins,
                "losses": res.losses,
                "draws": res.draws,
                "capped": res.capped,
                "effective_games": res.effective_games,
            }
        )

    # --- Programmatic Ratification Metrics ---
    eval_gens = [s.generation for s in b_snapshots]
    matching_internal = [
        internal_elo_dict.get(g, float(g_a_cfg["base_elo_anchor"])) for g in eval_gens
    ]

    # Binary external scores (0-win wall: not discriminative) ...
    external_elos = [res.estimated_elo for res in b_results]
    # ... and continuous external signals (discriminative even at a 0-win wall)
    continuous_external = [
        (res.avg_sf_eval_agent if res.avg_sf_eval_agent is not None else float(res.avg_game_length))
        for res in b_results
    ]
    survival_lengths = [res.avg_game_length for res in b_results]
    material_gaps = [
        res.avg_material_gap_agent if res.avg_material_gap_agent is not None else 0.0
        for res in b_results
    ]

    def _monotonic_ratio(series: list[float]) -> float:
        if len(series) < 2:
            return 1.0
        count = sum(1 for i in range(len(series) - 1) if series[i + 1] > series[i])
        return count / float(len(series) - 1)

    mono_a_ratio = _monotonic_ratio(matching_internal)
    mono_b_ratio = _monotonic_ratio(external_elos)
    mono_b_cont_ratio = _monotonic_ratio(continuous_external)

    if (
        len(continuous_external) > 1
        and np.std(continuous_external) > 0
        and np.std(matching_internal) > 0
    ):
        corr_matrix = np.corrcoef(matching_internal, continuous_external)
        pearson_r = float(corr_matrix[0, 1])
    else:
        pearson_r = 0.0

    # --- Verdict Evaluation (continuous external signals; binary 0-win wall flagged) ---
    pass_threshold = float(pre_reg["pass_criterion_monotonic_ratio"])
    wall_detected = all(res.score == 0.0 for res in b_results)
    if mono_a_ratio >= pass_threshold and mono_b_cont_ratio >= pass_threshold and pearson_r >= 0.80:
        verdict = "[bold green]VERDICT: LEARNING_CONFIRMED (PASS)[/bold green]"
    elif mono_a_ratio >= pass_threshold and (mono_b_cont_ratio < 0.50 or pearson_r < 0.30):
        verdict = "[bold red]VERDICT: SELF_PLAY_COLLAPSE (FALSIFIED)[/bold red]"
    else:
        verdict = "[bold red]VERDICT: NO_LEARNING (FAIL)[/bold red]"

    results = {
        "experiment_id": config["experiment_id"],
        "generations": generations,
        "generations_evaluated": eval_gens,
        "internal_elo_ground_a": matching_internal,
        "external_elo_ground_b": external_elos,
        "stockfish_score_ratios": scores_ratio,
        "win_loss_draw_stats": win_loss_draw_stats,
        "continuous_external": {
            "avg_sf_eval_agent": continuous_external,
            "avg_game_length": survival_lengths,
            "avg_material_gap_agent": material_gaps,
            "wall_detected": wall_detected,
        },
        "ground_a_termination_counts": dict(tournament_result.termination_counts),
        "train_telemetry": train_telemetry,
        "metrics": {
            "ground_a_monotonicity_ratio": mono_a_ratio,
            "ground_b_monotonicity_ratio_binary": mono_b_ratio,
            "ground_b_monotonicity_ratio_continuous": mono_b_cont_ratio,
            "pearson_correlation_r": pearson_r,
            "pass_threshold": pass_threshold,
            "wall_detected": wall_detected,
        },
        "verdict": verdict,
    }

    return results


def main() -> None:
    """Entry point for experiment script."""
    config = load_experiment_config()
    console.print(
        Panel.fit(
            f"[bold blue]Executing Dual Ground Chess Experiment [{config['experiment_id']}][/bold blue]",
            border_style="blue",
        )
    )

    try:
        results = evaluate_dual_grounds(config)
    except KeyboardInterrupt:
        console.print(
            "\n[bold red][Experiment Interrupted by User (Ctrl+C)] Gracefully shutting down.[/bold red]"
        )
        _kill_worker_children()
        os._exit(130)

    os.makedirs("exports", exist_ok=True)
    export_path = Path("exports/chess_experiment_results.json")
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    table_res = Table(title="EXPERIMENTAL RATIFICATION RESULTS", header_style="bold magenta")
    table_res.add_column("Metric", style="cyan")
    table_res.add_column("Value", style="bold white")

    table_res.add_row("Generations Evaluated", str(results["generations_evaluated"]))
    table_res.add_row(
        "Ground A Monotonicity",
        f"{results['metrics']['ground_a_monotonicity_ratio'] * 100:.1f}%",
    )
    table_res.add_row(
        "Ground B Monotonicity (Continuous)",
        f"{results['metrics']['ground_b_monotonicity_ratio_continuous'] * 100:.1f}%",
    )
    table_res.add_row(
        "Ground B Monotonicity (Binary Elo)",
        f"{results['metrics']['ground_b_monotonicity_ratio_binary'] * 100:.1f}%",
    )
    table_res.add_row(
        "Pearson Correlation (r)",
        f"{results['metrics']['pearson_correlation_r']:.3f}",
    )
    table_res.add_row("Stockfish 0-Win Wall Detected", str(results["metrics"]["wall_detected"]))
    table_res.add_row("Final Verdict", results["verdict"])

    console.print(table_res)
    console.print(f"\n[bold green]Results exported to:[/bold green] {export_path}")


if __name__ == "__main__":
    main()
