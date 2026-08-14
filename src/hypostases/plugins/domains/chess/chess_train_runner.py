"""Chess domain training runner — self-registers with DomainRegistry.

This module is the only place in the Chess plugin that ties together:
  - ChessSelfPlayTrainer
  - chess_experiment_config.yaml (Rule 006 data-driven defaults)
  - UnifiedRunManager (checkpoint, NNUE, metadata persistence)
  - Signal-safe mid-training exit handling (graceful + forced second Ctrl-C)
  - Interactive resume / extend prompts

Config precedence: CLI flags > loaded checkpoint (resume) > YAML > trainer defaults.

Import this module (or let cli/train.py do so via importlib) to activate
the ``@DomainRegistry.register_trainer("chess")`` decorator.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt

from hypostases.domains.registry import DomainRegistry
from hypostases.plugins.domains.chess.chess_config import (
    load_chess_experiment_config,
    resolve_chess_training_config,
)
from hypostases.plugins.domains.chess.chess_trainer import ChessSelfPlayTrainer
from hypostases.plugins.domains.chess.ground_a_self_play import run_ground_a_benchmark
from hypostases.simulation.run_manager import UnifiedRunManager

console = Console()


# ---------------------------------------------------------------------------
# Resume / extend interactive helpers
# ---------------------------------------------------------------------------


def _resolve_resume_target(
    run_manager: UnifiedRunManager,
    n: int,
    run_dir_override: str | None,
) -> Path | None:
    if run_dir_override:
        p = Path(run_dir_override)
        if not p.exists():
            console.print(f"[bold red][ERROR][/bold red] Run directory not found: {p}")
            sys.exit(1)
        return p
    return run_manager.get_nth_latest_run(n=n)


def _interactive_resume_or_extend(
    run_dir: Path,
    meta: dict[str, Any],
) -> tuple[str, int]:
    """Interactive prompt when a run directory is found.

    Returns (mode, extra_gens) where mode is one of:
      ``"resume"``  — resume remaining gens of an interrupted run
      ``"extend"``  — add N more gens to a completed (or interrupted) run
      ``"abort"``   — user chose not to proceed
    """
    status = meta.get("status", "unknown")
    last = meta.get("last_completed_gen", 0)
    target = meta.get("target_generations", last)
    interrupted_at = meta.get("interrupted_at_gen")

    console.print()
    console.print(
        Panel(
            f"[bold]Run:[/bold] [cyan]{run_dir.name}[/cyan]\n"
            f"[bold]Status:[/bold] {status}\n"
            f"[bold]Target:[/bold] {target} generations  "
            f"[bold]Completed:[/bold] {last}"
            + (
                f"  [bold yellow][interrupted at gen {interrupted_at}][/bold yellow]"
                if interrupted_at is not None
                else ""
            ),
            title="[bold yellow]Existing Run Detected[/bold yellow]",
            border_style="yellow",
        )
    )

    if status == UnifiedRunManager.STATUS_INTERRUPTED:
        remaining = target - last
        console.print(
            f"[bold yellow]This run was interrupted mid-training.[/bold yellow] "
            f"{remaining} generation(s) remaining."
        )
        console.print(f"  [R]esume remaining {remaining} gen(s) / add [N] more / [A]bort")
        choice = console.input("  Choice [r/n/a]: ").strip().lower()
        if choice in ("", "r"):
            return "resume", remaining
        elif choice == "n":
            extra = IntPrompt.ask("  How many additional generations to add?", console=console)
            return "extend", int(extra)
        else:
            return "abort", 0
    else:
        extra = IntPrompt.ask(
            f"  Run completed ({last} gens). How many additional generations to add?",
            console=console,
        )
        return "extend", int(extra)


def _kill_worker_children() -> None:
    """Terminates all live multiprocessing child processes (pool workers).

    Called on the forced second Ctrl-C so the hard exit never orphans worker
    processes (they ignore SIGINT and would otherwise outlive the parent).
    """
    with contextlib.suppress(Exception):
        for p in multiprocessing.active_children():
            p.terminate()


# ---------------------------------------------------------------------------
# Core training function — registered as the "chess" trainer
# ---------------------------------------------------------------------------


@DomainRegistry.register_trainer("chess")
def run_chess_training(
    *,
    resume_n: int | None,
    run_dir_override: str | None,
    total_gens: int | None,
    games_per_gen: int | None,
    snapshot_interval: int | None,
    seed: int | None,
    workers: int | None,
    verbose: bool,
    log_games: bool,
) -> None:
    """Full chess self-play training pipeline.

    Resolves mode (new / resume / extend), loads chess_experiment_config.yaml as
    the default parameter source (CLI flags override), restores checkpoint state
    when resuming, installs a SIGINT/SIGTERM handler for mid-training exits, runs
    ChessSelfPlayTrainer, and atomically persists all artifacts via
    UnifiedRunManager on completion or interruption.
    """
    run_manager = UnifiedRunManager()

    raw_config = load_chess_experiment_config()
    if raw_config:
        console.print(
            "[bold blue][CONFIG][/bold blue] Loaded [cyan]chess_experiment_config.yaml[/cyan]"
        )

    # CLI values take precedence only when explicitly provided (argparse defaults
    # to None; the effective values come from resolve_chess_training_config).
    cli_overrides: dict[str, Any] = {}
    if total_gens is not None:
        cli_overrides["gens"] = total_gens
    if games_per_gen is not None:
        cli_overrides["games"] = games_per_gen
    if snapshot_interval is not None:
        cli_overrides["snapshot_interval"] = snapshot_interval
    if seed is not None:
        cli_overrides["seed"] = seed
    if workers is not None:
        cli_overrides["workers"] = workers

    # ------------------------------------------------------------------
    # 1. Determine mode: new / resume / extend
    # ------------------------------------------------------------------
    resume_run_dir: Path | None = None
    mode = "new"
    resume_from_gen = 0

    if resume_n is not None:
        resume_run_dir = _resolve_resume_target(run_manager, resume_n, run_dir_override)
        if resume_run_dir is None:
            console.print(
                f"[bold red][ERROR][/bold red] No run #{resume_n} found in exports/runs/."
            )
            sys.exit(1)

        try:
            meta = run_manager.load_run_metadata(resume_run_dir)
        except FileNotFoundError:
            console.print(
                f"[bold red][ERROR][/bold red] No run_metadata.json in {resume_run_dir}. "
                "Cannot resume."
            )
            sys.exit(1)

        mode, extra_gens = _interactive_resume_or_extend(resume_run_dir, meta)

        if mode == "abort":
            console.print("[yellow]Aborted.[/yellow]")
            sys.exit(0)

        run_dir = resume_run_dir
    else:
        run_dir = run_manager.create_run()
        console.print(f"[bold cyan][NEW RUN][/bold cyan] Created [cyan]{run_dir.name}[/cyan]")

    # Effective configuration (CLI > YAML > defaults)
    cfg = resolve_chess_training_config(cli=cli_overrides, raw_config=raw_config)

    if resume_n is not None:
        last_completed = meta.get("last_completed_gen", 0)
        target_orig = meta.get("target_generations", last_completed)

        if mode == "resume":
            resume_from_gen = last_completed
            cfg["total_generations"] = target_orig + extra_gens
        else:  # extend
            resume_from_gen = last_completed
            cfg["total_generations"] = last_completed + extra_gens

        console.print(
            f"[bold cyan][RESUME][/bold cyan] Continuing [cyan]{run_dir.name}[/cyan] "
            f"from gen {resume_from_gen} → target {cfg['total_generations']}."
        )

    run_manager.save_run_config(run_dir, cfg)

    # ------------------------------------------------------------------
    # 2. Load or initialize checkpoint state
    # ------------------------------------------------------------------
    loaded_theta_meta = None
    loaded_temperature = None
    loaded_beta_efe = None
    loaded_nnue_weights = None
    loaded_telemetry: dict[int, dict[str, Any]] = {}

    if mode in ("resume", "extend") and resume_run_dir is not None:
        try:
            ck_state = run_manager.load_checkpoint_state(run_dir)
            loaded_theta_meta = ck_state["agent"]["theta_meta"]
            loaded_temperature = ck_state["agent"]["temperature"]
            loaded_beta_efe = ck_state["agent"]["beta_efe"]
            loaded_telemetry = {int(k): v for k, v in ck_state.get("telemetry", {}).items()}
            console.print(
                f"[bold green][CHECKPOINT][/bold green] Loaded agent state from "
                f"[cyan]{run_dir / 'checkpoint_state.yaml'}[/cyan]"
            )
        except FileNotFoundError:
            console.print(
                "[bold yellow][WARN][/bold yellow] No checkpoint_state.yaml found; "
                "starting with fresh agent parameters."
            )

        try:
            loaded_nnue_weights = run_manager.load_nnue_weights(run_dir, tag="latest")
            console.print(
                f"[bold green][NNUE][/bold green] Loaded NNUE weights from "
                f"[cyan]{run_dir / 'nnue' / 'nnue_latest.npz'}[/cyan]"
            )
        except FileNotFoundError:
            console.print(
                "[bold yellow][WARN][/bold yellow] No NNUE weights found; "
                "starting with fresh random net."
            )

    # ------------------------------------------------------------------
    # 3. Initialize trainer
    # ------------------------------------------------------------------
    trainer = ChessSelfPlayTrainer(
        learning_rate=cfg["learning_rate"],
        beta_efe=loaded_beta_efe or cfg["beta_efe"],
        initial_temperature=loaded_temperature or cfg["initial_temperature"],
        value_gamma=cfg["value_gamma"],
        max_workers=cfg["workers"],
    )
    if loaded_telemetry:
        trainer.telemetry = loaded_telemetry

    initial_priors: Any = (
        loaded_theta_meta if loaded_theta_meta is not None else cfg["initial_priors"]
    )

    started_at = datetime.now(timezone.utc).isoformat()
    run_manager.save_run_metadata(
        run_dir,
        status=UnifiedRunManager.STATUS_RUNNING,
        target_generations=cfg["total_generations"],
        last_completed_gen=resume_from_gen,
        started_at=started_at,
    )

    # ------------------------------------------------------------------
    # 4. Signal handler for mid-training exits
    # ------------------------------------------------------------------
    _interrupted = [False]

    def _flush_and_exit(signum: int, frame: Any) -> None:
        if _interrupted[0]:
            console.print(
                "\n[bold red][FORCED EXIT][/bold red] Second interrupt received; "
                "terminating immediately."
            )
            _kill_worker_children()
            os._exit(130)
        _interrupted[0] = True
        console.print(
            "\n[bold yellow][Ctrl-C][/bold yellow] Graceful stop requested — finishing "
            "current generation, then saving checkpoint."
        )

    signal.signal(signal.SIGINT, _flush_and_exit)
    signal.signal(signal.SIGTERM, _flush_and_exit)

    # ------------------------------------------------------------------
    # 5. Run training
    # ------------------------------------------------------------------
    snapshots_meta: list[dict[str, Any]] = run_manager.load_snapshots_index(run_dir)

    console.print()
    console.print(
        Panel(
            f"[bold]Run dir:[/bold] [cyan]{run_dir}[/cyan]\n"
            f"[bold]Total generations:[/bold] {cfg['total_generations']}  "
            f"[bold]Start gen:[/bold] {resume_from_gen}\n"
            f"[bold]Games / gen:[/bold] {cfg['games_per_generation']}  "
            f"[bold]Seed:[/bold] {cfg['seed']}  [bold]Workers:[/bold] {cfg['workers']}",
            title="[bold cyan]HYPOSTASES Chess Training[/bold cyan]",
            border_style="cyan",
        )
    )

    try:
        snapshots = trainer.execute_self_play_training_run(
            total_generations=cfg["total_generations"],
            snapshot_interval_k=cfg["snapshot_interval_k"],
            games_per_generation=cfg["games_per_generation"],
            seed=cfg["seed"],
            min_temperature=cfg["min_temperature"],
            max_moves_training=cfg["max_moves_training"],
            early_adjudication_material=cfg["early_adjudication_material"],
            initial_priors=initial_priors,
            value_gamma=cfg["value_gamma"],
            curriculum_probability=cfg["curriculum_probability"],
            resign_value_threshold=cfg["resign_value_threshold"],
            resign_confirm_moves=cfg["resign_confirm_moves"],
            nnue_epochs=cfg["nnue_epochs"],
            nnue_learning_rate=cfg["nnue_learning_rate"],
            replay_capacity=cfg["replay_capacity"],
            search_depth=cfg["search_depth"],
            adjudicate_bare_king_requires_mate=cfg["adjudicate_bare_king_requires_mate"],
            verbose=verbose,
            log_game_details=log_games,
            interrupt_flag=lambda: _interrupted[0],
        )
    except KeyboardInterrupt:
        _interrupted[0] = True
        snapshots = []

    # ------------------------------------------------------------------
    # 6. Finalize: save checkpoint + NNUE + metadata
    # ------------------------------------------------------------------
    final_gen = resume_from_gen
    final_theta = None
    final_temperature = loaded_temperature or cfg["initial_temperature"]
    final_beta_efe = loaded_beta_efe or cfg["beta_efe"]

    if snapshots:
        last_snap = snapshots[-1]
        final_gen = resume_from_gen + last_snap.generation
        final_theta = last_snap.theta_meta.tolist()
        final_temperature = last_snap.temperature
        nnue_weights = last_snap.nnue_weights
        for snap in snapshots:
            abs_gen = resume_from_gen + snap.generation
            snapshots_meta.append(
                {
                    "generation": abs_gen,
                    "temperature": snap.temperature,
                    "theta_meta": snap.theta_meta.tolist(),
                }
            )
    else:
        nnue_weights = loaded_nnue_weights or {}

    interrupted_at_gen = final_gen if _interrupted[0] else None
    status = (
        UnifiedRunManager.STATUS_INTERRUPTED
        if _interrupted[0]
        else UnifiedRunManager.STATUS_COMPLETED
    )

    run_manager.save_full_checkpoint(
        run_dir,
        status=status,
        target_generations=cfg["total_generations"],
        last_completed_gen=final_gen,
        total_games_played=final_gen * cfg["games_per_generation"],
        theta_meta=final_theta or (loaded_theta_meta or []),
        temperature=final_temperature,
        beta_efe=final_beta_efe,
        nnue_weights=nnue_weights,
        telemetry=trainer.telemetry,
        snapshots_meta=snapshots_meta,
        started_at=started_at,
        interrupted_at_gen=interrupted_at_gen,
        interrupted_at=datetime.now(timezone.utc).isoformat() if _interrupted[0] else None,
        completed_at=datetime.now(timezone.utc).isoformat() if not _interrupted[0] else None,
    )

    # ------------------------------------------------------------------
    # 7. PGN export (Ground A benchmark)
    # ------------------------------------------------------------------
    if not _interrupted[0] and snapshots:
        console.print(
            "\n[bold cyan][Ground A][/bold cyan] Running benchmark games across snapshots ..."
        )
        try:
            pgn_dir = run_dir / "pgn" / "ground_a"
            run_ground_a_benchmark(
                snapshots,
                pgn_output_dir=str(pgn_dir),
                games_per_pair=cfg["ground_a_games_per_pair"],
                max_moves=cfg["ground_a_max_moves_eval"],
                eval_temperature=cfg["ground_a_eval_temperature"],
                search_depth=cfg["ground_a_search_depth"],
                max_workers=cfg["workers"],
                verbose=verbose,
            )
            console.print(f"[bold green]✓[/bold green] PGN written to [cyan]{pgn_dir}[/cyan]")
        except Exception as exc:
            console.print(f"[bold yellow][WARN][/bold yellow] Ground A benchmark failed: {exc}")

    # ------------------------------------------------------------------
    # 8. Final status panel
    # ------------------------------------------------------------------
    console.print()
    if _interrupted[0]:
        console.print(
            Panel(
                f"[bold yellow]Training interrupted at generation {final_gen}.[/bold yellow]\n"
                f"State saved to [cyan]{run_dir}[/cyan].\n"
                "Resume with: [bold]hypostases train --chess resume[/bold]",
                title="[bold yellow]⚠ Interrupted[/bold yellow]",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold green]Training complete![/bold green] {final_gen} generation(s) finished.\n"
                f"Artifacts: [cyan]{run_dir}[/cyan]",
                title="[bold green]✓ Done[/bold green]",
                border_style="green",
            )
        )
