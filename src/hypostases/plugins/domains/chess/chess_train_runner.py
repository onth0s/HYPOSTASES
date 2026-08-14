"""Chess domain training runner — self-registers with DomainRegistry.

This module is the only place in the Chess plugin that ties together:
  - ChessSelfPlayTrainer
  - UnifiedRunManager (checkpoint, NNUE, metadata persistence)
  - Signal-safe mid-training exit handling
  - Interactive resume / extend prompts

Import this module (or let cli/train.py do so via importlib) to activate
the ``@DomainRegistry.register_trainer("chess")`` decorator.
"""

from __future__ import annotations

import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt

from hypostases.domains.registry import DomainRegistry
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


# ---------------------------------------------------------------------------
# Core training function — registered as the "chess" trainer
# ---------------------------------------------------------------------------


@DomainRegistry.register_trainer("chess")
def run_chess_training(
    *,
    resume_n: int | None,
    run_dir_override: str | None,
    total_gens: int,
    games_per_gen: int,
    snapshot_interval: int,
    seed: int,
    workers: int,
    verbose: bool,
    log_games: bool,
) -> None:
    """Full chess self-play training pipeline.

    Resolves mode (new / resume / extend), restores checkpoint state when
    resuming, installs a SIGINT/SIGTERM handler for mid-training exits, runs
    ChessSelfPlayTrainer, and atomically persists all artifacts via
    UnifiedRunManager on completion or interruption.
    """
    run_manager = UnifiedRunManager()

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

        last_completed = meta.get("last_completed_gen", 0)
        target_orig = meta.get("target_generations", last_completed)

        if mode == "resume":
            resume_from_gen = last_completed
            total_gens = target_orig + extra_gens
        else:  # extend
            resume_from_gen = last_completed
            total_gens = last_completed + extra_gens

        run_dir = resume_run_dir
        console.print(
            f"[bold cyan][RESUME][/bold cyan] Continuing [cyan]{run_dir.name}[/cyan] "
            f"from gen {resume_from_gen} → target {total_gens}."
        )
    else:
        run_dir = run_manager.create_run()
        console.print(f"[bold cyan][NEW RUN][/bold cyan] Created [cyan]{run_dir.name}[/cyan]")

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
        beta_efe=loaded_beta_efe or 0.05,
        initial_temperature=loaded_temperature or 0.8,
        max_workers=workers,
    )
    if loaded_telemetry:
        trainer.telemetry = loaded_telemetry

    initial_priors: Any = loaded_theta_meta if loaded_theta_meta is not None else "random"

    started_at = datetime.now(UTC).isoformat()
    run_manager.save_run_metadata(
        run_dir,
        status=UnifiedRunManager.STATUS_RUNNING,
        target_generations=total_gens,
        last_completed_gen=resume_from_gen,
        started_at=started_at,
    )

    # ------------------------------------------------------------------
    # 4. Signal handler for mid-training exits
    # ------------------------------------------------------------------
    _interrupted = [False]

    def _flush_and_exit(signum: int, frame: Any) -> None:
        _interrupted[0] = True

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
            f"[bold]Total generations:[/bold] {total_gens}  "
            f"[bold]Start gen:[/bold] {resume_from_gen}\n"
            f"[bold]Games / gen:[/bold] {games_per_gen}  "
            f"[bold]Seed:[/bold] {seed}  [bold]Workers:[/bold] {workers}",
            title="[bold cyan]HYPOSTASES Chess Training[/bold cyan]",
            border_style="cyan",
        )
    )

    try:
        snapshots = trainer.execute_self_play_training_run(
            total_generations=total_gens,
            snapshot_interval_k=snapshot_interval,
            games_per_generation=games_per_gen,
            seed=seed,
            verbose=verbose,
            log_game_details=log_games,
            initial_priors=initial_priors,
        )
    except KeyboardInterrupt:
        _interrupted[0] = True
        snapshots = []

    # ------------------------------------------------------------------
    # 6. Finalize: save checkpoint + NNUE + metadata
    # ------------------------------------------------------------------
    final_gen = resume_from_gen
    final_theta = None
    final_temperature = loaded_temperature or 0.8
    final_beta_efe = loaded_beta_efe or 0.05

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
        target_generations=total_gens,
        last_completed_gen=final_gen,
        total_games_played=final_gen * games_per_gen,
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
            run_ground_a_benchmark(snapshots, pgn_output_dir=str(pgn_dir), verbose=verbose)
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
