"""Configuration and mode resolution utilities for Chess training runner."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt

from hypostases.simulation.run_manager import UnifiedRunManager

console = Console()


def resolve_resume_target(
    run_manager: UnifiedRunManager,
    n: int,
    run_dir_override: str | None,
) -> Path | None:
    """Find the target run directory from an override path or nth latest run."""
    if run_dir_override:
        p = Path(run_dir_override)
        if not p.exists():
            console.print(f"[bold red][ERROR][/bold red] Run directory not found: {p}")
            sys.exit(1)
        return p
    return run_manager.get_nth_latest_run(n=n)


def interactive_resume_or_extend(
    run_dir: Path,
    meta: dict[str, Any],
) -> tuple[str, int]:
    """Interactive prompt when an existing run directory is found.

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
