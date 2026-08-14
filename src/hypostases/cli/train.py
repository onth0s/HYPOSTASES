"""HYPOSTASES CLI — train subcommand.

Dispatches domain-specific training via the DomainRegistry trainer registry.
This module is fully domain-agnostic: it contains zero imports of plugin
or chess-specific modules. The chess training implementation registers itself
in ``hypostases.plugins.domains.chess.chess_train_runner``.

Usage examples::

    # New training run (20 generations)
    hypostases train --chess

    # New training run with custom generation count
    hypostases train --chess --gens 40

    # Resume or extend the latest run (interactive prompt if interrupted)
    hypostases train --chess resume

    # Resume or extend the 2nd-most-recent run
    hypostases train --chess resume 2

    # Resume from a specific run directory
    hypostases train --chess resume --run-dir exports/runs/run_20250801_120000
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback

from rich.console import Console

from hypostases.domains.registry import DomainRegistry

console = Console()

# Plugin module that self-registers its trainer when imported.
_DOMAIN_PLUGIN_MODULES: dict[str, str] = {
    "chess": "hypostases.plugins.domains.chess.chess_train_runner",
}


def _load_domain_trainer(domain: str) -> None:
    """Imports the plugin module so it self-registers its trainer in DomainRegistry."""
    module_path = _DOMAIN_PLUGIN_MODULES.get(domain)
    if module_path is None:
        console.print(
            f"[bold red][ERROR][/bold red] No plugin module registered for domain '{domain}'."
        )
        sys.exit(1)
    try:
        importlib.import_module(module_path)
    except ImportError as exc:
        console.print(
            f"[bold red][ERROR][/bold red] Could not load plugin for domain '{domain}': {exc}\n"
            f"Install it with: pip install hypostases[{domain}]"
        )
        sys.exit(1)


def add_subparser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "train",
        help="Train a HYPOSTASES domain agent via self-play RL",
        description="Run domain-specific self-play reinforcement learning training.",
    )

    # Domain flags
    domain_group = parser.add_argument_group("domain")
    domain_group.add_argument(
        "--chess",
        action="store_true",
        help="Train the Chess domain agent.",
    )

    # Resume flags
    resume_group = parser.add_argument_group("resume")
    resume_group.add_argument(
        "resume",
        nargs="?",
        default=None,
        metavar="resume",
        help="'resume' keyword to resume the Nth most-recent run.",
    )
    resume_group.add_argument(
        "resume_n",
        nargs="?",
        type=int,
        default=None,
        metavar="N",
        help="Which run to resume (1=latest, 2=second latest, …). Default: 1.",
    )
    resume_group.add_argument(
        "--run-dir",
        default=None,
        metavar="PATH",
        help="Explicit run directory to resume from (overrides N).",
    )

    # Training hyper-parameters
    # Defaults (None) fall back to chess_experiment_config.yaml, then trainer
    # defaults — resolved inside the domain runner with CLI > YAML precedence.
    hp_group = parser.add_argument_group("hyperparameters")
    hp_group.add_argument(
        "--gens", type=int, default=None, help="Total generations (default: from config YAML)."
    )
    hp_group.add_argument(
        "--games",
        type=int,
        default=None,
        help="Games per generation (default: from config YAML).",
    )
    hp_group.add_argument(
        "--snapshot-interval",
        type=int,
        default=None,
        help="Snapshot every N generations (default: from config YAML).",
    )
    hp_group.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: from config YAML)."
    )
    hp_group.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes (default: from config YAML).",
    )

    # Output flags
    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--quiet", action="store_true", help="Suppress verbose generation-level output."
    )
    output_group.add_argument(
        "--log-games", action="store_true", help="Log individual game results to console."
    )

    parser.set_defaults(func=_cli_handler)


def _cli_handler(args: argparse.Namespace) -> None:
    """Dispatches train subcommand to domain-specific training via DomainRegistry."""
    if not args.chess:
        console.print(
            "[bold red][ERROR][/bold red] Please specify a domain flag. "
            "Currently supported: [bold]--chess[/bold]"
        )
        sys.exit(1)

    # Validate resume syntax: positional "resume" keyword + optional N
    resume_n: int | None = None
    if args.resume is not None:
        if args.resume != "resume":
            console.print(
                f"[bold red][ERROR][/bold red] Unknown positional argument '{args.resume}'.\n"
                "Did you mean: [bold]hypostases train --chess resume[/bold]?"
            )
            sys.exit(1)
        resume_n = args.resume_n if args.resume_n is not None else 1

    domain = "chess"

    # Load the plugin — this triggers @DomainRegistry.register_trainer("chess")
    _load_domain_trainer(domain)

    try:
        trainer_fn = DomainRegistry.get_trainer(domain)
    except KeyError as exc:
        console.print(f"[bold red][ERROR][/bold red] {exc}")
        sys.exit(1)

    try:
        trainer_fn(
            resume_n=resume_n,
            run_dir_override=args.run_dir,
            total_gens=args.gens,
            games_per_gen=args.games,
            snapshot_interval=args.snapshot_interval,
            seed=args.seed,
            workers=args.workers,
            verbose=not args.quiet,
            log_games=args.log_games,
        )
    except KeyboardInterrupt:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130)
    except BaseException as exc:
        console.print(f"[bold red][ERROR][/bold red] Chess training failed: {exc}")
        console.print(traceback.format_exc(), markup=False)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    # Do NOT return through normal interpreter shutdown: after a graceful interrupt
    # (or crash) the ProcessPoolExecutor's manager thread can be stuck joining a
    # queue feeder that blocks forever on CPython 3.10 (see
    # chess_trainer._terminate_pool_workers), which would hang threading._shutdown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
