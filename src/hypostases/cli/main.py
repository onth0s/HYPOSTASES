"""HYPOSTASES CLI — Main Command Dispatcher.

Entry point for the executable console script 'hypostases'.
Subcommands:
  - hypostases trace: Run forward simulation trace
  - hypostases infer: Run inverse inference particle filter
  - hypostases sweep: Run diagnostic / formal 3-condition sweep
  - hypostases spec merge: Merge specification markdown parts
  - hypostases train: Train a domain agent via self-play RL
"""

from __future__ import annotations

import argparse
import sys
from typing import ClassVar

from rich.console import Console
from rich_argparse import RichHelpFormatter

from hypostases.cli import infer, spec, sweep, sweep_memory, trace, train

console = Console(stderr=True)


class HypostasesHelpFormatter(RichHelpFormatter):
    """Custom rich-argparse help formatter with HYPOSTASES theme styling."""

    styles: ClassVar[dict[str, str]] = {
        "argparse.args": "bold cyan",
        "argparse.groups": "bold magenta",
        "argparse.help": "default",
        "argparse.metavar": "yellow",
        "argparse.syntax": "bold white",
        "argparse.text": "default",
        "argparse.prog": "bold cyan",
    }


class RichArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that outputs errors and help in rich formatted color."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("formatter_class", HypostasesHelpFormatter)
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        console.print(f"[bold red]error:[/bold red] {message}")
        self.exit(2)


def main() -> None:
    # Ensure subcommand errors and help are also rich formatted
    RichHelpFormatter.highlights.append(
        r"\b(?P<subcommand>trace|infer|sweep|sweep-memory|spec|train)\b"
    )

    parser = RichArgumentParser(
        prog="hypostases",
        description="HYPOSTASES — Agent-based Modeling Framework & Inverse Inference CLI",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        parser_class=RichArgumentParser,
    )

    trace.add_subparser(subparsers)
    infer.add_subparser(subparsers)
    sweep.add_subparser(subparsers)
    sweep_memory.add_subparser(subparsers)
    spec.add_subparser(subparsers)
    train.add_subparser(subparsers)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
