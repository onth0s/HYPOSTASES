"""HYPOSTASES CLI — Main Command Dispatcher.

Entry point for the executable console script 'hypostases'.
Subcommands:
  - hypostases trace: Run forward simulation trace
  - hypostases infer: Run inverse inference particle filter
  - hypostases sweep: Run diagnostic / formal 3-condition sweep
  - hypostases spec merge: Merge specification markdown parts
"""

from __future__ import annotations

import argparse
import sys

from hypostases.cli import infer, spec, sweep, sweep_memory, trace


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hypostases",
        description="HYPOSTASES — Agent-based Modeling Framework & Inverse Inference CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    trace.add_subparser(subparsers)
    infer.add_subparser(subparsers)
    sweep.add_subparser(subparsers)
    sweep_memory.add_subparser(subparsers)
    spec.add_subparser(subparsers)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
