"""HYPOSTASES CLI — Specification Tools (Spec Merge).

Spec Ref: Part I - VII document management.
"""

from __future__ import annotations

import argparse

from hypostases.utils.merge_spec import main as merge_main


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    spec_parser = subparsers.add_parser("spec", help="Specification document management tools")
    spec_subparsers = spec_parser.add_subparsers(dest="spec_command", required=True)

    merge_parser = spec_subparsers.add_parser(
        "merge", help="Merge spec/ markdown parts into HYPOSTASES_<epoch>.md"
    )
    merge_parser.add_argument(
        "--dry-run", action="store_true", help="Preview merge output without writing file"
    )
    merge_parser.add_argument(
        "--output-dir", type=str, default=None, help="Target output directory"
    )
    merge_parser.set_defaults(func=main_spec_merge)


def main_spec_merge(args: argparse.Namespace) -> None:
    # Forward arguments to merge_spec utility directly
    sys_args = []
    if args.dry_run:
        sys_args.append("--dry-run")
    if args.output_dir:
        sys_args.extend(["--output-dir", args.output_dir])

    merge_main(sys_args)
