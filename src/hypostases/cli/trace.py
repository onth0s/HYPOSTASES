"""HYPOSTASES CLI — Forward Simulation Trace Runner.

Spec Ref: Part V, Part VI §8.
Runs forward simulation of social agents sharing a pool.
"""

from __future__ import annotations

import argparse
import json

from hypostases.simulation import run_simulation_trace


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("trace", help="Run forward simulation trace (Part VI §8)")
    parser.add_argument(
        "--steps", type=int, default=12, help="Number of Tier-1 steps (default: 12)"
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed (default: 7)")
    parser.add_argument(
        "--export-json", type=str, default=None, help="File path to save JSON trace output"
    )
    parser.set_defaults(func=main_trace)


def main_trace(args: argparse.Namespace) -> None:
    res = run_simulation_trace(steps=args.steps, seed=args.seed)

    print(f"=== HYPOSTASES Forward Simulation Trace (Steps={res['steps']}, Seed={res['seed']}) ===")
    print(f"{'Step':>5} | {'Pool':>6} | {'Agent A Action':>18} | {'Agent B Action':>18}")
    print("-" * 55)
    for s in res["trace"]:
        print(
            f"{s['step']:>5} | {s['pool_after']:>6.2f} | "
            f"{s['actions'].get('Agent_A', ''):>18} | "
            f"{s['actions'].get('Agent_B', ''):>18}"
        )

    print("\n--- Final Agent States ---")
    for name, st in res["agents"].items():
        print(
            f"{name}: reserve={st['reserve']}, social_capital={st['social_capital']}, "
            f"mood={st['mood']}, u={st['u']}"
        )

    if args.export_json:
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved trace output to {args.export_json}")
