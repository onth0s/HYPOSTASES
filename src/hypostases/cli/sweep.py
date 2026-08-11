"""HYPOSTASES CLI — Diagnostic and Formal Sweep Runner.

Spec Ref: Part VII §12.7.
Formally tests whether the WITHDRAW identifiability gap is closed.
Z(theta_low, theta_high, n, N, seed) = (mu_high - mu_low) / sqrt((var_high + var_low)/2)

Gap closed at (n, N) iff over S >= 5 seeds:
  1. Directional correctness: Z > 0 in >= ceil(0.8*S) seeds
  2. Statistical separation: median(Z) > 1.0
  3. Non-degeneracy: var in [0.5, 20] in all but <= 1 seed
"""

from __future__ import annotations

import argparse
import json

from hypostases.simulation import run_sweep_benchmark


def run_sweep(
    steps_list: list[int],
    n_particles: int,
    seeds: list[int],
) -> list[dict]:
    eval_results = run_sweep_benchmark(
        steps_list=steps_list,
        n_particles=n_particles,
        seeds=seeds,
    )
    return eval_results


def format_sweep_output(eval_results: list[dict], output_format: str = "table") -> None:
    if output_format == "json":
        print(json.dumps(eval_results, indent=2))
    else:
        print("=== HYPOSTASES Phase A: Diagnostic Sweep (Specification §12.7) ===")
        print(
            f"{'n_steps':>8} | {'n_particles':>12} | {'median_Z':>10} | {'dir_ok':>8} | {'passed':>8}"
        )
        print("-" * 55)
        for r in eval_results:
            print(
                f"{r['n_steps']:>8} | {r['n_particles']:>12} | {r['cond2_median_z']:>10.3f} | "
                f"{r['cond1_count']}/{r['n_seeds']:>6} | {r['passed']!s:>8}"
            )


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "sweep", help="Run diagnostic / formal 3-condition sweep (Part VII §12.7)"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[10, 50, 200],
        help="List of step counts (default: 10 50 200)",
    )
    parser.add_argument(
        "--particles", type=int, default=300, help="Particle count N (default: 300)"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Random seeds (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--output-format", choices=["table", "json"], default="table", help="Output format"
    )
    parser.set_defaults(func=main_sweep)


def main_sweep(args: argparse.Namespace) -> None:
    res = run_sweep_benchmark(
        steps_list=args.steps,
        n_particles=args.particles,
        seeds=args.seeds,
    )
    format_sweep_output(res, output_format=args.output_format)
