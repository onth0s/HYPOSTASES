"""HYPOSTASES CLI — Memory Decay Stability Sweep.

Runs simulations over multiple steps to check how memory decay models impact w.sigma2 stability.
"""

from __future__ import annotations

import argparse

from hypostases.engine import Action, ActionType, evolve, feedback
from hypostases.simulation.harness import build_agent


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "sweep-memory", help="Run memory decay stability sweep (Phase 2)"
    )
    parser.add_argument(
        "--steps", type=int, default=30, help="Number of simulation steps (default: 30)"
    )
    parser.set_defaults(func=main_sweep_memory)


def run_memory_sweep(steps: int, decay_mode: str, memory_decay: float) -> list[float]:
    agent = build_agent(name="agent", memory_decay=memory_decay, reserve=10.0)
    agent.sigma.decay_mode = decay_mode

    sigma2_history = [agent.sigma.w.sigma2]

    for _ in range(steps):
        pool_before = 10.0
        pool_after = 10.0
        action = Action(ActionType.REQUEST, amount=1.0)
        delta_log = {
            "pool_before": pool_before,
            "pool_after_shares": pool_before,
            "pool_after": pool_after,
            "shares_total": 0.0,
            "requests_total": 1.0,
            "granted": {"agent": 1.0},
        }

        phi = feedback(
            agent.sigma,
            pool_before=pool_before,
            pool_after=pool_after,
            action=action,
            delta_log=delta_log,
            agent_name="agent",
        )
        evolve(agent.sigma, phi)
        sigma2_history.append(agent.sigma.w.sigma2)

    return sigma2_history


def main_sweep_memory(args: argparse.Namespace) -> None:
    decays = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
    modes = ["variance", "precision"]

    print("=== HYPOSTASES Memory Decay Stability Sweep ===")
    print(f"Running for {args.steps} steps. SIGMA2_MAX = 20.0, SIGMA2_MIN = 0.0001\n")

    for mode in modes:
        print(f"--- Mode: {mode.upper()} ---")
        print(f"{'Decay':<6} | {'Initial':<8} | {'Midpoint':<8} | {'Final':<8}")
        print("-" * 38)
        for decay in decays:
            history = run_memory_sweep(args.steps, mode, decay)
            mid = history[len(history) // 2]
            final = history[-1]
            print(f"{decay:<6.2f} | {history[0]:<8.4f} | {mid:<8.4f} | {final:<8.4f}")
        print()
