"""HYPOSTASES CLI — Inverse Inference Runner.

Spec Ref: Part VII §10, §11, §12.
Runs SMC particle filter inference on observed action & pool traces.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from hypostases.engine.constants import DEFAULT_XI
from hypostases.inference import (
    goal_posterior,
    infer,
    infer_hierarchical,
    summarize_kalman,
    summarize_map,
)
from hypostases.simulation import generate_sample_trace


def run_cli_infer(
    n_particles: int = 300,
    seed: int = 42,
    agent_name: str = "Agent_A",
    n_steps: int = 12,
    lag_window: int | None = None,
    use_hierarchical: bool = False,
    use_rao_blackwell: bool = False,
) -> dict:
    xi = DEFAULT_XI
    actions, pools = generate_sample_trace(n_steps=n_steps, seed=seed)

    rng = np.random.default_rng(seed)
    if use_hierarchical:
        particles = infer_hierarchical(
            observed_actions=actions,
            observed_pool_trace=pools,
            xi=xi,
            n_particles=n_particles,
            agent_name=agent_name,
            lag_window=lag_window,
            rng=rng,
        )
    else:
        particles = infer(
            observed_actions=actions,
            observed_pool_trace=pools,
            xi=xi,
            n_particles=n_particles,
            agent_name=agent_name,
            lag_window=lag_window,
            use_rao_blackwell=use_rao_blackwell,
            rng=rng,
        )

    map_state = summarize_map(particles)
    kalman = summarize_kalman(particles)
    g_post = goal_posterior(particles)

    res = {
        "n_particles": n_particles,
        "seed": seed,
        "agent_name": agent_name,
        "lag_window": lag_window,
        "hierarchical": use_hierarchical,
        "rao_blackwell": use_rao_blackwell,
        "map_estimate": {
            "reserve": round(map_state.c.reserve, 2),
            "mood": round(map_state.c.mood, 2),
            "sociality": round(map_state.c.sociality, 2),
            "u": [round(val, 2) for val in map_state.g.u],
        },
        "kalman_summary": {
            "reserve_mean": round(kalman["reserve_mean"], 2),
            "reserve_var": round(kalman["reserve_var"], 2),
            "mood_mean": round(kalman["mood_mean"], 2),
        },
        "goal_posterior": {k: round(v, 4) for k, v in g_post.items()},
    }
    return res


def format_inference_output(res: dict, output_format: str = "table") -> None:
    if output_format == "json":
        print(json.dumps(res, indent=2))
    else:
        print(
            f"=== HYPOSTASES Inverse Inference Report (Particles={res['n_particles']}, Seed={res['seed']}) ==="
        )
        print(f"Agent Name: {res['agent_name']}")
        print(
            f"Lag Window: {res['lag_window']} | Hierarchical: {res['hierarchical']} | Rao-Blackwell: {res['rao_blackwell']}"
        )
        print("\n--- MAP Estimate ---")
        print(f"Reserve: {res['map_estimate']['reserve']}")
        print(f"Mood: {res['map_estimate']['mood']}")
        print(f"Sociality: {res['map_estimate']['sociality']}")
        print(f"Latent Utilities u: {res['map_estimate']['u']}")
        print("\n--- Kalman Summary ---")
        print(
            f"Reserve Mean: {res['kalman_summary']['reserve_mean']} (var: {res['kalman_summary']['reserve_var']})"
        )
        print(f"Mood Mean: {res['kalman_summary']['mood_mean']}")
        print("\n--- Goal Posterior ---")
        for k, v in res["goal_posterior"].items():
            print(f"  {k:<12}: {v * 100:>5.1f}%")


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "infer", help="Run inverse inference particle filter (Part VII §10)"
    )
    parser.add_argument(
        "--particles", type=int, default=300, help="Particle count N (default: 300)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--agent-name", type=str, default="Agent_A", help="Target agent name")
    parser.add_argument("--steps", type=int, default=12, help="Trace length steps (default: 12)")
    parser.add_argument("--lag-window", type=int, default=None, help="Bounded history lag window")
    parser.add_argument(
        "--hierarchical", action="store_true", help="Use two-pass hierarchical particle filter"
    )
    parser.add_argument(
        "--use-rao-blackwell", action="store_true", help="Use Kalman world model updates"
    )
    parser.add_argument(
        "--output-format", choices=["table", "json"], default="table", help="Output format"
    )
    parser.set_defaults(func=main_infer)


def main_infer(args: argparse.Namespace) -> None:
    res = run_cli_infer(
        n_particles=args.particles,
        seed=args.seed,
        agent_name=args.agent_name,
        n_steps=args.steps,
        lag_window=args.lag_window,
        use_hierarchical=args.hierarchical,
        use_rao_blackwell=args.use_rao_blackwell,
    )
    format_inference_output(res, output_format=args.output_format)
