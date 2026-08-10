"""HYPOSTASES Simulation — Diagnostic & Benchmark Sweeps.

Spec Ref: Part VII §12.7.
Formally tests whether the WITHDRAW identifiability gap is closed.
Z(theta_low, theta_high, n, N, seed) = (mu_high - mu_low) / sqrt((var_high + var_low)/2)

Gap closed at (n, N) iff over S >= 5 seeds:
  1. Directional correctness: Z > 0 in >= ceil(0.8*S) seeds
  2. Statistical separation: median(Z) > 1.0
  3. Non-degeneracy: var in [0.5, 20] in all but <= 1 seed
"""

from __future__ import annotations

import math

import numpy as np

from hypostases.inference import infer, summarize_kalman
from hypostases.simulation.harness import (
    generate_forced_withdraw_trace,
    make_test_agent,
)


def single_trial(
    n_steps: int,
    n_particles: int,
    seed: int,
    reserve_low: float = 3.0,
    reserve_high: float = 14.0,
) -> dict:
    """Executes a single trial of the formal WITHDRAW gap test across low vs high reserve hypotheses."""
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    d_low = make_test_agent("D_low", reserve_low)
    d_high = make_test_agent("D_high", reserve_high)
    actions_low, pools_low = generate_forced_withdraw_trace(d_low, pool0=9.0, n_steps=n_steps)
    actions_high, pools_high = generate_forced_withdraw_trace(d_high, pool0=9.0, n_steps=n_steps)

    rng_low = np.random.default_rng(seed=seed)
    particles_low = infer(
        actions_low, pools_low, xi, n_particles=n_particles, agent_name="D_low", rng=rng_low
    )

    rng_high = np.random.default_rng(seed=seed + 100000)
    particles_high = infer(
        actions_high, pools_high, xi, n_particles=n_particles, agent_name="D_high", rng=rng_high
    )

    k_low = summarize_kalman(particles_low)
    k_high = summarize_kalman(particles_high)

    pooled_sd = math.sqrt((k_high["reserve_var"] + k_low["reserve_var"]) / 2)
    z = (
        (k_high["reserve_mean"] - k_low["reserve_mean"]) / pooled_sd
        if pooled_sd > 0
        else float("nan")
    )

    return {
        "z": z,
        "mu_low": k_low["reserve_mean"],
        "var_low": k_low["reserve_var"],
        "mu_high": k_high["reserve_mean"],
        "var_high": k_high["reserve_var"],
    }


def evaluate_config(n_steps: int, n_particles: int, seeds: list[int]) -> dict:
    """Evaluates the 3 formal gap-closure conditions for a specified (steps, particles) configuration across seeds."""
    results = [single_trial(n_steps, n_particles, s) for s in seeds]
    zs = [r["z"] for r in results]
    n_seeds = len(seeds)

    cond1_count = sum(1 for z in zs if z > 0)
    cond1 = cond1_count >= math.ceil(0.8 * n_seeds)

    median_z = float(np.median(zs))
    cond2 = median_z > 1.0

    degenerate_count = sum(
        1 for r in results if not (0.5 <= r["var_low"] <= 20) or not (0.5 <= r["var_high"] <= 20)
    )
    cond3 = degenerate_count <= 1

    passed = cond1 and cond2 and cond3
    return {
        "n_steps": n_steps,
        "n_particles": n_particles,
        "passed": passed,
        "cond1_directional": cond1,
        "cond1_count": cond1_count,
        "n_seeds": n_seeds,
        "cond2_median_z": round(median_z, 3),
        "cond2_pass": cond2,
        "cond3_degenerate_count": degenerate_count,
        "cond3_pass": cond3,
        "all_z": [round(z, 3) for z in zs],
    }


def run_sweep_benchmark(
    steps_list: list[int],
    n_particles: int,
    seeds: list[int],
) -> list[dict]:
    """Runs diagnostic sweep benchmark across multiple step counts."""
    return [evaluate_config(n_steps, n_particles, seeds) for n_steps in steps_list]
