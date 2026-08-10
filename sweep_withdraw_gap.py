"""
Part VII §12.7: formal "closing the gap" test, per the ratified specification.

Z(theta_low, theta_high, n, N, seed) = (mu_high - mu_low) / sqrt((var_high+var_low)/2)

Gap closed at (n, N) iff, over S>=5 seeds:
  1. Directional correctness: Z > 0 in >= ceil(0.8*S) seeds
  2. Statistical separation: median(Z) > 1.0
  3. Non-degeneracy: var in [0.5, 20] in all but <=1 seed

Step 1 (this file, PHASE A): diagnostic sweep over (n, N) to isolate which
lever -- trace length, particle count, or coupling strength -- is binding,
BEFORE tuning anything. Step 3 (PHASE B) applies whatever fix the diagnosis
indicates. Step 4 (PHASE C) re-runs the formal multi-seed test.
"""

import math

import numpy as np
from hypostases_ref import AgentState, Characteristics, GoalHierarchy, PowerExternal, WorldModel

from hypostases_infer import infer, reseed_infer, summarize_kalman


def make_test_agent(name, reserve0):
    """D_low/D_high construction from §12.5, factored out for reuse."""
    c = Characteristics(
        skill=0.6, resilience=0.5, sociality=0.5, memory_decay=0.9, reserve=reserve0, mood=0.0
    )
    w = WorldModel(mu=10.0, sigma2=1.0, replenish_rate_est=1.0)
    u = np.array([1.0, 1.0, 1.0, 0.9])
    g = GoalHierarchy(pi=np.exp(u) / np.sum(np.exp(u)), u=u)
    rho_ext = PowerExternal(social_capital=1.0, time_budget=999)
    return AgentState(c, w, g, rho_ext, name)


def generate_forced_withdraw_trace(agent, pool0, n_steps):
    from hypostases_ref import Action, ActionType, evolve, feedback, step_env

    pool = pool0
    actions, pools_before = [], []
    for _t in range(n_steps):
        act = Action(ActionType.WITHDRAW)
        new_pool, delta_log = step_env(pool, [(agent.name, act)])
        phi = feedback(agent, pool, new_pool, act, delta_log)
        evolve(agent, phi)
        actions.append(act)
        pools_before.append(pool)
        pool = new_pool
    return actions, pools_before


def single_trial(n_steps, n_particles, seed, reserve_low=3.0, reserve_high=14.0):
    """One (n, N, seed) trial. Trace generation is deterministic (forced
    WITHDRAW has no stochastic branch), so only Infer's own RNG varies by
    seed, per the spec's requirement."""
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    d_low = make_test_agent("D_low", reserve_low)
    d_high = make_test_agent("D_high", reserve_high)
    actions_low, pools_low = generate_forced_withdraw_trace(d_low, pool0=9.0, n_steps=n_steps)
    actions_high, pools_high = generate_forced_withdraw_trace(d_high, pool0=9.0, n_steps=n_steps)

    reseed_infer(seed)
    particles_low = infer(actions_low, pools_low, xi, n_particles=n_particles, agent_name="D_low")
    reseed_infer(seed + 100000)  # different but deterministic stream for the high-reserve run
    particles_high = infer(
        actions_high, pools_high, xi, n_particles=n_particles, agent_name="D_high"
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


def evaluate_config(n_steps, n_particles, seeds):
    """Runs the formal 3-condition test (§ ratified spec) for one (n, N)
    over the given seed list."""
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
        "cond2_median_z": median_z,
        "cond2_pass": cond2,
        "cond3_degenerate_count": degenerate_count,
        "cond3_pass": cond3,
        "all_z": zs,
    }


if __name__ == "__main__":
    SEEDS = [1, 2, 3, 4, 5]

    print("=== PHASE A: Diagnostic sweep (isolating the binding lever) ===\n")
    print(f"{'n_steps':>8} {'n_particles':>12} {'median_Z':>10} {'dir_ok':>8} {'passed':>8}")
    configs = [
        (10, 300),
        (50, 300),
        (200, 300),
    ]
    for n_steps, n_particles in configs:
        r = evaluate_config(n_steps, n_particles, SEEDS)
        print(
            f"{n_steps:>8} {n_particles:>12} {r['cond2_median_z']:>10.3f} "
            f"{r['cond1_count']}/{r['n_seeds']:>6} {r['passed']!s:>8}"
        )

    print("\n=== PHASE B: Coupling-strength sweep (holding n=50, N=300 fixed) ===\n")
    print("Testing whether the binding constraint is coupling strength, per Phase A's")
    print("diagnosis (more steps/particles alone did not reliably improve Z).\n")
    import hypostases_ref as href

    original_goal_probs = href._goal_probs

    def make_coupled_goal_probs(coupling):
        def _coupled(agent, xi):
            omega = agent.omega()
            u_effective = agent.g.u.copy()
            status_idx = href.K.index("STATUS")
            reserve_factor = 1.0 + coupling * max(0.0, agent.c.reserve - 5.0)
            u_effective[status_idx] *= reserve_factor
            logits = agent.g.pi * omega * u_effective
            temperature = 0.15 + float(np.mean(xi))
            return href._softmax(logits / max(temperature, 1e-3))

        return _coupled

    print(f"{'coupling':>10} {'median_Z':>10} {'dir_ok':>8} {'passed':>8} {'degen':>6}")
    for coupling in [0.08, 0.3, 0.6, 1.0, 2.0, 4.0]:
        href._goal_probs = make_coupled_goal_probs(coupling)
        r = evaluate_config(50, 300, SEEDS)
        print(
            f"{coupling:>10.2f} {r['cond2_median_z']:>10.3f} "
            f"{r['cond1_count']}/{r['n_seeds']:>6} {r['passed']!s:>8} {r['cond3_degenerate_count']:>6}"
        )
    href._goal_probs = original_goal_probs
