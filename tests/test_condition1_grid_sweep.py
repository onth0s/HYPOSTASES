"""Tests — Pre-Registered Condition 1: 3x3 Parameter Grid Attractor Sweep.

Pre-registered empirical test suite evaluating Condition 1 (Generic Attractor Mapping).
Evaluates a 3x3 grid over (kappa, lambda) across 10 random seeds over a 500-step horizon.

PRE-REGISTERED THRESHOLDS (commited prior to output inspection):
1. Stationarity Ratio = V(400, 500) / V(0, 100) < 0.20
2. Mean Pairwise Cosine Similarity S_cos > 0.70

PRE-REGISTERED NEGATIVE-RESULT BRANCH:
Cells failing either threshold are classified as NON-ATTRACTING / REGIME-BOUND DRIFT,
establishing that role formation is regime-dependent rather than a universal attractor.
"""

from __future__ import annotations

import numpy as np

from hypostases.engine.dynamics import evolve, feedback, pi_decision, step_env
from hypostases.inference import sample_prior


def _run_single_trajectory(
    kappa: float,
    lam: float,
    decay: float,
    seed: int,
    n_steps: int = 500,
    n_agents: int = 5,
) -> tuple[np.ndarray, float, float]:
    """Runs a single trajectory for given (kappa, lambda, decay) and seed."""
    rng = np.random.default_rng(seed)
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    agents = {f"Agent_{i}": sample_prior(rng=rng) for i in range(n_agents)}
    u_0 = np.mean([ag.g.u.copy() for ag in agents.values()], axis=0)

    u_history = [u_0.copy()]

    import hypostases.engine.constants as const

    orig_kappa = const.SCARCITY_COST_KAPPA
    orig_lam = const.GOVERNANCE_SCALING_LAMBDA
    orig_decay = const.UTILITY_DECAY_RATE

    const.SCARCITY_COST_KAPPA = kappa
    const.GOVERNANCE_SCALING_LAMBDA = lam
    const.UTILITY_DECAY_RATE = decay

    pool = 10.0
    try:
        for _step in range(n_steps):
            agent_actions = [
                (name, pi_decision(ag, pool_belief=pool, xi=xi, rng=rng))
                for name, ag in agents.items()
            ]
            pool, delta_log = step_env(pool, agent_actions, enable_withdraw_fee=(lam > 0))
            for name, ag in agents.items():
                act = delta_log["actions_log"][name]
                phi = feedback(ag, delta_log["pool_before"], pool, act, delta_log, agent_name=name)
                evolve(ag, phi)

            u_curr = np.mean([ag.g.u.copy() for ag in agents.values()], axis=0)
            u_history.append(u_curr.copy())
    finally:
        const.SCARCITY_COST_KAPPA = orig_kappa
        const.GOVERNANCE_SCALING_LAMBDA = orig_lam
        const.UTILITY_DECAY_RATE = orig_decay

    u_history_arr = np.array(u_history)
    u_0_vec = u_history_arr[0]
    u_100_vec = u_history_arr[100]
    u_400_vec = u_history_arr[400]
    u_500_vec = u_history_arr[500]

    v_initial = float(np.linalg.norm(u_100_vec - u_0_vec) / 100.0)
    v_final = float(np.linalg.norm(u_500_vec - u_400_vec) / 100.0)

    disp = u_500_vec - u_0_vec
    norm_disp = float(np.linalg.norm(disp))
    disp_unit = disp / (norm_disp + 1e-9)

    return disp_unit, v_initial, v_final


def evaluate_grid_cell(
    kappa: float,
    lam: float,
    decay: float,
    seeds: list[int],
) -> dict[str, float | str]:
    """Evaluates 10 seeds for a single (kappa, lambda, decay) cell."""
    displacements = []
    v_initials = []
    v_finals = []

    for s in seeds:
        disp_unit, v_init, v_fin = _run_single_trajectory(kappa, lam, decay, seed=s)
        displacements.append(disp_unit)
        v_initials.append(v_init)
        v_finals.append(v_fin)

    mean_v_init = float(np.mean(v_initials))
    mean_v_fin = float(np.mean(v_finals))
    stationarity_ratio = mean_v_fin / (mean_v_init + 1e-9)

    n_seeds = len(seeds)
    cos_sims = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            sim = float(np.dot(displacements[i], displacements[j]))
            cos_sims.append(sim)

    mean_cos_sim = float(np.mean(cos_sims))

    is_stationary = stationarity_ratio < 0.20
    is_clustered = mean_cos_sim > 0.70

    if is_stationary and is_clustered:
        classification = "ATTRACTOR CONFIRMED"
    else:
        classification = "NON-ATTRACTING / REGIME DRIFT"

    return {
        "kappa": kappa,
        "lambda": lam,
        "decay": decay,
        "stationarity_ratio": stationarity_ratio,
        "mean_cos_sim": mean_cos_sim,
        "classification": classification,
    }


class TestCondition1PreRegisteredGrid:
    def test_side_by_side_grid_sweep(self):
        """Executes side-by-side comparison between decay=0.0 (pure dynamics) and decay=0.05 (decay regularized)."""
        kappas = [0.0, 0.5, 1.0]
        lambdas = [0.0, 1.0, 2.0]
        seeds = list(range(100, 110))

        no_decay_results = [
            evaluate_grid_cell(k, l_val, 0.0, seeds) for k in kappas for l_val in lambdas
        ]
        decay_results = [
            evaluate_grid_cell(k, l_val, 0.05, seeds) for k in kappas for l_val in lambdas
        ]

        print("\n=== SIDE-BY-SIDE GRID SWEEP COMPARISON ===")
        print("\n--- RUN A: PURE DYNAMICS (UTILITY_DECAY_RATE = 0.0) ---")
        print("| Kappa | Lambda | Ratio (<0.20) | Cos Sim (>0.70) | Classification |")
        print("|-------|--------|---------------|-----------------|----------------|")
        for r in no_decay_results:
            print(
                f"| {r['kappa']:.1f}   | {r['lambda']:.1f}    | {r['stationarity_ratio']:.4f}        | {r['mean_cos_sim']:.4f}          | {r['classification']} |"
            )

        print("\n--- RUN B: LEAKY DECAY REGULARIZED (UTILITY_DECAY_RATE = 0.05) ---")
        print("| Kappa | Lambda | Ratio (<0.20) | Cos Sim (>0.70) | Classification |")
        print("|-------|--------|---------------|-----------------|----------------|")
        for r in decay_results:
            print(
                f"| {r['kappa']:.1f}   | {r['lambda']:.1f}    | {r['stationarity_ratio']:.4f}        | {r['mean_cos_sim']:.4f}          | {r['classification']} |"
            )

        assert len(no_decay_results) == 9
        assert len(decay_results) == 9
