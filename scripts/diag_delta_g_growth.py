"""Diagnostic 2: ‖Δg_t‖ growth characterization across 500 steps.

Logs the magnitude ‖Δg_t‖ of the feedback utility delta every step for one
representative trajectory to determine whether Δg growth is linear, superlinear,
or plateaus naturally.

NOT a test — pure logging diagnostic. Run with:
    python scripts/diag_delta_g_growth.py
"""

from __future__ import annotations

import numpy as np

import hypostases.engine.constants as const
from hypostases.engine.dynamics import feedback, pi_decision, step_env
from hypostases.engine.types import FeedbackDelta
from hypostases.inference import sample_prior

const.SCARCITY_COST_KAPPA = 0.0
const.GOVERNANCE_SCALING_LAMBDA = 0.0
const.UTILITY_DECAY_RATE = 0.0

SEED = 100
N_STEPS = 500
N_AGENTS = 5
REPORT_EVERY = 25

rng = np.random.default_rng(SEED)
xi = np.array([0.2, 0.2, 0.2, 0.2])

agents = {f"Agent_{i}": sample_prior(rng=rng) for i in range(N_AGENTS)}
pool = 10.0

delta_g_norms: list[float] = []
u_norms: list[float] = []

for step in range(1, N_STEPS + 1):
    agent_actions = [
        (name, pi_decision(ag, pool_belief=pool, xi=xi, rng=rng)) for name, ag in agents.items()
    ]
    pool, delta_log = step_env(pool, agent_actions, enable_withdraw_fee=False)

    step_delta_g_norms = []
    phis: list[FeedbackDelta] = []
    for name, ag in agents.items():
        act = delta_log["actions_log"][name]
        phi = feedback(ag, delta_log["pool_before"], pool, act, delta_log, agent_name=name)
        phis.append(phi)
        step_delta_g_norms.append(float(np.linalg.norm(phi.delta_g)))

    mean_dg = float(np.mean(step_delta_g_norms))
    delta_g_norms.append(mean_dg)

    # Integrate (without evolve to keep state for inspection)
    for ag, phi in zip(agents.values(), phis, strict=False):
        ag.g.u = ag.g.u + phi.delta_g  # pure additive, no decay
        if "reserve" in phi.delta_c:
            ag.c.reserve = max(0.0, ag.c.reserve + phi.delta_c["reserve"])
        if "mood" in phi.delta_c:
            ag.c.mood = max(-1.0, min(1.0, ag.c.mood + phi.delta_c["mood"]))
        from hypostases.engine.constants import MOOD_DECAY_RATE

        ag.c.mood *= 1.0 - MOOD_DECAY_RATE

    mean_u_norm = float(np.mean([np.linalg.norm(ag.g.u) for ag in agents.values()]))
    u_norms.append(mean_u_norm)

    if step % REPORT_EVERY == 0:
        print(f"  step={step:4d}: mean ‖Δg‖={mean_dg:.6f}  mean ‖u‖={mean_u_norm:.6f}")

# Report summary statistics
arr = np.array(delta_g_norms)
u_arr = np.array(u_norms)

print("\n=== DIAGNOSTIC 2: ‖Δg_t‖ growth summary ===")
print(f"  ‖Δg‖ min={arr.min():.6f}  max={arr.max():.6f}  mean={arr.mean():.6f}")
print(f"  ||Dg|| t=1-100 mean:  {arr[:100].mean():.6f}")
print(f"  ||Dg|| t=401-500 mean: {arr[400:].mean():.6f}")

growth_ratio = arr[400:].mean() / (arr[:100].mean() + 1e-9)
print(f"  tail/initial ratio: {growth_ratio:.4f}  (>1 = accelerating, <1 = decelerating)")

print(f"\n  ‖u‖ at t=1: {u_arr[0]:.4f}")
print(f"  ‖u‖ at t=100: {u_arr[99]:.4f}")
print(f"  ‖u‖ at t=500: {u_arr[499]:.4f}")
u_growth = u_arr[499] / (u_arr[0] + 1e-9)
print(f"  ‖u‖ growth factor over 500 steps: {u_growth:.4f}x")
