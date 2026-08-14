"""Diagnostic 1: λ=0 cos-sim=1.0 anomaly investigation.

Checks whether sample_prior() draws actually differ across seeds, and logs
u_t trajectories per seed at t=0, 10, 50, 100, 500 to detect when/whether
trajectories from distinct priors merge.

NOT a test — pure logging diagnostic. Run with:
    python scripts/light/diag_prior_seeding.py
"""

from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import hypostases.engine.constants as const
from hypostases.engine.dynamics import evolve, feedback, pi_decision, step_env
from hypostases.inference import sample_prior

SEEDS = list(range(100, 110))
KAPPA = 0.0
LAMBDA = 0.0
N_STEPS = 500
CHECKPOINTS = {0, 10, 50, 100, 500}

const.SCARCITY_COST_KAPPA = KAPPA
const.GOVERNANCE_SCALING_LAMBDA = LAMBDA
const.UTILITY_DECAY_RATE = 0.0

console = Console()

console.print(Panel("DIAGNOSTIC 1: sample_prior() per-seed uniqueness", style="bold cyan"))
trajectories: dict[int, dict[int, np.ndarray]] = {}

for s in SEEDS:
    rng = np.random.default_rng(s)
    n_agents = 5
    xi = np.array([0.2, 0.2, 0.2, 0.2])

    agents = {f"Agent_{i}": sample_prior(rng=rng) for i in range(n_agents)}
    u_0 = np.mean([ag.g.u.copy() for ag in agents.values()], axis=0)

    console.print(f"Seed {s}: u_0 = {u_0.round(4)}")

    ckpt_log: dict[int, np.ndarray] = {0: u_0.copy()}
    pool = 10.0

    for step in range(1, N_STEPS + 1):
        agent_actions = [
            (name, pi_decision(ag, pool_belief=pool, xi=xi, rng=rng)) for name, ag in agents.items()
        ]
        pool, delta_log = step_env(pool, agent_actions, enable_withdraw_fee=(LAMBDA > 0))
        for name, ag in agents.items():
            act = delta_log["actions_log"][name]
            phi = feedback(ag, delta_log["pool_before"], pool, act, delta_log, agent_name=name)
            evolve(ag, phi)

        if step in CHECKPOINTS:
            u_curr = np.mean([ag.g.u.copy() for ag in agents.values()], axis=0)
            ckpt_log[step] = u_curr.copy()

    trajectories[s] = ckpt_log

# Report: pairwise distance of u_0 across seeds
u0s = [trajectories[s][0] for s in SEEDS]
u0_table = Table(title="Pairwise ||u_0^i - u_0^j|| across seeds (should be nonzero)", style="cyan")
u0_table.add_column("Seed pair", style="magenta")
u0_table.add_column("||du_0||", style="green")
for i in range(len(SEEDS)):
    for j in range(i + 1, len(SEEDS)):
        d = float(np.linalg.norm(u0s[i] - u0s[j]))
        u0_table.add_row(f"({SEEDS[i]}, {SEEDS[j]})", f"{d:.6f}")
console.print(u0_table)

# Report: pairwise distance of u_500 across seeds
u500s = [trajectories[s][500] for s in SEEDS]
u500_table = Table(
    title="Pairwise ||u_500^i - u_500^j|| across seeds (attractor check)", style="cyan"
)
u500_table.add_column("Seed pair", style="magenta")
u500_table.add_column("||du_500||", style="green")
for i in range(len(SEEDS)):
    for j in range(i + 1, len(SEEDS)):
        d = float(np.linalg.norm(u500s[i] - u500s[j]))
        u500_table.add_row(f"({SEEDS[i]}, {SEEDS[j]})", f"{d:.6f}")
console.print(u500_table)

# Report: trajectory evolution per seed
console.print(Panel("u_t per seed at checkpoints {0, 10, 50, 100, 500}", style="bold cyan"))
for s in SEEDS:
    ckpt_table = Table(title=f"Seed {s}", style="cyan", show_header=False)
    ckpt_table.add_column("t")
    ckpt_table.add_column("u_t")
    for t in sorted(CHECKPOINTS):
        u = trajectories[s][t]
        ckpt_table.add_row(f"t={t:4d}", f"{u.round(6)}")
    console.print(ckpt_table)
