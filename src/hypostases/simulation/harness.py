"""HYPOSTASES Simulation — Agent & Trace Harness Utilities.

Spec Ref: Part V, Part VI §8, Part VII §10.
Contains helper utilities for instantiating agents, generating synthetic traces,
and running forward simulation traces.
"""

from __future__ import annotations

import numpy as np

from hypostases.engine import (
    Action,
    ActionType,
    Agent,
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
    evolve,
    feedback,
    pi_decision,
    step_env,
)


def build_agent(
    name: str,
    skill: float = 0.6,
    resilience: float = 0.5,
    sociality: float = 0.5,
    memory_decay: float = 0.9,
    reserve: float = 6.0,
    mood: float = 0.0,
    mu: float = 10.0,
    sigma2: float = 2.0,
    replenish_rate_est: float = 1.0,
    u: np.ndarray | None = None,
    social_capital: float = 1.0,
    time_budget: float = 12.0,
) -> Agent:
    """Consolidated builder to construct Agent with custom attributes and state."""
    if u is None:
        u = np.array([1.0, 1.0, 1.0, 1.0])
    c = Characteristics(
        skill=skill,
        resilience=resilience,
        sociality=sociality,
        memory_decay=memory_decay,
        reserve=reserve,
        mood=mood,
    )
    w = WorldModel(mu=mu, sigma2=sigma2, replenish_rate_est=replenish_rate_est)
    g = GoalHierarchy(u=u)
    rho_ext = PowerExternal(social_capital=social_capital, time_budget=time_budget)
    return Agent(name=name, sigma=AgentState(c=c, w=w, g=g, rho_ext=rho_ext))


def make_agent(
    name: str,
    sociality: float,
    status_u: float,
    reserve: float = 6.0,
) -> Agent:
    """Instantiates a standard simulation agent with custom sociality, status utility, and reserve."""
    return build_agent(
        name=name,
        sociality=sociality,
        reserve=reserve,
        u=np.array([1.0, 1.0, 1.0, status_u]),
    )


def make_test_agent(name: str, reserve0: float) -> Agent:
    """Instantiates a test agent for diagnostic sweep benchmark trials."""
    return build_agent(
        name=name,
        sociality=0.5,
        reserve=reserve0,
        sigma2=1.0,
        u=np.array([1.0, 1.0, 1.0, 0.9]),
        time_budget=999.0,
    )


def generate_sample_trace(
    n_steps: int = 12,
    reserve: float = 11.66,
    seed: int = 42,
) -> tuple[list[Action], list[float]]:
    """Generates a synthetic action & pool trace for demonstration and inference benchmarking."""
    rng = np.random.default_rng(seed)
    actions: list[Action] = []
    pools: list[float] = []
    pool = 10.0
    for _ in range(n_steps):
        act_choice = rng.choice(["REQUEST", "SHARE", "WITHDRAW"], p=[0.5, 0.3, 0.2])
        if act_choice == "REQUEST":
            act = Action(ActionType.REQUEST, amount=float(rng.uniform(1.0, 4.0)))
        elif act_choice == "SHARE":
            act = Action(ActionType.SHARE, amount=float(rng.uniform(0.5, 2.0)))
        else:
            act = Action(ActionType.WITHDRAW)
        actions.append(act)
        pools.append(pool)
        pool = max(1.0, pool + float(rng.normal(0, 0.5)))
    return actions, pools


def generate_forced_withdraw_trace(
    agent: Agent,
    pool0: float,
    n_steps: int,
) -> tuple[list[Action], list[float]]:
    """Generates a forced WITHDRAW action sequence trace while evolving agent state."""
    pool = pool0
    actions: list[Action] = []
    pools_before: list[float] = []
    for _ in range(n_steps):
        act = Action(ActionType.WITHDRAW)
        new_pool, delta_log = step_env(pool, [(agent.name, act)])
        phi = feedback(agent.sigma, pool, new_pool, act, delta_log, agent_name=agent.name)
        evolve(agent.sigma, phi)
        actions.append(act)
        pools_before.append(pool)
        pool = new_pool
    return actions, pools_before


def run_simulation_trace(
    steps: int = 12,
    seed: int = 7,
    pool_init: float = 10.0,
    xi: np.ndarray | None = None,
) -> dict:
    """Runs a full multi-agent forward simulation trace (Part VI §8)."""
    if xi is None:
        xi = np.array([0.2, 0.2, 0.2, 0.2])

    rng = np.random.default_rng(seed)

    agent_a = make_agent("Agent_A", sociality=0.8, status_u=0.2)
    agent_b = make_agent("Agent_B", sociality=0.1, status_u=1.2)
    agents = [agent_a, agent_b]

    pool = pool_init
    trace_log = []

    for step in range(1, steps + 1):
        # 1. Decision Policy stage per agent
        actions = [(ag.name, pi_decision(ag.sigma, pool, xi, rng=rng)) for ag in agents]

        # 2. Environment stage
        pool_after, delta_log = step_env(pool, actions)

        # 3. Feedback & State Evolution stage per agent
        step_record = {
            "step": step,
            "pool_before": pool,
            "pool_after": pool_after,
            "actions": {name: str(act) for name, act in actions},
            "agents": {},
        }

        for ag, (_, act) in zip(agents, actions, strict=True):
            phi = feedback(ag.sigma, pool, pool_after, act, delta_log, agent_name=ag.name)

            evolve(ag.sigma, phi)
            step_record["agents"][ag.name] = {
                "reserve": round(ag.sigma.c.reserve, 2),
                "social_capital": round(ag.sigma.rho_ext.social_capital, 2),
                "mood": round(ag.sigma.c.mood, 2),
                "u": [round(val, 2) for val in ag.sigma.g.u],
            }

        trace_log.append(step_record)
        pool = pool_after

    return {
        "steps": steps,
        "seed": seed,
        "final_pool": pool,
        "agents": {
            ag.name: {
                "reserve": round(ag.sigma.c.reserve, 2),
                "social_capital": round(ag.sigma.rho_ext.social_capital, 2),
                "mood": round(ag.sigma.c.mood, 2),
                "u": [round(val, 2) for val in ag.sigma.g.u],
            }
            for ag in agents
        },
        "trace": trace_log,
    }
