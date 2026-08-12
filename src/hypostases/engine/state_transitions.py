"""HYPOSTASES Engine — Internal State Transitions & Evolution Dynamics."""

from __future__ import annotations

import numpy as np

from hypostases.engine.constants import (
    MOOD_DECAY_RATE,
    PEER_BELIEF_ALPHA,
    SIGMA2_MAX,
    SIGMA2_MIN,
    UTILITY_DECAY_RATE,
)
from hypostases.engine.types import Action, ActionType, AgentState, DeltaLog, FeedbackDelta


def _integrate_non_world(agent: AgentState, phi: FeedbackDelta) -> None:
    """Integrates all primitive state fields EXCEPT the world model (shared by evolve and evolve_rb)."""
    # Integrate Characteristics c
    if "reserve" in phi.delta_c:
        agent.c.reserve = max(0.0, agent.c.reserve + phi.delta_c["reserve"])
    if "mood" in phi.delta_c:
        agent.c.mood = max(-1.0, min(1.0, agent.c.mood + phi.delta_c["mood"]))

    # Baseline mood decay toward zero (Phase 2.4)
    agent.c.mood *= 1.0 - MOOD_DECAY_RATE

    # Integrate Peer Beliefs (Theory of Mind & Bayesian Evidence Integration)
    for peer_name, val in phi.delta_peer_beliefs.items():
        prev = agent.w.peer_beliefs.get(peer_name, val)
        # Apply exponential smoothing as robust fallback baseline for direct scalar deltas
        agent.w.peer_beliefs[peer_name] = PEER_BELIEF_ALPHA * val + (1.0 - PEER_BELIEF_ALPHA) * prev

    # Integrate Goal Hierarchy latent utilities u with baseline decay regularization
    agent.g.u = (1.0 - UTILITY_DECAY_RATE) * agent.g.u + phi.delta_g

    # Integrate External Power ρ_ext
    if "social_capital" in phi.delta_rho_ext:
        agent.rho_ext.social_capital = max(
            0.0, agent.rho_ext.social_capital + phi.delta_rho_ext["social_capital"]
        )

    # Tier-1 tick: decrement time_budget (reset at Tier-2 epoch boundary — declared out of scope)
    agent.rho_ext.time_budget = max(0.0, agent.rho_ext.time_budget - 1.0)


def _record_memory_event(agent: AgentState, phi: FeedbackDelta) -> None:
    """Record transition event into Front 03 memory sub-systems if present on agent.w."""
    if agent.w.m_ep is None:
        return

    from hypostases.engine.memory import EpisodicEvent

    surprise = float(phi.delta_w.get("last_surprise", 0.0))
    utility_impact = float(np.linalg.norm(phi.delta_g))
    urgency = float(1.0 / max(1.0, agent.c.reserve))
    trust = float(sum(phi.delta_peer_beliefs.values())) if phi.delta_peer_beliefs else 0.5
    info_gain = float(abs(phi.delta_w.get("sigma2", 0.0)))
    novelty = float(abs(phi.delta_w.get("mu", 0.0)) * 0.05)

    gateway = agent.w.thalamic_gateway
    if gateway is not None:
        salience = gateway.compute_salience(
            surprise, info_gain, utility_impact, urgency, trust, novelty
        )
    else:
        salience = 0.5 * (abs(surprise) + utility_impact)

    event = EpisodicEvent(
        tick=0,
        state_snapshot={"reserve": agent.c.reserve, "mu": agent.w.mu},
        action=Action(action_type=ActionType.REQUEST, amount=0.0),
        surprise=surprise,
        utility_delta=utility_impact,
        next_state_snapshot={"reserve": agent.c.reserve, "mu": agent.w.mu},
        salience_score=salience,
    )
    agent.w.m_ep.add_event(event)

    if (
        agent.w.m_work is not None
        and gateway is not None
        and gateway.should_gate_to_working_memory(salience)
    ):
        agent.w.m_work.recent_events.append(event)


def evolve(agent: AgentState, phi: FeedbackDelta) -> None:
    """Part II §3.4: State Evolution Stage integrating deltas into primitives."""
    _integrate_non_world(agent, phi)

    # Integrate World Model w
    if "mu" in phi.delta_w:
        agent.w.mu += phi.delta_w["mu"]
    if "replenish_rate_est" in phi.delta_w:
        agent.w.replenish_rate_est += phi.delta_w["replenish_rate_est"]
    if "sigma2" in phi.delta_w:
        agent.w.sigma2 = max(SIGMA2_MIN, agent.w.sigma2 + phi.delta_w["sigma2"])
    if "last_surprise" in phi.delta_w:
        agent.w.last_surprise = phi.delta_w["last_surprise"]

    # Apply memory decay to belief confidence (Phase 2)
    decay_mode = getattr(agent, "decay_mode", "variance")
    if decay_mode == "variance":
        agent.w.sigma2 = agent.w.sigma2 + (SIGMA2_MAX - agent.w.sigma2) * (
            1.0 - agent.c.memory_decay
        )
    elif decay_mode == "precision":
        tau = 1.0 / max(agent.w.sigma2, 1e-9)
        tau_min = 1.0 / SIGMA2_MAX
        tau = tau + (tau_min - tau) * (1.0 - agent.c.memory_decay)
        agent.w.sigma2 = max(SIGMA2_MIN, 1.0 / max(tau, 1e-9))

    _record_memory_event(agent, phi)


def step_env(
    pool_state: float,
    agent_actions: list[tuple[str, Action]],
    replenish_rate: float = 1.0,
) -> tuple[float, DeltaLog]:
    """Part II §3.2: Environment Stage (Shared Field Resource Step).

    Processes joint actions over shared resource pool S_t and updates pool state.

    Returns:
        tuple (next_pool_state, delta_log)
    """
    total_requested = 0.0
    total_shared = 0.0
    total_withdrawn = 0.0
    total_punished = 0.0

    for _, act in agent_actions:
        if act.action_type == ActionType.REQUEST:
            total_requested += act.amount
        elif act.action_type == ActionType.SHARE:
            total_shared += act.amount
        elif act.action_type == ActionType.WITHDRAW:
            total_withdrawn += act.amount
        elif act.action_type == ActionType.PUNISH:
            total_punished += act.amount

    # Net extraction and replenishment
    net_extraction = total_requested + total_withdrawn - total_shared
    next_pool = max(0.0, pool_state - net_extraction + replenish_rate)

    delta_log: DeltaLog = {
        "total_requested": total_requested,
        "total_shared": total_shared,
        "total_withdrawn": total_withdrawn,
        "total_punished": total_punished,
        "replenish_rate": replenish_rate,
        "net_extraction": net_extraction,
    }
    return next_pool, delta_log
