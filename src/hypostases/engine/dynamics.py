"""HYPOSTASES Engine — v4 Core Update Dynamics.

Spec Ref: Part II §3 (v4), Part IV §6 (Schema v1).
Implements the core loop stages:
  1. Policy Stage: pi_decision (forward simulation)
  2. Environment Stage: step_env (spatial field / shared resource update)
  3. Feedback Stage: feedback (delta generation)
  4. State Evolution Stage: evolve (primitive state integration)
"""

from __future__ import annotations

import numpy as np

from hypostases.engine._math import softmax
from hypostases.engine.constants import (
    ACQUISITION_U_GAIN,
    CROWDING_OUT_HYSTERESIS_GAIN,
    GOVERNANCE_SCALING_LAMBDA,
    INEQUITY_AVERSION_GAIN,
    KALMAN_OBS_NOISE_R,
    KALMAN_PROCESS_NOISE_Q,
    PUNISH_MOOD_GAIN,
    PUNISH_RESERVE_COST,
    PUNISH_TARGET_PENALTY,
    REGIME_SHIFT_GAIN,
    RELATIONAL_U_GAIN,
    REQUEST_MOOD_PENALTY,
    REQUEST_SOCIAL_COST,
    SHARE_MOOD_BONUS,
    SHARE_SOCIAL_GAIN,
    SIGMA2_MIN,
    STATUS_U_GAIN,
    SURVIVAL_U_GAIN,
    WITHDRAW_DEGRADE,
    WITHDRAW_FEE,
    WITHDRAW_MOOD_PENALTY,
    WITHDRAW_SOCIAL_COST,
    WORLD_MU_GAIN,
    WORLD_REPLENISH_GAIN,
    WORLD_SIGMA2_UPDATE_GAIN,
)
from hypostases.engine.state_transitions import (
    _integrate_non_world as _integrate_non_world,
)
from hypostases.engine.state_transitions import (
    evolve as evolve,
)
from hypostases.engine.types import (
    EPISTEMIC_ACTION_TYPES,
    Action,
    ActionType,
    AgentState,
    DeltaCharacteristics,
    DeltaLog,
    DeltaPowerExternal,
    DeltaWorldModel,
    FeedbackDelta,
    K,
)
from hypostases.engine.utility_dynamics import (
    _ACQUISITION_IDX,
    _RELATIONAL_IDX,
    _STATUS_IDX,
    _SURVIVAL_IDX,
    GOAL_SPEC,
    _effective_utilities,
    goal_probs,
)


def pi_decision(
    agent: AgentState,
    pool_belief: float,
    xi: np.ndarray,
    rng: np.random.Generator | None = None,
) -> Action:
    """Part II §3.1: Decision Policy (Forward Simulation).

    Selects a goal category k ~ π_t, then generates an action parameterized by state.

    Parameters:
        agent: The AgentState containing primitives and weights.
        pool_belief: Observed or estimated resources in the shared pool.
        xi: The Index of Exploration context vector.
        rng: Generator for random choices.
    """
    if rng is None:
        rng = np.random.default_rng()

    probs = goal_probs(agent, xi, pool_belief=pool_belief)
    chosen_goal_idx = int(rng.choice(len(K), p=probs))

    goal = K[chosen_goal_idx]
    branch = GOAL_SPEC[goal]
    amt = branch.amount_fn(agent, pool_belief)
    return Action(branch.action_type, amount=amt)


def _resolve_shares_first(
    pool_before_adj: float,
    shares_total: float,
    requests: list[tuple[str, float]],
) -> tuple[float, float, dict[str, float]]:
    """Shares replenish pool first, requests served after."""
    pool_after_shares = pool_before_adj + shares_total
    total_requested = sum(amt for _, amt in requests)
    granted: dict[str, float] = {}

    if total_requested <= 0:
        return pool_after_shares, pool_after_shares, granted
    if total_requested <= pool_after_shares:
        for name, amt in requests:
            granted[name] = amt
        return pool_after_shares - total_requested, pool_after_shares, granted

    ration_ratio = pool_after_shares / total_requested
    for name, amt in requests:
        granted[name] = amt * ration_ratio
    return 0.0, pool_after_shares, granted


def _resolve_pro_rata(
    pool_before_adj: float,
    shares_total: float,
    requests: list[tuple[str, float]],
) -> tuple[float, float, dict[str, float]]:
    """Requests served from pool_before_adj; shares added after requests."""
    pool_after_shares = pool_before_adj
    total_requested = sum(amt for _, amt in requests)
    granted: dict[str, float] = {}

    if total_requested <= 0:
        return pool_after_shares + shares_total, pool_after_shares, granted
    if total_requested <= pool_after_shares:
        for name, amt in requests:
            granted[name] = amt
        return pool_after_shares - total_requested + shares_total, pool_after_shares, granted

    ration_ratio = pool_after_shares / total_requested
    for name, amt in requests:
        granted[name] = amt * ration_ratio
    return shares_total, pool_after_shares, granted


def _resolve_priority(
    pool_before_adj: float,
    shares_total: float,
    requests: list[tuple[str, float]],
    priorities: dict[str, float] | None = None,
) -> tuple[float, float, dict[str, float]]:
    """Greedy allocation sorted by priorities (highest first)."""
    pool_after_shares = pool_before_adj + shares_total
    if priorities is None:
        sorted_requests = sorted(requests, key=lambda x: x[0])
    else:
        sorted_requests = sorted(requests, key=lambda x: priorities.get(x[0], 0.0), reverse=True)

    granted: dict[str, float] = {}
    pool_temp = pool_after_shares
    for name, amt in sorted_requests:
        if amt <= pool_temp:
            granted[name] = amt
            pool_temp -= amt
        else:
            granted[name] = pool_temp
            pool_temp = 0.0
    return pool_temp, pool_after_shares, granted


def _resolve_lottery(
    pool_before_adj: float,
    shares_total: float,
    requests: list[tuple[str, float]],
    rng: np.random.Generator | None = None,
) -> tuple[float, float, dict[str, float]]:
    """Greedy allocation in shuffled random order."""
    if rng is None:
        rng = np.random.default_rng()
    pool_after_shares = pool_before_adj + shares_total
    shuffled_requests = list(requests)
    rng.shuffle(shuffled_requests)

    granted: dict[str, float] = {}
    pool_temp = pool_after_shares
    for name, amt in shuffled_requests:
        if amt <= pool_temp:
            granted[name] = amt
            pool_temp -= amt
        else:
            granted[name] = pool_temp
            pool_temp = 0.0
    return pool_temp, pool_after_shares, granted


def step_env(
    pool_before: float,
    agent_actions: list[tuple[str, Action]],
    enable_withdraw_fee: bool = False,
    enable_withdraw_degrade: bool = False,
    concurrency_operator: str = "shares-first",
    priorities: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, DeltaLog]:
    """Part II §3.2, Part III §5.6: Environment Stage under concurrency.

    Configurable Concurrency Operators:
      - "shares-first": Shares replenish pool first, requests served after (default).
      - "pro-rata": Requests served from pool_before_adj; shares added to pool after requests.
      - "priority": Greedy allocation sorted by priorities (highest first).
      - "lottery": Greedy allocation in shuffled random order.
    """
    n_agents = max(1, len(agent_actions))
    withdraws_count = sum(1 for _, act in agent_actions if act.action_type == ActionType.WITHDRAW)
    withdraw_prevalence = withdraws_count / n_agents  # Contention 3: defection ratio

    withdraw_deductions = 0.0
    if enable_withdraw_fee:
        # Contention 3: Dynamic governance scaling — fee amplified by defection prevalence
        dynamic_fee = WITHDRAW_FEE * (1.0 + GOVERNANCE_SCALING_LAMBDA * withdraw_prevalence)
        withdraw_deductions += withdraws_count * dynamic_fee
    if enable_withdraw_degrade:
        withdraw_deductions += withdraws_count * WITHDRAW_DEGRADE

    pool_before_adj = max(0.0, pool_before - withdraw_deductions)
    shares_total = sum(
        act.amount for _, act in agent_actions if act.action_type == ActionType.SHARE
    )

    requests = [
        (name, act.amount) for name, act in agent_actions if act.action_type == ActionType.REQUEST
    ]
    total_requested = sum(amt for _, amt in requests)

    if concurrency_operator == "shares-first":
        pool_final, pool_after_shares, granted = _resolve_shares_first(
            pool_before_adj, shares_total, requests
        )
    elif concurrency_operator == "pro-rata":
        pool_final, pool_after_shares, granted = _resolve_pro_rata(
            pool_before_adj, shares_total, requests
        )
    elif concurrency_operator == "priority":
        pool_final, pool_after_shares, granted = _resolve_priority(
            pool_before_adj, shares_total, requests, priorities=priorities
        )
    elif concurrency_operator == "lottery":
        pool_final, pool_after_shares, granted = _resolve_lottery(
            pool_before_adj, shares_total, requests, rng=rng
        )
    else:
        raise ValueError(f"Unknown concurrency_operator: {concurrency_operator}")

    punishments: dict[str, float] = {}
    for _, act in agent_actions:
        if act.action_type == ActionType.PUNISH and act.target is not None:
            punishments[act.target] = punishments.get(act.target, 0.0) + PUNISH_TARGET_PENALTY

    delta_log: DeltaLog = {
        "pool_before": pool_before,
        "pool_after_shares": pool_after_shares,
        "pool_after": pool_final,
        "shares_total": shares_total,
        "requests_total": total_requested,
        "granted": granted,
        "punishments": punishments,
        "enable_withdraw_fee": enable_withdraw_fee,
        "actions_log": {name: act for name, act in agent_actions},
    }
    return pool_final, delta_log


def _feedback_request(
    agent: AgentState,
    action: Action,
    delta_log: DeltaLog,
    agent_name: str,
) -> tuple[DeltaCharacteristics, np.ndarray, DeltaPowerExternal]:
    granted_amt = delta_log.get("granted", {}).get(agent_name, 0.0)
    shortfall = max(0.0, action.amount - granted_amt)

    delta_c: DeltaCharacteristics = {
        "reserve": granted_amt,
        "mood": -REQUEST_MOOD_PENALTY * shortfall * (1.0 - agent.c.resilience),
    }
    delta_rho_ext: DeltaPowerExternal = {"social_capital": -REQUEST_SOCIAL_COST}

    delta_g = np.zeros(len(K))
    fill_ratio = granted_amt / (action.amount + 1e-9)
    delta_g[_SURVIVAL_IDX] = SURVIVAL_U_GAIN * (2.0 * fill_ratio - 1.0)
    delta_g[_ACQUISITION_IDX] += ACQUISITION_U_GAIN * fill_ratio

    return delta_c, delta_g, delta_rho_ext


def _feedback_share(
    agent: AgentState,
    action: Action,
) -> tuple[DeltaCharacteristics, np.ndarray, DeltaPowerExternal]:
    delta_c: DeltaCharacteristics = {
        "reserve": -action.amount,
        "mood": SHARE_MOOD_BONUS * agent.c.sociality,
    }
    delta_rho_ext: DeltaPowerExternal = {"social_capital": SHARE_SOCIAL_GAIN}
    delta_g = np.zeros(len(K))
    delta_g[_RELATIONAL_IDX] = RELATIONAL_U_GAIN * agent.c.sociality
    return delta_c, delta_g, delta_rho_ext


def _feedback_withdraw(
    agent: AgentState,
    action: Action,
    delta_log: DeltaLog,
) -> tuple[DeltaCharacteristics, np.ndarray, DeltaPowerExternal]:
    delta_c: DeltaCharacteristics = {
        "reserve": 0.0,
        "mood": -WITHDRAW_MOOD_PENALTY * agent.c.sociality,
    }
    delta_rho_ext: DeltaPowerExternal = {"social_capital": -WITHDRAW_SOCIAL_COST}
    delta_g = np.zeros(len(K))

    status_delta = STATUS_U_GAIN * (1.0 - agent.c.sociality)
    if delta_log.get("enable_withdraw_fee", False):
        status_delta = -STATUS_U_GAIN * (1.0 - agent.c.sociality)
    delta_g[_STATUS_IDX] = status_delta
    return delta_c, delta_g, delta_rho_ext


def _feedback_punish(
    agent: AgentState,
    action: Action,
) -> tuple[DeltaCharacteristics, np.ndarray, DeltaPowerExternal]:
    delta_c: DeltaCharacteristics = {
        "reserve": -PUNISH_RESERVE_COST,
        "mood": PUNISH_MOOD_GAIN * agent.c.sociality,
    }
    delta_rho_ext: DeltaPowerExternal = {"social_capital": 0.1}
    delta_g = np.zeros(len(K))
    return delta_c, delta_g, delta_rho_ext


def _apply_cross_cutting_c(
    agent: AgentState,
    agent_name: str,
    delta_log: DeltaLog,
    delta_c: DeltaCharacteristics,
) -> None:
    if agent_name in delta_log.get("punishments", {}):
        penalty = delta_log["punishments"][agent_name]
        delta_c["reserve"] = delta_c.get("reserve", 0.0) - penalty

    numeric_beliefs = [val for val in agent.w.peer_beliefs.values() if isinstance(val, int | float)]
    if numeric_beliefs:
        max_peer_est = max(numeric_beliefs)
        if max_peer_est > agent.c.reserve:
            deprivation = max_peer_est - agent.c.reserve
            delta_c["mood"] = delta_c.get("mood", 0.0) - INEQUITY_AVERSION_GAIN * deprivation


def _apply_crowding_out(
    agent: AgentState,
    action: Action,
    delta_log: DeltaLog,
    delta_g: np.ndarray,
) -> None:
    if delta_log.get("enable_withdraw_fee", False) and action.action_type == ActionType.WITHDRAW:
        shift = CROWDING_OUT_HYSTERESIS_GAIN * (1.0 - agent.c.sociality)
        delta_g[_ACQUISITION_IDX] += shift
        delta_g[_RELATIONAL_IDX] -= shift


def _compute_world_surprise(
    agent: AgentState,
    action: Action,
    delta_log: DeltaLog,
    pool_before: float,
    pool_after: float,
    agent_name: str,
) -> DeltaWorldModel:
    own_impact = 0.0
    if action.action_type == ActionType.REQUEST:
        own_impact = -delta_log.get("granted", {}).get(agent_name, 0.0)
    elif action.action_type == ActionType.SHARE:
        own_impact = action.amount

    observed_delta = (pool_after - pool_before) - own_impact
    predicted_delta = agent.w.replenish_rate_est
    surprise = observed_delta - predicted_delta

    acceleration = abs(surprise - agent.w.last_surprise)
    regime_expansion = REGIME_SHIFT_GAIN * acceleration

    return {
        "mu": WORLD_MU_GAIN * surprise,
        "replenish_rate_est": WORLD_REPLENISH_GAIN * surprise,
        "sigma2": WORLD_SIGMA2_UPDATE_GAIN * (abs(surprise) - agent.w.sigma2) + regime_expansion,
        "last_surprise": surprise,
    }


def feedback(
    agent: AgentState,
    pool_before: float,
    pool_after: float,
    action: Action,
    delta_log: DeltaLog,
    agent_name: str = "agent",
) -> FeedbackDelta:
    """Part II §3.3, Part IV §6.7b: Feedback Stage emitting primitive deltas.

    Directive 003 Branch Audit (Part III §5.8):
      - ActionType.REQUEST: State-dependent reserve gain and resilience-weighted mood penalty.
      - ActionType.SHARE: State-dependent reserve cost, sociality-weighted mood/capital, and RELATIONAL utility gain.
      - ActionType.WITHDRAW: State-dependent sociality mood penalty and social capital cost.
      - Peer beliefs: Explicitly tracks observed grant draws per peer agent.
    """
    if action.action_type in EPISTEMIC_ACTION_TYPES:
        from hypostases.active_perception import execute_epistemic_action

        return execute_epistemic_action(agent, action, ground_truth_val=pool_after)

    if action.action_type == ActionType.REQUEST:
        delta_c, delta_g, delta_rho_ext = _feedback_request(agent, action, delta_log, agent_name)
    elif action.action_type == ActionType.SHARE:
        delta_c, delta_g, delta_rho_ext = _feedback_share(agent, action)
    elif action.action_type == ActionType.WITHDRAW:
        delta_c, delta_g, delta_rho_ext = _feedback_withdraw(agent, action, delta_log)
    elif action.action_type == ActionType.PUNISH:
        delta_c, delta_g, delta_rho_ext = _feedback_punish(agent, action)
    else:
        delta_c, delta_g, delta_rho_ext = {}, np.zeros(len(K)), {}

    _apply_cross_cutting_c(agent, agent_name, delta_log, delta_c)
    _apply_crowding_out(agent, action, delta_log, delta_g)
    delta_w = _compute_world_surprise(agent, action, delta_log, pool_before, pool_after, agent_name)

    delta_peer_beliefs: dict[str, float] = {}
    for peer_name, granted_amt in delta_log.get("granted", {}).items():
        if peer_name != agent_name:
            delta_peer_beliefs[peer_name] = granted_amt

    # Softmax Jacobian Attenuation
    u_eff = _effective_utilities(agent)
    pi_current = softmax(u_eff)
    sensitivity = pi_current * (1.0 - pi_current)
    delta_g = delta_g * sensitivity

    return FeedbackDelta(
        delta_c=delta_c,
        delta_w=delta_w,
        delta_g=delta_g,
        delta_rho_ext=delta_rho_ext,
        delta_peer_beliefs=delta_peer_beliefs,
    )


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


def evolve_rb(
    agent: AgentState,
    phi: FeedbackDelta,
    surprise: float,
    process_noise_q: float = KALMAN_PROCESS_NOISE_Q,
    obs_noise_r: float = KALMAN_OBS_NOISE_R,
) -> None:
    """Phase 4 — Rao-Blackwellized State Evolution (opt-in variant of ``evolve``).

    Replaces the EMA world-model update in ``evolve`` with a closed-form Kalman
    predict-update cycle conditioned on the observed ``surprise``. All other state
    fields (c, g, rho_ext, peer_beliefs) are integrated identically to ``evolve``.

    The Kalman equations are:
      Predict:  mu_pred   = mu + replenish_rate_est
                sig2_pred = sigma2 + Q
      Update:   K_gain    = sig2_pred / (sig2_pred + R)
                mu_post   = mu_pred + K_gain * (surprise - mu_pred)
                sig2_post = (1 - K_gain) * sig2_pred

    Parameters:
        agent: Agent state to update in-place.
        phi: FeedbackDelta emitted by ``feedback`` (used for c, g, rho_ext, peers).
        surprise: Observed surprise scalar = (pool_after - pool_before) - own_impact
            - w.replenish_rate_est, computed once per tick from delta_log and shared
            across all particles (it is observation-fixed, not particle-specific).
        process_noise_q: Kalman process noise Q (default: KALMAN_PROCESS_NOISE_Q).
        obs_noise_r: Kalman observation noise R (default: KALMAN_OBS_NOISE_R).
    """
    _integrate_non_world(agent, phi)

    # --- World Model — Kalman predict-update (replaces EMA) ---
    # Predict
    mu_pred = agent.w.mu + agent.w.replenish_rate_est
    sig2_pred = max(SIGMA2_MIN, agent.w.sigma2 + process_noise_q)

    # Update
    k_gain = sig2_pred / (sig2_pred + obs_noise_r)
    agent.w.mu = mu_pred + k_gain * (surprise - mu_pred)
    agent.w.sigma2 = max(SIGMA2_MIN, (1.0 - k_gain) * sig2_pred)

    # Replenish rate estimate: carry forward EMA from phi (consistent with evolve)
    if "replenish_rate_est" in phi.delta_w:
        agent.w.replenish_rate_est += phi.delta_w["replenish_rate_est"]

    # Track last_surprise for regime-shift detection (Contention 2)
    agent.w.last_surprise = surprise
