"""HYPOSTASES Inference — Sequential Monte Carlo Bootstrap Particle Filter.

Spec Ref: Part VII §10.2, §11, §12.7 (Specification v4 Target).
Canonical estimator: bootstrap particle filter approximating Δ(Σ).
Reweighting uses action_likelihood; state propagation uses the unmodified
step_env / feedback / evolve triplet.

RNG state is explicitly passed via numpy.random.Generator (no global RNG state).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from hypostases.engine import (
    Action,
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
    action_likelihood,
    evolve,
    evolve_rb,
    feedback,
    step_env,
)
from hypostases.engine.constants import ROUGHEN_RESERVE_SD


@dataclass
class Particle:
    """Single hypothesis particle for full primitive state σ = (c, w, g, ρ_ext)."""

    sigma: AgentState
    weight: float


def sample_prior(
    reserve_range: tuple[float, float] = (1.0, 20.0),
    prior_type: str = "uniform",
    goal_bias: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> AgentState:
    """Declared prior over Σ = C × W × G × R_ext (Part VII §10.2 step 1).

    Uses explicit rng generator parameter.

    Parameters:
        goal_bias: Optional mapping from GoalCategory name (str) to additive bias
            applied to the sampled ``u`` vector. Used by ``infer_hierarchical`` to
            concentrate the micro-pass prior around the macro-pass dominant goal cluster.
    """
    if rng is None:
        rng = np.random.default_rng()

    if prior_type == "truncated_normal":
        while True:
            val = rng.normal(loc=10.0, scale=4.0)
            if reserve_range[0] <= val <= reserve_range[1]:
                reserve_val = float(val)
                break
    elif prior_type == "log_normal":
        val = rng.lognormal(mean=2.2226, sigma=0.4)
        reserve_val = float(np.clip(val, reserve_range[0], reserve_range[1]))
    else:
        reserve_val = float(rng.uniform(*reserve_range))

    c = Characteristics(
        sociality=float(rng.uniform(0.0, 1.0)),
        reserve=reserve_val,
    )
    w = WorldModel()
    u = rng.normal(loc=[1.0, 1.0, 1.0, 1.0], scale=0.8)

    if goal_bias is not None:
        from hypostases.engine.types import GoalCategory, K

        for k_name, bias_val in goal_bias.items():
            try:
                goal_cat = GoalCategory(k_name)
                idx = list(K).index(goal_cat)
                u[idx] += bias_val
            except (ValueError, IndexError):
                pass  # unknown goal name — silently skip

    g = GoalHierarchy(u=u)
    rho_ext = PowerExternal()
    return AgentState(c=c, w=w, g=g, rho_ext=rho_ext)


def infer(
    observed_actions: list[Action],
    observed_pool_trace: list[float],
    xi: np.ndarray,
    n_particles: int = 300,
    agent_name: str = "unknown",
    ess_threshold_ratio: float = 0.5,
    reserve_range: tuple[float, float] = (1.0, 20.0),
    prior_type: str = "uniform",
    validate_invariants: bool = False,
    enable_withdraw_fee: bool = False,
    enable_withdraw_degrade: bool = False,
    lag_window: int | None = None,
    use_rao_blackwell: bool = False,
    goal_bias: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> list[Particle]:
    """Part VII §10.2: Sequential Monte Carlo bootstrap particle filter.

    Returns the final weighted particle set approximating Δ(Σ).

    Parameters:
        lag_window: If set, truncates the observation trace to the last
            ``lag_window`` steps before filtering. Aligned with ``c.memory_decay``:
            older observations carry exponentially diminishing information about
            current latent state.
        use_rao_blackwell: If True, replaces the EMA world-model update with a
            closed-form Kalman predict-update step (``evolve_rb``). The surprise
            value is computed once per tick from ``delta_log`` and is shared
            across all particles (it is observation-fixed, not particle-specific).
        goal_bias: Optional dict mapping GoalCategory names to additive ``u`` bias.
            Used by ``infer_hierarchical`` to concentrate the prior on the dominant
            macro-pass cluster. Passed directly to ``sample_prior``.
    """
    if lag_window is not None:
        observed_actions = observed_actions[-lag_window:]
        observed_pool_trace = observed_pool_trace[-lag_window:]

    if rng is None:
        rng = np.random.default_rng()

    particles = [
        Particle(
            sigma=sample_prior(
                reserve_range=reserve_range,
                prior_type=prior_type,
                goal_bias=goal_bias,
                rng=rng,
            ),
            weight=1.0 / n_particles,
        )
        for _ in range(n_particles)
    ]

    if validate_invariants:
        from hypostases.schemas import assert_invariants

    for a_obs, pool_t in zip(observed_actions, observed_pool_trace, strict=True):
        # 1. Reweight (§10.2 step 3): score current belief against observed action
        for p in particles:
            lik = action_likelihood(p.sigma, a_obs, xi, pool_belief=pool_t)
            p.weight *= lik

        total_w = sum(p.weight for p in particles)
        if total_w <= 0:
            warnings.warn(
                "Particle weight collapse detected — resetting to uniform",
                stacklevel=2,
            )
            for p in particles:
                p.weight = 1.0 / n_particles
        else:
            for p in particles:
                p.weight /= total_w

        # 2. Resample if effective sample size collapsed (§10.2 step 4)
        ess = 1.0 / sum(p.weight**2 for p in particles)
        if ess < ess_threshold_ratio * n_particles:
            particles = _resample(particles, n_particles, rng=rng)

        # 3. Propose (§10.2 step 2): propagate using step_env / feedback / evolve
        _, delta_log = step_env(
            pool_t,
            [(agent_name, a_obs)],
            enable_withdraw_fee=enable_withdraw_fee,
            enable_withdraw_degrade=enable_withdraw_degrade,
        )

        # Surprise is observation-fixed: computed once, shared across all particles
        pool_after = delta_log["pool_after"]
        if use_rao_blackwell:
            own_impact = a_obs.amount if a_obs.action_type.value == "SHARE" else 0.0
            surprise_val = float((pool_after - pool_t) - own_impact)

        for p in particles:
            phi = feedback(
                p.sigma,
                pool_t,
                pool_after,
                a_obs,
                delta_log,
                agent_name=agent_name,
            )
            if use_rao_blackwell:
                evolve_rb(p.sigma, phi, surprise=surprise_val)
            else:
                evolve(p.sigma, phi)
            if validate_invariants:
                assert_invariants(p.sigma)

    return particles


def _resample(
    particles: list[Particle],
    n: int,
    roughen_reserve_sd: float = ROUGHEN_RESERVE_SD,
    rng: np.random.Generator | None = None,
) -> list[Particle]:
    """Part VII §12.7: Systematic resampling with post-resample reserve roughening."""
    if rng is None:
        rng = np.random.default_rng()

    reserves = np.array([p.sigma.c.reserve for p in particles])
    std_reserve = float(np.std(reserves))

    weights = np.array([p.weight for p in particles])
    cumsum = np.cumsum(weights)
    u0 = rng.uniform(0.0, 1.0 / n)
    positions = u0 + np.arange(n) / n
    idx = np.searchsorted(cumsum, positions)
    idx = np.clip(idx, 0, len(particles) - 1)
    resampled = [Particle(sigma=particles[i].sigma.clone(), weight=1.0 / n) for i in idx]

    # Dynamic scaling based on incoming sample spread
    dynamic_sd = max(0.05, roughen_reserve_sd * std_reserve)

    if dynamic_sd > 0:
        for p in resampled:
            p.sigma.c.reserve = max(0.0, p.sigma.c.reserve + float(rng.normal(0, dynamic_sd)))

    return resampled


@dataclass
class JointParticle:
    """Single particle representing joint hypotheses over all agents' primitive states."""

    sigmas: dict[str, AgentState]
    weight: float


def infer_joint(
    observed_actions: list[dict[str, Action]],
    observed_pool_trace: list[float],
    xi: np.ndarray,
    agent_names: list[str],
    n_particles: int = 300,
    ess_threshold_ratio: float = 0.5,
    reserve_range: tuple[float, float] = (1.0, 20.0),
    prior_type: str = "uniform",
    concurrency_operator: str = "shares-first",
    lag_window: int | None = None,
    use_rao_blackwell: bool = False,
    rng: np.random.Generator | None = None,
) -> list[JointParticle]:
    """Exact joint multi-agent inference over the product of agent state spaces.

    Parameters:
        lag_window: If set, truncates the observation trace to the last
            ``lag_window`` steps before filtering.
        use_rao_blackwell: If True, uses ``evolve_rb`` (Kalman world-model update)
            instead of ``evolve`` (EMA) during particle propagation.
    """
    if lag_window is not None:
        observed_actions = observed_actions[-lag_window:]
        observed_pool_trace = observed_pool_trace[-lag_window:]

    if rng is None:
        rng = np.random.default_rng()

    particles = []
    for _ in range(n_particles):
        sigmas = {
            name: sample_prior(reserve_range=reserve_range, prior_type=prior_type, rng=rng)
            for name in agent_names
        }
        particles.append(JointParticle(sigmas=sigmas, weight=1.0 / n_particles))

    for a_obs_dict, pool_t in zip(observed_actions, observed_pool_trace, strict=True):
        # 1. Reweight particles based on joint action likelihood
        for p in particles:
            lik = 1.0
            for name in agent_names:
                act = a_obs_dict[name]
                lik *= action_likelihood(p.sigmas[name], act, xi, pool_belief=pool_t)
            p.weight *= lik

        total_w = sum(p.weight for p in particles)
        if total_w <= 0:
            warnings.warn("Joint particle weight collapse — resetting to uniform", stacklevel=2)
            for p in particles:
                p.weight = 1.0 / n_particles
        else:
            for p in particles:
                p.weight /= total_w

        # 2. Resample if ESS collapsed
        ess = 1.0 / sum(p.weight**2 for p in particles)
        if ess < ess_threshold_ratio * n_particles:
            particles = _resample_joint(particles, n_particles, rng=rng)

        # 3. Propagate all particles
        actions_list = list(a_obs_dict.items())
        _, delta_log = step_env(pool_t, actions_list, concurrency_operator=concurrency_operator)
        pool_after = delta_log["pool_after"]

        for p in particles:
            for name in agent_names:
                act = a_obs_dict[name]
                phi = feedback(
                    p.sigmas[name],
                    pool_t,
                    pool_after,
                    act,
                    delta_log,
                    agent_name=name,
                )
                if use_rao_blackwell:
                    own_impact = act.amount if act.action_type.value == "SHARE" else 0.0
                    surprise_val = float((pool_after - pool_t) - own_impact)
                    evolve_rb(p.sigmas[name], phi, surprise=surprise_val)
                else:
                    evolve(p.sigmas[name], phi)

    return particles


def _resample_joint(
    particles: list[JointParticle],
    n: int,
    roughen_reserve_sd: float = ROUGHEN_RESERVE_SD,
    rng: np.random.Generator | None = None,
) -> list[JointParticle]:
    """Resamples joint particles and applies adaptive roughening per agent state."""
    if rng is None:
        rng = np.random.default_rng()

    weights = np.array([p.weight for p in particles])
    cumsum = np.cumsum(weights)
    u0 = rng.uniform(0.0, 1.0 / n)
    positions = u0 + np.arange(n) / n
    idx = np.searchsorted(cumsum, positions)
    idx = np.clip(idx, 0, len(particles) - 1)

    resampled = []
    for i in idx:
        sigmas_clone = {name: state.clone() for name, state in particles[i].sigmas.items()}
        resampled.append(JointParticle(sigmas=sigmas_clone, weight=1.0 / n))

    for name in particles[0].sigmas:
        reserves = np.array([p.sigmas[name].c.reserve for p in particles])
        std_reserve = float(np.std(reserves))
        dynamic_sd = max(0.05, roughen_reserve_sd * std_reserve)

        if dynamic_sd > 0:
            for p in resampled:
                p.sigmas[name].c.reserve = max(
                    0.0, p.sigmas[name].c.reserve + float(rng.normal(0, dynamic_sd))
                )

    return resampled


def infer_mean_field(
    observed_actions: list[dict[str, Action]],
    observed_pool_trace: list[float],
    xi: np.ndarray,
    agent_names: list[str],
    n_particles: int = 300,
    ess_threshold_ratio: float = 0.5,
    reserve_range: tuple[float, float] = (1.0, 20.0),
    prior_type: str = "uniform",
    concurrency_operator: str = "shares-first",
    lag_window: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, list[Particle]]:
    """Performs factorized Mean-Field particle filtering for each agent.

    Parameters:
        lag_window: If set, truncates the observation trace to the last
            ``lag_window`` steps before filtering.
    """
    if lag_window is not None:
        observed_actions = observed_actions[-lag_window:]
        observed_pool_trace = observed_pool_trace[-lag_window:]

    if rng is None:
        rng = np.random.default_rng()

    filters = {
        name: [
            Particle(
                sigma=sample_prior(reserve_range=reserve_range, prior_type=prior_type, rng=rng),
                weight=1.0 / n_particles,
            )
            for _ in range(n_particles)
        ]
        for name in agent_names
    }

    for a_obs_dict, pool_t in zip(observed_actions, observed_pool_trace, strict=True):
        for name in agent_names:
            act = a_obs_dict[name]
            particles = filters[name]
            for p in particles:
                lik = action_likelihood(p.sigma, act, xi, pool_belief=pool_t)
                p.weight *= lik

            total_w = sum(p.weight for p in particles)
            if total_w <= 0:
                warnings.warn(f"Mean-Field particle collapse for {name} — resetting", stacklevel=2)
                for p in particles:
                    p.weight = 1.0 / n_particles
            else:
                for p in particles:
                    p.weight /= total_w

            ess = 1.0 / sum(p.weight**2 for p in particles)
            if ess < ess_threshold_ratio * n_particles:
                filters[name] = _resample(particles, n_particles, rng=rng)

        actions_list = list(a_obs_dict.items())
        _, delta_log = step_env(pool_t, actions_list, concurrency_operator=concurrency_operator)

        for name in agent_names:
            act = a_obs_dict[name]
            for p in filters[name]:
                phi = feedback(
                    p.sigma,
                    pool_t,
                    delta_log["pool_after"],
                    act,
                    delta_log,
                    agent_name=name,
                )
                evolve(p.sigma, phi)

    return filters


def infer_hierarchical(
    observed_actions: list[Action],
    observed_pool_trace: list[float],
    xi: np.ndarray,
    n_particles: int = 300,
    agent_name: str = "unknown",
    ess_threshold_ratio: float = 0.5,
    reserve_range: tuple[float, float] = (1.0, 20.0),
    prior_type: str = "uniform",
    validate_invariants: bool = False,
    enable_withdraw_fee: bool = False,
    enable_withdraw_degrade: bool = False,
    lag_window: int | None = None,
    macro_n_particles: int = 50,
    goal_bias_strength: float = 2.0,
    rng: np.random.Generator | None = None,
) -> list[Particle]:
    """Two-pass hierarchical particle filter (Amazon Condor analogy, np-rock-hard.md).

    Macro-pass: a cheap small-N filter identifies the dominant GoalCategory cluster
    from the posterior goal distribution.
    Micro-pass: the full N-particle filter runs with a biased prior concentrated
    on the dominant cluster(s), reducing wasted probability mass in irrelevant regions.

    Parameters:
        macro_n_particles: Number of particles for the cheap macro-pass (default 50).
        goal_bias_strength: Additive bias applied to dominant goal's ``u`` index in
            the micro-pass prior (default 2.0).

    Note:
        SURVIVAL and ACQUISITION both map to ``REQUEST`` and are distinguishable only
        by action amount, not type. The macro-pass clusters by action type; within the
        REQUEST cluster, both goals receive equal bias. This is documented as a known
        limitation and does not affect SHARE (RELATIONAL) or WITHDRAW (STATUS) clusters.
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- Macro-pass: identify dominant goal cluster ---
    macro_particles = infer(
        observed_actions=observed_actions,
        observed_pool_trace=observed_pool_trace,
        xi=xi,
        n_particles=macro_n_particles,
        agent_name=agent_name,
        ess_threshold_ratio=ess_threshold_ratio,
        reserve_range=reserve_range,
        prior_type=prior_type,
        validate_invariants=validate_invariants,
        enable_withdraw_fee=enable_withdraw_fee,
        enable_withdraw_degrade=enable_withdraw_degrade,
        lag_window=lag_window,
        rng=rng,
    )

    from hypostases.engine.types import K
    from hypostases.inference.summaries import goal_posterior

    macro_post = goal_posterior(macro_particles)

    # Build goal_bias: apply bias_strength to any goal whose macro posterior mass > 0.2
    goal_bias: dict[str, float] = {
        g.value: goal_bias_strength for g in K if macro_post.get(g.value, 0.0) > 0.2
    }

    # --- Micro-pass: full filter with biased prior ---
    return infer(
        observed_actions=observed_actions,
        observed_pool_trace=observed_pool_trace,
        xi=xi,
        n_particles=n_particles,
        agent_name=agent_name,
        ess_threshold_ratio=ess_threshold_ratio,
        reserve_range=reserve_range,
        prior_type=prior_type,
        validate_invariants=validate_invariants,
        enable_withdraw_fee=enable_withdraw_fee,
        enable_withdraw_degrade=enable_withdraw_degrade,
        lag_window=lag_window,
        goal_bias=goal_bias,
        rng=rng,
    )
