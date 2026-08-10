"""
HYPOSTASES Infer -- Part VII reference implementation.

Canonical estimator: bootstrap particle filter (Part VII §10.2). Kalman-style
and MAP/grid estimators are read-outs of the SAME particle set, not separate
code paths (§10.1) -- see summarize_kalman() and summarize_map() below.

Reuses the exact generative model from hypostases_ref.py (pi_decision's
internals via action_likelihood, step_env, feedback, evolve) -- Infer adds
no new dynamics, only a weighting/resampling loop around existing ones,
per Part II §4.3's "one generative model, two directions" framing.
"""

from dataclasses import dataclass

import numpy as np
from hypostases_ref import (
    Action,
    AgentState,
    Characteristics,
    GoalHierarchy,
    K,
    PowerExternal,
    WorldModel,
    action_likelihood,
    evolve,
    feedback,
    step_env,
)

infer_rng = np.random.default_rng(seed=42)


def reseed_infer(seed: int):
    """Allows the sweep (Part VII §12.7) to vary Infer's own randomness
    (prior sampling + resampling) independently per trial, per the formal
    spec's multi-seed requirement. Existing worked examples (§12.1-12.5)
    are unaffected unless this is called."""
    global infer_rng
    infer_rng = np.random.default_rng(seed=seed)


@dataclass
class Particle:
    sigma: AgentState  # one hypothesis for the full primitive state sigma = (c, w, g, rho_ext)
    weight: float


def sample_prior(name: str, reserve_range=(1.0, 20.0)) -> AgentState:
    """Declared prior over Sigma = C x W x G x R_ext (Part VII §10.2 step 1).
    Wide, weakly-informative: reserve and goal-utilities are the parameters
    we most want Infer to recover, so they're drawn broadly; other fields
    use schema-typical defaults to keep the search space tractable.

    v3 fix (Part VII §12.7): reserve_range widened from (2,12) to (1,20).
    The original (2,12) range structurally excluded any true reserve above
    12 -- meaning the §12.5 adversarial test's D_high (reserve=14) could
    NEVER be represented by any particle, capping mu_high regardless of
    evidence strength. This was the actual root cause of §12.5's failed
    test, not weak coupling (see §12.7 for the diagnostic that found this)."""
    c = Characteristics(
        skill=0.6,
        resilience=0.5,
        sociality=infer_rng.uniform(0.0, 1.0),
        memory_decay=0.9,
        reserve=infer_rng.uniform(*reserve_range),
        mood=0.0,
    )
    w = WorldModel(mu=10.0, sigma2=2.0, replenish_rate_est=1.0)
    u = infer_rng.normal(loc=[1.0, 1.0, 1.0, 1.0], scale=0.8)
    pi = np.exp(u) / np.sum(np.exp(u))
    g = GoalHierarchy(pi=pi, u=u)
    rho_ext = PowerExternal(social_capital=1.0, time_budget=12)
    return AgentState(c, w, g, rho_ext, name)


def infer(
    observed_actions: list[Action],
    observed_pool_trace: list[float],
    xi: np.ndarray,
    n_particles: int = 300,
    agent_name: str = "unknown",
    ess_threshold_ratio: float = 0.5,
) -> list[Particle]:
    """
    Part VII §10.2: sequential Monte Carlo bootstrap particle filter.
    observed_actions[t], observed_pool_trace[t] are the evidence at each
    Tier-1 step. Returns the final weighted particle set approximating
    Delta(Sigma) (Part II §4.3).
    """
    particles = [
        Particle(sigma=sample_prior(agent_name), weight=1.0 / n_particles)
        for _ in range(n_particles)
    ]

    for _t, (a_obs, pool_t) in enumerate(zip(observed_actions, observed_pool_trace, strict=False)):
        # --- Reweight (§10.2 step 3): score each particle's CURRENT belief
        # against the observed action, before propagating forward.
        for p in particles:
            lik = action_likelihood(p.sigma, a_obs, xi, pool_belief=pool_t)
            p.weight *= lik

        total_w = sum(p.weight for p in particles)
        if total_w <= 0:
            for p in particles:
                p.weight = 1.0 / n_particles
        else:
            for p in particles:
                p.weight /= total_w

        # --- Resample if effective sample size has degenerated (§10.2 step 4)
        ess = 1.0 / sum(p.weight**2 for p in particles)
        if ess < ess_threshold_ratio * n_particles:
            particles = _resample(particles, n_particles)

        # --- Propose (§10.2 step 2): propagate each particle one step using
        # the SAME evolve()/feedback() functions used in forward simulation.
        # We propagate using the OBSERVED action (not a freshly sampled one)
        # since we're conditioning on what actually happened.
        _, delta_log = step_env(pool_t, [(agent_name, a_obs)])
        for p in particles:
            phi = feedback(p.sigma, pool_t, delta_log["pool_after"], a_obs, delta_log)
            evolve(p.sigma, phi)

    return particles


def _resample(particles: list[Particle], n: int, roughen_reserve_sd: float = 0.3) -> list[Particle]:
    """
    v3 fix (Part VII §12.7): adds post-resample roughening (jitter) on
    reserve, the standard SMC correction for particle impoverishment.
    Diagnosis (§12.7): weak per-step WITHDRAW evidence combined with a wide
    prior caused ESS to collapse fast (300->~20 within 10 steps), so
    repeated resampling was collapsing onto a small, seed-dependent set of
    survivors well before evidence had a chance to separate D_low/D_high --
    producing noise-dominated, non-monotonic Z across seeds regardless of
    coupling strength. Roughening re-injects diversity after each resample
    so the filter doesn't prematurely commit to a handful of particles.
    """
    weights = np.array([p.weight for p in particles])
    idx = infer_rng.choice(len(particles), size=n, p=weights)
    resampled = [Particle(sigma=particles[i].sigma.clone(), weight=1.0 / n) for i in idx]
    if roughen_reserve_sd > 0:
        for p in resampled:
            p.sigma.c.reserve = max(
                0.0, p.sigma.c.reserve + infer_rng.normal(0, roughen_reserve_sd)
            )
    return resampled


# ---------------------------------------------------------------------------
# §10.1 -- MAP and Kalman-style summaries as READ-OUTS of the same particle
# set, not separate estimators.
# ---------------------------------------------------------------------------
def summarize_map(particles: list[Particle]) -> AgentState:
    """Degenerate case: report only the single highest-weight particle."""
    return max(particles, key=lambda p: p.weight).sigma


def summarize_kalman(particles: list[Particle]) -> dict:
    """Degenerate case: weighted mean/variance over the continuous sub-blocks
    (reserve, mood), collapsing the multimodal goal posterior into a single
    Gaussian summary -- valid only when a goal story is already fixed/assumed,
    per §10.1."""
    weights = np.array([p.weight for p in particles])
    reserves = np.array([p.sigma.c.reserve for p in particles])
    moods = np.array([p.sigma.c.mood for p in particles])
    return {
        "reserve_mean": float(np.average(reserves, weights=weights)),
        "reserve_var": float(
            np.average((reserves - np.average(reserves, weights=weights)) ** 2, weights=weights)
        ),
        "mood_mean": float(np.average(moods, weights=weights)),
    }


def goal_posterior(particles: list[Particle]) -> dict:
    """Not a degenerate summary -- the genuinely multimodal readout §10.1
    argues Kalman/MAP cannot represent: weighted distribution over which
    goal category is dominant (argmax of pi) across the particle set."""
    weights = np.array([p.weight for p in particles])
    dominant = [K[int(np.argmax(p.sigma.g.pi))] for p in particles]
    out = {k: 0.0 for k in K}
    for d, w in zip(dominant, weights, strict=False):
        out[d] += w
    return out
