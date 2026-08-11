"""HYPOSTASES Inference — Prior Distribution Sampler over Agent State Space Σ."""

from __future__ import annotations

import numpy as np

from hypostases.engine import (
    AgentState,
    Characteristics,
    GoalCategory,
    GoalHierarchy,
    K,
    PowerExternal,
    WorldModel,
)


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
