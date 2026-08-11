import numpy as np
import pytest

from hypostases.engine import (
    AgentState,
    Characteristics,
    DeltaLog,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)
from hypostases.simulation.harness import build_agent


@pytest.fixture
def make_state():
    """Pytest factory fixture for instantiating customized AgentState instances."""

    def _factory(
        reserve: float = 10.0,
        sociality: float = 0.5,
        mood: float = 0.0,
        resilience: float = 0.5,
        skill: float = 0.6,
        memory_decay: float = 0.9,
        mu: float = 10.0,
        sigma2: float = 2.0,
        replenish_rate_est: float = 1.0,
        peer_beliefs: dict[str, float] | None = None,
        u: np.ndarray | None = None,
        social_capital: float = 1.0,
        time_budget: float = 12.0,
    ) -> AgentState:
        c = Characteristics(
            skill=skill,
            resilience=resilience,
            sociality=sociality,
            memory_decay=memory_decay,
            reserve=reserve,
            mood=mood,
        )
        w = WorldModel(
            mu=mu,
            sigma2=sigma2,
            replenish_rate_est=replenish_rate_est,
            peer_beliefs=peer_beliefs or {},
        )
        g = GoalHierarchy(u=u if u is not None else np.array([1.0, 1.0, 1.0, 1.0]))
        rho_ext = PowerExternal(social_capital=social_capital, time_budget=time_budget)
        return AgentState(c=c, w=w, g=g, rho_ext=rho_ext)

    return _factory


@pytest.fixture
def default_agent_state(make_state) -> AgentState:
    return make_state(reserve=10.0, sociality=0.5)


@pytest.fixture
def default_delta_log() -> DeltaLog:
    return {
        "pool_before": 10.0,
        "pool_after_shares": 10.0,
        "pool_after": 10.0,
        "shares_total": 0.0,
        "requests_total": 0.0,
        "granted": {},
        "punishments": {},
        "enable_withdraw_fee": False,
        "actions_log": {},
    }


@pytest.fixture
def default_xi() -> np.ndarray:
    return np.array([0.2, 0.2, 0.2, 0.2])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


__all__ = ["build_agent"]
