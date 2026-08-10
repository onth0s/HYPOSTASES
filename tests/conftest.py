import numpy as np
import pytest

from hypostases.engine import AgentState, Characteristics, GoalHierarchy, PowerExternal, WorldModel


@pytest.fixture
def default_agent_state() -> AgentState:
    return AgentState(
        c=Characteristics(reserve=10.0, sociality=0.5),
        w=WorldModel(),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )


@pytest.fixture
def default_xi() -> np.ndarray:
    return np.array([0.2, 0.2, 0.2, 0.2])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
