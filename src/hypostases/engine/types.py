"""HYPOSTASES Engine — v4 Primitive State Types and Spaces.

Spec Ref: Part I §2 (v4), Part IV §6 (Schema v1).
Persistent per-agent state is strictly the four-tuple:
  σ = (c, w, g, ρ_ext) ∈ C × W × G × R_ext

Key v4 Invariant: GoalHierarchy g = u ∈ ℝ^{n_k} stores ONLY latent utility weights.
Goal allocation probabilities π ∈ Δ(K) are transient policy allocations computed
dynamically inside pi_decision / _goal_probs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)


from typing import Any, TypedDict

import numpy as np

from hypostases.engine._math import compute_omega, softmax


class GoalCategory(StrEnum):
    SURVIVAL = "SURVIVAL"
    ACQUISITION = "ACQUISITION"
    RELATIONAL = "RELATIONAL"
    STATUS = "STATUS"


K: tuple[GoalCategory, ...] = tuple(GoalCategory)
N_K: int = len(GoalCategory)


class ActionType(StrEnum):
    REQUEST = "REQUEST"
    SHARE = "SHARE"
    WITHDRAW = "WITHDRAW"
    PUNISH = "PUNISH"
    INSPECT = "INSPECT"
    PROBE = "PROBE"
    MONITOR = "MONITOR"
    QUERY = "QUERY"
    EXPERIMENT = "EXPERIMENT"
    VERIFY = "VERIFY"
    OBSERVE = "OBSERVE"
    SPY = "SPY"


EPISTEMIC_ACTION_TYPES: set[ActionType] = {
    ActionType.INSPECT,
    ActionType.PROBE,
    ActionType.MONITOR,
    ActionType.QUERY,
    ActionType.EXPERIMENT,
    ActionType.VERIFY,
    ActionType.OBSERVE,
    ActionType.SPY,
}


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    amount: float = 0.0
    target: str | None = None

    def __repr__(self) -> str:
        if self.action_type == ActionType.WITHDRAW:
            return "WITHDRAW"
        if self.action_type == ActionType.PUNISH:
            return f"PUNISH({self.target})"
        if self.action_type in EPISTEMIC_ACTION_TYPES:
            target_str = f", target='{self.target}'" if self.target else ""
            return f"{self.action_type.value}({target_str})"
        return f"{self.action_type.value}({self.amount:.2f})"


@dataclass
class Characteristics:
    """c ∈ C = ℝ^6 (Part I §2.2.1, Part IV §6.2)"""

    skill: float = 0.6
    resilience: float = 0.5
    sociality: float = 0.5
    memory_decay: float = 0.9
    reserve: float = 10.0
    mood: float = 0.0

    def clone(self) -> Characteristics:
        return replace(self)


@dataclass
class WorldModel:
    """w ∈ W = Δ(S × ∏_{j≠i} Σ^{(j)}) × F (Part I §2.2.1, Part IV §6.3)

    Parametric Gaussian belief N(μ, σ²) over environment state, paired
    with a learned replenish rate estimate, peer latent belief tracking,
    and Front 03 Memory Architecture sub-systems (M_ep, M_sem, M_proc, M_work).
    """

    mu: float = 10.0
    sigma2: float = 2.0
    replenish_rate_est: float = 1.0
    peer_beliefs: dict[str, float] = field(default_factory=dict)
    last_surprise: float = 0.0  # Contention 2: tracks previous surprise for regime-shift detection
    m_ep: Any = field(default=None)  # EpisodicMemory instance
    m_sem: Any = field(default=None)  # SemanticMemory instance
    m_proc: Any = field(default=None)  # ProceduralMemory instance
    m_work: Any = field(default=None)  # WorkingMemory instance
    thalamic_gateway: Any = field(default=None)  # ThalamicGateway instance

    def clone(self) -> WorldModel:
        return replace(
            self,
            peer_beliefs=self.peer_beliefs.copy(),
            m_ep=self.m_ep.clone()
            if self.m_ep is not None and hasattr(self.m_ep, "clone")
            else self.m_ep,
            m_sem=self.m_sem.clone()
            if self.m_sem is not None and hasattr(self.m_sem, "clone")
            else self.m_sem,
            m_proc=self.m_proc.clone()
            if self.m_proc is not None and hasattr(self.m_proc, "clone")
            else self.m_proc,
            m_work=self.m_work.clone()
            if self.m_work is not None and hasattr(self.m_work, "clone")
            else self.m_work,
            thalamic_gateway=replace(self.thalamic_gateway)
            if self.thalamic_gateway is not None
            and hasattr(self.thalamic_gateway, "__dataclass_fields__")
            else self.thalamic_gateway,
        )


@dataclass
class GoalHierarchy:
    """g = u ∈ G = ℝ^{n_k} (v4 Target, Part I §2.0, §2.2.1, §5.5)

    Stores strictly persistent latent utility parameters `u`.
    Softmax policy allocation π ∈ Δ(K) is computed dynamically inside
    pi_decision and likelihood routines.
    """

    u: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 1.0]))

    def __post_init__(self) -> None:
        self.u = np.asarray(self.u, dtype=float)
        if self.u.shape != (N_K,):
            raise ValueError(f"u must be a vector of length {N_K}, got shape {self.u.shape}")

    @property
    def pi(self) -> np.ndarray:
        """Raw, temperature-free policy allocation view π_raw = softmax(u).

        Note: Forward decision dynamics and particle inference readouts use temperature-scaled
        and affordability-adjusted allocations via ``goal_probs(sigma, xi, pool_belief)``.
        """
        return softmax(self.u)

    def clone(self) -> GoalHierarchy:
        return replace(self, u=self.u.copy())


@dataclass
class PowerExternal:
    """ρ_ext ∈ R_ext = ℝ≥0^2 (Part I §2.2.1, Part IV §6.6a)"""

    social_capital: float = 1.0
    time_budget: float = 12.0

    def clone(self) -> PowerExternal:
        return replace(self)


class DeltaLog(TypedDict, total=False):
    pool_before: float
    pool_after_shares: float
    pool_after: float
    shares_total: float
    requests_total: float
    granted: dict[str, float]
    punishments: dict[str, float]
    enable_withdraw_fee: bool
    actions_log: dict[str, Action]


class DeltaCharacteristics(TypedDict, total=False):
    reserve: float
    mood: float


class DeltaWorldModel(TypedDict, total=False):
    mu: float
    sigma2: float
    replenish_rate_est: float
    last_surprise: float


class DeltaPowerExternal(TypedDict, total=False):
    social_capital: float
    time_budget: float


@dataclass
class FeedbackDelta:
    """φ ∈ Φ = ΔC × ΔW × ΔG × ΔR_ext (Part I §2.2.1, Part II §3.3)"""

    delta_c: DeltaCharacteristics = field(default_factory=dict)
    delta_w: DeltaWorldModel = field(default_factory=dict)
    delta_g: np.ndarray = field(default_factory=lambda: np.zeros(N_K))
    delta_rho_ext: DeltaPowerExternal = field(default_factory=dict)
    delta_peer_beliefs: dict[str, float] = field(default_factory=dict)


@dataclass
class AgentState:
    """σ^(i)_t = (c^(i)_t, w^(i)_t, g^(i)_t, ρ_ext^(i)_t) (v4 four-tuple, Part I §2.3)"""

    c: Characteristics
    w: WorldModel
    g: GoalHierarchy
    rho_ext: PowerExternal
    decay_mode: str = "variance"

    def power_internal(self) -> dict[str, float]:
        """ρ_int = proj_int(c) (derived read-only view, Part I §2.2.2, Part IV §6.6b)"""
        return {"reserve_capacity": self.c.reserve}

    def omega(self, xi: np.ndarray | None = None, pool_belief: float = 10.0) -> np.ndarray:
        """ω = derive_Ω(u, ρ_ext, ρ_int, c) (derived view, Part I §2.2.2, Part IV §6.5).

        In v4, willingness scales transient goal allocation π by affordability against
        dynamically scarcity-adjusted action costs (Contention 1).

        Parameters:
            pool_belief: Current pool estimate S_t used to compute endogenous scarcity costs.
        """
        return compute_omega(self.g.u, self.c.reserve, xi, pool_belief=pool_belief)

    def clone(self) -> AgentState:
        return replace(
            self,
            c=self.c.clone(),
            w=self.w.clone(),
            g=self.g.clone(),
            rho_ext=self.rho_ext.clone(),
        )


@dataclass
class Agent:
    """Named agent entity wrapping identity metadata and primitive state tuple σ."""

    name: str
    sigma: AgentState

    def clone(self) -> Agent:
        return replace(self, sigma=self.sigma.clone())
