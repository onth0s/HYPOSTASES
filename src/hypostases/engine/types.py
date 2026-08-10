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
from enum import Enum
from typing import TypedDict

import numpy as np

from hypostases.engine._math import compute_omega, softmax


class GoalCategory(str, Enum):
    SURVIVAL = "SURVIVAL"
    ACQUISITION = "ACQUISITION"
    RELATIONAL = "RELATIONAL"
    STATUS = "STATUS"


K: tuple[GoalCategory, ...] = tuple(GoalCategory)
N_K: int = len(GoalCategory)


class ActionType(str, Enum):
    REQUEST = "REQUEST"
    SHARE = "SHARE"
    WITHDRAW = "WITHDRAW"


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    amount: float = 0.0

    def __repr__(self) -> str:
        if self.action_type == ActionType.WITHDRAW:
            return "WITHDRAW"
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
    with a learned replenish rate estimate and peer latent belief tracking.
    """

    mu: float = 10.0
    sigma2: float = 2.0
    replenish_rate_est: float = 1.0
    peer_beliefs: dict[str, float] = field(default_factory=dict)

    def clone(self) -> WorldModel:
        return replace(self, peer_beliefs=self.peer_beliefs.copy())


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
        """Dynamic/transient policy allocation π = softmax(u) (v4 read-only helper)."""
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
    actions_log: dict[str, Action]


@dataclass
class FeedbackDelta:
    """φ ∈ Φ = ΔC × ΔW × ΔG × ΔR_ext (Part I §2.2.1, Part II §3.3)"""

    delta_c: dict[str, float] = field(default_factory=dict)
    delta_w: dict[str, float] = field(default_factory=dict)
    delta_g: np.ndarray = field(default_factory=lambda: np.zeros(N_K))
    delta_rho_ext: dict[str, float] = field(default_factory=dict)
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

    def omega(self, xi: np.ndarray | None = None) -> np.ndarray:
        """ω = derive_Ω(u, ρ_ext, ρ_int, c) (derived view, Part I §2.2.2, Part IV §6.5).

        In v4, willingness scales transient goal allocation π by affordability.
        """
        return compute_omega(self.g.u, self.c.reserve, xi)

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
