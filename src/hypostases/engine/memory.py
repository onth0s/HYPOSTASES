"""HYPOSTASES Engine — Front 03 Memory Architecture Sub-systems.

Spec Ref: Front 03 (docs/front_03_memory_architecture.md), docs/WAVE_1_FRONT_03.
Grounded in SOTA literature (ICLR 2026 Lifelong Memory, Voyager, MemGPT, Generative Agents).

All memory mechanisms operate strictly as derived computational views and functional layers
over the primitive state tuple σ = (c, w, g, ρ_ext) without violating Rule 005 (Strict
Prohibition of Artificial Human Cognitive Deficiencies).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from hypostases.engine.types import N_K, Action, ActionType, GoalCategory

if TYPE_CHECKING:
    from hypostases.engine.types import AgentState


@dataclass
class ValenceVector:
    """Valence Vector v (ICLR 2026 Lifelong Memory §4 & Damasio Somatic Marker Hypothesis).

    In HYPOSTASES (Rule 005 compliant), 'valence' is formalized as a vector of computable,
    game-theoretic payoff and uncertainty metrics:
      - utility_gradient: ∇_c g payoff impact
      - precision_scalar: Conviction / confidence level at last recalculation π_G ∈ [0, 1]
      - density_scalar: Local neighborhood connectivity in Knowledge Graph
      - associative_pointers: Dict mapping connected gist IDs to edge weights
    """

    utility_gradient: np.ndarray = field(default_factory=lambda: np.zeros(N_K))
    precision_scalar: float = 0.8
    density_scalar: float = 1.0
    associative_pointers: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.utility_gradient = np.asarray(self.utility_gradient, dtype=float)
        if self.utility_gradient.shape != (N_K,):
            raise ValueError(
                f"utility_gradient must have shape ({N_K},), got {self.utility_gradient.shape}"
            )

    def clone(self) -> ValenceVector:
        return replace(
            self,
            utility_gradient=self.utility_gradient.copy(),
            associative_pointers=self.associative_pointers.copy(),
        )


@dataclass
class Gist:
    """Hierarchical Knowledge Graph Gist (ICLR 2026 Lifelong Memory §3.5 & Beck CBT Belief Hierarchy).

    Core, intermediate, and contextual beliefs emerge from the gateway selection weight W_G:
      - Core Beliefs: Extremely high weight, active across contexts.
      - Intermediate Beliefs: High weight in specific domains.
      - Contextual Gists: Activated transiently by matching cues.
    """

    gist_id: str
    concept: str
    valence: ValenceVector
    weight: float = 1.0
    labile: bool = False  # Set to True during cathartic update reconsolidation window
    last_updated_tick: int = 0

    def clone(self) -> Gist:
        return replace(self, valence=self.valence.clone())


@dataclass
class EpisodicEvent:
    """Episodic Trajectory Log Event e_t (Park et al. 2023 / CLS Theory)."""

    tick: int
    state_snapshot: dict[str, Any]
    action: Action
    surprise: float  # |δ_TD|
    utility_delta: float  # Δg impact
    next_state_snapshot: dict[str, Any]
    salience_score: float = 0.0

    def clone(self) -> EpisodicEvent:
        return replace(
            self,
            state_snapshot=self.state_snapshot.copy(),
            next_state_snapshot=self.next_state_snapshot.copy(),
        )


@dataclass
class SkillArtifact:
    """Reusable Procedural Macro-Action Skill (Voyager, Wang et al. 2023; Zelman et al. 2013 kMPs).

    Encapsulates macro-policy execution routines, continuous Kinematic Motion Primitives (kMPs),
    precondition triggers, and expected utility gain.
    """

    skill_id: str
    description: str
    preconditions: dict[str, Any]  # Logic predicates: min_reserve, min_world_mu, etc.
    macro_policy: list[dict[str, Any]]  # Sequence of macro steps
    expected_utility_gain: np.ndarray = field(default_factory=lambda: np.zeros(N_K))
    confidence: float = 0.5
    execution_count: int = 0
    # Kinematic Motion Primitives (kMPs) & Wave Dynamics (Zelman et al. 2013, Gutfreund et al. 1998)
    gaussian_weights: np.ndarray = field(
        default_factory=lambda: np.ones(4)
    )  # Default K=4 (Rule 008)
    gaussian_means: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 2))
    )  # (K, 2) for (s, t)
    gaussian_stds: np.ndarray = field(default_factory=lambda: np.ones((4, 2)))  # (K, 2) for (s, t)
    stiffness_wave_speed: float = 1.0

    def __post_init__(self) -> None:
        self.expected_utility_gain = np.asarray(self.expected_utility_gain, dtype=float)
        self.gaussian_weights = np.asarray(self.gaussian_weights, dtype=float)
        self.gaussian_means = np.asarray(self.gaussian_means, dtype=float)
        self.gaussian_stds = np.clip(np.asarray(self.gaussian_stds, dtype=float), 1e-6, None)

    def evaluate_kmp_trajectory(self, s: float, t: float) -> float:
        """Evaluate spatiotemporal Gaussian basis sum for continuous trajectory s at time t (Zelman et al. 2013)."""
        val = 0.0
        k_dim = len(self.gaussian_weights)
        for k in range(k_dim):
            w_k = self.gaussian_weights[k]
            mu_s, mu_t = self.gaussian_means[k, 0], self.gaussian_means[k, 1]
            sigma_s, sigma_t = self.gaussian_stds[k, 0], self.gaussian_stds[k, 1]
            diff_s = (s - mu_s) / sigma_s
            diff_t = (t * self.stiffness_wave_speed - mu_t) / sigma_t
            val += w_k * float(np.exp(-0.5 * (diff_s**2 + diff_t**2)))
        return float(val)

    def matches_preconditions(self, sigma: AgentState) -> bool:
        """Check if current state σ satisfies the skill's precondition predicates."""
        min_reserve = float(self.preconditions.get("min_reserve", 0.0))
        if sigma.c.reserve < min_reserve:
            return False

        min_mu = float(self.preconditions.get("min_world_mu", 0.0))
        if sigma.w.mu < min_mu:
            return False

        min_social = float(self.preconditions.get("min_social_capital", 0.0))
        if sigma.rho_ext.social_capital < min_social:
            return False

        req_goal = self.preconditions.get("required_goal")
        if req_goal:
            dominant_goal_idx = int(np.argmax(sigma.g.u))
            dominant_goal_name = GoalCategory[list(GoalCategory)[dominant_goal_idx].name].value
            if dominant_goal_name != req_goal:
                return False

        return True

    def resolve_action(self, step_idx: int, sigma: AgentState) -> Action | None:
        """Resolve the next micro-action step from the macro-policy sequence."""
        if not self.macro_policy or step_idx >= len(self.macro_policy):
            return None

        step_spec = self.macro_policy[step_idx]
        action_type_str = step_spec.get("action_type", "REQUEST")
        action_type = ActionType(action_type_str)
        amount_factor = float(step_spec.get("amount_factor", 1.0))
        target_rule = step_spec.get("target_rule", "SELF")

        target: str | None = None
        if target_rule == "HIGHEST_RESERVE_PEER" and sigma.w.peer_beliefs:
            target = max(sigma.w.peer_beliefs, key=sigma.w.peer_beliefs.get)  # type: ignore[arg-type]
        elif target_rule == "LOWEST_RESERVE_PEER" and sigma.w.peer_beliefs:
            target = min(sigma.w.peer_beliefs, key=sigma.w.peer_beliefs.get)  # type: ignore[arg-type]

        amount = amount_factor * max(1.0, sigma.c.reserve * 0.1)
        return Action(action_type=action_type, amount=amount, target=target)

    def clone(self) -> SkillArtifact:
        return replace(
            self,
            preconditions=self.preconditions.copy(),
            macro_policy=[step.copy() for step in self.macro_policy],
            expected_utility_gain=self.expected_utility_gain.copy(),
            gaussian_weights=self.gaussian_weights.copy(),
            gaussian_means=self.gaussian_means.copy(),
            gaussian_stds=self.gaussian_stds.copy(),
        )


@dataclass
class WorkingMemory:
    """Executive Function Context Window Buffer M_work (MemGPT / ICLR 2026 Lifelong Memory §3.1).

    Capacity-limited active workspace tracking active gists, recent items, and topological displacement.
    """

    capacity_limit: int = 8
    active_gists: list[Gist] = field(default_factory=list)
    recent_events: list[EpisodicEvent] = field(default_factory=list)

    def add_gist(self, gist: Gist) -> None:
        """Inject a gist into working memory. Displaces lowest-salience item if capacity reached."""
        existing_ids = [g.gist_id for g in self.active_gists]
        if gist.gist_id in existing_ids:
            idx = existing_ids.index(gist.gist_id)
            self.active_gists[idx] = gist
            return

        if len(self.active_gists) >= self.capacity_limit:
            # Capacity displacement: remove gist with lowest weight * precision
            self.active_gists.sort(key=lambda g: g.weight * g.valence.precision_scalar)
            self.active_gists.pop(0)

        self.active_gists.append(gist)

    def clone(self) -> WorkingMemory:
        return replace(
            self,
            active_gists=[g.clone() for g in self.active_gists],
            recent_events=[e.clone() for e in self.recent_events],
        )


@dataclass
class EpisodicMemory:
    """Salience-indexed Episodic Trajectory Store M_ep (Park et al. 2023 / CLS Theory)."""

    events: list[EpisodicEvent] = field(default_factory=list)
    max_history: int = 100

    def add_event(self, event: EpisodicEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.max_history:
            self.events.pop(0)

    def retrieve_salient(self, top_k: int = 5) -> list[EpisodicEvent]:
        """Retrieve top-k events sorted by salience score."""
        sorted_events = sorted(self.events, key=lambda e: e.salience_score, reverse=True)
        return sorted_events[:top_k]

    def clone(self) -> EpisodicMemory:
        return replace(self, events=[e.clone() for e in self.events])


@dataclass
class SemanticMemory:
    """Knowledge Graph Store M_sem (ICLR 2026 Lifelong Memory §3.2).

    Persistent store of gists, associative links, and spreading activation lookup.
    """

    gists: dict[str, Gist] = field(default_factory=dict)

    def add_gist(self, gist: Gist) -> None:
        self.gists[gist.gist_id] = gist

    def get_gist(self, gist_id: str) -> Gist | None:
        return self.gists.get(gist_id)

    def spreading_activation(
        self, seed_gist_ids: list[str], max_depth: int = 2, decay: float = 0.7
    ) -> dict[str, float]:
        """System 1 Spreading Activation (Collins & Loftus, 1975 / ICLR 2026 §5).

        Propagates activation through associative edges proportional to pointer weights.
        """
        activation_map: dict[str, float] = {gid: 1.0 for gid in seed_gist_ids if gid in self.gists}
        frontier = list(seed_gist_ids)

        for _depth in range(max_depth):
            next_frontier = []
            for current_id in frontier:
                if current_id not in self.gists:
                    continue
                current_gist = self.gists[current_id]
                current_act = activation_map[current_id]
                for target_id, weight in current_gist.valence.associative_pointers.items():
                    propagated = current_act * weight * decay
                    if propagated > activation_map.get(target_id, 0.0):
                        activation_map[target_id] = propagated
                        next_frontier.append(target_id)
            frontier = next_frontier

        return activation_map

    def clone(self) -> SemanticMemory:
        return replace(self, gists={gid: g.clone() for gid, g in self.gists.items()})


@dataclass
class ProceduralMemory:
    """Skill Artifact Library M_proc (Voyager, Wang et al. 2023)."""

    skills: dict[str, SkillArtifact] = field(default_factory=dict)

    def add_skill(self, skill: SkillArtifact) -> None:
        self.skills[skill.skill_id] = skill

    def find_applicable_skills(self, sigma: AgentState) -> list[SkillArtifact]:
        """Find all skills whose preconditions are satisfied by state σ."""
        return [skill for skill in self.skills.values() if skill.matches_preconditions(sigma)]

    def clone(self) -> ProceduralMemory:
        return replace(self, skills={sid: s.clone() for sid, s in self.skills.items()})


@dataclass
class ThalamicGateway:
    """Multi-Channel Epistemic Filter & Gating Service (ICLR 2026 Lifelong Memory §3.3).

    Tags incoming transitions across 6 independent salience channels:
      1. Temporal Difference Surprise: |δ_TD|
      2. Information Gain / Entropy Reduction: ΔH
      3. Goal Utility Impact: |Δg|
      4. Resource Urgency: κ_urgency
      5. Source Trust / Reputation: R_peer
      6. Topological Novelty: Novelty scalar
    """

    w_surp: float = 0.35
    w_info: float = 0.20
    w_util: float = 0.20
    w_urg: float = 0.10
    w_trust: float = 0.10
    w_nov: float = 0.05
    gating_threshold: float = 0.40

    def compute_salience(
        self,
        surprise: float,
        info_gain: float,
        utility_impact: float,
        urgency: float,
        trust: float,
        novelty: float,
    ) -> float:
        """Compute weighted multi-channel salience score."""
        score = (
            self.w_surp * min(1.0, abs(surprise))
            + self.w_info * min(1.0, abs(info_gain))
            + self.w_util * min(1.0, abs(utility_impact))
            + self.w_urg * min(1.0, abs(urgency))
            + self.w_trust * min(1.0, abs(trust))
            + self.w_nov * min(1.0, abs(novelty))
        )
        return float(score)

    def should_gate_to_working_memory(self, salience_score: float) -> bool:
        return salience_score >= self.gating_threshold


class CatharticUpdateEngine:
    """Precision-Weighted Belief Revision Engine (ICLR 2026 Lifelong Memory §4 / Beck CBT).

    Updates gist conviction parameters when empirical mismatch exceeds precision-weighted thresholds.
    """

    @staticmethod
    def evaluate_catharsis(
        gist: Gist,
        empirical_mismatch: float,
        tau_catharsis: float = 0.65,
    ) -> bool:
        """Evaluate if empirical contradiction in working memory triggers a cathartic update."""
        threshold = tau_catharsis * gist.valence.precision_scalar
        return empirical_mismatch > threshold

    @staticmethod
    def execute_cathartic_update(
        gist: Gist,
        new_utility_gradient: np.ndarray,
        empirical_mismatch: float,
        current_tick: int,
    ) -> Gist:
        """Perform cathartic update on gist parameters and recalculate precision scalar."""
        updated_gist = gist.clone()
        updated_gist.valence.utility_gradient = np.asarray(new_utility_gradient, dtype=float)
        # Recalculate precision: higher mismatch reduces immediate conviction until consolidated
        updated_gist.valence.precision_scalar = max(
            0.1, min(1.0, gist.valence.precision_scalar * (1.0 - 0.5 * empirical_mismatch))
        )
        updated_gist.labile = False
        updated_gist.last_updated_tick = current_tick
        return updated_gist
