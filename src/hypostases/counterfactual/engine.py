"""HYPOSTASES Engine — v4 Counterfactual Simulation & Multi-Future Lookahead."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hypostases.engine.dynamics import evolve, feedback, pi_decision, step_env
from hypostases.engine.types import Action, ActionType, AgentState
from hypostases.schemas import assert_invariants


@dataclass
class VirtualEnvironmentSandbox:
    """Ephemeral sandbox for zero-side-effect forward rollouts of agent state σ and environment."""

    agent_state: AgentState
    pool_state: float
    xi: np.ndarray
    kappa_max: float = 1.0  # Dynamic elastica curvature bound (Cacace et al. 2020)

    def step(self, action: Action) -> float:
        """Simulate one discrete environment tick, return scalar utility delta Δu."""
        # 1. Environment step
        agent_actions = [("agent", action)]
        next_pool, delta_log = step_env(self.pool_state, agent_actions)

        # 2. Feedback stage
        agent_delta = feedback(
            self.agent_state, self.pool_state, next_pool, action, delta_log, agent_name="agent"
        )

        # Calculate immediate utility before evolution
        u_before = float(np.sum(self.agent_state.g.u))

        # 3. State Evolution stage
        evolve(self.agent_state, agent_delta)
        self.pool_state = next_pool

        u_after = float(np.sum(self.agent_state.g.u))
        return u_after - u_before

    def clone(self) -> VirtualEnvironmentSandbox:
        return VirtualEnvironmentSandbox(
            agent_state=self.agent_state.clone(),
            pool_state=self.pool_state,
            xi=self.xi.copy(),
            kappa_max=self.kappa_max,
        )


@dataclass
class CounterfactualBranch:
    """Represents a simulated hypothetical future trajectory."""

    action_sequence: list[Action] = field(default_factory=list)
    accumulated_utility: float = 0.0
    utilities: list[float] = field(default_factory=list)
    final_state: AgentState | None = None

    def evaluate_discounted_value(
        self, discount_factor: float = 0.95, risk_lambda: float = 0.1
    ) -> float:
        """Calculate risk-adjusted discounted expected utility."""
        if not self.utilities:
            return 0.0
        discounted_sum = sum((discount_factor**t) * u for t, u in enumerate(self.utilities))
        variance = float(np.var(self.utilities)) if len(self.utilities) > 1 else 0.0
        return discounted_sum - (risk_lambda * variance)


class CounterfactualEngine:
    """Multi-future counterfactual simulator and EvoCF rollout manager."""

    def __init__(
        self,
        lookahead_depth: int = 3,
        branching_factor: int = 4,
        mcts_simulations: int = 25,
        discount_factor: float = 0.95,
        risk_penalty_lambda: float = 0.1,
        c_puct: float = 1.414,
        evocf_mutation_rate: float = 0.2,
    ) -> None:
        self.lookahead_depth = lookahead_depth
        self.branching_factor = branching_factor
        self.mcts_simulations = mcts_simulations
        self.discount_factor = discount_factor
        self.risk_penalty_lambda = risk_penalty_lambda
        self.c_puct = c_puct
        self.evocf_mutation_rate = evocf_mutation_rate

    def simulate_lookahead(
        self,
        initial_agent_state: AgentState,
        initial_pool: float,
        xi: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> Action:
        """Runs multi-future lookahead search and returns optimal physical action.

        Guarantees zero side effects on initial_agent_state.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Check invariant before sandbox cloning
        assert_invariants(initial_agent_state)

        # Direct reactive decision if depth == 0
        if self.lookahead_depth <= 0:
            return pi_decision(initial_agent_state, initial_pool, xi, rng=rng)

        branches: list[CounterfactualBranch] = []
        action_space = [
            Action(ActionType.REQUEST, amount=2.0),
            Action(ActionType.SHARE, amount=1.0),
            Action(ActionType.WITHDRAW, amount=0.0),
            Action(ActionType.PUNISH, target="peer", amount=1.0),
        ]

        # Sample trajectories up to branching_factor
        for b in range(self.branching_factor):
            sandbox = VirtualEnvironmentSandbox(
                agent_state=initial_agent_state.clone(),
                pool_state=initial_pool,
                xi=xi.copy(),
            )
            branch = CounterfactualBranch()

            # First step action choice
            first_action = action_space[b % len(action_space)]
            branch.action_sequence.append(first_action)
            u_delta = sandbox.step(first_action)
            branch.utilities.append(u_delta)

            # Rollout up to lookahead_depth
            for _ in range(1, self.lookahead_depth):
                rollout_action = pi_decision(
                    sandbox.agent_state, sandbox.pool_state, sandbox.xi, rng=rng
                )
                branch.action_sequence.append(rollout_action)
                u_delta = sandbox.step(rollout_action)
                branch.utilities.append(u_delta)

            branch.final_state = sandbox.agent_state
            branch.accumulated_utility = branch.evaluate_discounted_value(
                discount_factor=self.discount_factor,
                risk_lambda=self.risk_penalty_lambda,
            )
            branches.append(branch)

        # Select action sequence with maximum expected discounted utility
        best_branch = max(branches, key=lambda br: br.accumulated_utility)
        selected_action = best_branch.action_sequence[0]

        # Verify physical state invariant remained untouched
        assert_invariants(initial_agent_state)

        return selected_action

    def evaluate_dubins_reachability(
        self,
        start_pose: np.ndarray,
        target_pose: np.ndarray,
        kappa_max: float = 1.0,
    ) -> float:
        """Computes Dubins reachability path distance under curvature bound kappa_max (Cacace et al. 2020).

        Returns estimated continuous path length under physical curvature constraints.
        """
        start = np.asarray(start_pose, dtype=float)
        target = np.asarray(target_pose, dtype=float)
        euclidean_dist = float(np.linalg.norm(target[:2] - start[:2]))

        # Curvature penalty scaling (Dubins car model lower bound approximation)
        r_min = 1.0 / max(1e-6, kappa_max)
        curvature_penalty = 0.5 * r_min * abs(target[-1] - start[-1]) if len(start) > 2 else 0.0
        return euclidean_dist + curvature_penalty

    def mutate_plan_evocf(
        self,
        branch: CounterfactualBranch,
        rng: np.random.Generator,
    ) -> CounterfactualBranch:
        """EvoCF: Perform evolutionary plan mutation over candidate action trajectory."""
        mutated_actions = list(branch.action_sequence)
        action_types = tuple(ActionType)

        for i in range(len(mutated_actions)):
            if rng.random() < self.evocf_mutation_rate:
                new_type = rng.choice(action_types)
                mutated_actions[i] = Action(new_type, amount=1.0)

        return CounterfactualBranch(action_sequence=mutated_actions)
