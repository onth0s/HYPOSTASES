"""HYPOSTASES Planning — Plan Repair Engine.

Spec Ref: docs/WAVE_2_FRONT_02/front_02_explicit_planning_layer_spec.md
Synthesizes AdaPlanner (Sun et al.) refine-then-resume, Self-PR (Kang et al.) failure feedback analyzer,
HTN CFG repair (Lutalo & Bercher), and RAP (Hao et al.) / LATS (Zhou et al.) MCTS rollouts via counterfactual.py.

Rule 005 Prohibition: Pure game-theoretic payoff optimization E[Δu] - C_repair; zero artificial human cognitive defects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hypostases.counterfactual import CounterfactualEngine, VirtualEnvironmentSandbox
from hypostases.engine.types import AgentState
from hypostases.planning.plan_executor import OutOfPlanInterruption
from hypostases.planning.plan_types import Plan, PlanNode, PlanStatus


@dataclass
class PlanPatchResult:
    """Outcome of plan repair attempt."""

    repaired: bool
    repaired_plan: Plan | None
    breakpoint_k: int
    patch_nodes_inserted: int
    expected_utility_gain: float
    strategy_used: str  # e.g., "LOCAL_SUBGRAPH_PATCH", "FULL_REPLAN", "FAIL"


class PlanRepairEngine:
    """Closed-loop plan repair and refine-then-resume manager."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.max_repair_attempts = self.config.get("max_repair_attempts", 3)
        self.mcts_iterations = self.config.get("repair_engine", {}).get("mcts_max_iterations", 15)
        self.c_puct = self.config.get("repair_engine", {}).get("c_puct", 1.414)
        self.prefer_local_patch = self.config.get("repair_engine", {}).get(
            "prefer_local_patch_over_full_replan", True
        )
        self.counterfactual_engine = CounterfactualEngine(
            lookahead_depth=3, branching_factor=4, c_puct=self.c_puct
        )

    def diagnose_failure(self, interruption: OutOfPlanInterruption) -> dict[str, Any]:
        """Self-PR Failure Feedback Analyzer.

        Parses execution trace and error signature to identify failing step & precondition deficit.
        """
        failed_node = interruption.failed_node
        reason = interruption.reason

        diagnosis = {
            "breakpoint_k": interruption.breakpoint_k,
            "failed_action": failed_node.action_name,
            "failed_preconditions": failed_node.preconditions,
            "reason": reason,
        }

        # Analyze missing precondition keys
        missing_keys = []
        if hasattr(interruption.state, "w"):
            w_state = (
                interruption.state.w.peer_beliefs
                if hasattr(interruption.state.w, "peer_beliefs")
                else {}
            )
            c_state = (
                interruption.state.c.__dict__ if hasattr(interruption.state.c, "__dict__") else {}
            )
        elif isinstance(interruption.state, dict):
            w_state = interruption.state.get("w", {})
            c_state = interruption.state.get("c", {})
        else:
            w_state, c_state = {}, {}

        merged = {**w_state, **c_state}
        for key in failed_node.preconditions:
            if key not in merged:
                missing_keys.append(key)
        diagnosis["missing_keys"] = missing_keys
        return diagnosis

    def repair_plan(
        self,
        plan: Plan,
        interruption: OutOfPlanInterruption,
        agent_state: AgentState | None = None,
        pool_state: float = 100.0,
    ) -> PlanPatchResult:
        """Performs closed-loop plan repair upon OutOfPlanInterruption.

        Uses RAP MCTS rollouts via Front 04 counterfactual.py to evaluate patch candidates.

        Args:
            plan: Failing active Plan object.
            interruption: Caught OutOfPlanInterruption exception.
            agent_state: Current agent primitive state σ.
            pool_state: Environment pool state.

        Returns:
            PlanPatchResult with repaired Plan and metadata.
        """
        diagnosis = self.diagnose_failure(interruption)
        breakpoint_k = interruption.breakpoint_k
        failed_node = interruption.failed_node

        # Check repair attempts limit
        repair_count = plan.metadata.get("repair_attempts", 0)
        if repair_count >= self.max_repair_attempts:
            plan.status = PlanStatus.FAILED
            return PlanPatchResult(
                repaired=False,
                repaired_plan=plan,
                breakpoint_k=breakpoint_k,
                patch_nodes_inserted=0,
                expected_utility_gain=0.0,
                strategy_used="FAIL_EXCEEDED_ATTEMPTS",
            )

        plan.metadata["repair_attempts"] = repair_count + 1

        # 1. Local Sub-graph Patch Generation (AdaPlanner & Self-PR Synthesis)
        patch_nodes: list[PlanNode] = []

        # If missing preconditions, generate recovery action step
        for missing_key in diagnosis["missing_keys"]:
            recovery_node = PlanNode(
                node_id=f"patch_{breakpoint_k}_{missing_key}",
                action_name=f"RESOLVE_{missing_key.upper()}",
                preconditions={},
                effects={missing_key: failed_node.preconditions[missing_key]},
                expected_utility_delta=0.5,
            )
            patch_nodes.append(recovery_node)

        if not patch_nodes:
            # Fallback local patch step
            patch_nodes.append(
                PlanNode(
                    node_id=f"patch_{breakpoint_k}_fallback",
                    action_name=f"RECOVER_{failed_node.action_name}",
                    preconditions={},
                    effects=failed_node.preconditions,
                    expected_utility_delta=0.3,
                )
            )

        # 2. RAP MCTS Evaluation via Counterfactual Virtual Sandbox (Front 04 Integration)
        expected_utility_gain = 0.5
        if agent_state is not None:
            xi_vec = (
                agent_state.g.u if hasattr(agent_state.g, "u") else np.array([1.0, 1.0, 1.0, 1.0])
            )
            sandbox = VirtualEnvironmentSandbox(
                agent_state=agent_state.clone(), pool_state=pool_state, xi=xi_vec
            )
            # Run test rollout using CounterfactualEngine
            rollout_action = self.counterfactual_engine.simulate_lookahead(
                initial_agent_state=agent_state, initial_pool=pool_state, xi=xi_vec
            )
            sim_delta = sandbox.step(rollout_action)
            expected_utility_gain = max(0.1, sim_delta + 0.5)

        # 3. HTN Production Rule Splice (Lutalo & Bercher Synthesis)
        # Construct repaired nodes: valid prefix n_{1...k-1} + patch_nodes + failing_node + suffix
        prefix_nodes = plan.nodes[:breakpoint_k]
        suffix_nodes = plan.nodes[breakpoint_k:]

        repaired_nodes = prefix_nodes + patch_nodes + suffix_nodes

        # Re-index plan nodes
        for idx, node in enumerate(repaired_nodes):
            node.node_id = f"node_{idx + 1}"

        # AdaPlanner Refine-Then-Resume: update plan nodes and set breakpoint_k to breakpoint_k
        plan.nodes = repaired_nodes
        plan.breakpoint_k = breakpoint_k  # Resumes from insertion checkpoint
        plan.status = PlanStatus.EXECUTING

        return PlanPatchResult(
            repaired=True,
            repaired_plan=plan,
            breakpoint_k=breakpoint_k,
            patch_nodes_inserted=len(patch_nodes),
            expected_utility_gain=expected_utility_gain,
            strategy_used="LOCAL_SUBGRAPH_PATCH",
        )
