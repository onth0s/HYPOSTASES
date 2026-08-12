"""HYPOSTASES Planning — Closed-Loop Plan Executor.

Spec Ref: docs/WAVE_2_FRONT_02/front_02_explicit_planning_layer_spec.md
Synthesizes AdaPlanner (Sun et al.) assertion checks and ask_LLM() in-plan refinement,
GOAP 2006 (Orkin) real-time tick precondition validation, and Plan-and-Solve (Wang et al.) step tracking.

Rule 005 Prohibition: Pure game-theoretic execution dynamics; zero artificial human cognitive defects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hypostases.planning.plan_types import Plan, PlanNode, PlanStatus


class OutOfPlanInterruption(Exception):  # noqa: N818
    """Raised when an out-of-plan feedback assertion or state invariant fails during tick execution."""

    def __init__(self, breakpoint_k: int, failed_node: PlanNode, reason: str, state: dict):
        self.breakpoint_k = breakpoint_k
        self.failed_node = failed_node
        self.reason = reason
        self.state = state
        super().__init__(
            f"OutOfPlanInterruption at step {breakpoint_k} [{failed_node.action_name}]: {reason}"
        )


@dataclass
class ExecutionResult:
    """Outcome of single tick plan execution step."""

    success: bool
    action_taken: str
    action_params: dict[str, Any]
    in_plan_refined: bool = False
    contingency_rerouted: bool = False
    message: str = ""


class PlanExecutor:
    """Closed-loop plan execution & real-time monitoring engine."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.deviation_threshold = self.config.get("deviation_threshold_epsilon", 0.15)
        self.verify_preconditions = self.config.get("executor", {}).get(
            "verify_preconditions_every_tick", True
        )
        self.verify_invariants = self.config.get("executor", {}).get(
            "verify_global_invariants_every_tick", True
        )

    def verify_state_match(
        self, required: dict[str, Any], current: dict[str, Any]
    ) -> tuple[bool, str]:
        """Checks whether all required state key-value pairs match the current state."""
        for key, val in required.items():
            if key not in current:
                return False, f"Missing required state key '{key}'"
            if current[key] != val:
                return False, f"State mismatch for '{key}': expected {val}, got {current[key]}"
        return True, "OK"

    def execute_step(
        self, plan: Plan, agent_state: dict[str, Any], environment_callback: Any | None = None
    ) -> ExecutionResult:
        """Executes a single step of the plan with closed-loop monitoring.

        Args:
            plan: Active Plan object.
            agent_state: Current agent primitive state dict σ = (c, w, g, ρ_ext).
            environment_callback: Optional environment step function.

        Returns:
            ExecutionResult containing execution status.

        Raises:
            OutOfPlanInterruption: If precondition or global invariant fails (AdaPlanner out-of-plan).
        """
        if plan.is_finished():
            return ExecutionResult(
                success=False,
                action_taken="NONE",
                action_params={},
                message="Plan already completed or failed",
            )

        plan.status = PlanStatus.EXECUTING
        current_node = plan.current_node()

        if current_node is None:
            plan.status = PlanStatus.COMPLETED
            return ExecutionResult(
                success=True,
                action_taken="NONE",
                action_params={},
                message="Plan reached completion",
            )

        merged_state: dict[str, Any] = {}
        w_state: dict[str, Any] = {}
        c_state: dict[str, Any] = {}

        if hasattr(agent_state, "w") and hasattr(agent_state, "c"):
            w_obj = agent_state.w
            c_obj = agent_state.c

            if hasattr(w_obj, "__dict__"):
                w_state.update(w_obj.__dict__)
            if hasattr(w_obj, "peer_beliefs") and isinstance(w_obj.peer_beliefs, dict):
                w_state.update(w_obj.peer_beliefs)

            if hasattr(c_obj, "__dict__"):
                c_state.update(c_obj.__dict__)

            merged_state = {**w_state, **c_state}
        elif isinstance(agent_state, dict):
            w_val = agent_state.get("w", {})
            c_val = agent_state.get("c", {})
            w_state = w_val if isinstance(w_val, dict) else {}
            c_state = c_val if isinstance(c_val, dict) else {}
            merged_state = {**agent_state, **w_state, **c_state}
        else:
            w_state, c_state, merged_state = {}, {}, {}

        # 1. GOAP 2006 / AdaPlanner Tick-Level Global Invariant Check
        if self.verify_invariants and plan.global_invariants:
            valid, msg = self.verify_state_match(plan.global_invariants, merged_state)
            if not valid:
                plan.status = PlanStatus.INTERRUPTED
                raise OutOfPlanInterruption(
                    breakpoint_k=plan.breakpoint_k,
                    failed_node=current_node,
                    reason=f"Global Invariant Breach: {msg}",
                    state=agent_state,
                )

        # 2. GOAP 2006 Precondition Verification
        if self.verify_preconditions and current_node.preconditions:
            valid, msg = self.verify_state_match(current_node.preconditions, merged_state)
            if not valid:
                plan.status = PlanStatus.INTERRUPTED
                raise OutOfPlanInterruption(
                    breakpoint_k=plan.breakpoint_k,
                    failed_node=current_node,
                    reason=f"Precondition Violation: {msg}",
                    state=agent_state,
                )

        # 3. LATS Dynamic Contingency Branching Check
        contingency_triggered = False
        for branch in current_node.contingency_branches:
            if branch.condition_predicate(w_state, c_state):
                # Reroute to target node index
                for idx, node in enumerate(plan.nodes):
                    if node.node_id == branch.target_node_id:
                        plan.breakpoint_k = idx
                        current_node = node
                        contingency_triggered = True
                        break

        # 4. In-Plan Refinement (AdaPlanner ask_LLM / State Extraction Synthesis)
        in_plan_refined = False
        if "dynamic_info_key" in current_node.action_params:
            key_to_extract = current_node.action_params["dynamic_info_key"]
            if key_to_extract in merged_state:
                current_node.action_params["extracted_info"] = merged_state[key_to_extract]
                in_plan_refined = True

        # 5. Step Execution Action Dispatch
        action_name = current_node.action_name
        action_params = current_node.action_params

        if environment_callback is not None:
            # Execute step via environment callback
            environment_callback(action_name, action_params)

        # Advance breakpoint to next node
        plan.advance_step()

        return ExecutionResult(
            success=True,
            action_taken=action_name,
            action_params=action_params,
            in_plan_refined=in_plan_refined,
            contingency_rerouted=contingency_triggered,
            message="Step executed cleanly",
        )
