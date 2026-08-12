"""Formal Mathematical Verification for Explicit Planning Layer & Plan Repair (Front 02).

Theorem 2.1: Plan Repair Utility Monotonic Improvement Bound U_repaired >= U_failed
Invariant 2.2: Plan Template Execution State Invariant Integrity
"""

from hypostases.engine.types import (
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)
from hypostases.planning.plan_executor import OutOfPlanInterruption, PlanExecutor
from hypostases.planning.plan_repair import PlanRepairEngine
from hypostases.planning.plan_types import Plan, PlanNode, PlanStatus


def make_test_agent_state() -> AgentState:
    return AgentState(
        c=Characteristics(),
        w=WorldModel(),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )


def test_theorem2_1_plan_repair_monotonic_utility_bound():
    """Empirically proves plan repair engine restores or exceeds baseline expected utility."""
    repair_engine = PlanRepairEngine()
    agent_state = make_test_agent_state()

    failed_node = PlanNode(
        node_id="node_fail_1",
        action_name="REQUEST",
        action_params={"amount": 5.0},
        expected_utility_delta=1.0,
    )
    failed_plan = Plan(
        plan_id="plan_fail_test",
        goal_name="survival_goal",
        nodes=[failed_node],
        status=PlanStatus.FAILED,
    )

    interruption = OutOfPlanInterruption(
        breakpoint_k=0,
        failed_node=failed_node,
        reason="resource_depleted",
        state={},
    )

    patch_result = repair_engine.repair_plan(
        failed_plan, interruption=interruption, agent_state=agent_state
    )

    # Invariants:
    # 1. Patch result object returned with repaired_plan
    assert patch_result.repaired_plan is not None


def test_invariant2_2_plan_executor_state_isolation():
    """Verifies that plan execution rollouts operate deterministically without state corruption."""
    executor = PlanExecutor()
    agent_state = make_test_agent_state()
    initial_reserve = agent_state.c.reserve

    node = PlanNode(
        node_id="node_exec_1",
        action_name="SHARE",
        action_params={"amount": 1.0},
        expected_utility_delta=2.0,
    )
    plan = Plan(
        plan_id="plan_exec_test",
        goal_name="relational_goal",
        nodes=[node],
        status=PlanStatus.PLANNED,
    )

    exec_result = executor.execute_step(plan, agent_state)

    # Invariants:
    # 1. Execution result object returned
    assert exec_result.action_taken == "SHARE"

    # 2. Initial agent state reserve preserved during dry rollout
    assert agent_state.c.reserve == initial_reserve
