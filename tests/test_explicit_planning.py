"""HYPOSTASES Planning — Pytest Test Suite for Wave 2 Front 02 Explicit Planning Layer.

Spec Ref: docs/WAVE_2_FRONT_02/front_02_explicit_planning_layer_spec.md
Synthesizes AdaPlanner, GOAP 2003/2006, HTN, RAP, LATS, and Voyager.
Verifies Rules 001, 002, 003, 004, 005, 006, and 007.
"""

import pytest

from hypostases.engine.types import (
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)
from hypostases.planning import (
    ContingencyBranch,
    OutOfPlanInterruption,
    Plan,
    PlanExecutor,
    PlanLibrary,
    PlanNode,
    PlanRepairEngine,
    PlanStatus,
)
from hypostases.schemas.loader import load_planning_config


@pytest.fixture
def mock_agent_state() -> AgentState:
    """Fixture providing valid AgentState σ = (c, w, g, ρ_ext)."""
    return AgentState(
        c=Characteristics(reserve=10.0),
        w=WorldModel(
            mu=10.0, sigma2=2.0, peer_beliefs={"location": "base", "has_item": False, "energy": 100}
        ),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(social_capital=1.0),
    )


@pytest.fixture
def sample_plan() -> Plan:
    """Fixture providing multi-node sample Plan."""
    nodes = [
        PlanNode(
            node_id="node_1",
            action_name="GOTO_LOCATION",
            action_params={"target": "resource_zone"},
            preconditions={"location": "base"},
            effects={"location": "resource_zone"},
            expected_utility_delta=1.0,
        ),
        PlanNode(
            node_id="node_2",
            action_name="ACQUIRE_ITEM",
            action_params={"item": "mineral_1"},
            preconditions={"location": "resource_zone"},
            effects={"has_item": True},
            expected_utility_delta=2.5,
        ),
        PlanNode(
            node_id="node_3",
            action_name="RETURN_BASE",
            action_params={"target": "base"},
            preconditions={"has_item": True},
            effects={"location": "base"},
            expected_utility_delta=1.5,
        ),
    ]
    return Plan(
        plan_id="test_plan_001",
        goal_name="GATHER_RESOURCES",
        nodes=nodes,
        global_invariants={"energy": 100},
        status=PlanStatus.PLANNED,
    )


def test_load_planning_config():
    """Verify loading declarative planning configuration (Rule 006)."""
    cfg = load_planning_config()
    assert "max_planning_horizon" in cfg
    assert cfg["max_planning_horizon"] == 20
    assert "library" in cfg
    assert "repair_engine" in cfg


def test_plan_types_and_node_advancement(sample_plan):
    """Test Plan object structure and step advancement."""
    assert sample_plan.breakpoint_k == 0
    assert sample_plan.current_node().action_name == "GOTO_LOCATION"

    sample_plan.advance_step()
    assert sample_plan.breakpoint_k == 1
    assert sample_plan.current_node().action_name == "ACQUIRE_ITEM"

    sample_plan.advance_step()
    sample_plan.advance_step()
    assert sample_plan.is_finished()
    assert sample_plan.status == PlanStatus.COMPLETED


def test_closed_loop_executor_successful_execution(sample_plan, mock_agent_state):
    """Test step-by-step execution under valid state preconditions."""
    executor = PlanExecutor()

    # Step 1: GOTO_LOCATION
    res1 = executor.execute_step(sample_plan, mock_agent_state)
    assert res1.success
    assert res1.action_taken == "GOTO_LOCATION"
    assert sample_plan.breakpoint_k == 1

    # Update mock state to reflect effect
    mock_agent_state.w.peer_beliefs["location"] = "resource_zone"

    # Step 2: ACQUIRE_ITEM
    res2 = executor.execute_step(sample_plan, mock_agent_state)
    assert res2.success
    assert res2.action_taken == "ACQUIRE_ITEM"
    assert sample_plan.breakpoint_k == 2


def test_executor_out_of_plan_precondition_interruption(sample_plan, mock_agent_state):
    """Test out-of-plan feedback assertion when precondition fails (AdaPlanner Synthesis)."""
    executor = PlanExecutor()

    # Corrupt location precondition to trigger assertion error
    mock_agent_state.w.peer_beliefs["location"] = "unknown_wilderness"

    with pytest.raises(OutOfPlanInterruption) as exc_info:
        executor.execute_step(sample_plan, mock_agent_state)

    assert exc_info.value.breakpoint_k == 0
    assert "Precondition Violation" in str(exc_info.value)
    assert sample_plan.status == PlanStatus.INTERRUPTED


def test_executor_global_invariant_breach(sample_plan, mock_agent_state):
    """Test instant plan interruption upon global invariant breach (GOAP 2006)."""
    executor = PlanExecutor()

    # Breach global invariant (energy depleted)
    mock_agent_state.w.peer_beliefs["energy"] = 0

    with pytest.raises(OutOfPlanInterruption) as exc_info:
        executor.execute_step(sample_plan, mock_agent_state)

    assert "Global Invariant Breach" in str(exc_info.value)


def test_lats_contingency_branching(sample_plan, mock_agent_state):
    """Test dynamic contingency branching (LATS Synthesis)."""
    executor = PlanExecutor()

    # Add contingency branch to node_1
    branch = ContingencyBranch(
        branch_id="emergency_branch",
        condition_predicate=lambda w, c: w.get("threat_level", 0) > 5,
        target_node_id="node_3",
        description="Reroute directly to base upon threat",
    )
    sample_plan.nodes[0].contingency_branches.append(branch)

    # Set threat condition
    mock_agent_state.w.peer_beliefs["threat_level"] = 10

    res = executor.execute_step(sample_plan, mock_agent_state)
    assert res.success
    assert res.contingency_rerouted


def test_plan_repair_engine_counterfactual_patch(sample_plan, mock_agent_state):
    """Test closed-loop plan repair with RAP MCTS counterfactual rollouts."""
    executor = PlanExecutor()
    repair_engine = PlanRepairEngine()

    # Trigger interruption at node_2
    sample_plan.breakpoint_k = 1
    mock_agent_state.w.peer_beliefs["location"] = (
        "wrong_zone"  # Causes precondition failure for node_2
    )

    try:
        executor.execute_step(sample_plan, mock_agent_state)
    except OutOfPlanInterruption as interruption:
        patch_result = repair_engine.repair_plan(
            plan=sample_plan, interruption=interruption, agent_state=mock_agent_state
        )

        assert patch_result.repaired
        assert patch_result.strategy_used == "LOCAL_SUBGRAPH_PATCH"
        assert patch_result.patch_nodes_inserted >= 1
        assert sample_plan.status == PlanStatus.EXECUTING
        assert sample_plan.breakpoint_k == 1


def test_plan_library_goap_matching_and_yaml_persistence(sample_plan):
    """Test PlanLibrary A* matching, skill discovery, and YAML serialization (Rules 006 & 007)."""
    library = PlanLibrary()

    current_state = {"location": "base"}
    template = library.discover_and_archive_skill(
        plan=sample_plan, utility_gain=1.5, prerequisite_state=current_state
    )

    assert template is not None
    assert template.goal_name == "GATHER_RESOURCES"
    assert template.average_utility_gain == 1.5

    # Test GOAP A* template matching
    matched = library.match_template("GATHER_RESOURCES", current_state)
    assert matched is not None
    assert matched.template_id == template.template_id

    # Test Plan instantiation from template
    new_plan = library.instantiate_plan(matched, plan_id="plan_instantiated_001")
    assert new_plan.goal_name == "GATHER_RESOURCES"
    assert len(new_plan.nodes) == len(sample_plan.nodes)

    # Test YAML reloading
    reloaded_count = library.load_templates_from_yaml()
    assert reloaded_count >= 1
