"""HYPOSTASES — Front 03 Memory Architecture Test Suite.

Spec Ref: docs/front_03_memory_architecture.md, docs/WAVE_1_FRONT_03.
Ground truth checks for Working, Episodic, Semantic, Procedural Memory, Thalamic Gateway,
Cathartic Update Engine, SkillArtifacts, YAML Presets, and Rule 005 Invariant Enforcement.
"""

import numpy as np

from hypostases.engine.dynamics import FeedbackDelta, evolve
from hypostases.engine.memory import (
    CatharticUpdateEngine,
    EpisodicMemory,
    Gist,
    ProceduralMemory,
    SemanticMemory,
    SkillArtifact,
    ThalamicGateway,
    ValenceVector,
    WorkingMemory,
)
from hypostases.engine.types import (
    ActionType,
    AgentState,
    Characteristics,
    GoalHierarchy,
    PowerExternal,
    WorldModel,
)
from hypostases.schemas.loader import load_memory_preset, load_skill_artifact_schema


def test_valence_vector_and_gist_creation() -> None:
    """Verify ValenceVector and Gist initialization and cloning."""
    vv = ValenceVector(
        utility_gradient=np.array([0.5, 0.2, 0.0, 0.1]),
        precision_scalar=0.85,
        density_scalar=1.2,
        associative_pointers={"gist_b": 0.6},
    )
    gist = Gist(gist_id="gist_a", concept="RESOURCE_CONSERVATION", valence=vv, weight=2.5)

    assert gist.gist_id == "gist_a"
    assert gist.weight == 2.5
    assert gist.valence.precision_scalar == 0.85

    cloned_gist = gist.clone()
    assert cloned_gist.gist_id == gist.gist_id
    assert np.allclose(cloned_gist.valence.utility_gradient, gist.valence.utility_gradient)
    assert cloned_gist is not gist
    assert cloned_gist.valence is not gist.valence


def test_thalamic_gateway_salience_and_channel_independence() -> None:
    """Verify multi-channel salience scoring and single-channel independence protection."""
    gateway = ThalamicGateway(gating_threshold=0.40)

    # Low overall signals
    low_salience = gateway.compute_salience(
        surprise=0.1, info_gain=0.1, utility_impact=0.1, urgency=0.1, trust=0.1, novelty=0.1
    )
    assert low_salience < 0.40
    assert not gateway.should_gate_to_working_memory(low_salience)

    # Single high channel (e.g. high surprise |δ_TD|)
    high_surprise_salience = gateway.compute_salience(
        surprise=1.0, info_gain=0.0, utility_impact=0.0, urgency=0.0, trust=0.0, novelty=0.0
    )
    assert high_surprise_salience >= 0.35  # w_surp is 0.35
    high_util_salience = gateway.compute_salience(
        surprise=0.0, info_gain=0.0, utility_impact=1.0, urgency=1.0, trust=1.0, novelty=1.0
    )
    assert high_util_salience >= gateway.gating_threshold
    assert gateway.should_gate_to_working_memory(high_util_salience)


def test_working_memory_capacity_displacement() -> None:
    """Verify WorkingMemory capacity displacement of lowest-salience/precision items."""
    wm = WorkingMemory(capacity_limit=2)

    g1 = Gist("g1", "C1", ValenceVector(precision_scalar=0.9), weight=1.0)
    g2 = Gist("g2", "C2", ValenceVector(precision_scalar=0.3), weight=1.0)
    g3 = Gist("g3", "C3", ValenceVector(precision_scalar=0.95), weight=1.0)

    wm.add_gist(g1)
    wm.add_gist(g2)
    assert len(wm.active_gists) == 2

    # g3 should displace g2 (lowest weight * precision = 0.3)
    wm.add_gist(g3)
    assert len(wm.active_gists) == 2
    active_ids = [g.gist_id for g in wm.active_gists]
    assert "g1" in active_ids
    assert "g3" in active_ids
    assert "g2" not in active_ids


def test_semantic_memory_spreading_activation() -> None:
    """Verify System 1 spreading activation across connected Gists in Knowledge Graph."""
    sem = SemanticMemory()

    g_a = Gist("g_a", "PUNISHMENT", ValenceVector(associative_pointers={"g_b": 0.8}), weight=2.0)
    g_b = Gist("g_b", "COOPERATION", ValenceVector(associative_pointers={"g_c": 0.5}), weight=1.5)
    g_c = Gist("g_c", "TRUST", ValenceVector(), weight=1.0)

    sem.add_gist(g_a)
    sem.add_gist(g_b)
    sem.add_gist(g_c)

    activations = sem.spreading_activation(seed_gist_ids=["g_a"], max_depth=2, decay=0.7)
    assert "g_a" in activations
    assert activations["g_a"] == 1.0
    assert "g_b" in activations
    assert np.isclose(activations["g_b"], 1.0 * 0.8 * 0.7)
    assert "g_c" in activations
    assert activations["g_c"] < activations["g_b"]


def test_cathartic_update_engine() -> None:
    """Verify precision-weighted cathartic belief updating upon empirical mismatch."""
    v = ValenceVector(utility_gradient=np.array([1.0, 0.0, 0.0, 0.0]), precision_scalar=0.5)
    gist = Gist("g1", "CONCEPT", v, weight=1.0)

    # Low mismatch should not trigger catharsis
    low_mismatch = 0.2
    assert not CatharticUpdateEngine.evaluate_catharsis(
        gist, empirical_mismatch=low_mismatch, tau_catharsis=0.65
    )

    # High mismatch (> 0.65 * 0.5 = 0.325) triggers catharsis
    high_mismatch = 0.4
    assert CatharticUpdateEngine.evaluate_catharsis(
        gist, empirical_mismatch=high_mismatch, tau_catharsis=0.65
    )

    new_grad = np.array([0.2, 0.8, 0.0, 0.0])
    updated_gist = CatharticUpdateEngine.execute_cathartic_update(
        gist, new_utility_gradient=new_grad, empirical_mismatch=high_mismatch, current_tick=10
    )

    assert np.allclose(updated_gist.valence.utility_gradient, new_grad)
    assert updated_gist.last_updated_tick == 10
    assert updated_gist.valence.precision_scalar < gist.valence.precision_scalar


def test_skill_artifact_preconditions_and_macro_resolution() -> None:
    """Verify Voyager-style SkillArtifact precondition matching and action resolution."""
    skill = SkillArtifact(
        skill_id="DEFENSIVE_SHARE",
        description="Share resources when reserve is high and goal is SURVIVAL",
        preconditions={"min_reserve": 5.0, "min_world_mu": 2.0, "required_goal": "SURVIVAL"},
        macro_policy=[
            {"action_type": "SHARE", "amount_factor": 0.5, "target_rule": "HIGHEST_RESERVE_PEER"},
            {"action_type": "REQUEST", "amount_factor": 0.2, "target_rule": "SELF"},
        ],
        expected_utility_gain=np.array([0.5, 0.0, 0.5, 0.0]),
        confidence=0.9,
    )

    state = AgentState(
        c=Characteristics(reserve=10.0),
        w=WorldModel(mu=5.0, peer_beliefs={"peer_1": 8.0, "peer_2": 15.0}),
        g=GoalHierarchy(u=np.array([10.0, 1.0, 1.0, 1.0])),  # SURVIVAL dominant
        rho_ext=PowerExternal(),
    )

    assert skill.matches_preconditions(state)

    a0 = skill.resolve_action(0, state)
    assert a0 is not None
    assert a0.action_type == ActionType.SHARE
    assert a0.target == "peer_2"  # HIGHEST_RESERVE_PEER

    a1 = skill.resolve_action(1, state)
    assert a1 is not None
    assert a1.action_type == ActionType.REQUEST

    a2 = skill.resolve_action(2, state)
    assert a2 is None


def test_memory_integration_with_agent_state_and_evolve() -> None:
    """Verify end-to-end integration of Front 03 memory sub-systems with AgentState and evolve()."""
    ep_mem = EpisodicMemory()
    sem_mem = SemanticMemory()
    proc_mem = ProceduralMemory()
    work_mem = WorkingMemory()
    gateway = ThalamicGateway()

    state = AgentState(
        c=Characteristics(reserve=10.0),
        w=WorldModel(
            mu=10.0,
            m_ep=ep_mem,
            m_sem=sem_mem,
            m_proc=proc_mem,
            m_work=work_mem,
            thalamic_gateway=gateway,
        ),
        g=GoalHierarchy(),
        rho_ext=PowerExternal(),
    )

    phi = FeedbackDelta(
        delta_w={"last_surprise": 0.8, "mu": 0.2, "sigma2": 0.1},
        delta_g=np.array([0.5, 0.1, 0.0, 0.0]),
    )

    evolve(state, phi)

    # Episodic event should be logged
    assert len(state.w.m_ep.events) == 1
    event = state.w.m_ep.events[0]
    assert event.surprise == 0.8

    # High surprise event should be gated to working memory
    assert len(state.w.m_work.recent_events) == 1

    # Verify cloning preserves state invariants σ = (c, w, g, ρ_ext)
    cloned_state = state.clone()
    assert cloned_state.c.reserve == state.c.reserve
    assert len(cloned_state.w.m_ep.events) == 1
    assert cloned_state is not state


def test_schema_and_preset_loading() -> None:
    """Verify data-driven YAML loader compliance for Front 03 memory presets (Rule 006)."""
    preset = load_memory_preset()
    assert "working_memory" in preset
    assert preset["working_memory"]["capacity_limit"] == 8
    assert "thalamic_gateway" in preset
    assert "catharsis_engine" in preset

    skill_schema = load_skill_artifact_schema()
    assert "fields" in skill_schema
    assert "skill_id" in skill_schema["fields"]
