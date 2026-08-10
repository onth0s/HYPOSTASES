"""Tests — Institutional Crowding-Out ("The Fine Dilemma").

Verifies multi-epoch utility hysteresis: toggling enable_withdraw_fee ON then OFF
shifts latent utilities g.u from RELATIONAL to ACQUISITION, testing voluntary cooperation.
"""

from __future__ import annotations

from hypostases.engine.dynamics import evolve, feedback, step_env
from hypostases.engine.types import Action, ActionType, GoalCategory, K
from hypostases.simulation.scenarios import create_scenario_agents


class TestInstitutionalCrowdingOut:
    def test_scenario_crowding_out_loading(self):
        agents = create_scenario_agents("crowding_out")
        assert "Agent_1" in agents
        assert "Agent_2" in agents

    def test_fee_toggling_utility_hysteresis(self):
        agents = create_scenario_agents("crowding_out")
        agent_1 = agents["Agent_1"]

        rel_idx = K.index(GoalCategory.RELATIONAL)
        acq_idx = K.index(GoalCategory.ACQUISITION)

        initial_u_rel = agent_1.g.u[rel_idx]
        initial_u_acq = agent_1.g.u[acq_idx]

        # Phase 1: Institutional fine active (enable_withdraw_fee=True) - agent withdraws under fine
        withdraw_action = Action(ActionType.WITHDRAW)

        for _ in range(10):
            pool_after, delta_log = step_env(
                10.0, [("Agent_1", withdraw_action)], enable_withdraw_fee=True
            )
            phi = feedback(
                agent_1, 10.0, pool_after, withdraw_action, delta_log, agent_name="Agent_1"
            )
            evolve(agent_1, phi)

        # Phase 2: Remove fine (enable_withdraw_fee=False)
        for _ in range(10):
            pool_after, delta_log = step_env(
                10.0, [("Agent_1", withdraw_action)], enable_withdraw_fee=False
            )
            phi = feedback(
                agent_1, 10.0, pool_after, withdraw_action, delta_log, agent_name="Agent_1"
            )
            evolve(agent_1, phi)

        u_rel_final = agent_1.g.u[rel_idx]
        u_acq_final = agent_1.g.u[acq_idx]

        # Hysteresis check: after fine is removed, ACQUISITION weight should be higher or RELATIONAL lower than initial
        assert (u_acq_final > initial_u_acq) or (u_rel_final < initial_u_rel), (
            "Utility hysteresis expected: fine removal should not immediately reset utilities"
        )
