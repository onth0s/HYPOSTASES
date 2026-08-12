"""Demo script executing a multi-agent simulation run and exporting the compartmentalized state bundle."""

import random

from hypostases.meta_learning.meta_optimizer import MetaLearner
from hypostases.meta_learning.meta_state import MetaParameterVector
from hypostases.simulation.exporter import RunExporter


def main():
    run_id = "demo_run_001"
    exporter = RunExporter(run_id=run_id)

    agent_names = ["Agent_A", "Agent_B"]
    manifest_path = exporter.initialize_run(
        scenario_name="Active_Sensing_Swarm_Demo", seed=42, agent_names=agent_names
    )
    print(f"Initialized run manifest: {manifest_path}")

    # Set up meta-learners for both agents
    meta_learners = {
        "Agent_A": MetaLearner(
            initial_params=MetaParameterVector(learning_rate=0.01, particle_count=16, efe_beta=0.4)
        ),
        "Agent_B": MetaLearner(
            initial_params=MetaParameterVector(learning_rate=0.02, particle_count=32, efe_beta=0.7)
        ),
    }

    random.seed(42)

    # Execute 10 simulation ticks
    for tick in range(1, 11):
        tick_meta = {}
        for agent_name, learner in meta_learners.items():
            # Simulate tick feedback: utility gain, belief variance reduction, compute cost
            u_gain = random.uniform(0.5, 2.5)
            var_red = random.uniform(0.02, 0.15)
            c_cost = random.uniform(1.0, 5.0)

            updated_params = learner.adapt_step(
                utility_gain=u_gain, belief_var_reduction=var_red, compute_cost=c_cost
            )
            tick_meta[agent_name] = updated_params

        # Save checkpoint snapshot every 5 ticks
        if tick % 5 == 0:
            chk_path = exporter.save_checkpoint(tick, tick_meta)
            print(f"Saved checkpoint at tick {tick}: {chk_path}")

    # Finalize run
    summary_path = exporter.finalize_run(
        summary_metrics={
            "ticks_completed": 10,
            "agent_a_final_particle_count": meta_learners["Agent_A"].meta_params.particle_count,
            "agent_b_final_particle_count": meta_learners["Agent_B"].meta_params.particle_count,
            "status": "SUCCESS",
        }
    )
    print(f"Finalized run summary: {summary_path}")


if __name__ == "__main__":
    main()
