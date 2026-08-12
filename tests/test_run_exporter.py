"""Unit tests for RunExporter module."""

import os

from hypostases.meta_learning.meta_state import MetaParameterVector
from hypostases.simulation.exporter import RunExporter


def test_run_exporter_flow(tmp_path):
    base_dir = str(tmp_path / "exports")
    exporter = RunExporter(run_id="test_run_123", base_dir=base_dir)

    # 1. Test initialize_run
    manifest_path = exporter.initialize_run(
        scenario_name="TestScenario", seed=100, agent_names=["Agent_1", "Agent_2"]
    )
    assert os.path.exists(manifest_path)

    # 2. Test save_checkpoint
    params = MetaParameterVector(learning_rate=0.05, particle_count=20)
    agent_params = {"Agent_1": params, "Agent_2": params}
    chk_dir = exporter.save_checkpoint(tick=10, agent_meta_params=agent_params)
    assert os.path.exists(chk_dir)
    assert os.path.exists(os.path.join(chk_dir, "Agent_1_meta.yaml"))
    assert os.path.exists(os.path.join(chk_dir, "Agent_2_meta.yaml"))

    # 3. Test finalize_run
    summary_path = exporter.finalize_run(summary_metrics={"score": 100})
    assert os.path.exists(summary_path)
