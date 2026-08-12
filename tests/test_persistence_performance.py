"""Performance and Dual-Persistence Benchmarks for Plan Storage & Meta-Parameters (Rules 007 & 011).

Validates:
  1. Rule 007: Plan Storage Persistence Format YAML serialization overhead is bounded under simulation ticks.
  2. Rule 011: Dual persistence for meta-parameters (in-memory projection + YAML serialization).
"""

import time
from pathlib import Path

import yaml

from hypostases.natural_language_compression import SymbolicCompressionEngine
from hypostases.planning.plan_library import PlanLibrary, PlanTemplate


def test_rule007_plan_storage_yaml_serialization_performance(tmp_path: Path):
    """Benchmarks YAML serialization latency of Plan Storage Persistence Format (Rule 007)."""
    cfg = {"library": {"storage_directory": str(tmp_path / "plans")}}
    library = PlanLibrary(config=cfg)

    t0 = time.perf_counter()
    n_iterations = 100
    for i in range(n_iterations):
        tpl_copy = PlanTemplate(
            template_id=f"template_perf_{i}",
            goal_name="survival_goal",
            node_specs=[{"node_id": "n1", "action_name": "REQUEST", "expected_utility_delta": 1.5}],
            average_utility_gain=1.5,
        )
        library.save_template_to_yaml(tpl_copy)

    elapsed = time.perf_counter() - t0
    avg_latency_ms = (elapsed / n_iterations) * 1000.0

    # Rule 007 validation: Average serialization latency per plan is under 5ms
    assert avg_latency_ms < 5.0, f"YAML plan serialization too slow: {avg_latency_ms:.2f}ms/plan"


def test_rule011_meta_parameters_dual_persistence(tmp_path: Path):
    """Validates dual persistence for meta-parameters theta_meta (in-memory + persistent YAML snapshot)."""
    engine = SymbolicCompressionEngine()
    snapshot_path = tmp_path / "meta_params_snapshot.yaml"

    # 1. In-memory meta-parameters projection
    assert hasattr(engine, "meta_params")
    assert isinstance(engine.meta_params, tuple)
    assert len(engine.meta_params) == 4

    # 2. Persistent human-readable YAML serialization
    engine.save_snapshot_yaml(snapshot_path)
    assert snapshot_path.exists()

    with open(snapshot_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "meta_parameters" in data
    assert data["meta_parameters"]["lambda_mdl"] == engine.lambda_mdl

    # 3. Snapshot reload verification
    new_engine = SymbolicCompressionEngine()
    new_engine.load_snapshot_yaml(snapshot_path)
    assert new_engine.lambda_mdl == engine.lambda_mdl
