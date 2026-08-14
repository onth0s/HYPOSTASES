"""Tests for UnifiedRunManager: creation, atomic saves, resume resolution, and metadata.

Rule 012 compliant: all assertions are empirical, not just surface-level.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from hypostases.simulation.run_manager import UnifiedRunManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mgr(tmp_path: Path) -> UnifiedRunManager:
    return UnifiedRunManager(base_dir=tmp_path / "runs")


# ---------------------------------------------------------------------------
# Run creation
# ---------------------------------------------------------------------------


def test_create_run_produces_subdirs(mgr: UnifiedRunManager) -> None:
    run_dir = mgr.create_run()
    for sub in ("nnue", "pgn/self_play", "pgn/ground_a", "pgn/ground_b", "telemetry"):
        assert (run_dir / sub).is_dir(), f"Missing subdirectory: {sub}"


def test_create_run_default_name_is_timestamped(mgr: UnifiedRunManager) -> None:
    run_dir = mgr.create_run()
    assert run_dir.name.startswith("run_"), f"Unexpected name: {run_dir.name}"


def test_create_run_custom_id(mgr: UnifiedRunManager) -> None:
    run_dir = mgr.create_run(run_id="run_custom_abc")
    assert run_dir.name == "run_custom_abc"
    assert run_dir.is_dir()


def test_latest_pointer_updated_on_create(mgr: UnifiedRunManager, tmp_path: Path) -> None:
    run_dir = mgr.create_run()
    latest_path = tmp_path / "runs" / "latest"
    assert latest_path.exists()
    assert latest_path.read_text(encoding="utf-8") == run_dir.name


# ---------------------------------------------------------------------------
# Nth-latest resolution
# ---------------------------------------------------------------------------


def test_get_nth_latest_run_empty(mgr: UnifiedRunManager) -> None:
    assert mgr.get_nth_latest_run(1) is None


def test_get_nth_latest_run_single(mgr: UnifiedRunManager) -> None:
    run_dir = mgr.create_run(run_id="run_20250101_000001")
    assert mgr.get_nth_latest_run(1) == run_dir
    assert mgr.get_nth_latest_run(2) is None


def test_get_nth_latest_run_ordering(mgr: UnifiedRunManager) -> None:
    # Create three runs with deterministic names so sorting is predictable
    r1 = mgr.create_run(run_id="run_20250101_000001")
    time.sleep(0.01)
    r2 = mgr.create_run(run_id="run_20250101_000002")
    time.sleep(0.01)
    r3 = mgr.create_run(run_id="run_20250101_000003")

    assert mgr.get_nth_latest_run(1) == r3
    assert mgr.get_nth_latest_run(2) == r2
    assert mgr.get_nth_latest_run(3) == r1
    assert mgr.get_nth_latest_run(4) is None


def test_list_runs_newest_first(mgr: UnifiedRunManager) -> None:
    r1 = mgr.create_run(run_id="run_20250101_000001")
    r2 = mgr.create_run(run_id="run_20250101_000002")
    r3 = mgr.create_run(run_id="run_20250101_000003")
    result = mgr.list_runs()
    assert result == [r3, r2, r1]


# ---------------------------------------------------------------------------
# Run metadata (atomic JSON)
# ---------------------------------------------------------------------------


def test_save_and_load_run_metadata_roundtrip(mgr: UnifiedRunManager) -> None:
    run_dir = mgr.create_run()
    mgr.save_run_metadata(
        run_dir,
        status=UnifiedRunManager.STATUS_RUNNING,
        target_generations=20,
        last_completed_gen=7,
        total_games_played=105,
        started_at="2025-01-01T00:00:00+00:00",
    )
    meta = mgr.load_run_metadata(run_dir)
    assert meta["status"] == "running"
    assert meta["target_generations"] == 20
    assert meta["last_completed_gen"] == 7
    assert meta["total_games_played"] == 105


def test_load_run_metadata_missing_raises(mgr: UnifiedRunManager) -> None:
    run_dir = mgr.create_run()
    with pytest.raises(FileNotFoundError):
        mgr.load_run_metadata(run_dir)


def test_metadata_file_is_valid_json(mgr: UnifiedRunManager) -> None:
    run_dir = mgr.create_run()
    mgr.save_run_metadata(
        run_dir,
        status=UnifiedRunManager.STATUS_COMPLETED,
        target_generations=10,
        last_completed_gen=10,
    )
    raw = (run_dir / "run_metadata.json").read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert parsed["status"] == "completed"


def test_interrupted_metadata_saves_interrupted_at_gen(mgr: UnifiedRunManager) -> None:
    run_dir = mgr.create_run()
    mgr.save_run_metadata(
        run_dir,
        status=UnifiedRunManager.STATUS_INTERRUPTED,
        target_generations=20,
        last_completed_gen=5,
        interrupted_at_gen=5,
        interrupted_at="2025-01-01T12:00:00+00:00",
    )
    meta = mgr.load_run_metadata(run_dir)
    assert meta["status"] == "interrupted"
    assert meta["interrupted_at_gen"] == 5


# ---------------------------------------------------------------------------
# Checkpoint state (atomic YAML)
# ---------------------------------------------------------------------------


def test_save_and_load_checkpoint_state_roundtrip(mgr: UnifiedRunManager) -> None:
    run_dir = mgr.create_run()
    theta = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    telemetry = {3: {"avg_game_length": 45.2, "theta_norm_delta": 0.012}}

    mgr.save_checkpoint_state(
        run_dir,
        gen_idx=3,
        theta_meta=theta,
        temperature=0.5,
        beta_efe=0.07,
        telemetry=telemetry,
    )

    state = mgr.load_checkpoint_state(run_dir)
    assert state["last_saved_gen"] == 3
    assert pytest.approx(state["agent"]["temperature"]) == 0.5
    assert pytest.approx(state["agent"]["beta_efe"]) == 0.07
    assert pytest.approx(state["agent"]["theta_meta"]) == theta
    assert "3" in state["telemetry"]


def test_checkpoint_state_file_is_valid_yaml(mgr: UnifiedRunManager) -> None:
    run_dir = mgr.create_run()
    mgr.save_checkpoint_state(
        run_dir,
        gen_idx=1,
        theta_meta=[0.0] * 10,
        temperature=0.8,
        beta_efe=0.05,
        telemetry={},
    )
    raw = (run_dir / "checkpoint_state.yaml").read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    assert parsed["last_saved_gen"] == 1


def test_load_checkpoint_state_missing_raises(mgr: UnifiedRunManager) -> None:
    run_dir = mgr.create_run()
    with pytest.raises(FileNotFoundError):
        mgr.load_checkpoint_state(run_dir)


# ---------------------------------------------------------------------------
# NNUE weights (atomic npz)
# ---------------------------------------------------------------------------


def test_save_and_load_nnue_weights_roundtrip(mgr: UnifiedRunManager) -> None:
    run_dir = mgr.create_run()
    weights = {
        "W_white": np.random.randn(10, 5).astype(np.float32),
        "W_black": np.random.randn(10, 5).astype(np.float32),
        "W_l1": np.random.randn(5, 5).astype(np.float32),
        "b_l1": np.random.randn(5).astype(np.float32),
        "W_l2": np.random.randn(1, 5).astype(np.float32),
        "b_l2": np.random.randn(1).astype(np.float32),
    }
    mgr.save_nnue_weights(run_dir, weights, tag="latest")
    loaded = mgr.load_nnue_weights(run_dir, tag="latest")
    for key in weights:
        np.testing.assert_array_almost_equal(weights[key], loaded[key])


def test_load_nnue_weights_missing_raises(mgr: UnifiedRunManager) -> None:
    run_dir = mgr.create_run()
    with pytest.raises(FileNotFoundError):
        mgr.load_nnue_weights(run_dir, tag="latest")


def test_nnue_save_is_atomic_no_partial_read(mgr: UnifiedRunManager, tmp_path: Path) -> None:
    """Atomic write: a .tmp.npz file must not persist after save_nnue_weights returns."""
    run_dir = mgr.create_run()
    weights = {"W": np.zeros((3, 3), dtype=np.float32)}
    mgr.save_nnue_weights(run_dir, weights, tag="latest")
    tmp_artifacts = list((run_dir / "nnue").glob("*.tmp*"))
    assert tmp_artifacts == [], f"Stale tmp files found: {tmp_artifacts}"


# ---------------------------------------------------------------------------
# Snapshots index
# ---------------------------------------------------------------------------


def test_save_and_load_snapshots_index(mgr: UnifiedRunManager) -> None:
    run_dir = mgr.create_run()
    snaps = [
        {"generation": 0, "temperature": 0.8, "theta_meta": [0.0] * 10},
        {"generation": 5, "temperature": 0.6, "theta_meta": [0.1] * 10},
    ]
    mgr.save_snapshots_index(run_dir, snaps)
    loaded = mgr.load_snapshots_index(run_dir)
    assert len(loaded) == 2
    assert loaded[1]["generation"] == 5


def test_load_snapshots_index_empty_when_missing(mgr: UnifiedRunManager) -> None:
    run_dir = mgr.create_run()
    assert mgr.load_snapshots_index(run_dir) == []


# ---------------------------------------------------------------------------
# Full atomic checkpoint bundle
# ---------------------------------------------------------------------------


def test_save_full_checkpoint_creates_all_files(mgr: UnifiedRunManager) -> None:
    run_dir = mgr.create_run()
    weights = {"W": np.zeros((2, 2), dtype=np.float32)}
    mgr.save_full_checkpoint(
        run_dir,
        status=UnifiedRunManager.STATUS_RUNNING,
        target_generations=20,
        last_completed_gen=5,
        total_games_played=75,
        theta_meta=[0.1] * 10,
        temperature=0.7,
        beta_efe=0.05,
        nnue_weights=weights,
        telemetry={5: {"avg_game_length": 50.0}},
        snapshots_meta=[{"generation": 5, "temperature": 0.7, "theta_meta": [0.1] * 10}],
        started_at="2025-01-01T00:00:00+00:00",
    )
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "checkpoint_state.yaml").exists()
    assert (run_dir / "snapshots_index.json").exists()
    assert (run_dir / "nnue" / "nnue_latest.npz").exists()


def test_save_full_checkpoint_latest_pointer_updated(
    mgr: UnifiedRunManager, tmp_path: Path
) -> None:
    run_dir = mgr.create_run()
    weights = {"W": np.zeros((2, 2), dtype=np.float32)}
    mgr.save_full_checkpoint(
        run_dir,
        status=UnifiedRunManager.STATUS_COMPLETED,
        target_generations=10,
        last_completed_gen=10,
        total_games_played=150,
        theta_meta=[0.0] * 10,
        temperature=0.8,
        beta_efe=0.05,
        nnue_weights=weights,
        telemetry={},
        snapshots_meta=[],
    )
    latest = (tmp_path / "runs" / "latest").read_text(encoding="utf-8")
    assert latest == run_dir.name


# ---------------------------------------------------------------------------
# Legacy migration helper
# ---------------------------------------------------------------------------


def test_migrate_legacy_pgn_dirs_removes_directory(tmp_path: Path) -> None:
    mgr = UnifiedRunManager(base_dir=tmp_path / "runs")
    legacy_dir = tmp_path / "exports" / "pgn"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "game001.pgn").write_text("test", encoding="utf-8")

    removed = mgr.migrate_legacy_pgn_dirs([str(legacy_dir)])
    assert str(legacy_dir) in removed
    assert not legacy_dir.exists()


def test_migrate_legacy_pgn_dirs_skips_nonexistent(tmp_path: Path) -> None:
    mgr = UnifiedRunManager(base_dir=tmp_path / "runs")
    result = mgr.migrate_legacy_pgn_dirs([str(tmp_path / "does_not_exist")])
    assert result == []
