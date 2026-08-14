"""Regression tests for the ChessSelfPlayTrainer in-process fast path and the
per-generation θ-barrier training run.

Covers the unpacking of the 5-tuple returned by _worker_run_training_game, the
baseline-corrected REINFORCE update on theta_meta, and the generation-barrier
ordering of execute_self_play_training_run (P2.2 sync fix).
"""

import numpy as np

from hypostases.plugins.domains.chess.chess_agent_adapter import ChessAgentAdapter
from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.plugins.domains.chess.chess_trainer import ChessSelfPlayTrainer


def test_train_generation_fast_path_unpacks_worker_tuple() -> None:
    """Regression: train_generation must not crash on the 5-tuple worker return."""
    domain = ChessDomain()
    agent = ChessAgentAdapter(
        domain=domain,
        beta_efe=0.2,
        temperature=0.8,
        theta_meta=np.ones(10, dtype=np.float64),
    )
    trainer = ChessSelfPlayTrainer(learning_rate=0.01, beta_efe=0.2, initial_temperature=0.8)

    updated = trainer.train_generation(agent, games_per_gen=3, max_moves=12, seed=7)

    assert updated.theta_meta.shape == (10,)
    assert np.all(np.isfinite(updated.theta_meta))
    assert np.all(updated.theta_meta[:8] >= 0.0)
    assert np.allclose(updated.theta_meta[8:], agent.theta_meta[8:])


def test_execute_run_generation_barrier_ordering() -> None:
    """The per-generation θ barrier must finalize each gen exactly once and leave no
    dangling gradient batches; snapshots record the finalized θ per generation."""
    trainer = ChessSelfPlayTrainer(
        learning_rate=0.05,
        beta_efe=0.05,
        initial_temperature=0.8,
        max_workers=2,
    )
    snapshots = trainer.execute_self_play_training_run(
        total_generations=2,
        snapshot_interval_k=1,
        games_per_generation=2,
        seed=11,
        max_moves_training=20,
        curriculum_probability=0.0,
        nnue_epochs=1,
        nnue_learning_rate=0.001,
        replay_capacity=100,
        search_depth=1,
        verbose=False,
    )

    assert [s.generation for s in snapshots] == [0, 1, 2]
    assert trainer._grad_batches == {}
    for gen in (1, 2):
        entry = trainer.telemetry[gen]
        assert entry["theta_norm_delta"] >= 0.0
        assert np.isfinite(entry["theta_norm_delta"])
        assert entry["terminations"]
        assert entry["avg_game_length"] > 0
    # Gen-1 background SGD must have completed (polled during Gen-2's drain).
    assert trainer.telemetry[1]["nnue_loss"] is not None
    # Each snapshot carries the θ finalized at its generation boundary.
    for snap in snapshots:
        assert np.all(np.isfinite(snap.theta_meta))
        assert np.all(snap.theta_meta[:8] >= 0.0)


def test_execute_run_interrupt_flag_short_circuits() -> None:
    """A tripped interrupt_flag must stop training before generation 1 begins.

    The run returns only the generation-0 snapshot, does not hang, and leaves no
    dangling gradient batches from a partial generation.
    """
    trainer = ChessSelfPlayTrainer(
        learning_rate=0.05,
        beta_efe=0.05,
        initial_temperature=0.8,
        max_workers=2,
    )
    snapshots = trainer.execute_self_play_training_run(
        total_generations=10,
        snapshot_interval_k=2,
        games_per_generation=2,
        seed=11,
        max_moves_training=20,
        curriculum_probability=0.0,
        nnue_epochs=1,
        nnue_learning_rate=0.001,
        replay_capacity=100,
        search_depth=1,
        verbose=False,
        interrupt_flag=lambda: True,
    )

    assert [s.generation for s in snapshots] == [0]
    assert trainer._grad_batches == {}
