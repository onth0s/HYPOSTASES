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
    # Temperature slot is meta-learned (non-negative floor); β stays a valid probability.
    assert updated.theta_meta[8] >= 0.0
    assert 0.01 <= updated.beta_efe <= 0.99


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


def test_execute_run_interrupt_preserves_live_state() -> None:
    """An interrupt must preserve the LIVE agent state, not the last periodic snapshot.

    Regression: interrupted checkpoints previously fell back to the last periodic
    snapshot (gen-0 initial θ when snapshot_interval_k > gens elapsed), so a Ctrl-C
    during gen 5 saved `last_completed_gen: 0` and discarded all finalized progress.
    The trainer must append a live snapshot labeled with the last FULLY-finalized
    generation (gen_idx - 1).

    Interrupt fires at the generation-2 boundary (after gen 1 finalized θ, before any
    gen-2 game is submitted) — no in-flight games are terminated, so the CPython 3.10
    feeder-thread shutdown hazard in ``_terminate_pool_workers`` is avoided.
    """
    trainer = ChessSelfPlayTrainer(
        learning_rate=0.05,
        beta_efe=0.05,
        initial_temperature=0.8,
        max_workers=2,
    )

    def interrupt_flag() -> bool:
        # Trip once generation 1 has fully finalized (telemetry[1] written), i.e. at
        # the boundary BEFORE generation 2 — while gen-0 + finalized gen-1 θ live.
        return 1 in trainer.telemetry

    snapshots = trainer.execute_self_play_training_run(
        total_generations=4,
        snapshot_interval_k=10,
        games_per_generation=4,
        seed=11,
        max_moves_training=20,
        curriculum_probability=0.0,
        nnue_epochs=1,
        nnue_learning_rate=0.001,
        replay_capacity=100,
        search_depth=1,
        verbose=False,
        interrupt_flag=interrupt_flag,
    )

    # Live state preserved: [gen-0 initial, live-agent snapshot labeled gen 1].
    # The buggy path returned only [gen-0] here (the "interrupted at generation 0" panel).
    assert [s.generation for s in snapshots] == [0, 1]
    live = snapshots[-1]
    assert live.theta_meta.shape == (10,)
    assert np.all(np.isfinite(live.theta_meta))
    assert trainer._grad_batches == {}


def test_execute_run_interrupt_during_drain_preserves_live_state(capsys) -> None:
    """A mid-drain interrupt must drain, discard the partial gen, and print feedback.

    Regression: the mid-drain path force-killed workers mid-flight (CPython 3.10
    feeder-thread shutdown hang) and was never exercised in-process — the old
    interrupt test only covered the boundary. Now it cancels not-yet-started games,
    lets in-flight games finish, discards their partial results (a partial batch
    would bias θ), and appends the live snapshot labeled with the last
    FULLY-finalized generation. The drain prints unconditional STOPPING feedback so
    the console never goes silent while games wind down (a silent drain previously
    made users mash Ctrl-C into the forced-exit path, aborting the checkpoint save).
    """
    trainer = ChessSelfPlayTrainer(
        learning_rate=0.05,
        beta_efe=0.05,
        initial_temperature=0.8,
        max_workers=2,
    )

    def interrupt_flag() -> bool:
        # Trip DURING generation 2's drain: once its first result has been consumed
        # (_grad_batches gets a "2" entry) at least one more game is still pending,
        # so this is a genuine mid-generation interrupt with games in flight.
        return 2 in trainer._grad_batches

    snapshots = trainer.execute_self_play_training_run(
        total_generations=2,
        snapshot_interval_k=10,
        games_per_generation=6,
        seed=11,
        max_moves_training=20,
        curriculum_probability=0.0,
        nnue_epochs=1,
        nnue_learning_rate=0.001,
        replay_capacity=100,
        search_depth=1,
        verbose=False,
        interrupt_flag=interrupt_flag,
    )

    assert [s.generation for s in snapshots] == [0, 1]
    live = snapshots[-1]
    assert live.theta_meta.shape == (10,)
    assert np.all(np.isfinite(live.theta_meta))
    assert trainer._grad_batches == {}
    assert trainer.telemetry[1]["theta_norm_delta"] >= 0.0

    captured = capsys.readouterr()
    assert "STOPPING" in captured.out
    assert "STOPPED" in captured.out
    assert "Do NOT press Ctrl-C again" in captured.out
