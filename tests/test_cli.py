"""Tests for HYPOSTASES CLI Commands Execution."""

import argparse

from hypostases.cli import infer, spec, sweep, sweep_memory, trace


def test_cli_trace_execution(capsys):
    args = argparse.Namespace(steps=3, seed=7, export_json=None)
    trace.main_trace(args)
    captured = capsys.readouterr()
    assert "Forward Simulation Trace" in captured.out
    assert "Final Agent States" in captured.out


def test_cli_infer_execution(capsys):
    args = argparse.Namespace(
        particles=20,
        seed=42,
        agent_name="TestAgent",
        steps=3,
        lag_window=2,
        hierarchical=False,
        use_rao_blackwell=True,
        output_format="table",
    )
    infer.main_infer(args)
    captured = capsys.readouterr()
    assert "Inverse Inference Report" in captured.out
    assert "Goal Posterior" in captured.out


def test_cli_sweep_execution(capsys):
    args = argparse.Namespace(steps=[5, 10], particles=20, seeds=[1, 2], output_format="table")
    sweep.main_sweep(args)
    captured = capsys.readouterr()
    assert "Phase A: Diagnostic Sweep" in captured.out


def test_cli_spec_merge_dry_run(capsys):
    args = argparse.Namespace(dry_run=True, output_dir=None)
    spec.main_spec_merge(args)
    captured = capsys.readouterr()
    assert "[DRY RUN] Would merge" in captured.out


def test_cli_infer_json_output(capsys):
    args = argparse.Namespace(
        particles=20,
        seed=42,
        agent_name="TestAgent",
        steps=3,
        lag_window=None,
        hierarchical=True,
        use_rao_blackwell=False,
        output_format="json",
    )
    infer.main_infer(args)
    captured = capsys.readouterr()
    assert '"agent_name": "TestAgent"' in captured.out


def test_cli_sweep_json_output(capsys):
    args = argparse.Namespace(steps=[5], particles=20, seeds=[1], output_format="json")
    sweep.main_sweep(args)
    captured = capsys.readouterr()
    assert '"n_steps": 5' in captured.out


def test_cli_sweep_memory_execution(capsys):
    args = argparse.Namespace(steps=5)
    sweep_memory.main_sweep_memory(args)
    captured = capsys.readouterr()
    assert "Memory Decay Stability Sweep" in captured.out
    assert "VARIANCE" in captured.out
    assert "PRECISION" in captured.out
