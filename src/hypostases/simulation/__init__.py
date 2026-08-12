"""HYPOSTASES Simulation — Forward Trace Harnesses & Diagnostic Benchmarks.

Provides reusable agent instantiation, synthetic trace generation, and formal
3-condition sweep benchmarks independent of CLI argument parsers.
"""

from __future__ import annotations

from hypostases.simulation.benchmarks import (
    evaluate_config,
    run_sweep_benchmark,
    single_trial,
)
from hypostases.simulation.exporter import RunExporter
from hypostases.simulation.harness import (
    build_agent,
    generate_forced_withdraw_trace,
    generate_sample_trace,
    make_agent,
    make_test_agent,
    run_simulation_trace,
)
from hypostases.simulation.scenarios import create_scenario_agents

__all__ = [
    "RunExporter",
    "build_agent",
    "create_scenario_agents",
    "evaluate_config",
    "generate_forced_withdraw_trace",
    "generate_sample_trace",
    "make_agent",
    "make_test_agent",
    "run_simulation_trace",
    "run_sweep_benchmark",
    "single_trial",
]
