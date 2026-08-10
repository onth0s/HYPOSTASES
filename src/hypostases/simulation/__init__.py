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
from hypostases.simulation.harness import (
    build_agent,
    generate_forced_withdraw_trace,
    generate_sample_trace,
    make_agent,
    make_test_agent,
    run_simulation_trace,
)

__all__ = [
    "build_agent",
    "evaluate_config",
    "generate_forced_withdraw_trace",
    "generate_sample_trace",
    "make_agent",
    "make_test_agent",
    "run_simulation_trace",
    "run_sweep_benchmark",
    "single_trial",
]
