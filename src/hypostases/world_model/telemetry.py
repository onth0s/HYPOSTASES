"""HYPOSTASES Search Telemetry and Microtiming Profiler.

Spec Ref: scratch/DISSONANCES.md D-002, D-003, D-004
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class TelemetryMode(Enum):
    OFF = auto()
    FAST = auto()
    FULL = auto()


@dataclass
class EvaluatorTiming:
    feature_extraction_ns: int = 0
    delta_update_ns: int = 0
    refresh_ns: int = 0
    dense_forward_ns: int = 0
    ordering_eval_ns: int = 0
    tt_lookup_ns: int = 0


@dataclass
class SearchTelemetry:
    mode: TelemetryMode = TelemetryMode.FAST
    searched_nodes: int = 0
    internal_nodes: int = 0
    leaf_evals: int = 0
    ordering_evals: int = 0
    cutoffs: int = 0
    tt_hits: int = 0
    tt_misses: int = 0
    quiescence_nodes: int = 0
    max_pv_depth: int = 0
    ebf: float = 0.0

    refreshes: int = 0
    deltas: int = 0
    refresh_ratio: float = 0.0

    timing: EvaluatorTiming = field(default_factory=EvaluatorTiming)

    def compute_derived_metrics(self) -> None:
        total_acc_ops = self.refreshes + self.deltas
        if total_acc_ops > 0:
            self.refresh_ratio = float(self.refreshes) / float(total_acc_ops)
        else:
            self.refresh_ratio = 0.0

        # Frozen Spec Ref: EBF = (N_internal_searched)^(1 / D_PV) where internal nodes exclude quiescence nodes and TT hits
        if self.max_pv_depth > 0 and self.internal_nodes > 0:
            self.ebf = float(self.internal_nodes) ** (1.0 / float(self.max_pv_depth))
        else:
            self.ebf = 0.0


class TelemetryRecorder:
    """Decoupled SearchProfiler/TelemetryRecorder for counters and microtiming stats."""

    def __init__(self, mode: TelemetryMode = TelemetryMode.FAST) -> None:
        self.telemetry = SearchTelemetry(mode=mode)

    def record_node(self, is_quiescence: bool = False) -> None:
        if self.telemetry.mode == TelemetryMode.OFF:
            return
        self.telemetry.searched_nodes += 1
        if is_quiescence:
            self.telemetry.quiescence_nodes += 1
        else:
            self.telemetry.internal_nodes += 1

    def record_leaf_eval(self) -> None:
        if self.telemetry.mode == TelemetryMode.OFF:
            return
        self.telemetry.leaf_evals += 1

    def record_ordering_eval(self) -> None:
        if self.telemetry.mode == TelemetryMode.OFF:
            return
        self.telemetry.ordering_evals += 1

    def record_cutoff(self) -> None:
        if self.telemetry.mode == TelemetryMode.OFF:
            return
        self.telemetry.cutoffs += 1

    def record_tt_lookup(self, hit: bool) -> None:
        if self.telemetry.mode == TelemetryMode.OFF:
            return
        if hit:
            self.telemetry.tt_hits += 1
        else:
            self.telemetry.tt_misses += 1

    def record_accumulator_op(self, is_refresh: bool) -> None:
        if self.telemetry.mode == TelemetryMode.OFF:
            return
        if is_refresh:
            self.telemetry.refreshes += 1
        else:
            self.telemetry.deltas += 1

    def measure(self, timing_attr: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if self.telemetry.mode != TelemetryMode.FULL:
            return fn(*args, **kwargs)

        t0 = time.perf_counter_ns()
        res = fn(*args, **kwargs)
        dt = time.perf_counter_ns() - t0

        curr = getattr(self.telemetry.timing, timing_attr, 0)
        setattr(self.telemetry.timing, timing_attr, curr + dt)
        return res

    def get_telemetry(self) -> SearchTelemetry:
        self.telemetry.compute_derived_metrics()
        return self.telemetry
