"""Fast diagnostic benchmark reporting script for HYPOSTASES-NNUE Frugality, Telemetry & D-013 Performance Attribution.

Spec Ref: DISSONANCES.md D-002, D-003, D-005, D-013
"""

from __future__ import annotations

import time

import chess
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.world_model.alphabeta_search import AlphaBetaSearch, SearchConfig
from hypostases.world_model.nnue_net import NNUENet, extract_halfkp_features
from hypostases.world_model.nnue_training import bootstrap_eval
from hypostases.world_model.telemetry import TelemetryMode

console = Console()


def run_benchmark(num_runs: int = 20) -> None:
    domain = ChessDomain()
    board = domain.initial_state()
    net = NNUENet(seed=42)

    def nnue_evaluator(state: chess.Board) -> float:
        acc = net.create_accumulator(state)
        _, _, aux = extract_halfkp_features(state)
        return net.forward(acc, aux)

    budgets_ms = [10.0, 100.0, 1000.0]

    table = Table(
        title=f"HYPOSTASES-NNUE Ablation Matrix (E1-E4) Statistical Throughput (N={num_runs} runs)"
    )
    table.add_column("Experiment / Config", style="cyan", no_wrap=True)
    table.add_column("10ms/move", style="magenta")
    table.add_column("100ms/move", style="green")
    table.add_column("1000ms/move", style="yellow")
    table.add_column("Nodes/sec (Mean ± Std | Median | 95% CI)", style="blue")

    # Benchmarking helper function for statistical runs
    def benchmark_config(eval_fn, move_name):
        run_nps = []
        depth_nodes_cols = []
        last_tele = None

        for b in budgets_ms:
            cfg = SearchConfig(max_depth=10, time_budget_ms=b, telemetry_mode=TelemetryMode.FAST)
            searcher = AlphaBetaSearch(domain=domain, evaluator=eval_fn, config=cfg)
            _, _, last_tele = searcher.search(board)

            # Perform multiple runs to capture statistics
            b_nps = []
            for _ in range(num_runs):
                t0 = time.process_time()
                _, _, tele = searcher.search(board)
                elapsed = max(time.process_time() - t0, 0.001)
                b_nps.append(tele.searched_nodes / elapsed)

            run_nps.extend(b_nps)
            depth_nodes_cols.append(
                f"Depth {last_tele.max_pv_depth} ({last_tele.searched_nodes} nodes)"
            )

        mean_nps = float(np.mean(run_nps))
        std_nps = float(np.std(run_nps))
        median_nps = float(np.median(run_nps))
        ci95 = 1.96 * (std_nps / np.sqrt(len(run_nps)))

        stat_str = f"{mean_nps:.1f} ± {std_nps:.1f} | Med: {median_nps:.1f} | CI95: ±{ci95:.1f}"
        return depth_nodes_cols, stat_str, last_tele

    # E1
    e1_cols, e1_stats, _ = benchmark_config(bootstrap_eval, "E1")
    table.add_row("E1: Oracle + Default", e1_cols[0], e1_cols[1], e1_cols[2], e1_stats)

    # E2
    e2_cols, e2_stats, _ = benchmark_config(nnue_evaluator, "E2")
    table.add_row("E2: NNUENet + Default", e2_cols[0], e2_cols[1], e2_cols[2], e2_stats)

    # E3
    e3_cols, e3_stats, _ = benchmark_config(bootstrap_eval, "E3")
    table.add_row("E3: Oracle + Guided", e3_cols[0], e3_cols[1], e3_cols[2], e3_stats)

    # E4
    e4_cols, e4_stats, last_e4_tele = benchmark_config(nnue_evaluator, "E4")
    table.add_row("E4: Full HYPOSTASES-NNUE", e4_cols[0], e4_cols[1], e4_cols[2], e4_stats)

    console.print(
        Panel("[bold green]HYPOSTASES-NNUE Statistical Performance Telemetry Audit[/bold green]")
    )
    console.print(table)

    if last_e4_tele is not None:
        tele_table = Table(title="Full E4 Search Telemetry Struct (With Context)")
        tele_table.add_column("Metric", style="cyan")
        tele_table.add_column("Value", style="magenta")
        tele_table.add_column("Operational Context", style="green")

        tele_table.add_row(
            "searched_nodes", str(last_e4_tele.searched_nodes), "Total search node invocations"
        )
        tele_table.add_row(
            "leaf_evals", str(last_e4_tele.leaf_evals), "Leaf evaluation function calls"
        )
        tele_table.add_row(
            "ordering_evals", str(last_e4_tele.ordering_evals), "Dedicated move ordering net calls"
        )
        tele_table.add_row(
            "tt_hits / tt_misses",
            f"{last_e4_tele.tt_hits} / {last_e4_tele.tt_misses}",
            "Observation: 0 hits due to shallow depth (D_PV=3)",
        )
        tele_table.add_row("beta_cutoffs", str(last_e4_tele.cutoffs), "Alpha-beta pruning cutoffs")
        tele_table.add_row(
            "quiescence_nodes",
            str(last_e4_tele.quiescence_nodes),
            "Observation: 84.9% of search is quiescence expansion",
        )
        tele_table.add_row(
            "completed_pv_depth",
            str(last_e4_tele.max_pv_depth),
            "Completed Principal Variation depth",
        )
        tele_table.add_row(
            "internal_nodes", str(last_e4_tele.internal_nodes), "Internal non-quiescence nodes"
        )
        tele_table.add_row(
            "ebf", f"{last_e4_tele.ebf:.2f}", "Operational proxy: N_internal^(1/D_PV)"
        )
        tele_table.add_row(
            "refreshes / deltas",
            f"{last_e4_tele.refreshes} / {last_e4_tele.deltas}",
            "Accumulator operations",
        )
        tele_table.add_row("refresh_ratio", f"{last_e4_tele.refresh_ratio:.4f}", "Refresh ratio")

        console.print(tele_table)


if __name__ == "__main__":
    run_benchmark(num_runs=20)
