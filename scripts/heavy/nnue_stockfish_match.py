"""Heavy script to train NNUENet and execute a Stockfish 1400 Elo match tournament.

Spec Ref: Rule 006 (Data-driven YAML Configuration via schema/stockfish_match_config.yaml)
Rule 013 Ref: Color user-facing terminal output using `rich`.
Rule 014 Ref: Heavy script under scripts/heavy/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import chess
import yaml

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.plugins.domains.chess.ground_a_self_play import PolicySnapshot
from hypostases.plugins.domains.chess.ground_b_stockfish import GroundBStockfish
from hypostases.world_model.alphabeta_search import AlphaBetaSearch, SearchConfig
from hypostases.world_model.nnue_net import NNUENet, extract_halfkp_features
from hypostases.world_model.nnue_training import generate_and_audit_dataset, train_nnue

console = Console()
DEFAULT_CONFIG_PATH = Path("src/hypostases/plugins/domains/chess/stockfish_match_config.yaml")


def load_config(config_path: Path) -> dict:
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "dataset": {"num_positions": 5000, "seed": 42},
        "training": {"epochs": 5, "learning_rate": 0.001, "seed": 42},
        "match": {
            "target_elo": 1400.0,
            "games": 20,
            "time_control_sec": 0.02,
            "max_workers": 16,
            "stockfish_threads": 1,
        },
        "search": {"max_depth": 4, "time_budget_ms": 50.0, "tt_size_mb": 16, "quiescence_depth": 4},
    }


def run_training_and_match(cfg: dict) -> None:
    ds_cfg = cfg.get("dataset", {})
    tr_cfg = cfg.get("training", {})
    match_cfg = cfg.get("match", {})
    search_cfg = cfg.get("search", {})

    target_elo = float(match_cfg.get("target_elo", 1400.0))
    positions = int(ds_cfg.get("num_positions", 5000))
    epochs = int(tr_cfg.get("epochs", 5))
    lr = float(tr_cfg.get("learning_rate", 0.001))
    games = int(match_cfg.get("games", 20))
    time_ctrl = float(match_cfg.get("time_control_sec", 0.02))
    max_workers = int(match_cfg.get("max_workers", 16))
    sf_threads = int(match_cfg.get("stockfish_threads", 1))

    console.print(
        Panel(
            f"[bold cyan]HYPOSTASES-NNUE Training & Stockfish {int(target_elo)} Match Session[/bold cyan]"
        )
    )

    # Phase 1: Train NNUENet
    console.print(
        f"[yellow]Phase 1/2: Generating {positions} positions and training NNUENet for {epochs} epochs (lr={lr})...[/yellow]"
    )
    dataset = generate_and_audit_dataset(num_positions=positions, seed=ds_cfg.get("seed", 42))
    net = NNUENet(seed=tr_cfg.get("seed", 42))
    final_loss = train_nnue(net, dataset, epochs=epochs, lr=lr)
    console.print(
        f"[bold green][OK] Training Complete! Final MSE Loss: {final_loss:.6f}[/bold green]\n"
    )

    # Phase 2: Prepare Policy Snapshot for Search
    def nnue_evaluator(s: chess.Board) -> float:
        acc = net.create_accumulator(s)
        _, _, aux = extract_halfkp_features(s)
        return net.forward(acc, aux)

    s_config = SearchConfig(
        max_depth=int(search_cfg.get("max_depth", 4)),
        time_budget_ms=float(search_cfg.get("time_budget_ms", 50.0)),
        tt_size_mb=int(search_cfg.get("tt_size_mb", 16)),
        quiescence_depth=int(search_cfg.get("quiescence_depth", 4)),
    )
    searcher = AlphaBetaSearch(domain=ChessDomain(), evaluator=nnue_evaluator, config=s_config)

    def agent_policy(state: chess.Board, legal_moves: list[chess.Move]) -> chess.Move:
        move, _, _ = searcher.search(state)
        return move if move in legal_moves else legal_moves[0]

    snapshot = PolicySnapshot(generation=1, policy_fn=agent_policy)

    # Phase 3: Match against Stockfish
    console.print(
        f"[yellow]Phase 2/2: Executing {games}-game match vs Stockfish (Target Elo: {target_elo}) across {max_workers} CPU workers...[/yellow]"
    )
    harness = GroundBStockfish(
        reference_elo=target_elo,
        time_control=time_ctrl,
        max_workers=max_workers,
        stockfish_threads=sf_threads,
    )
    result = harness.evaluate_snapshot(snapshot=snapshot, games_n=games, verbose=True)

    # Render Summary Results Table
    table = Table(title=f"Match Results vs Stockfish {int(target_elo)} Elo")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Games Played", str(result.games_played))
    table.add_row(
        "Wins / Losses / Draws", f"{result.wins} W / {result.losses} L / {result.draws} D"
    )
    table.add_row("Win Rate Score", f"{result.score * 100.0:.1f}%")
    table.add_row("Estimated Agent Elo", f"{result.estimated_elo:.1f}")
    table.add_row("Stockfish Target Elo", f"{result.reference_elo:.1f}")

    console.print("\n")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train HYPOSTASES-NNUE and run Stockfish match tournament"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    run_training_and_match(cfg)


if __name__ == "__main__":
    main()
