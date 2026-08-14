"""Ground A — Self-Play Training and Snapshot Tournament Evaluation.

Tests policy improvement against internal past selves without external reference.
Computes Bradley-Terry (Bayeselo-style) internal Elo scale anchored at the oldest
evaluated generation = 1000. Logs draw-rate and game-length collapse metrics.
The compared generations are configured via an explicit evaluate_generations list,
decoupled from the snapshot storage cadence.
"""

from __future__ import annotations

import os
import random
import signal
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import numpy as np
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from hypostases.plugins.domains.chess.chess_agent_adapter import (
    SEARCH_DEPTH,
    ChessAgentAdapter,
)
from hypostases.plugins.domains.chess.chess_domain import ChessDomain

console = Console()


def _init_worker_process() -> None:
    """Worker process initializer ignoring SIGINT so main process handles Ctrl+C cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _worker_run_ground_a_game(
    theta_white: np.ndarray,
    theta_black: np.ndarray,
    beta_efe: float,
    temp_white: float,
    temp_black: float,
    max_moves: int,
    game_seed: int,
    nnue_weights_white: dict[str, np.ndarray] | None = None,
    nnue_weights_black: dict[str, np.ndarray] | None = None,
    search_depth: int = SEARCH_DEPTH,
) -> tuple[float, float, int, str]:
    """Top-level worker running one Ground A self-play game between two policy snapshots."""
    import random

    from hypostases.plugins.domains.chess.nnue_net import NNUENet

    random.seed(game_seed)
    np.random.seed(game_seed)

    domain = ChessDomain()
    agent_w = ChessAgentAdapter(
        domain=domain, beta_efe=beta_efe, temperature=temp_white, theta_meta=theta_white
    )
    agent_b = ChessAgentAdapter(
        domain=domain, beta_efe=beta_efe, temperature=temp_black, theta_meta=theta_black
    )

    net_w = NNUENet() if nnue_weights_white is not None else None
    if net_w and nnue_weights_white:
        net_w.W_white = nnue_weights_white["W_white"]
        net_w.W_black = nnue_weights_white["W_black"]
        net_w.W_l1 = nnue_weights_white["W_l1"]
        net_w.b_l1 = nnue_weights_white["b_l1"]
        net_w.W_l2 = nnue_weights_white["W_l2"]
        net_w.b_l2 = nnue_weights_white["b_l2"]

    net_b = NNUENet() if nnue_weights_black is not None else None
    if net_b and nnue_weights_black:
        net_b.W_white = nnue_weights_black["W_white"]
        net_b.W_black = nnue_weights_black["W_black"]
        net_b.W_l1 = nnue_weights_black["W_l1"]
        net_b.b_l1 = nnue_weights_black["b_l1"]
        net_b.W_l2 = nnue_weights_black["W_l2"]
        net_b.b_l2 = nnue_weights_black["b_l2"]

    board = domain.initial_state()
    num_moves = 0
    done = False

    game_pgn = chess.pgn.Game()
    game_pgn.headers["Event"] = "HYPOSTASES Ground A Self-Play Tournament"
    game_pgn.headers["White"] = f"HYPOSTASES Gen {game_seed % 1000:02d}"
    game_pgn.headers["Black"] = f"HYPOSTASES Gen {game_seed % 1000:02d}"
    node = game_pgn

    while not done and num_moves < max_moves:
        legal_moves = domain.valid_actions(board)
        if not legal_moves:
            break

        if board.turn == chess.WHITE:
            chosen_move = agent_w.select_move(
                board, legal_moves, depth=search_depth, nnue_net=net_w
            )
        else:
            chosen_move = agent_b.select_move(
                board, legal_moves, depth=search_depth, nnue_net=net_b
            )

        node = node.add_variation(chosen_move)
        board, _reward, _done, _info = domain.step(board, chosen_move)
        num_moves += 1

    outcome = board.outcome()
    res_str = outcome.result() if outcome is not None else "*"
    game_pgn.headers["Result"] = res_str

    pgn_text = str(game_pgn)

    if outcome is not None:
        if outcome.winner is not None:
            if outcome.winner == chess.WHITE:
                return 1.0, 0.0, num_moves, outcome.termination.name, pgn_text
            else:
                return 0.0, 1.0, num_moves, outcome.termination.name, pgn_text
        return 0.5, 0.5, num_moves, outcome.termination.name, pgn_text

    return 0.5, 0.5, num_moves, "max_moves_exceeded", pgn_text


def _select_tournament_pairs(
    snapshots: list[PolicySnapshot],
    evaluate_generations: list[int] | None = None,
    eval_last_n_snapshots: int = 0,
) -> list[tuple[PolicySnapshot, PolicySnapshot]]:
    """Selects ordered matchup pairs for the internal-Elo tournament.

    Decouples which generations are compared from the snapshot storage cadence:
    - evaluate_generations (preferred): explicit list of generations to pit against
      each other in a mini round-robin (all i<j pairs), at any spacing. Entries not
      present among the stored snapshots are ignored.
    - eval_last_n_snapshots > 0 (fallback alias): the last N stored snapshots.
    - Otherwise (0 / None): full round-robin over all stored snapshots.
    """
    by_gen = {s.generation: s for s in snapshots}

    if evaluate_generations is not None and len(evaluate_generations) > 0:
        target_gens = [g for g in evaluate_generations if g in by_gen]
        target_snaps = [by_gen[g] for g in sorted(target_gens)]
    elif eval_last_n_snapshots > 0:
        target_snaps = snapshots[-eval_last_n_snapshots:]
    else:
        target_snaps = snapshots

    pairs: list[tuple[PolicySnapshot, PolicySnapshot]] = []
    for i, s1 in enumerate(target_snaps):
        for j, s2 in enumerate(target_snaps):
            if i < j and s1.generation != s2.generation:
                pairs.append((s1, s2))
    return pairs


@dataclass
class PolicySnapshot:
    """Snapshot of agent policy at a given generation."""

    generation: int
    policy_fn: Callable[[Any, list[Any]], Any]
    theta_meta: np.ndarray | None = None
    temperature: float = 0.5
    nnue_weights: dict[str, np.ndarray] | None = None


@dataclass
class TournamentResult:
    """Outcome matrix of round-robin tournament between snapshots."""

    snapshot_ids: list[int]
    wins: dict[tuple[int, int], float] = field(
        default_factory=dict
    )  # (p1, p2) -> score (wins + 0.5 draws)
    games: dict[tuple[int, int], int] = field(default_factory=dict)  # (p1, p2) -> total games
    draw_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    game_lengths: list[int] = field(default_factory=list)
    termination_counts: dict[str, int] = field(default_factory=dict)


class GroundASelfPlay:
    """Ground A Self-Play harness and internal Elo tournament solver with Parallel Workers."""

    def __init__(
        self, chess_domain: ChessDomain | None = None, max_workers: int | None = None
    ) -> None:
        self.domain = chess_domain or ChessDomain()
        self.max_workers = max_workers or (os.cpu_count() or 20)

    def run_self_play_game(
        self,
        policy_white: Callable[[Any, list[Any]], Any],
        policy_black: Callable[[Any, list[Any]], Any],
        max_moves: int = 200,
    ) -> tuple[float, float, int, str]:
        """Plays one game between policy_white and policy_black."""
        board = self.domain.initial_state()
        num_moves = 0
        done = False
        info = {}

        while not done and num_moves < max_moves:
            legal_moves = self.domain.valid_actions(board)
            if not legal_moves:
                break

            current_policy = (
                policy_white if board.turn == self.domain.initial_state().turn else policy_black
            )
            chosen_move = current_policy(board, legal_moves)

            board, _reward, _done, _info = self.domain.step(board, chosen_move)
            num_moves += 1

        if not done:
            return 0.5, 0.5, num_moves, "max_moves_exceeded"

        winner = info.get("winner")
        if winner is True:
            return 1.0, 0.0, num_moves, info.get("done_reason", "checkmate")
        elif winner is False:
            return 0.0, 1.0, num_moves, info.get("done_reason", "checkmate")
        else:
            return 0.5, 0.5, num_moves, info.get("done_reason", "draw")

    def run_snapshot_tournament(
        self,
        snapshots: list[PolicySnapshot],
        games_per_pair: int = 8,
        max_moves: int = 120,
        seed: int = 42,
        eval_temperature: float | None = None,
        verbose: bool = True,
        search_depth: int = SEARCH_DEPTH,
        evaluate_generations: list[int] | None = None,
        eval_last_n_snapshots: int = 0,
        export_pgn_dir: str | Path | None = "exports/pgn/ground_a",
        timestamp_runs: bool = True,
    ) -> TournamentResult:
        """Runs a parallel mini round-robin tournament between the evaluated generations.

        The compared generations are decoupled from the snapshot storage cadence via
        evaluate_generations (explicit list) or eval_last_n_snapshots (last N fallback).

        eval_temperature holds the softmax temperature fixed across all evaluated
        snapshots so internal-Elo differences measure policy quality rather than
        exploration-temperature decay.
        """
        random.seed(seed)
        np.random.seed(seed)

        pgn_out_dir: Path | None = None
        if export_pgn_dir:
            pgn_out_dir = Path(export_pgn_dir)
            if timestamp_runs:
                from datetime import datetime

                ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                pgn_out_dir = pgn_out_dir / f"run_{ts}"
            pgn_out_dir.mkdir(parents=True, exist_ok=True)

        snapshot_ids = [s.generation for s in snapshots]
        result = TournamentResult(snapshot_ids=snapshot_ids)

        pairs = _select_tournament_pairs(snapshots, evaluate_generations, eval_last_n_snapshots)
        participating = {s.generation for pair in pairs for s in pair}

        if verbose:
            console.print(
                f"\n[Ground A] Starting Parallel Self-Play Tournament ({len(participating)} snapshots, {len(pairs)} matchups across {self.max_workers} workers)..."
            )

        total_games = len(pairs) * games_per_pair

        # Default theta_meta if not provided
        default_theta = np.array([1.0, 0.3, 0.5, 0.8, 0.2, 0.4, 0.5, 0.3], dtype=np.float32)

        futures_map = {}
        pair_matchup_stats = {}

        for s1, s2 in pairs:
            pair_matchup_stats[(s1.generation, s2.generation)] = {
                "score_g1": 0.0,
                "g1_wins": 0,
                "g2_wins": 0,
                "draws": 0,
                "lengths": [],
            }

        with (
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                console=console,
                disable=not verbose,
            ) as progress,
            ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=_init_worker_process,
            ) as executor,
        ):
            task_id = progress.add_task(
                f"[green]Ground A Tournament ({len(pairs)} matchups)[/green]",
                total=total_games,
            )

            game_counter = 0
            for s1, s2 in pairs:
                th1 = s1.theta_meta if s1.theta_meta is not None else default_theta
                th2 = s2.theta_meta if s2.theta_meta is not None else default_theta
                t1 = s1.temperature if eval_temperature is None else eval_temperature
                t2 = s2.temperature if eval_temperature is None else eval_temperature

                for game_idx in range(games_per_pair):
                    g_seed = seed + game_counter * 101
                    if game_idx % 2 == 0:
                        fut = executor.submit(
                            _worker_run_ground_a_game,
                            th1,
                            th2,
                            0.2,
                            t1,
                            t2,
                            max_moves,
                            g_seed,
                            s1.nnue_weights,
                            s2.nnue_weights,
                            search_depth,
                        )
                        futures_map[fut] = (s1.generation, s2.generation, True)
                    else:
                        fut = executor.submit(
                            _worker_run_ground_a_game,
                            th2,
                            th1,
                            0.2,
                            t2,
                            t1,
                            max_moves,
                            g_seed,
                            s2.nnue_weights,
                            s1.nnue_weights,
                            search_depth,
                        )
                        futures_map[fut] = (s1.generation, s2.generation, False)
                    game_counter += 1

            for future in as_completed(futures_map):
                g1, g2, is_s1_white = futures_map[future]
                w_score, b_score, moves, reason, pgn_text = future.result()

                stats = pair_matchup_stats[(g1, g2)]
                stats["lengths"].append(moves)
                result.termination_counts[reason] = result.termination_counts.get(reason, 0) + 1

                if pgn_out_dir:
                    pgn_file = pgn_out_dir / f"ground_a_gen{g1:02d}_vs_gen{g2:02d}.pgn"
                    with open(pgn_file, "a", encoding="utf-8") as f:
                        f.write(pgn_text + "\n\n")

                if is_s1_white:
                    stats["score_g1"] += w_score
                    if w_score == 1.0:
                        stats["g1_wins"] += 1
                    elif b_score == 1.0:
                        stats["g2_wins"] += 1
                    else:
                        stats["draws"] += 1
                else:
                    stats["score_g1"] += b_score
                    if b_score == 1.0:
                        stats["g1_wins"] += 1
                    elif w_score == 1.0:
                        stats["g2_wins"] += 1
                    else:
                        stats["draws"] += 1

                progress.update(task_id, advance=1)
                _p1_gen, _p2_gen = (g1, g2) if is_s1_white else (g2, g1)
                agent_res = (
                    "W"
                    if (w_score == 1.0 if is_s1_white else b_score == 1.0)
                    else ("L" if (b_score == 1.0 if is_s1_white else w_score == 1.0) else "D")
                )
                res_style = (
                    "bold green"
                    if agent_res == "W"
                    else ("bold red" if agent_res == "L" else "yellow")
                )
                if verbose:
                    console.print(
                        f"  [cyan][Ground A Game][/cyan] Gen {g1:02d} vs Gen {g2:02d} | Result: [{res_style}]{agent_res}[/{res_style}] | Moves: {moves:02d} | Reason: {reason}"
                    )

        digits = len(str(max(snapshot_ids))) if snapshot_ids else 2
        for (g1, g2), stats in pair_matchup_stats.items():
            pair_key = (g1, g2)
            result.wins[pair_key] = stats["score_g1"]
            result.games[pair_key] = games_per_pair
            result.draw_counts[pair_key] = stats["draws"]
            result.game_lengths.extend(stats["lengths"])

            avg_len = float(np.mean(stats["lengths"])) if stats["lengths"] else 0.0
            if verbose:
                console.print(
                    f"  [bold green][Ground A Matchup][/bold green] Gen {g1:0{digits}d} vs Gen {g2:0{digits}d} | Record: [bold white]{stats['g1_wins']}W - {stats['g2_wins']}L - {stats['draws']}D[/bold white] (Score: [bold cyan]{stats['score_g1']:.1f}/{games_per_pair}[/bold cyan], Avg Length: {avg_len:.1f} moves)"
                )

        return result

    @staticmethod
    def compute_internal_elo(
        result: TournamentResult,
        base_elo: float = 1000.0,
        max_iter: int = 100,
        tol: float = 1e-5,
    ) -> dict[int, float]:
        """Computes maximum likelihood Bradley-Terry Elo ratings for snapshot tournament.

        Anchored such that generation 0 (or first snapshot) = base_elo.
        """
        snapshots = result.snapshot_ids
        if not snapshots:
            return {}

        n = len(snapshots)
        idx_map = {gen: i for i, gen in enumerate(snapshots)}

        # Build pairwise match matrix w_mat[i, j] = score of i against j
        w_mat = np.zeros((n, n))
        n_mat = np.zeros((n, n))

        for (g1, g2), score in result.wins.items():
            i, j = idx_map[g1], idx_map[g2]
            tot = result.games[(g1, g2)]
            w_mat[i, j] += score
            w_mat[j, i] += tot - score
            n_mat[i, j] += tot
            n_mat[j, i] += tot

        # MM Iteration for Bradley-Terry
        p = np.ones(n)
        for _ in range(max_iter):
            p_old = p.copy()
            for i in range(n):
                wins_i = np.sum(w_mat[i, :])
                denom = 0.0
                for j in range(n):
                    if i != j and n_mat[i, j] > 0:
                        denom += n_mat[i, j] / (p_old[i] + p_old[j])
                if denom > 0:
                    p[i] = wins_i / denom

            # Normalize to avoid numerical drift
            p = p / np.mean(p)
            if np.max(np.abs(p - p_old)) < tol:
                break

        # Convert Bradley-Terry strength p to Elo scale: Elo = 400 * log10(p)
        elo_raw = 400.0 * np.log10(np.maximum(p, 1e-10))

        # Anchor first snapshot to base_elo
        anchor_offset = base_elo - elo_raw[0]
        elo_ratings = {snapshots[i]: float(elo_raw[i] + anchor_offset) for i in range(n)}

        return elo_ratings
