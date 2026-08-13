"""Ground A — Self-Play Training and Snapshot Tournament Evaluation.

Tests policy improvement against internal past selves without external reference.
Computes Bradley-Terry (Bayeselo-style) internal Elo scale anchored at generation 0 = 1000.
Logs draw-rate and game-length collapse metrics.
"""

from __future__ import annotations

import os
import random
import signal
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import chess
import numpy as np
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from hypostases.plugins.domains.chess.chess_agent_adapter import ChessAgentAdapter
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
) -> tuple[float, float, int, str]:
    """Top-level worker running one Ground A self-play game between two policy snapshots."""
    import random

    random.seed(game_seed)
    np.random.seed(game_seed)

    domain = ChessDomain()
    agent_w = ChessAgentAdapter(
        domain=domain, beta_efe=beta_efe, temperature=temp_white, theta_meta=theta_white
    )
    agent_b = ChessAgentAdapter(
        domain=domain, beta_efe=beta_efe, temperature=temp_black, theta_meta=theta_black
    )

    board = domain.initial_state()
    num_moves = 0
    done = False
    info: dict[str, Any] = {}

    while not done and num_moves < max_moves:
        legal_moves = domain.valid_actions(board)
        if not legal_moves:
            break

        cur_agent = agent_w if board.turn == chess.WHITE else agent_b
        chosen_move = cur_agent.select_move(board, legal_moves)

        board, reward, done, info = domain.step(board, chosen_move)
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


@dataclass
class PolicySnapshot:
    """Snapshot of agent policy at a given generation."""

    generation: int
    policy_fn: Callable[[Any, list[Any]], Any]
    theta_meta: np.ndarray | None = None
    temperature: float = 0.5


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

            board, reward, done, info = self.domain.step(board, chosen_move)
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
        games_per_pair: int = 10,
        max_moves: int = 200,
        seed: int = 42,
        verbose: bool = False,
    ) -> TournamentResult:
        """Runs parallel round-robin tournament between all provided policy snapshots."""
        random.seed(seed)
        np.random.seed(seed)

        snapshot_ids = [s.generation for s in snapshots]
        result = TournamentResult(snapshot_ids=snapshot_ids)

        if verbose:
            console.print(
                f"\n[Ground A] Starting Parallel Self-Play Tournament ({len(snapshots)} snapshots across {self.max_workers} workers)..."
            )

        pairs = []
        for i, s1 in enumerate(snapshots):
            for j, s2 in enumerate(snapshots):
                if i < j:
                    pairs.append((s1, s2))

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
                t1, t2 = s1.temperature, s2.temperature

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
                        )
                        futures_map[fut] = (s1.generation, s2.generation, False)
                    game_counter += 1

            for future in as_completed(futures_map):
                g1, g2, is_s1_white = futures_map[future]
                w_score, b_score, moves, reason = future.result()

                stats = pair_matchup_stats[(g1, g2)]
                stats["lengths"].append(moves)

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

        for (g1, g2), stats in pair_matchup_stats.items():
            pair_key = (g1, g2)
            result.wins[pair_key] = stats["score_g1"]
            result.games[pair_key] = games_per_pair
            result.draw_counts[pair_key] = stats["draws"]
            result.game_lengths.extend(stats["lengths"])

            avg_len = float(np.mean(stats["lengths"])) if stats["lengths"] else 0.0
            if verbose:
                console.print(
                    f"  [bold green][Ground A Matchup][/bold green] Gen {g1:>2} vs Gen {g2:>2} | Gen {g1}: [bold white]{stats['g1_wins']}W[/bold white] - Gen {g2}: [bold white]{stats['g2_wins']}W[/bold white] - Draws: [yellow]{stats['draws']}D[/yellow] (Score: [bold cyan]{stats['score_g1']:.1f}/{games_per_pair}[/bold cyan], Avg Length: {avg_len:.1f} moves)"
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
