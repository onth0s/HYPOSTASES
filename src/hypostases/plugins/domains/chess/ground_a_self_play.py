"""Ground A — Self-Play Training and Snapshot Tournament Evaluation.

Tests policy improvement against internal past selves without external reference.
Computes Bradley-Terry (Bayeselo-style) internal Elo scale anchored at generation 0 = 1000.
Logs draw-rate and game-length collapse metrics.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import numpy as np
from rich.console import Console

from hypostases.plugins.domains.chess.chess_domain import ChessDomain

console = Console()


@dataclass
class PolicySnapshot:
    """Snapshot of agent policy at a given generation."""

    generation: int
    policy_fn: Callable[[Any, list[Any]], Any]


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

    def __init__(self, chess_domain: ChessDomain | None = None, max_workers: int = 10) -> None:
        self.domain = chess_domain or ChessDomain()
        self.max_workers = max(1, max_workers)

    def run_self_play_game(
        self,
        policy_white: Callable[[Any, list[Any]], Any],
        policy_black: Callable[[Any, list[Any]], Any],
        max_moves: int = 200,
    ) -> tuple[float, float, int, str]:
        """Plays one game between policy_white and policy_black.

        Returns: (score_white, score_black, num_moves, termination_reason)
        """
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
            # Max moves reached -> draw
            return 0.5, 0.5, num_moves, "max_moves_exceeded"

        winner = info.get("winner")
        if winner is True:  # White won
            return 1.0, 0.0, num_moves, info.get("done_reason", "checkmate")
        elif winner is False:  # Black won
            return 0.0, 1.0, num_moves, info.get("done_reason", "checkmate")
        else:  # Draw
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
            print(
                f"\n[Ground A] Starting Parallel Self-Play Tournament ({len(snapshots)} snapshots across {self.max_workers} workers)..."
            )

        pairs = []
        for i, s1 in enumerate(snapshots):
            for j, s2 in enumerate(snapshots):
                if i < j:
                    pairs.append((s1, s2))

        lock = Lock()

        def _run_pair_games(
            s1: PolicySnapshot, s2: PolicySnapshot
        ) -> tuple[int, int, float, int, int, int, list[int]]:
            g1, g2 = s1.generation, s2.generation
            score_g1 = 0.0
            g1_wins = 0
            g2_wins = 0
            draws = 0
            lengths = []

            for game_idx in range(games_per_pair):
                if game_idx % 2 == 0:
                    w_score, b_score, moves, reason = self.run_self_play_game(
                        s1.policy_fn, s2.policy_fn, max_moves
                    )
                    score_g1 += w_score
                    if w_score == 1.0:
                        g1_wins += 1
                    elif b_score == 1.0:
                        g2_wins += 1
                    else:
                        draws += 1
                else:
                    w_score, b_score, moves, reason = self.run_self_play_game(
                        s2.policy_fn, s1.policy_fn, max_moves
                    )
                    score_g1 += b_score
                    if b_score == 1.0:
                        g1_wins += 1
                    elif w_score == 1.0:
                        g2_wins += 1
                    else:
                        draws += 1
                lengths.append(moves)

            return g1, g2, score_g1, g1_wins, g2_wins, draws, lengths

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_run_pair_games, s1, s2) for s1, s2 in pairs]

            for future in as_completed(futures):
                g1, g2, score_g1, g1_wins, g2_wins, draws, lengths = future.result()
                pair_key = (g1, g2)

                with lock:
                    result.wins[pair_key] = score_g1
                    result.games[pair_key] = games_per_pair
                    result.draw_counts[pair_key] = draws
                    result.game_lengths.extend(lengths)

                    avg_len = float(np.mean(lengths)) if lengths else 0.0
                    if verbose:
                        console.print(
                            f"  [bold green][Ground A Matchup][/bold green] Gen {g1:>2} vs Gen {g2:>2} | Gen {g1}: [bold white]{g1_wins}W[/bold white] - Gen {g2}: [bold white]{g2_wins}W[/bold white] - Draws: [yellow]{draws}D[/yellow] (Score: [bold cyan]{score_g1:.1f}/{games_per_pair}[/bold cyan], Avg Length: {avg_len:.1f} moves)"
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
