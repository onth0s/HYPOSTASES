"""Ground B — HYPOSTASES Agent vs. Stockfish External Benchmark.

Tests whether self-play policy improvement corresponds to genuine chess skill
by benchmarking policy snapshots against a fixed, calibrated Stockfish reference.
Includes deterministic MockStockfishEngine for CI/headless environments without Stockfish binary.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any

import chess
import chess.engine
import numpy as np
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.plugins.domains.chess.ground_a_self_play import PolicySnapshot

console = Console()


@dataclass
class StockfishBenchmarkResult:
    """Benchmark results for a snapshot against Stockfish reference."""

    generation: int
    score: float  # Score ratio in [0, 1] (wins + 0.5 draws) / N
    games_played: int
    wins: int
    losses: int
    draws: int
    estimated_elo: float
    reference_elo: float


class MockStockfishEngine:
    """Deterministic fallback engine simulating Stockfish for headless CI environments."""

    def __init__(self, target_elo: float = 1300.0) -> None:
        self.target_elo = target_elo

    def play(self, board: chess.Board, limit: chess.engine.Limit) -> chess.engine.PlayResult:
        """Selects a legal move deterministically based on board hash and piece values."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.engine.PlayResult(move=None, ponder=None, info={})

        # Simple deterministic heuristic choice: captures > checks > center control
        best_move = legal_moves[0]
        best_score = -9999.0

        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,
        }

        for move in legal_moves:
            score = 0.0
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured:
                    score += piece_values[captured.piece_type] * 10
            if board.gives_check(move):
                score += 5.0

            # Hash tiebreaker for variation
            score += (hash((move.uci(), board.fen())) % 100) / 100.0

            if score > best_score:
                best_score = score
                best_move = move

        return chess.engine.PlayResult(move=best_move, ponder=None, info={})

    def quit(self) -> None:
        pass


class GroundBStockfish:
    """Ground B External Stockfish Benchmark Harness with Persistent Parallel UCI Engine Pool."""

    def __init__(
        self,
        stockfish_path: str | None = None,
        reference_elo: float = 1300.0,
        time_control: float = 0.1,
        stockfish_threads: int = 2,
        max_workers: int = 4,
        chess_domain: ChessDomain | None = None,
    ) -> None:
        if stockfish_path is not None:
            resolved_path = (
                str(Path(stockfish_path).resolve()) if Path(stockfish_path).exists() else ""
            )
        else:
            local_stockfish = (Path(__file__).parent / "stockfish.exe").resolve()
            if local_stockfish.exists():
                resolved_path = str(local_stockfish)
            else:
                resolved_path = shutil.which("stockfish") or ""

        self.stockfish_path = resolved_path
        self.reference_elo = reference_elo
        self.time_control = time_control
        self.stockfish_threads = max(1, stockfish_threads)
        self.max_workers = max(1, max_workers)
        self.domain = chess_domain or ChessDomain()

    def get_engine(self) -> tuple[Any, bool]:
        """Spawns Stockfish UCI engine or fallback MockStockfishEngine.

        Returns (engine_instance, is_real_stockfish).
        """
        if self.stockfish_path and os.path.exists(self.stockfish_path):
            try:
                engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                uci_options: dict[str, Any] = {"Threads": self.stockfish_threads}

                # Stockfish 18 UCI_Elo range is [1320, 3190]
                if self.reference_elo >= 1320:
                    target_elo = int(np.clip(self.reference_elo, 1320, 3190))
                    uci_options.update({"UCI_LimitStrength": True, "UCI_Elo": target_elo})
                else:
                    # Map lower Elo (< 1320) to UCI Skill Level [0, 20]
                    skill_level = int(np.clip((self.reference_elo - 800) / 26.0, 0, 20))
                    uci_options.update({"Skill Level": skill_level})

                try:
                    engine.configure(uci_options)
                except Exception:
                    engine.configure({"Threads": self.stockfish_threads, "Skill Level": 0})

                return engine, True
            except Exception as err:
                console.print(
                    f"[bold red][Ground B Warning][/bold red] Stockfish launch error: {err}"
                )

        return MockStockfishEngine(target_elo=self.reference_elo), False

    def _create_engine_pool(self, count: int) -> tuple[list[Any], bool]:
        """Spawns a fixed pool of persistent engine instances."""
        engines = []
        is_real = False
        for _ in range(count):
            eng, real = self.get_engine()
            engines.append(eng)
            if real:
                is_real = True
        return engines, is_real

    @staticmethod
    def _close_engine_pool(engines: list[Any]) -> None:
        """Cleanly terminates all engine processes and transport pipes in the pool."""
        for eng in engines:
            if hasattr(eng, "close"):
                with contextlib.suppress(Exception):
                    eng.close()
            elif hasattr(eng, "quit"):
                with contextlib.suppress(Exception):
                    eng.quit()

    def _play_single_game_with_engine(
        self,
        engine: Any,
        snapshot: PolicySnapshot,
        agent_is_white: bool,
        max_moves: int,
    ) -> tuple[float, str]:
        """Executes a single game using a provided persistent engine instance."""
        limit = chess.engine.Limit(time=self.time_control)
        board = self.domain.initial_state()
        num_moves = 0

        while not board.is_game_over() and num_moves < max_moves:
            legal_moves = self.domain.valid_actions(board)
            if not legal_moves:
                break

            if (board.turn == chess.WHITE) == agent_is_white:
                chosen_move = snapshot.policy_fn(board, legal_moves)
            else:
                play_result = engine.play(board, limit)
                chosen_move = (
                    play_result.move if play_result.move in legal_moves else legal_moves[0]
                )

            board, reward, done, info = self.domain.step(board, chosen_move)
            num_moves += 1

        outcome = board.outcome()
        if outcome is not None:
            agent_won = (
                (outcome.winner == chess.WHITE)
                if agent_is_white
                else (outcome.winner == chess.BLACK)
            )
            if agent_won:
                return 1.0, "W"
            elif outcome.winner is None:
                return 0.5, "D"
            else:
                return 0.0, "L"
        else:
            return 0.5, "D"

    def evaluate_snapshot(
        self,
        snapshot: PolicySnapshot,
        games_n: int = 30,
        max_moves: int = 200,
        seed: int = 42,
        verbose: bool = False,
    ) -> StockfishBenchmarkResult:
        """Evaluates a policy snapshot against Stockfish over N games reusing a persistent engine pool."""
        pool_size = min(self.max_workers, games_n)
        engines, is_real = self._create_engine_pool(pool_size)
        engine_name = "Stockfish 18" if is_real else "MockStockfishEngine"

        gen_formatted = f"{snapshot.generation:02d}"
        if verbose:
            console.print(
                f"\n  [bold yellow][Ground B Parallel Benchmark][/bold yellow] Gen {gen_formatted} vs [bold cyan]{engine_name}[/bold cyan] ({games_n} games across {pool_size} persistent engine workers, {self.stockfish_threads} threads/engine)..."
            )

        wins = 0
        losses = 0
        draws = 0
        total_score = 0.0
        completed_count = 0
        lock = Lock()

        engine_queue: Queue[Any] = Queue()
        for eng in engines:
            engine_queue.put(eng)

        def worker_task(game_idx: int) -> tuple[float, str]:
            eng = engine_queue.get()
            try:
                return self._play_single_game_with_engine(
                    engine=eng,
                    snapshot=snapshot,
                    agent_is_white=(game_idx % 2 == 0),
                    max_moves=max_moves,
                )
            finally:
                engine_queue.put(eng)

        try:
            with (
                Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    console=console,
                    disable=not verbose,
                ) as progress,
                ThreadPoolExecutor(max_workers=pool_size) as executor,
            ):
                task_id = progress.add_task(
                    f"[yellow]Ground B Benchmark (Gen {gen_formatted} vs {engine_name})[/yellow]",
                    total=games_n,
                )
                futures = [executor.submit(worker_task, game_idx) for game_idx in range(games_n)]

                for future in as_completed(futures):
                    score, res_char = future.result()
                    with lock:
                        completed_count += 1
                        total_score += score
                        if res_char == "W":
                            wins += 1
                        elif res_char == "L":
                            losses += 1
                        else:
                            draws += 1
                        progress.update(task_id, advance=1)

                        if verbose and completed_count == games_n:
                            console.print(
                                f"    --> Completed {games_n} Games | Results: [bold green]{wins}W[/bold green]-[bold red]{losses}L[/bold red]-[bold yellow]{draws}D[/bold yellow]"
                            )
        finally:
            self._close_engine_pool(engines)

        score_ratio = total_score / float(games_n)
        estimated_elo = self.logistic_elo_fit(score_ratio, self.reference_elo)

        if verbose:
            console.print(
                f"    [bold yellow]Final Score:[/bold yellow] {wins}W-{losses}L-{draws}D ({score_ratio * 100:.1f}%), Est. Elo: [bold cyan]{estimated_elo:.1f}[/bold cyan]"
            )

        return StockfishBenchmarkResult(
            generation=snapshot.generation,
            score=score_ratio,
            games_played=games_n,
            wins=wins,
            losses=losses,
            draws=draws,
            estimated_elo=estimated_elo,
            reference_elo=self.reference_elo,
        )

    @staticmethod
    def logistic_elo_fit(score_ratio: float, reference_elo: float) -> float:
        """Computes estimated Elo rating from game score ratio via logistic fit.

        Formula: Elo_agent = Elo_ref + 400 * log10(S / (1 - S))
        Clamped score ratio in [0.01, 0.99] to prevent logarithmic divergence.
        """
        clamped_score = float(np.clip(score_ratio, 0.01, 0.99))
        delta_elo = 400.0 * np.log10(clamped_score / (1.0 - clamped_score))
        return float(reference_elo + delta_elo)
