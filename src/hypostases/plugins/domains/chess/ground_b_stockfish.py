"""Ground B — HYPOSTASES Agent vs. Stockfish External Benchmark.

Tests whether self-play policy improvement corresponds to genuine chess skill
by benchmarking policy snapshots against a fixed, calibrated Stockfish reference.
Includes deterministic MockStockfishEngine for CI/headless environments without Stockfish binary.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any

import chess
import chess.engine
import chess.pgn
import numpy as np
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from hypostases.plugins.domains.chess.chess_agent_adapter import SEARCH_DEPTH
from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.plugins.domains.chess.ground_a_self_play import PolicySnapshot

console = Console()


@dataclass
class StockfishBenchmarkResult:
    """Benchmark results for a snapshot against Stockfish reference."""

    generation: int
    score: float  # Score ratio in [0, 1] (wins + 0.5 draws) / effective games
    games_played: int
    wins: int
    losses: int
    draws: int
    estimated_elo: float
    reference_elo: float
    avg_game_length: float = 0.0
    capped: int = 0
    effective_games: int = 0
    avg_sf_eval_agent: float | None = None  # mean SF eval (agent perspective) over games
    last_sf_eval_agent: float | None = None  # mean of last SF eval (agent perspective) per game
    avg_material_gap_agent: float | None = None  # mean final material balance (agent perspective)


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
        stockfish_threads: int = 1,
        max_workers: int | None = None,
        chess_domain: ChessDomain | None = None,
        stockfish_multipv: int = 1,
        eval_temperature: float | None = None,
        search_depth: int = SEARCH_DEPTH,
    ) -> None:
        if max_workers is None:
            max_workers = os.cpu_count() or 8
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
        self.stockfish_multipv = max(1, stockfish_multipv)
        self.eval_temperature = eval_temperature
        self.search_depth = max(1, search_depth)
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

                # MultiPV randomization: >1 weakens Stockfish by forcing it to spread
                # its evaluation across top-N candidate moves (config-gated weaker tier).
                if self.stockfish_multipv > 1:
                    uci_options["MultiPV"] = self.stockfish_multipv

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
    ) -> dict[str, Any]:
        """Executes a single game using a provided persistent engine instance and returns PGN + continuous signals.

        Continuous signals (make Ground B discriminative even at a 0-win wall):
          avg_sf_eval_agent / last_sf_eval_agent: Stockfish eval in agent perspective
            (centipawns; higher = agent keeps the position closer).
          material_gap_agent: final material balance in agent perspective.
        """
        limit = chess.engine.Limit(time=self.time_control)
        board = self.domain.initial_state()
        num_moves = 0

        game_pgn = chess.pgn.Game()
        game_pgn.headers["Event"] = "HYPOSTASES Ground B Stockfish Benchmark"
        game_pgn.headers["White"] = (
            f"HYPOSTASES Gen {snapshot.generation:02d}" if agent_is_white else "Stockfish 18"
        )
        game_pgn.headers["Black"] = (
            "Stockfish 18" if agent_is_white else f"HYPOSTASES Gen {snapshot.generation:02d}"
        )
        node = game_pgn

        from hypostases.plugins.domains.chess.chess_agent_adapter import ChessAgentAdapter
        from hypostases.world_model.nnue_net import NNUENet

        agent = ChessAgentAdapter(
            domain=self.domain,
            beta_efe=0.2,
            temperature=snapshot.temperature
            if self.eval_temperature is None
            else self.eval_temperature,
            theta_meta=snapshot.theta_meta,
        )
        net = NNUENet() if snapshot.nnue_weights is not None else None
        if net and snapshot.nnue_weights:
            net.W_white = snapshot.nnue_weights["W_white"]
            net.W_black = snapshot.nnue_weights["W_black"]
            net.W_l1 = snapshot.nnue_weights["W_l1"]
            net.b_l1 = snapshot.nnue_weights["b_l1"]
            net.W_l2 = snapshot.nnue_weights["W_l2"]
            net.b_l2 = snapshot.nnue_weights["b_l2"]

        sf_evals_agent: list[float] = []

        while not board.is_game_over() and num_moves < max_moves:
            legal_moves = self.domain.valid_actions(board)
            if not legal_moves:
                break

            if (board.turn == chess.WHITE) == agent_is_white:
                chosen_move = agent.select_move(
                    board, legal_moves, depth=self.search_depth, nnue_net=net
                )
            else:
                play_result = engine.play(board, limit)
                chosen_move = (
                    play_result.move if play_result.move in legal_moves else legal_moves[0]
                )
                # Capture Stockfish's evaluation in agent perspective
                info = play_result.info
                if isinstance(info, dict):
                    score = info.get("score")
                    if score is not None:
                        with contextlib.suppress(AttributeError, ValueError):
                            sf_evals_agent.append(
                                score.pov(agent_is_white).score(mate_score=100000)
                            )

            node = node.add_variation(chosen_move)
            board, _reward, _done, _info = self.domain.step(board, chosen_move)
            num_moves += 1

        outcome = board.outcome()
        res_str = outcome.result() if outcome is not None else "*"
        game_pgn.headers["Result"] = res_str

        material_gap = self._material_balance(board, agent_is_white)

        if outcome is not None:
            agent_won = (
                (outcome.winner == chess.WHITE)
                if agent_is_white
                else (outcome.winner == chess.BLACK)
            )
            if agent_won:
                result_char = "W"
                score = 1.0
            elif outcome.winner is None:
                result_char = "D"
                score = 0.5
            else:
                result_char = "L"
                score = 0.0
        else:
            result_char = "C"
            score = 0.0

        return {
            "score": score,
            "res_char": result_char,
            "moves": num_moves,
            "pgn": game_pgn,
            "avg_sf_eval_agent": float(np.mean(sf_evals_agent)) if sf_evals_agent else None,
            "last_sf_eval_agent": float(sf_evals_agent[-1]) if sf_evals_agent else None,
            "material_gap_agent": float(material_gap),
        }

    @staticmethod
    def _material_balance(board: chess.Board, agent_is_white: bool) -> float:
        """Computes final material balance from the agent's perspective."""
        piece_vals = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.25,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
        }
        balance = 0.0
        for p_type, val in piece_vals.items():
            balance += len(board.pieces(p_type, chess.WHITE)) * val
            balance -= len(board.pieces(p_type, chess.BLACK)) * val
        return balance if agent_is_white else -balance

    def evaluate_snapshot(
        self,
        snapshot: PolicySnapshot,
        games_n: int = 30,
        max_moves: int = 200,
        seed: int = 42,
        verbose: bool = False,
        export_pgn_dir: str | Path | None = "exports/pgn/ground_b",
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
        capped = 0
        total_score = 0.0
        completed_count = 0
        game_lengths: list[int] = []
        sf_eval_means: list[float] = []
        sf_eval_lasts: list[float] = []
        material_gaps: list[float] = []
        all_pgns: list[chess.pgn.Game] = []
        lock = Lock()

        engine_queue: Queue[Any] = Queue()
        for eng in engines:
            engine_queue.put(eng)

        def worker_task(game_idx: int) -> dict[str, Any]:
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

        # Progress bar showing live score, games finished, and elapsed time per game
        progress_columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn(
                "[bold green]Score: {task.fields[wins]}W-{task.fields[losses]}L-{task.fields[draws]}D[/bold green]"
            ),
        ]

        try:
            with (
                Progress(
                    *progress_columns,
                    console=console,
                    disable=not verbose,
                ) as progress,
                ThreadPoolExecutor(max_workers=pool_size) as executor,
            ):
                task_id = progress.add_task(
                    "match",
                    total=games_n,
                    wins=0,
                    losses=0,
                    draws=0,
                )
                futures = [executor.submit(worker_task, game_idx) for game_idx in range(games_n)]

                for future in as_completed(futures):
                    game_result = future.result()
                    score = game_result["score"]
                    res_char = game_result["res_char"]
                    moves = game_result["moves"]
                    pgn_game = game_result["pgn"]
                    with lock:
                        completed_count += 1
                        game_lengths.append(moves)
                        all_pgns.append(pgn_game)
                        if game_result["avg_sf_eval_agent"] is not None:
                            sf_eval_means.append(game_result["avg_sf_eval_agent"])
                        if game_result["last_sf_eval_agent"] is not None:
                            sf_eval_lasts.append(game_result["last_sf_eval_agent"])
                        material_gaps.append(game_result["material_gap_agent"])
                        if res_char == "W":
                            wins += 1
                            total_score += score
                        elif res_char == "L":
                            losses += 1
                            total_score += score
                        elif res_char == "D":
                            draws += 1
                            total_score += score
                        else:
                            capped += 1
                        progress.update(task_id, advance=1, wins=wins, losses=losses, draws=draws)
                        console.print(
                            f"  [cyan][Game {completed_count}/{games_n} Completed][/cyan] Result: [bold]{res_char}[/bold] | Moves: {moves} | Score: {wins}W-{losses}L-{draws}D [bold magenta]({capped} capped)[/bold magenta]"
                        )
        except KeyboardInterrupt:
            console.print(
                "\n[bold red]Interrupted by User (Ctrl+C). Terminating Stockfish engine processes...[/bold red]"
            )
            self._close_engine_pool(engines)
            sys.exit(1)
        finally:
            self._close_engine_pool(engines)

        # Export all Ground B PGN games
        if export_pgn_dir and all_pgns:
            out_dir = Path(export_pgn_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            pgn_file_path = out_dir / f"ground_b_gen{snapshot.generation:02d}_vs_stockfish.pgn"
            with open(pgn_file_path, "w", encoding="utf-8") as f:
                for pgn_game in all_pgns:
                    print(pgn_game, file=f, end="\n\n")

        effective_games = games_n - capped
        score_ratio = total_score / float(effective_games) if effective_games > 0 else 0.0
        estimated_elo = self.logistic_elo_fit(score_ratio, self.reference_elo)
        avg_len = float(np.mean(game_lengths)) if game_lengths else 0.0

        if verbose:
            console.print(
                f"    [bold yellow]Final Score:[/bold yellow] {wins}W-{losses}L-{draws}D ({score_ratio * 100:.1f}%) | Capped: {capped} | Est. Elo: [bold cyan]{estimated_elo:.1f}[/bold cyan] (Avg Length: {avg_len:.1f} moves)"
            )
            if sf_eval_means:
                console.print(
                    f"    [bold cyan]Continuous:[/bold cyan] Avg SF eval (agent) [bold]{np.mean(sf_eval_means):+.0f}[/bold] | Last [bold]{np.mean(sf_eval_lasts):+.0f}[/bold] | Material gap [bold]{np.mean(material_gaps):+.1f}[/bold]"
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
            avg_game_length=avg_len,
            capped=capped,
            effective_games=effective_games,
            avg_sf_eval_agent=float(np.mean(sf_eval_means)) if sf_eval_means else None,
            last_sf_eval_agent=float(np.mean(sf_eval_lasts)) if sf_eval_lasts else None,
            avg_material_gap_agent=float(np.mean(material_gaps)) if material_gaps else None,
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
