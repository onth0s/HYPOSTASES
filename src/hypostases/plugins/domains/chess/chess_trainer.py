"""HYPOSTASES Chess Self-Play Reinforcement/Meta-Learning Trainer.

Executes authentic self-play reinforcement learning and meta-parameter updates (θ_meta)
on HYPOSTASES agent state σ = (c, w, g, ρ_ext).
Saves trained agent policy checkpoints across generations for Ground A & Ground B benchmarking.
"""

from __future__ import annotations

import copy
import random
import signal
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from typing import Any

import chess
import numpy as np
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from hypostases.plugins.domains.chess.chess_agent_adapter import ChessAgentAdapter
from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.plugins.domains.chess.ground_a_self_play import PolicySnapshot

console = Console()


def _init_worker_process() -> None:
    """Worker process initializer ignoring SIGINT so only main process handles Ctrl+C cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _compute_material_balance(board: chess.Board, agent_color: chess.Color) -> float:
    """Computes material advantage balance for agent_color."""
    piece_vals = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.25,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
    }
    balance = 0.0
    for p_type, val in piece_vals.items():
        balance += len(board.pieces(p_type, agent_color)) * val
        balance -= len(board.pieces(p_type, not agent_color)) * val
    return balance


def _worker_run_training_game(
    theta_meta: np.ndarray,
    beta_efe: float,
    temperature: float,
    max_moves: int,
    seed_val: int,
    early_adjudication_material: float = 6.0,
) -> tuple[float, list[np.ndarray]]:
    """Isolated process worker function running 1 self-play training game on a separate CPU process."""
    random.seed(seed_val)
    np.random.seed(seed_val)

    domain = ChessDomain()
    agent = ChessAgentAdapter(
        domain=domain,
        beta_efe=beta_efe,
        temperature=temperature,
        theta_meta=theta_meta,
    )
    opponent = copy.deepcopy(agent)
    board = domain.initial_state()
    agent_color = board.turn
    num_moves = 0
    done = False
    trajectory_features = []

    while not done and num_moves < max_moves:
        legal_moves = domain.valid_actions(board)
        if not legal_moves:
            break

        if board.turn == agent_color:
            chosen_move = agent.select_move(board, legal_moves)
            feats = agent.extract_move_features(board, chosen_move)
            if len(theta_meta) >= 9:
                feats = np.append(feats, 0.5)
            if len(theta_meta) >= 10:
                u_tot, u_prag, u_epis = agent.evaluate_efe_utility(board, chosen_move)
                feats = np.append(feats, float(u_epis / 3.0))
            trajectory_features.append(feats)
        else:
            chosen_move = opponent.select_move(board, legal_moves)

        board, reward, done, info = domain.step(board, chosen_move)
        num_moves += 1

        # Early adjudication after move 15 if material lead >= early_adjudication_material
        if num_moves >= 15:
            mat_bal = _compute_material_balance(board, agent_color)
            if abs(mat_bal) >= early_adjudication_material:
                final_reward = 1.0 if mat_bal > 0 else -1.0
                return final_reward, trajectory_features

    outcome = board.outcome()
    final_reward = 0.0
    if outcome is not None:
        if outcome.winner == agent_color:
            final_reward = 1.0
        elif outcome.winner is not None:
            final_reward = -1.0

    return final_reward, trajectory_features


class ChessSelfPlayTrainer:
    """Trainer executing parallel multi-process self-play reinforcement learning on HYPOSTASES agents."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta_efe: float = 0.2,
        initial_temperature: float = 0.8,
        max_workers: int = 20,
        chess_domain: ChessDomain | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.beta_efe = beta_efe
        self.initial_temperature = initial_temperature
        self.max_workers = max(1, max_workers)
        self.domain = chess_domain or ChessDomain()

    def train_generation(
        self,
        agent: ChessAgentAdapter,
        games_per_gen: int = 15,
        max_moves: int = 60,
        seed: int = 42,
        progress: Progress | None = None,
        task_id: Any = None,
        executor: ProcessPoolExecutor | None = None,
    ) -> ChessAgentAdapter:
        """Trains HYPOSTASES agent for one generation via parallel self-play policy gradient updates."""
        updated_agent = copy.deepcopy(agent)

        if games_per_gen <= 5 or self.max_workers == 1:
            # Fast in-process execution for small batches / unit tests (avoids Windows process spawn overhead)
            for g_idx in range(games_per_gen):
                final_reward, trajectory_features = _worker_run_training_game(
                    updated_agent.theta_meta,
                    self.beta_efe,
                    updated_agent.temperature,
                    max_moves,
                    seed + g_idx,
                )
                if trajectory_features and final_reward != 0.0:
                    mean_feat = np.mean(trajectory_features, axis=0)
                    grad = self.learning_rate * final_reward * mean_feat
                    if len(updated_agent.theta_meta) >= 10:
                        grad[9] *= 5.0
                    new_theta = updated_agent.theta_meta + grad
                    new_theta[:9] = np.maximum(0.0, new_theta[:9])
                    updated_agent.theta_meta = new_theta
                if progress is not None and task_id is not None:
                    progress.update(task_id, advance=1)
        else:
            try:

                def _run_with_exec(exec_inst: ProcessPoolExecutor) -> None:
                    futures = [
                        exec_inst.submit(
                            _worker_run_training_game,
                            updated_agent.theta_meta,
                            self.beta_efe,
                            updated_agent.temperature,
                            max_moves,
                            seed + g_idx,
                        )
                        for g_idx in range(games_per_gen)
                    ]

                    for future in as_completed(futures):
                        final_reward, trajectory_features = future.result()

                        if trajectory_features and final_reward != 0.0:
                            mean_feat = np.mean(trajectory_features, axis=0)
                            grad = self.learning_rate * final_reward * mean_feat
                            if len(updated_agent.theta_meta) >= 10:
                                grad[9] *= 5.0
                            new_theta = updated_agent.theta_meta + grad
                            new_theta[:9] = np.maximum(0.0, new_theta[:9])
                            updated_agent.theta_meta = new_theta

                        if progress is not None and task_id is not None:
                            progress.update(task_id, advance=1)

                if executor is not None:
                    _run_with_exec(executor)
                else:
                    with ProcessPoolExecutor(
                        max_workers=min(self.max_workers, games_per_gen),
                        initializer=_init_worker_process,
                    ) as local_executor:
                        _run_with_exec(local_executor)
            except KeyboardInterrupt:
                raise

        return updated_agent

    def execute_self_play_training_run(
        self,
        total_generations: int = 20,
        snapshot_interval_k: int = 5,
        games_per_generation: int = 15,
        seed: int = 42,
        min_temperature: float = 0.20,
        max_moves_training: int = 35,
        early_adjudication_material: float = 6.0,
        initial_priors: Any = "random",
        verbose: bool = True,
    ) -> list[PolicySnapshot]:
        """Runs continuous asynchronous streaming multi-generation self-play training run and returns PolicySnapshots."""
        snapshots = []

        # Create initial Gen 0 agent
        agent = ChessAgentAdapter(
            domain=self.domain,
            beta_efe=self.beta_efe,
            temperature=self.initial_temperature,
            theta_meta=initial_priors,
        )

        def _format_agent_theta(cur_agent: ChessAgentAdapter) -> str:
            feat_vals = cur_agent.theta_meta[:8]
            return "[" + " ".join(f"{v:04.2f}" for v in feat_vals) + "]"

        if verbose:
            formatted_gen0 = _format_agent_theta(agent)
            console.print(
                f"\n[bold cyan][HYPOSTASES Trainer][/bold cyan] Initialized Gen 0 Agent ([bold yellow]theta_meta = {formatted_gen0}[/bold yellow] [magenta]temp={agent.temperature:.3f}[/magenta] [blue]beta={agent.beta_efe:.2f}[/blue])"
            )

        def make_policy_fn(cur_agent: ChessAgentAdapter) -> Any:
            def policy(board: Any, legal_moves: Any) -> Any:
                return cur_agent.select_move(board, legal_moves)

            return policy

        snapshots.append(
            PolicySnapshot(
                generation=0,
                policy_fn=make_policy_fn(copy.deepcopy(agent)),
                theta_meta=agent.theta_meta.copy(),
                temperature=agent.temperature,
            )
        )

        digits = len(str(total_generations))
        total_games = total_generations * games_per_generation
        max_capacity = min(self.max_workers * 2, games_per_generation * 2)

        pending_futures: dict[Any, int] = {}
        completed_per_gen: dict[int, int] = defaultdict(int)
        task_ids: dict[int, Any] = {}
        next_game_id = 0

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
            ) as pool_executor,
        ):
            try:
                # Helper to submit a single game to the worker pool
                def _submit_game(game_id: int) -> None:
                    gen_idx = (game_id // games_per_generation) + 1
                    if gen_idx not in task_ids and gen_idx <= total_generations:
                        formatted_theta = _format_agent_theta(agent)
                        task_ids[gen_idx] = progress.add_task(
                            f"[cyan]Gen {gen_idx:0{digits}d}/{total_generations}[/cyan] [yellow]theta={formatted_theta}[/yellow] [magenta]temp={agent.temperature:.3f}[/magenta] [blue]beta={agent.beta_efe:.2f}[/blue]",
                            total=games_per_generation,
                        )

                    fut = pool_executor.submit(
                        _worker_run_training_game,
                        agent.theta_meta,
                        self.beta_efe,
                        agent.temperature,
                        max_moves_training,
                        seed + game_id,
                        early_adjudication_material,
                    )
                    pending_futures[fut] = gen_idx

                # Fill initial streaming pool capacity
                while next_game_id < total_games and len(pending_futures) < max_capacity:
                    _submit_game(next_game_id)
                    next_game_id += 1

                # Continuous streaming loop: process FIRST_COMPLETED game and immediately backfill worker
                while pending_futures:
                    done_set, _ = wait(pending_futures.keys(), return_when=FIRST_COMPLETED)
                    for fut in done_set:
                        gen_idx = pending_futures.pop(fut)
                        final_reward, trajectory_features = fut.result()

                        if trajectory_features and final_reward != 0.0:
                            mean_feat = np.mean(trajectory_features, axis=0)
                            grad = self.learning_rate * final_reward * mean_feat
                            if len(agent.theta_meta) >= 10:
                                grad[9] *= 5.0
                            new_theta = agent.theta_meta + grad
                            new_theta[:9] = np.maximum(0.0, new_theta[:9])
                            agent.theta_meta = new_theta

                        if gen_idx in task_ids:
                            progress.update(task_ids[gen_idx], advance=1)

                        completed_per_gen[gen_idx] += 1
                        if (
                            completed_per_gen[gen_idx] == games_per_generation
                            and gen_idx % snapshot_interval_k == 0
                        ):
                            chk_theta = _format_agent_theta(agent)
                            console.print(
                                f"    --> [bold green][CHECKPOINT SAVED][/bold green] Gen {gen_idx:0{digits}d} Checkpoint Staged ([bold yellow]theta = {chk_theta}[/bold yellow] [magenta]temp={agent.temperature:.3f}[/magenta] [blue]beta={agent.beta_efe:.2f}[/blue])"
                            )
                            snapshots.append(
                                PolicySnapshot(
                                    generation=gen_idx,
                                    policy_fn=make_policy_fn(copy.deepcopy(agent)),
                                )
                            )

                        # Immediately backfill next game to keep workers 100% saturated
                        if next_game_id < total_games:
                            _submit_game(next_game_id)
                            next_game_id += 1

            except KeyboardInterrupt:
                raise

        return snapshots
