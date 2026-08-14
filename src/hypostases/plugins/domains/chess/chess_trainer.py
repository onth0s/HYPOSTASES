"""HYPOSTASES Chess Self-Play Reinforcement/Meta-Learning Trainer.

Executes authentic self-play reinforcement learning and meta-parameter updates (θ_meta)
on HYPOSTASES agent state σ = (c, w, g, ρ_ext).
Saves trained agent policy checkpoints across generations for Ground A & Ground B benchmarking.

Training-signal design (see AGENTS.md 003/006/012 and chess_experiment_config.yaml):
  * Games run to real outcomes: long horizon (max_moves_training), NNUE-value-based
    resignation, and a material-adjudication fallback — so value labels encode
    winning/conversion rather than raw material counting.
  * Discounted-outcome value labels: γ^(remaining plies) * terminal reward, STM-normalized.
  * Endgame curriculum: a configurable fraction of training games start from
    schema/chess/endgame_curriculum.yaml mate-able presets so mate conversion appears
    in the training distribution.
  * Feature-normalized baseline-corrected REINFORCE over the full K=10 meta-parameter
    vector (8 tactical features + temperature + beta logit): β staticity is emergent
    from zero reward-feature covariance, never a manual pin.
"""

from __future__ import annotations

import contextlib
import copy
import multiprocessing
import random
import signal
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any

import chess
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from hypostases.plugins.domains.chess.chess_agent_adapter import (
    SEARCH_DEPTH,
    ChessAgentAdapter,
)
from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.plugins.domains.chess.ground_a_self_play import PolicySnapshot

console = Console()

REPLAY_CAPACITY = 3000

# Heartbeat interval for the streaming-loop status line: when no game completes
# within this window (all workers grinding long games), print live pool/gen state
# so the terminal does not look frozen while CPU stays at ~100%.
HEARTBEAT_INTERVAL = 5.0

# Heartbeat throttling: only print a status line when the longest in-flight game has
# exceeded HEARTBEAT_MIN_GAME_S AND the previous beat was at least HEARTBEAT_PRINT_GAP_S
# ago — keeps long-tail games visible without flooding the terminal every refresh.
HEARTBEAT_MIN_GAME_S = 60.0
HEARTBEAT_PRINT_GAP_S = 20.0

# Max-magnitude scale per tactical feature (see ChessAgentAdapter.extract_move_features):
#   feat0 capture_val (0..9), feat1 center (0..1), feat2 check (0..1), feat3 mate (0..10),
#   feat4 mobility (~0..0.95), feat5 defended (-1..1), feat6 capture_delta (-10..10),
#   feat7 king_attack (0..1)
FEATURE_SCALES = np.array([9.0, 1.0, 1.0, 10.0, 0.95, 1.0, 10.0, 1.0], dtype=np.float32)

# Meta-parameter learning spans the full K=10 vector: indices 8-9 (temperature,
# beta logit) carry default scale 1.0. The beta logit gradient receives the
# historical x5 boost (commit 25f2754) because the epistemic feature is ~0.28 in
# magnitude, and the resulting logit value is clamped to +/-BETA_LOGIT_MAX for
# numeric safety (a warning fires whenever the clamp actually engages).
BETA_LOGIT_GRADIENT_BOOST = 5.0
BETA_LOGIT_MAX = 10.0


def _init_worker_process() -> None:
    """Worker process initializer ignoring SIGINT so only main process handles Ctrl+C cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _terminate_pool_workers(executor: ProcessPoolExecutor) -> None:
    """Force-kills all pool worker processes so Ctrl+C or crashes leave no orphans.

    Workers are spawned with SIGINT ignored and the runner's SIGINT handler exits via
    os._exit(130), which would otherwise orphan them. TerminateProcess kills them
    unconditionally before the executor is torn down without waiting for running games.
    """
    with contextlib.suppress(Exception):
        for p in multiprocessing.active_children():
            p.terminate()
    with contextlib.suppress(Exception):
        executor.shutdown(wait=False, cancel_futures=True)


def _stm_normalized_label(board: chess.Board, white_perspective_reward: float) -> float:
    """Converts a white-perspective outcome reward into a side-to-move-perspective label.

    NNUENet.forward evaluates from the perspective of the side to move, so value labels
    must follow the same convention: flip the sign for black-to-move positions.
    """
    if board.turn == chess.WHITE:
        return white_perspective_reward
    return -white_perspective_reward


def _build_discounted_labels(
    game_positions: list[chess.Board],
    terminal_board: chess.Board,
    final_reward: float,
    value_gamma: float,
) -> list[tuple[chess.Board, float]]:
    """Builds discounted-outcome value labels γ^(remaining plies) * final_reward.

    The terminal board is appended with remaining=0 so it receives the full reward,
    teaching the value net terminal/mate states. Labels are STM-normalized.
    """
    positions = [*game_positions, terminal_board]
    n = len(positions)
    labels: list[tuple[chess.Board, float]] = []
    for i, b in enumerate(positions):
        remaining = n - 1 - i
        discounted = final_reward * float(value_gamma**remaining)
        labels.append((b, _stm_normalized_label(b, discounted)))
    return labels


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


def _has_non_king_material(board: chess.Board, color: chess.Color) -> bool:
    """True if `color` still holds at least one piece other than its king."""
    return any(
        board.pieces(p_type, color)
        for p_type in (
            chess.PAWN,
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
        )
    )


def _should_material_adjudicate(
    board: chess.Board,
    agent_color: chess.Color,
    threshold: float,
    adjudicate_bare_king_requires_mate: bool,
) -> bool:
    """Predicate for the material-adjudication fallback (Rule 003 branch audit).

    Adjudicates only when |material edge| >= threshold AND the position is not a
    bare-king endgame in which mate is still reachable (P1.2): a bare-king loser
    with a materially-winning opponent must play to mate so the value net learns
    conversion rather than a truncated "material edge = win" label.
    """
    mat_bal = _compute_material_balance(board, agent_color)
    if abs(mat_bal) < threshold:
        return False
    if not adjudicate_bare_king_requires_mate:
        return True
    losing_side = not agent_color if mat_bal > 0 else agent_color
    return _has_non_king_material(board, losing_side)


DEFAULT_CURRICULUM_PATH = Path(__file__).parent / "schemas" / "endgame_curriculum.yaml"


def load_endgame_curriculum(
    path: str | Path | None = None,
) -> list[str]:
    """Loads mate-able endgame preset FENs from a YAML curriculum file (Rule 006).

    Filters to legal positions with White to move and not in check.
    """
    if path is None:
        curriculum_path = DEFAULT_CURRICULUM_PATH
        if not curriculum_path.exists():
            curriculum_path = Path("schema/chess/endgame_curriculum.yaml")
    else:
        curriculum_path = Path(path)

    if not curriculum_path.exists():
        return []

    import yaml

    with open(curriculum_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    presets = data.get("curriculum", {}).get("presets", [])
    fens = [p.get("fen") for p in presets if isinstance(p, dict) and p.get("fen")]

    valid: list[str] = []
    for fen in fens:
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        if board.turn != chess.WHITE or board.is_check() or not board.is_valid():
            continue
        valid.append(fen)
    return valid


def _worker_run_training_game(
    theta_meta: np.ndarray,
    beta_efe: float,
    temperature: float,
    max_moves: int,
    seed_val: int,
    early_adjudication_material: float = 15.0,
    nnue_weights: dict[str, np.ndarray] | None = None,
    value_gamma: float = 0.97,
    start_fen: str | None = None,
    resign_value_threshold: float | None = None,
    resign_confirm_moves: int = 6,
    search_depth: int = SEARCH_DEPTH,
    adjudicate_bare_king_requires_mate: bool = True,
) -> tuple[float, list[np.ndarray], list[tuple[chess.Board, float]], int, float, str]:
    """Isolated process worker function running 1 self-play training game on a separate CPU process.

    Returns (final_reward, trajectory_features, discounted_labels, num_moves, mat_bal, termination).
    """
    from hypostases.plugins.domains.chess.nnue_net import NNUENet, extract_halfkp_features

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

    net = NNUENet() if nnue_weights is not None else None
    if net and nnue_weights:
        net.W_white = nnue_weights["W_white"]
        net.W_black = nnue_weights["W_black"]
        net.W_l1 = nnue_weights["W_l1"]
        net.b_l1 = nnue_weights["b_l1"]
        net.W_l2 = nnue_weights["W_l2"]
        net.b_l2 = nnue_weights["b_l2"]

    board = chess.Board(start_fen) if start_fen else domain.initial_state()
    agent_color = board.turn
    num_moves = 0
    done = False
    trajectory_features = []
    game_positions: list[chess.Board] = []
    hopeless_streak = {chess.WHITE: 0, chess.BLACK: 0}

    while not done and num_moves < max_moves:
        legal_moves = domain.valid_actions(board)
        if not legal_moves:
            break

        game_positions.append(board.copy())

        if board.turn == agent_color:
            chosen_move = agent.select_move(board, legal_moves, depth=search_depth, nnue_net=net)
            feats = agent.extract_move_features(board, chosen_move)
            if len(theta_meta) >= 9:
                feats = np.append(feats, 0.5)
            if len(theta_meta) >= 10:
                # Epistemic utility is search-independent (branching entropy * skill):
                # compute it directly on the post-move board instead of re-searching.
                post_move = board.copy()
                post_move.push(chosen_move)
                feats = np.append(feats, float(agent.epistemic_utility(post_move) / 3.0))
            trajectory_features.append(feats)
        else:
            chosen_move = opponent.select_move(board, legal_moves, depth=search_depth, nnue_net=net)

        board, _reward, _done, _info = domain.step(board, chosen_move)
        num_moves += 1

        # Value-based resignation: the side that just moved resigns when the value
        # net rates its position hopeless for `resign_confirm_moves` consecutive plies.
        if net is not None and resign_value_threshold is not None and not done:
            mover = not board.turn
            acc = net.create_accumulator(board)
            _, _, aux = extract_halfkp_features(board)
            val_stm = net.forward(acc, aux)
            val_mover = val_stm if board.turn == mover else -val_stm
            if val_mover < resign_value_threshold:
                hopeless_streak[mover] += 1
            else:
                hopeless_streak[mover] = 0
            if hopeless_streak[mover] >= resign_confirm_moves:
                winner = not mover
                final_reward = 1.0 if winner == agent_color else -1.0
                mat_bal = _compute_material_balance(board, agent_color)
                labeled_positions = _build_discounted_labels(
                    game_positions, board, final_reward, value_gamma
                )
                return (
                    final_reward,
                    trajectory_features,
                    labeled_positions,
                    num_moves,
                    mat_bal,
                    "RESIGNATION",
                )

        # Early material adjudication after move 15 (fallback; primary signal is resignation).
        # P1.1: curriculum games (seeded from endgame presets) are NEVER adjudicated — they
        # must play to a real outcome so the value net learns mate conversion, not just a
        # material edge. P1.2: never adjudicate when the losing side has a bare king and the
        # winner holds mating material — mate is still reachable, so the game plays on.
        if num_moves >= 15 and start_fen is None:
            mat_bal = _compute_material_balance(board, agent_color)
            if _should_material_adjudicate(
                board,
                agent_color,
                early_adjudication_material,
                adjudicate_bare_king_requires_mate,
            ):
                final_reward = 1.0 if mat_bal > 0 else -1.0
                labeled_positions = _build_discounted_labels(
                    game_positions, board, final_reward, value_gamma
                )
                return (
                    final_reward,
                    trajectory_features,
                    labeled_positions,
                    num_moves,
                    mat_bal,
                    "MATERIAL_ADJUDICATION",
                )

    outcome = board.outcome()
    mat_bal = _compute_material_balance(board, agent_color)
    if outcome is None:
        final_reward = float(np.tanh(mat_bal / 3.0))
        termination = "max_moves_exceeded"
    elif outcome.winner == agent_color:
        final_reward = 1.0
        termination = outcome.termination.name
    elif outcome.winner is None:
        final_reward = 0.0
        termination = outcome.termination.name
    else:
        final_reward = -1.0
        termination = outcome.termination.name

    labeled_positions = _build_discounted_labels(game_positions, board, final_reward, value_gamma)
    return (
        final_reward,
        trajectory_features,
        labeled_positions,
        num_moves,
        mat_bal,
        termination,
    )


class ChessSelfPlayTrainer:
    """Trainer executing parallel multi-process self-play reinforcement learning on HYPOSTASES agents."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta_efe: float = 0.05,
        initial_temperature: float = 0.8,
        max_workers: int = 20,
        value_gamma: float = 0.97,
        chess_domain: ChessDomain | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.beta_efe = beta_efe
        self.initial_temperature = initial_temperature
        self.max_workers = max(1, max_workers)
        self.value_gamma = value_gamma
        self.domain = chess_domain or ChessDomain()
        self._grad_batches: dict[Any, tuple[list[np.ndarray], list[float]]] = {}
        self.telemetry: dict[str, Any] = {}

    def _accumulate_gradient(
        self,
        gen_key: Any,
        trajectory_features: list[np.ndarray],
        final_reward: float,
    ) -> None:
        """Accumulates one game's discounted feature-normalized trace into a per-generation batch.

        The eligibility trace `norm` is the discounted weighted feature mean divided by
        per-index scale (FEATURE_SCALES for the 8 tactical features, 1.0 for the
        temperature/beta slots). The full K=10 meta-parameter vector is learned by the
        general-purpose estimator: no index is manually pinned — a feature whose
        covariance with the centered advantage is zero converges to staticity on its own.
        Games from concurrently running generations are keyed independently by `gen_key`;
        the batch is finalized once per generation by `_finalize_generation_gradient`.
        """
        if not trajectory_features:
            return
        n = len(trajectory_features)
        discounts = np.power(self.value_gamma, np.arange(n - 1, -1, -1)).astype(np.float32)
        w_sum = float(discounts.sum())
        if w_sum <= 0.0:
            return
        stacked = np.stack(trajectory_features)
        weighted_mean = np.sum(stacked * discounts[:, None], axis=0) / w_sum
        k = weighted_mean.shape[0]
        if k <= len(FEATURE_SCALES):
            scale = FEATURE_SCALES[:k]
        else:
            scale = np.concatenate(
                [FEATURE_SCALES, np.ones(k - len(FEATURE_SCALES), dtype=np.float32)]
            )
        norm = weighted_mean / scale

        batch = self._grad_batches.setdefault(gen_key, (list(), list()))
        batch[0].append(norm)
        batch[1].append(float(final_reward))

    def _finalize_generation_gradient(
        self,
        gen_key: Any,
        agent: ChessAgentAdapter,
    ) -> None:
        """Applies the batch-normalized REINFORCE update once per generation.

        Per-game advantages are centered by the batch-mean reward (the optimal constant
        baseline) and scaled to unit variance, so the update is the covariance of the
        advantage with each normalized feature trace:
            Δθ_f = η · mean_i( adv_i' · norm_{i,f} ),   adv' = (R - mean(R)) / std(R)

        This removes the persistent-EMA ratchet that biased every feature in proportion
        to its mean trace magnitude (mobility/defended crashed from losses/draws carrying
        full negative advantage): with centered advantages the mean(adv)·mean(norm_f)
        term is identically zero, leaving only the genuine signal. All-equal rewards
        (zero variance) produce no update. The final per-feature step is clipped.

        The full K=10 meta-parameter vector is learned by this general-purpose estimator
        — including the beta logit (θ[9]). Staticity of β is an EMERGENT property of zero
        reward-feature covariance, never a manual pin. The logit gradient receives the
        historical x5 boost (small epistemic feature magnitude) before clipping, and the
        resulting logit is clamped to ±BETA_LOGIT_MAX with a warning on actual clamp hits.
        """
        batch = self._grad_batches.pop(gen_key, None)
        if batch is None:
            return
        norms, rewards = batch
        if not norms:
            return

        grads = np.stack(norms)
        advantages = np.asarray(rewards, dtype=np.float64)
        advantages = advantages - advantages.mean()
        adv_std = advantages.std()
        if adv_std < 1e-9:
            return
        advantages = advantages / adv_std

        k = grads.shape[1]
        grad = np.zeros(k, dtype=np.float64)
        for i in range(len(grads)):
            grad += self.learning_rate * advantages[i] * grads[i]

        if k >= 10:
            grad[9] *= BETA_LOGIT_GRADIENT_BOOST

        grad = np.clip(grad, -0.5, 0.5)

        new_theta = agent.theta_meta.copy()
        # Bounds: tactical features + temperature floor at 0; the beta logit is
        # unbounded below (a negative logit simply shrinks β) and value-clamped to
        # ±BETA_LOGIT_MAX above/below for numeric safety.
        capped = min(k, 9)
        new_theta[:capped] = np.maximum(0.0, new_theta[:capped] + grad[:capped])
        if k >= 10:
            new_theta[9] = new_theta[9] + grad[9]
            if new_theta[9] > BETA_LOGIT_MAX or new_theta[9] < -BETA_LOGIT_MAX:
                clamped = max(-BETA_LOGIT_MAX, min(BETA_LOGIT_MAX, new_theta[9]))
                console.print(
                    f"  [bold yellow][WARN][/bold yellow] beta logit clamped to "
                    f"{clamped:+.2f} (raw {new_theta[9]:+.3f}) — investigate"
                )
                new_theta[9] = clamped
        agent.theta_meta = new_theta

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
        self._grad_batches.pop("single", None)

        if games_per_gen <= 5 or self.max_workers == 1:
            # Fast in-process execution for small batches / unit tests (avoids Windows process spawn overhead)
            for g_idx in range(games_per_gen):
                (
                    final_reward,
                    trajectory_features,
                    _labeled_positions,
                    _num_moves,
                    _mat_bal,
                    _termination,
                ) = _worker_run_training_game(
                    updated_agent.theta_meta,
                    self.beta_efe,
                    updated_agent.temperature,
                    max_moves,
                    seed + g_idx,
                    value_gamma=self.value_gamma,
                )
                self._accumulate_gradient("single", trajectory_features, final_reward)
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
                            value_gamma=self.value_gamma,
                        )
                        for g_idx in range(games_per_gen)
                    ]

                    for future in as_completed(futures):
                        (
                            final_reward,
                            trajectory_features,
                            _labeled_positions,
                            _num_moves,
                            _mat_bal,
                            _termination,
                        ) = future.result()

                        self._accumulate_gradient("single", trajectory_features, final_reward)

                        if progress is not None and task_id is not None:
                            progress.update(task_id, advance=1)

                if executor is not None:
                    _run_with_exec(executor)
                else:
                    local_executor = ProcessPoolExecutor(
                        max_workers=min(self.max_workers, games_per_gen),
                        initializer=_init_worker_process,
                    )
                    try:
                        _run_with_exec(local_executor)
                    except BaseException:
                        _terminate_pool_workers(local_executor)
                        raise
                    finally:
                        local_executor.shutdown(wait=False, cancel_futures=True)
            except KeyboardInterrupt:
                raise

        self._finalize_generation_gradient("single", updated_agent)
        return updated_agent

    def execute_self_play_training_run(
        self,
        total_generations: int = 20,
        snapshot_interval_k: int = 5,
        games_per_generation: int = 15,
        seed: int = 42,
        min_temperature: float = 0.20,
        max_moves_training: int = 400,
        early_adjudication_material: float = 15.0,
        initial_priors: Any = "random",
        value_gamma: float = 0.97,
        curriculum_probability: float = 0.25,
        curriculum_path: str | Path | None = None,
        resign_value_threshold: float = -3.0,
        resign_confirm_moves: int = 6,
        nnue_epochs: int = 30,
        nnue_learning_rate: float = 0.003,
        replay_capacity: int = 6000,
        search_depth: int = SEARCH_DEPTH,
        adjudicate_bare_king_requires_mate: bool = True,
        log_game_details: bool = False,
        verbose: bool = True,
    ) -> list[PolicySnapshot]:
        """Runs per-generation synchronous self-play training (θ barrier) with a background
        NNUE SGD learner, returning PolicySnapshots.

        Ordering (Rule 003 branch audit):
          * Hard per-generation barrier on θ_meta: generation N's games are submitted only
            after generation N-1's boundary finalized θ, so the REINFORCE batch that updates
            θ_{N-1}->θ_N was played by exactly θ_{N-1} (on-policy credit assignment).
          * The NNUE value learner runs as a background daemon thread (A3C-style staleness):
            generation N plays with the most recently committed net (at most ~1 generation
            behind), which never blocks the pool or the progress display.
        """
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
            return "[" + " ".join(f"{v:04.3f}" for v in feat_vals) + "]"

        if verbose:
            formatted_gen0 = _format_agent_theta(agent)
            console.print(
                f"\n[bold cyan][HYPOSTASES Trainer][/bold cyan] Initialized Gen 0 Agent ([bold yellow]theta_meta = {formatted_gen0}[/bold yellow] [magenta]temp={agent.temperature:.3f}[/magenta] [blue]beta={agent.beta_efe:.2f}[/blue])"
            )

        def make_policy_fn(cur_agent: ChessAgentAdapter) -> Any:
            def policy(board: Any, legal_moves: Any) -> Any:
                return cur_agent.select_move(board, legal_moves)

            return policy

        from hypostases.plugins.domains.chess.nnue_net import NNUENet
        from hypostases.plugins.domains.chess.nnue_training import train_nnue

        nnue_net = NNUENet(seed=seed)

        def _extract_nnue_weights(net: NNUENet) -> dict[str, np.ndarray]:
            return {
                "W_white": net.W_white.copy(),
                "W_black": net.W_black.copy(),
                "W_l1": net.W_l1.copy(),
                "b_l1": net.b_l1.copy(),
                "W_l2": net.W_l2.copy(),
                "b_l2": net.b_l2.copy(),
            }

        snapshots.append(
            PolicySnapshot(
                generation=0,
                policy_fn=make_policy_fn(copy.deepcopy(agent)),
                theta_meta=agent.theta_meta.copy(),
                temperature=agent.temperature,
                nnue_weights=_extract_nnue_weights(nnue_net),
            )
        )

        # Endgame curriculum presets (Rule 006 YAML data source)
        curriculum_fens = load_endgame_curriculum(curriculum_path)
        if verbose:
            if curriculum_fens:
                console.print(
                    f"[bold blue][Curriculum][/bold blue] Loaded {len(curriculum_fens)} endgame presets "
                    f"(p={curriculum_probability:.2f}): {', '.join(material_tag(fen) for fen in curriculum_fens)}"
                )
            else:
                console.print(
                    f"[bold yellow][Curriculum][/bold yellow] No endgame presets found at {curriculum_path}; "
                    "training from standard start only."
                )

        digits = len(str(total_generations))
        total_games = total_generations * games_per_generation

        pending_futures: dict[Any, tuple[int, bool, float]] = {}
        task_ids: dict[int, Any] = {}
        run_started = time.monotonic()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            disable=not verbose,
        ) as progress:
            pool_executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=_init_worker_process,
            )
            try:
                # Committed NNUE learner: only fully-finished SGD results are published.
                # Generation games snapshot the committed net once at gen start (uniform
                # per generation), so a running background SGD never exposes torn weights.
                committed_lock = threading.Lock()
                committed_weights: dict[str, np.ndarray] = _extract_nnue_weights(nnue_net)
                sgd_ev: threading.Event | None = None
                sgd_gen: int | None = None
                sgd_loss: float | None = None

                def _kick_background_sgd(
                    gen_idx: int,
                    train_positions: list[tuple[chess.Board, float]],
                ) -> None:
                    """Starts NNUE SGD on a daemon thread; publishes the net only on completion."""
                    nonlocal sgd_ev, sgd_gen, sgd_loss
                    if not train_positions:
                        return
                    with committed_lock:
                        base = {k: v.copy() for k, v in committed_weights.items()}
                    dataset = list(train_positions)
                    sgd_gen = gen_idx
                    sgd_loss = None
                    ev = threading.Event()

                    def _worker() -> None:
                        net = NNUENet()
                        for key, val in base.items():
                            setattr(net, key, val.copy())
                        loss = train_nnue(
                            net,
                            dataset,
                            epochs=nnue_epochs,
                            lr=nnue_learning_rate,
                            verbose=False,
                        )
                        with committed_lock:
                            committed_weights.clear()
                            committed_weights.update(_extract_nnue_weights(net))
                        nonlocal sgd_loss
                        sgd_loss = float(loss)
                        ev.set()

                    threading.Thread(
                        target=_worker,
                        name=f"nnue-sgd-gen{gen_idx}",
                        daemon=True,
                    ).start()
                    sgd_ev = ev

                def _drain_sgd_result() -> None:
                    """Prints + records a finished background SGD loss (called from the drain loop)."""
                    nonlocal sgd_ev, sgd_gen, sgd_loss
                    if sgd_ev is not None and sgd_ev.is_set():
                        gen = sgd_gen
                        loss = sgd_loss
                        sgd_ev = None
                        sgd_gen = None
                        sgd_loss = None
                        if loss is not None and gen is not None:
                            console.print(
                                f"    --> [bold blue][NNUE SGD][/bold blue] Gen {gen:0{digits}d} Neural Loss: [bold yellow]{loss:.4f}[/bold yellow]"
                            )
                            if gen in self.telemetry:
                                self.telemetry[gen]["nnue_loss"] = loss

                def _heartbeat_status(gen_idx: int, done: int, stats: dict[str, Any]) -> str:
                    """Live pool/gen status line for the per-generation drain loop."""
                    now = time.monotonic()
                    inflight = len(pending_futures)
                    longest = max(
                        now - start for _f, (_gi, _is_cur, start) in pending_futures.items()
                    )
                    total_done = (gen_idx - 1) * games_per_generation + done
                    elapsed = now - run_started
                    rate = total_done / elapsed if elapsed > 0.0 else 0.0
                    remaining = total_games - total_done
                    eta_min = (remaining / rate) / 60.0 if rate > 0.0 else 0.0
                    term_str = ", ".join(
                        f"{t}={c}"
                        for t, c in sorted(stats["terminations"].items(), key=lambda x: -x[1])
                    )
                    ending_part = f" | {term_str}" if term_str else ""
                    return (
                        f"  [dim][HEARTBEAT] Gen {gen_idx:02d}: {done}/{games_per_generation:02d}+{inflight} | "
                        f"longest {longest:.0f}s | ETA ~{eta_min:.0f}m{ending_part}[/dim]"
                    )

                # Run-global replay buffers carry positions across generations (FIFO capped).
                replay_positions: list[tuple[chess.Board, float]] = []
                curriculum_positions: list[tuple[chess.Board, float]] = []

                # ------------------------------------------------------------------
                # Per-generation barrier: submit ONE generation's games at a time, wait
                # for all to finish, then finalize θ (instant) and kick background SGD.
                # ------------------------------------------------------------------
                for gen_idx in range(1, total_generations + 1):
                    agent.temperature = max(
                        min_temperature,
                        self.initial_temperature * (0.93 ** (gen_idx - 1)),
                    )
                    agent.beta_efe = self.beta_efe
                    formatted_theta = _format_agent_theta(agent)
                    task_ids[gen_idx] = progress.add_task(
                        f"[cyan]Gen {gen_idx:0{digits}d}/{total_generations}[/cyan] [yellow]theta={formatted_theta}[/yellow] [magenta]temp={agent.temperature:.3f}[/magenta] [blue]beta={agent.beta_efe:.3f}[/blue]",
                        total=games_per_generation,
                    )

                    theta_before_gen = agent.theta_meta.copy()
                    with committed_lock:
                        gen_net_weights = {k: v.copy() for k, v in committed_weights.items()}

                    stats: dict[str, Any] = {
                        "terminations": defaultdict(int),
                        "moves": [],
                        "abs_material": [],
                        "curriculum_games": 0,
                    }

                    # Submit every game of this generation (θ + net fixed for the whole gen)
                    for g_local in range(games_per_generation):
                        game_id = (gen_idx - 1) * games_per_generation + g_local
                        is_curriculum = bool(
                            curriculum_fens and random.random() < curriculum_probability
                        )
                        start_fen = random.choice(curriculum_fens) if is_curriculum else None
                        game_num = g_local + 1

                        fut = pool_executor.submit(
                            _worker_run_training_game,
                            agent.theta_meta,
                            self.beta_efe,
                            agent.temperature,
                            max_moves_training,
                            seed + game_id,
                            early_adjudication_material,
                            gen_net_weights,
                            value_gamma,
                            start_fen,
                            resign_value_threshold,
                            resign_confirm_moves,
                            search_depth,
                            adjudicate_bare_king_requires_mate,
                        )
                        pending_futures[fut] = (gen_idx, is_curriculum, time.monotonic())

                        if verbose and log_game_details and (is_curriculum or game_num == 1):
                            seed_tag = material_tag(start_fen) if is_curriculum else "opening"
                            console.print(
                                f"  [dim][Gen {gen_idx:0{digits}d} Game {game_num:02d}/{games_per_generation:02d}] submitted | seed={seed_tag} | pool busy {len(pending_futures)}/{games_per_generation}[/dim]"
                            )

                    # Drain until this generation's games are all complete
                    completed = 0
                    last_beat = 0.0
                    while pending_futures:
                        done_set, _ = wait(
                            pending_futures.keys(),
                            return_when=FIRST_COMPLETED,
                            timeout=HEARTBEAT_INTERVAL,
                        )
                        _drain_sgd_result()
                        if not done_set:
                            now = time.monotonic()
                            longest = max(
                                now - start for _f, (_gi, _is_cur, start) in pending_futures.items()
                            )
                            if (
                                verbose
                                and longest > HEARTBEAT_MIN_GAME_S
                                and now - last_beat > HEARTBEAT_PRINT_GAP_S
                            ):
                                last_beat = now
                                console.print(_heartbeat_status(gen_idx, completed, stats))
                            continue

                        for fut in done_set:
                            _gi, is_curriculum, start_time = pending_futures.pop(fut)
                            game_duration = time.monotonic() - start_time
                            (
                                final_reward,
                                trajectory_features,
                                labeled_positions,
                                num_moves,
                                mat_bal,
                                termination,
                            ) = fut.result()
                            completed += 1

                            stats["terminations"][termination] += 1
                            stats["moves"].append(num_moves)
                            stats["abs_material"].append(abs(mat_bal))
                            if is_curriculum:
                                stats["curriculum_games"] += 1

                            target_buffer = (
                                curriculum_positions if is_curriculum else replay_positions
                            )
                            target_buffer.extend(labeled_positions)
                            del target_buffer[: -max(replay_capacity // 2, 1)]

                            self._accumulate_gradient(gen_idx, trajectory_features, final_reward)

                            if gen_idx in task_ids:
                                formatted_theta = _format_agent_theta(agent)
                                progress.update(
                                    task_ids[gen_idx],
                                    advance=1,
                                    description=f"[cyan]Gen {gen_idx:0{digits}d}/{total_generations}[/cyan] [yellow]theta={formatted_theta}[/yellow] [magenta]temp={agent.temperature:.3f}[/magenta] [blue]beta={agent.beta_efe:.3f}[/blue]",
                                )

                            if verbose and log_game_details:
                                res_char = (
                                    "W"
                                    if final_reward > 0.5
                                    else ("L" if final_reward < -0.5 else "D")
                                )
                                res_style = (
                                    "bold green"
                                    if res_char == "W"
                                    else ("bold red" if res_char == "L" else "yellow")
                                )
                                console.print(
                                    f"  [cyan][Gen {gen_idx:0{digits}d} Game {completed:02d}/{games_per_generation:02d}][/cyan] Result: [{res_style}]{res_char}[/{res_style}] | Moves: {num_moves:02d} | Reward: {final_reward:+.2f} | Mat: {mat_bal:+.1f} | End: {termination} (took {game_duration:.0f}s, in-flight {len(pending_futures)})"
                                )

                    # ---- generation boundary: θ finalize is instantaneous ----
                    self._finalize_generation_gradient(gen_idx, agent)
                    theta_delta = float(np.linalg.norm(agent.theta_meta - theta_before_gen))

                    train_positions = replay_positions + curriculum_positions
                    if train_positions:
                        console.print(
                            f"    --> [bold blue][NNUE SGD][/bold blue] Gen {gen_idx:0{digits}d} training "
                            f"({len(train_positions)} FENs: {len(replay_positions)} opening / {len(curriculum_positions)} curriculum) in background ..."
                        )
                        _kick_background_sgd(gen_idx, train_positions)

                    term_summary = ", ".join(
                        f"{t}={c}"
                        for t, c in sorted(stats["terminations"].items(), key=lambda x: -x[1])
                    )
                    avg_len = float(np.mean(stats["moves"])) if stats["moves"] else 0.0
                    avg_mat = (
                        float(np.mean(stats["abs_material"])) if stats["abs_material"] else 0.0
                    )
                    self.telemetry[gen_idx] = {
                        "terminations": dict(stats["terminations"]),
                        "avg_game_length": avg_len,
                        "avg_abs_material": avg_mat,
                        "curriculum_games": stats["curriculum_games"],
                        "theta_norm_delta": theta_delta,
                        "nnue_loss": None,
                    }
                    console.print(
                        f"    --> [bold green][GEN SUMMARY][/bold green] Gen {gen_idx:0{digits}d} Endings: {term_summary} | "
                        f"AvgLen: {avg_len:.0f} | Avg|Mat|: {avg_mat:.1f} | |dTheta|: {theta_delta:.3f}"
                    )

                    if gen_idx % snapshot_interval_k == 0:
                        with committed_lock:
                            snapshot_net = {k: v.copy() for k, v in committed_weights.items()}
                        chk_theta = _format_agent_theta(agent)
                        console.print(
                            f"    --> [bold green][CHECKPOINT SAVED][/bold green] Gen {gen_idx:0{digits}d} Checkpoint Staged ([bold yellow]theta = {chk_theta}[/bold yellow] [magenta]temp={agent.temperature:.3f}[/magenta] [blue]beta={agent.beta_efe:.3f}[/blue])"
                        )
                        snapshots.append(
                            PolicySnapshot(
                                generation=gen_idx,
                                policy_fn=make_policy_fn(copy.deepcopy(agent)),
                                theta_meta=agent.theta_meta.copy(),
                                temperature=agent.temperature,
                                nnue_weights=snapshot_net,
                            )
                        )

                    progress.remove_task(task_ids.pop(gen_idx))

            except BaseException:
                _terminate_pool_workers(pool_executor)
                raise
            finally:
                pool_executor.shutdown(wait=False, cancel_futures=True)

        return snapshots


def material_tag(fen: str) -> str:
    """Extracts a short material tag from a curriculum FEN for console display."""
    try:
        board = chess.Board(fen)
    except ValueError:
        return "?"
    pieces = [p.piece_type for p in board.piece_map().values() if p.piece_type != chess.KING]
    order = {
        chess.QUEEN: "Q",
        chess.ROOK: "R",
        chess.BISHOP: "B",
        chess.KNIGHT: "N",
        chess.PAWN: "P",
    }
    return "K" + "".join(order[p] for p in sorted(pieces, key=lambda p: -p)) + "K"
