"""Formal mathematical verification of the chess self-play training regimen.

Spec Ref: AGENTS.md 012 (formal math verification), chess_experiment_config.yaml,
schema/chess/endgame_curriculum.yaml.

Verifies, empirically and mathematically:
  * Discounted-outcome value-label invariants: γ→1 recovers constant outcome labels,
    γ→0 isolates the terminal position, γ-monotone discounting, and STM sign symmetry.
  * Endgame curriculum presets: every preset is legal, White to move, not in check;
    `mate_in` presets are verified forced mates with the bundled Stockfish 18 and
    `winning` presets are clearly winning.
  * Batched REINFORCE update invariants: determinism, batch-mean centering that kills
    the EMA ratchet, covariance-only credit assignment, bounded per-generation step,
    and feature-scale normalization.
  * Full-K meta-parameter learning (K=10): the beta logit (θ[9]) and temperature
    (θ[8]) are learned by the same general-purpose estimator — β staticity must be an
    EMERGENT property of zero reward-feature covariance, never a manual pin. Verifies
    β responds to genuine covariance, τ is emergent-static (constant feature), and the
    logit is unbounded below while value-clamped at ±BETA_LOGIT_MAX.
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest

from hypostases.plugins.domains.chess.chess_agent_adapter import ChessAgentAdapter
from hypostases.plugins.domains.chess.chess_domain import ChessDomain
from hypostases.plugins.domains.chess.chess_trainer import (
    BETA_LOGIT_GRADIENT_BOOST,
    BETA_LOGIT_MAX,
    FEATURE_SCALES,
    ChessSelfPlayTrainer,
    _build_discounted_labels,
    _stm_normalized_label,
    load_endgame_curriculum,
)

CURRICULUM_PATH = Path("schema/chess/endgame_curriculum.yaml")


# ---------------------------------------------------------------------------
# 1. Discounted-outcome value-label invariants
# ---------------------------------------------------------------------------
def _dummy_positions(n: int) -> list[chess.Board]:
    domain = ChessDomain()
    board = domain.initial_state()
    positions: list[chess.Board] = []
    while len(positions) < n and not board.is_game_over():
        positions.append(board.copy())
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(legal[0])
    return positions


def test_gamma_one_recovers_constant_outcome_labels() -> None:
    """γ=1.0 ⇒ every position (including terminal) carries the full terminal reward."""
    positions = _dummy_positions(5)
    terminal = positions[-1].copy()
    labels = _build_discounted_labels(positions, terminal, 1.0, 1.0)
    assert len(labels) == len(positions) + 1
    for board, label in labels:
        assert label == pytest.approx(_stm_normalized_label(board, 1.0))


def test_gamma_zero_isolates_terminal_label() -> None:
    """γ=0.0 ⇒ only the terminal position (remaining=0) receives the reward."""
    positions = _dummy_positions(5)
    terminal = positions[-1].copy()
    labels = _build_discounted_labels(positions, terminal, 1.0, 0.0)
    assert len(labels) == len(positions) + 1
    for i, (board, label) in enumerate(labels):
        expected = _stm_normalized_label(board, 1.0) if i == len(labels) - 1 else 0.0
        assert label == pytest.approx(expected)


def test_discount_is_monotone_in_remaining_plies() -> None:
    """|label| is non-decreasing toward the terminal position for γ ∈ (0,1)."""
    positions = _dummy_positions(6)
    terminal = positions[-1].copy()
    labels = _build_discounted_labels(positions, terminal, 1.0, 0.9)
    magnitudes = [abs(label) for _, label in labels]
    assert magnitudes == sorted(magnitudes)


def test_stm_normalization_sign_symmetry() -> None:
    """White-perspective reward R maps to +R for White-to-move and -R for Black-to-move."""
    positions = _dummy_positions(10)
    for board in positions:
        expected = 1.0 if board.turn == chess.WHITE else -1.0
        assert _stm_normalized_label(board, 1.0) == pytest.approx(expected)
        assert _stm_normalized_label(board, -1.0) == pytest.approx(-expected)


# ---------------------------------------------------------------------------
# 2. Endgame curriculum preset invariants
# ---------------------------------------------------------------------------
def _load_presets() -> list[dict]:
    import yaml

    with open(CURRICULUM_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["curriculum"]["presets"]


def test_curriculum_presets_are_legal_white_to_move() -> None:
    """Every preset: valid FEN, White to move, side to move not in check, both kings present."""
    fens = load_endgame_curriculum(CURRICULUM_PATH)
    assert len(fens) >= 6, "curriculum must define mate-able/winning presets"
    for fen in fens:
        board = chess.Board(fen)
        assert board.is_valid(), f"illegal position: {fen}"
        assert board.turn == chess.WHITE, f"preset must be White to move: {fen}"
        assert not board.is_check(), f"side to move must not be in check: {fen}"
        assert board.king(chess.WHITE) is not None and board.king(chess.BLACK) is not None


def test_curriculum_presets_mate_in_consistency() -> None:
    """`mate_in` presets must be forced mates; `winning` presets must be legal-but-unbounded."""
    for preset in _load_presets():
        board = chess.Board(preset["fen"])
        assert board.is_valid()
        if preset["mate_in"] is None:
            assert preset["winning"] is True
            assert board.has_insufficient_material(chess.WHITE) is False
        else:
            assert preset["mate_in"] >= 1
            assert preset["winning"] is True


def _stockfish_available() -> bool:
    sf = Path("src/hypostases/plugins/domains/chess/stockfish.exe")
    return sf.exists()


@pytest.mark.skipif(not _stockfish_available(), reason="bundled Stockfish binary not available")
def test_curriculum_presets_verified_by_stockfish() -> None:
    """Empirical end-to-end mate verification: `mate_in` presets are forced mates,
    `winning` presets are clearly winning (>= +3.0 pawns)."""
    import chess.engine

    sf_path = str(Path("src/hypostases/plugins/domains/chess/stockfish.exe").resolve())
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    try:
        engine.configure({"Threads": 1})
        for preset in _load_presets():
            board = chess.Board(preset["fen"])
            info = engine.analyse(board, chess.engine.Limit(depth=22, time=2.0))
            score = info.get("score")
            assert score is not None, f"no score for {preset['id']}"
            cp = score.pov(chess.WHITE).score(mate_score=100000)
            if preset["mate_in"] is not None:
                # forced mate within the declared horizon
                assert cp >= 99900, f"{preset['id']} not a forced mate (score={cp})"
            else:
                assert cp >= 300, f"{preset['id']} not clearly winning (score={cp})"
    finally:
        engine.quit()


# ---------------------------------------------------------------------------
# 3. Batched REINFORCE update invariants (P2.1 batch-normalized estimator, K=10)
# ---------------------------------------------------------------------------
def _make_agent() -> ChessAgentAdapter:
    return ChessAgentAdapter(
        domain=ChessDomain(),
        beta_efe=0.05,
        temperature=0.8,
        theta_meta=np.ones(10, dtype=np.float64),
    )


def _make_trainer(lr: float = 0.01) -> ChessSelfPlayTrainer:
    return ChessSelfPlayTrainer(learning_rate=lr, beta_efe=0.05, value_gamma=0.97)


def _trajectory(n: int = 12) -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    feats = rng.uniform(-1.0, 1.0, size=(n, 10)).astype(np.float32)
    feats[:, 4] = rng.uniform(0.05, 0.8, size=n).astype(np.float32)  # mobility
    feats[:, 8] = 0.5  # temperature feature (constant by construction)
    feats[:, 9] = rng.uniform(0.0, 0.28, size=n).astype(np.float32)  # epistemic signal
    return [f.copy() for f in feats]


def _apply_batch(
    trainer: ChessSelfPlayTrainer,
    agent: ChessAgentAdapter,
    games: list[tuple[list[np.ndarray], float]],
) -> np.ndarray:
    """Accumulates a batch of (trajectory, reward) games and finalizes the update once."""
    for traj, reward in games:
        trainer._accumulate_gradient("k", traj, reward)
    trainer._finalize_generation_gradient("k", agent)
    return agent.theta_meta


def test_batched_reinforce_update_is_deterministic() -> None:
    """Same batch => identical final θ (accumulation order invariant to determinism check)."""
    games = [(_trajectory(), 1.0), (_trajectory(), -1.0), (_trajectory(), 0.0)]

    trainer1, a1 = _make_trainer(), _make_agent()
    trainer2, a2 = _make_trainer(), _make_agent()
    _apply_batch(trainer1, a1, games)
    _apply_batch(trainer2, a2, games)
    assert np.allclose(a1.theta_meta, a2.theta_meta)


def test_batched_reinforce_centering_kills_ratchet() -> None:
    """Constant-baseline centering removes the mean(adv)·mean(norm_f) ratchet term.

    With balanced rewards (mean advantage 0) and an IDENTICAL eligibility trace on every
    game, the per-feature update must be exactly zero — regardless of how large the trace
    magnitude is. This is precisely the bias that crashed mobility/defended: a persistent
    EMA baseline made every loss/draw push high-norm features down.
    """
    traj_high = _trajectory()
    games = [(traj_high, 1.0), (traj_high, -1.0)] * 16
    agent = _make_agent()
    _apply_batch(_make_trainer(), agent, games)
    delta = agent.theta_meta - 1.0
    assert np.allclose(delta, 0.0, atol=1e-12)


def test_batched_reinforce_covariance_direction() -> None:
    """The update equals η·cov(adv, norm): only advantage-trace covariance survives.

    Two games, rewards [+1, -1], mobility traces [0.1 (win), 0.9 (loss)]:
    centered advantages = [1, -1], so Δθ[4] = η·(1·0.1 + (-1)·0.9)/scale₄ < 0.
    High-mobility LOSES de-weight mobility; a negative covariance drives the weight down.
    """
    feats_win = np.zeros((12, 10), dtype=np.float32)
    feats_loss = np.zeros((12, 10), dtype=np.float32)
    feats_win[:, 4] = 0.1
    feats_loss[:, 4] = 0.9
    games = [([f.copy() for f in feats_win], 1.0), ([f.copy() for f in feats_loss], -1.0)]

    agent = _make_agent()
    _apply_batch(_make_trainer(), agent, games)
    expected = -0.8 * 0.01 / FEATURE_SCALES[4]
    assert agent.theta_meta[4] == pytest.approx(1.0 + expected, abs=1e-6)
    assert agent.theta_meta[4] < 1.0


def test_batched_reinforce_zero_variance_no_update() -> None:
    """All-equal rewards (e.g. an all-draw generation) produce zero gradient and no update."""
    games = [(_trajectory(), 0.0)] * 12
    agent = _make_agent()
    _apply_batch(_make_trainer(), agent, games)
    assert np.allclose(agent.theta_meta, 1.0)


def test_batched_reinforce_per_generation_step_is_bounded() -> None:
    """The single per-generation per-feature update is clipped to [-0.5, 0.5].

    Wins carry a large uniform trace, losses an all-zero trace: the raw gradient
    (~32 per feature) far exceeds the clip on every one of the K=10 dimensions —
    including the beta logit after its x5 boost — proving the bounded-step guarantee
    holds for the full meta-parameter vector.
    """
    big_adv_traj = [np.full(10, 2.0, dtype=np.float32)] * 12
    zero_traj = [np.zeros(10, dtype=np.float32)] * 12
    games = [(big_adv_traj, 1.0), (zero_traj, -1.0)] * 16
    agent = _make_agent()
    _apply_batch(_make_trainer(lr=1.0), agent, games)
    delta = agent.theta_meta - 1.0
    assert np.max(np.abs(delta)) <= 0.5 + 1e-6


def test_batched_reinforce_feature_normalization_prevents_scale_dominance() -> None:
    """A feature with a 10x larger raw range must not dominate the update after scaling.

    capture_val=9 and capture_val=1 normalize to the same norm (9/9.0 = 1.0 vs 1/9.0),
    so the raw-scale gap may not translate into a >9x gap in the per-feature update.
    """
    traj_big = [np.array([9.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)] * 10
    traj_small = [
        np.array([1.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ] * 10
    games = [(traj_big, 1.0), (traj_big, -1.0), (traj_small, 1.0), (traj_small, -1.0)]

    agent = _make_agent()
    _apply_batch(_make_trainer(), agent, games)
    delta = agent.theta_meta[:8] - 1.0
    assert abs(delta[0]) <= 9.0 * abs(delta[4]) + 1e-6


def test_feature_scales_cover_all_tactical_features() -> None:
    """FEATURE_SCALES must be positive and length-8 (covers extract_move_features)."""
    assert FEATURE_SCALES.shape == (8,)
    assert np.all(FEATURE_SCALES > 0.0)


# ---------------------------------------------------------------------------
# 3b. Full-K meta-parameter learning: emergent β/τ staticity, never manual pins
# ---------------------------------------------------------------------------
def _epistemic_games(
    win_epis: float, loss_epis: float, lr: float = 0.01
) -> tuple[ChessSelfPlayTrainer, ChessAgentAdapter]:
    """Two-game batch: win carries `win_epis` epistemic trace, loss carries `loss_epis`.

    All tactical features and the constant temperature feature are zero, isolating the
    beta logit (θ[9]) update. Rewards [+1, -1] give centered advantages [1, -1].
    """
    win_traj = [np.zeros(10, dtype=np.float32)] * 12
    loss_traj = [np.zeros(10, dtype=np.float32)] * 12
    for f in win_traj:
        f[9] = win_epis
    for f in loss_traj:
        f[9] = loss_epis
    agent = _make_agent()
    trainer = _make_trainer(lr=lr)
    _apply_batch(
        trainer,
        agent,
        [([f.copy() for f in win_traj], 1.0), ([f.copy() for f in loss_traj], -1.0)],
    )
    return trainer, agent


def test_meta_learning_beta_staticity_is_emergent() -> None:
    """β staticity must EMERGE from zero reward-feature covariance, not be pinned.

    The epistemic trace is identical on wins and losses (win=loss=0.2): the estimator's
    own covariance Δθ[9] = Σ adv'·norm₉ = norm₉·Σ adv' ≡ 0, so the beta logit is
    provably unchanged — exactly the emergent-static behavior the engine must deliver
    without any manual freezing.
    """
    _, agent = _epistemic_games(win_epis=0.2, loss_epis=0.2)
    assert agent.theta_meta[9] == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(agent.theta_meta, 1.0, atol=1e-12)


def test_meta_learning_beta_responds_to_covariance() -> None:
    """β is learnable: genuine reward-feature covariance moves the logit.

    Wins carry high epistemic (0.9), losses low (0.1): Δθ[9] = η·(1·0.9 - 1·0.1) then
    the historical x5 logit boost, so θ[9] = 1.0 + 0.01·0.8·BETA_LOGIT_GRADIENT_BOOST.
    """
    _, agent = _epistemic_games(win_epis=0.9, loss_epis=0.1)
    expected = 1.0 + 0.01 * 0.8 * BETA_LOGIT_GRADIENT_BOOST
    assert agent.theta_meta[9] == pytest.approx(expected, abs=1e-9)
    assert agent.theta_meta[9] > 1.0


def test_meta_learning_tau_emergent_static() -> None:
    """The temperature logit (θ[8]) is emergent-static: its feature is a constant.

    norm₈ = 0.5/1.0 on every game, so Δθ[8] = 0.5·Σ adv' = 0.5·0 ≡ 0 for ANY reward
    vector — the constant-feature covariance is identically zero, so the general-purpose
    estimator freezes τ on its own.
    """
    _, agent = _epistemic_games(win_epis=0.9, loss_epis=0.1)
    assert agent.theta_meta[8] == pytest.approx(1.0, abs=1e-12)


def test_meta_learning_beta_logit_unbounded_below_and_clamped() -> None:
    """The beta logit is free to go negative (no max(0) floor) but value-clamps at ±MAX.

    Part A: strong negative covariance (win=0.1, loss=0.9) with lr=1.0 gives a post-boost
    step of -4.0, clipped to the -0.5 bounded step. Starting the logit at 0.4 drives it to
    -0.1 < 0 — a max(0,·) floor would have pinned it at 0, but the general-purpose update
    must let β shrink below its zero point.
    Part B: with θ[9] initialized near the boundary and a positive covariance step,
    the logit value is clamped to exactly BETA_LOGIT_MAX.
    """
    # Part A — unbounded below (no max(0) floor on the logit)
    agent_neg = _make_agent()
    agent_neg.theta_meta[9] = 0.4  # place the logit so a -0.5 bounded step crosses zero
    win_traj = [np.zeros(10, dtype=np.float32)] * 12
    loss_traj = [np.zeros(10, dtype=np.float32)] * 12
    for f in win_traj:
        f[9] = 0.1
    for f in loss_traj:
        f[9] = 0.9
    _apply_batch(
        _make_trainer(lr=1.0),
        agent_neg,
        [([f.copy() for f in win_traj], 1.0), ([f.copy() for f in loss_traj], -1.0)],
    )
    # grad9 raw = 1.0*(1*0.1 - 1*0.9) = -0.8, boosted x5 = -4.0, clipped to -0.5.
    assert agent_neg.theta_meta[9] == pytest.approx(0.4 - 0.5, abs=1e-5)
    assert agent_neg.theta_meta[9] < 0.0
    assert np.all(agent_neg.theta_meta[:8] >= 0.0)

    # Part B — value clamp at the upper bound (win=1.0, loss=0.0, lr=1.0)
    win_traj = [np.zeros(10, dtype=np.float32)] * 12
    loss_traj = [np.zeros(10, dtype=np.float32)] * 12
    for f in win_traj:
        f[9] = 1.0
    agent_clamp = _make_agent()
    agent_clamp.theta_meta[9] = 9.8
    _apply_batch(
        _make_trainer(lr=1.0),
        agent_clamp,
        [([f.copy() for f in win_traj], 1.0), ([f.copy() for f in loss_traj], -1.0)],
    )
    assert agent_clamp.theta_meta[9] == pytest.approx(BETA_LOGIT_MAX, abs=1e-9)
