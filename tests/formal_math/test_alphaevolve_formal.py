"""Formal Mathematical Verification Suite for Wave 5 Front 13 — AlphaEvolve Engine.

Rule 012 Compliance: Mandatory Formal Mathematical Implementation Verification.

Empirically verifies end-to-end mathematical theorems, limit behavior, game-theoretic equilibria,
MAP-Elites quality-diversity coverage monotonicity, regularized aging evolution bounds,
and Rule 005 state invariant preservation over sigma = (c, w, g, rho_ext).
"""

import numpy as np

from hypostases.alphaevolve.engine import AlphaEvolveEngine
from hypostases.alphaevolve.evaluator import GameTheoreticOracleEvaluator
from hypostases.alphaevolve.mutator import ASTMutator, FECEvaluator
from hypostases.alphaevolve.qd_archive import MAPElitesArchive, NoveltyArchive
from hypostases.alphaevolve.reservoir import MorphologicalReservoir


def test_formal_map_elites_fitness_monotonicity() -> None:
    """Verify theorem: MAP-Elites grid coverage and maximum cell fitness are non-decreasing."""
    archive = MAPElitesArchive(grid_dims=(5, 5, 5))
    rng = np.random.default_rng(42)

    previous_max_score = -np.inf
    previous_coverage = 0.0

    for step in range(50):
        code_str = f"def policy(state): return {float(step * 0.1)}"
        fitness = float(step * 0.2 + rng.uniform(-0.05, 0.05))
        behavior = rng.uniform([0.0, 0.0, -5.0], [100.0, 1.0, 5.0])

        archive.add_candidate(code_str, fitness, behavior)

        current_coverage = archive.coverage()
        current_max_score = (
            max([sol["score"] for sol in archive.solutions.values()])
            if archive.solutions
            else -np.inf
        )

        # Monotonicity invariants
        assert current_coverage >= previous_coverage - 1e-9
        assert current_max_score >= previous_max_score - 1e-9

        previous_coverage = current_coverage
        previous_max_score = current_max_score


def test_formal_incentive_compatibility_regret_bounds() -> None:
    """Verify theorem: Incentive Compatibility Regret R_IC >= 0 for all candidate AST policies."""
    evaluator = GameTheoreticOracleEvaluator(simulation_ticks=10, efe_mode=True)
    mutator = ASTMutator(seed=42)

    code = "def candidate_policy(state): return float(np.sum(state))"
    fn = mutator.compile_executable_function(code)

    eval_res = evaluator.evaluate_candidate(fn, code_length=len(code))

    assert "r_ic" in eval_res
    assert eval_res["r_ic"] >= 0.0
    assert eval_res["delta_kappa"] >= 0.0
    assert np.isfinite(eval_res["fitness"])


def test_formal_regularized_aging_purging_bounds() -> None:
    """Verify theorem: Aging evolution purges the oldest candidate after N_pop cycles."""
    engine = AlphaEvolveEngine(seed=42)
    queue_capacity = engine.population_capacity

    # Step engine through queue capacity iterations
    for _ in range(queue_capacity + 5):
        engine.step_generation()

    # FIFO queue size must strictly observe maximum capacity constraint
    assert len(engine.population_queue) <= queue_capacity


def test_formal_fec_discrepancy_invariance() -> None:
    """Verify theorem: Functional Equivalence Checking (FEC) correctly identifies identical ASTs."""
    fec = FECEvaluator(num_probes=16, tolerance=1e-6, seed=42)

    def fn1(state: np.ndarray) -> float:
        return float(state[0] + 2.0 * state[1])

    def fn2(state: np.ndarray) -> float:
        return float(state[0] + 2.0 * state[1])

    def fn3(state: np.ndarray) -> float:
        return float(state[0] - 5.0 * state[1])

    sig1 = fec.evaluate_signature(fn1)
    sig2 = fec.evaluate_signature(fn2)
    sig3 = fec.evaluate_signature(fn3)

    assert fec.are_equivalent(sig1, sig2)
    assert not fec.are_equivalent(sig1, sig3)


def test_formal_novelty_distance_positivity() -> None:
    """Verify theorem: Novelty distance rho(x, S) is non-negative and zero for identical points."""
    novelty = NoveltyArchive(k_neighbors=5, threshold=0.2)
    p1 = np.array([1.0, 0.5, 0.0])
    p2 = np.array([5.0, 2.5, 1.0])

    novelty.add_if_novel(p1, [])
    dist = novelty.compute_novelty(p2, [p1])

    assert dist >= 0.0
    assert np.isfinite(dist)

    # Explicitly verify zero distance for identical point set
    dist_same = novelty.compute_novelty(p1, [p1])
    assert dist_same == 0.0


def test_formal_fec_pruning_step() -> None:
    """Verify theorem: FEC pruning prevents population queue growth on equivalent ASTs."""
    engine = AlphaEvolveEngine(seed=42)

    # Force identical AST compilation signature
    signature = engine.fec_evaluator.evaluate_signature(
        engine.mutator.compile_executable_function(engine.skeleton_code)
    )

    # Add duplicate signature entry to population queue
    engine.population_queue[0]["signature"] = signature
    initial_queue_len = len(engine.population_queue)

    # Patch mutator to return un-mutated skeleton code to trigger equivalence
    engine.mutator.mutate_ast_code = lambda code: code

    res = engine.step_generation()
    assert res["status"] == "fec_pruned"
    assert len(engine.population_queue) == initial_queue_len


def test_formal_morphological_reservoir_bounds() -> None:
    """Verify theorem: Morphological computation index MC_1 is bounded in [0, 1]."""
    res = MorphologicalReservoir(reservoir_dim=16, seed=42)
    res.reset()

    trajectories = []
    for _ in range(10):
        u = np.array([0.5, -0.2])
        x_next = res.step_reservoir(u)
        trajectories.append(x_next)

    mc1 = res.compute_morphological_computation_index(trajectories)
    assert 0.0 <= mc1 <= 1.0


def test_formal_state_invariant_preservation_and_rule_005() -> None:
    """Verify Rule 005 compliance & state invariant preservation under AlphaEvolve engine."""
    engine = AlphaEvolveEngine(seed=42)

    # Step evolutionary engine
    res = engine.step_generation()
    assert res["status"] in ("success", "fec_pruned")

    # Compile top elite to SkillArtifact
    skill = engine.compile_elite_to_skill_artifact()
    assert skill.skill_id == "alphaevolve_elite_policy"
    assert skill.confidence >= 0.1 and skill.confidence <= 0.99
    # Verify Rule 008 Gaussian basis dimension K=4
    assert len(skill.gaussian_weights) == 4


def test_formal_multi_generation_fitness_non_regression() -> None:
    """Theorem 13.7: the MAP-Elites best elite cannot regress over generations."""
    engine = AlphaEvolveEngine(seed=42)
    best_scores = []

    for _ in range(20):
        engine.step_generation()
        best_scores.append(
            max(solution["score"] for solution in engine.qd_archive.solutions.values())
        )

    assert np.all(np.diff(best_scores) >= -1e-12), best_scores
