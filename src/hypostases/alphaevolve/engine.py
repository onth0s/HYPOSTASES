"""AlphaEvolve Engine Core Orchestration.

References:
- Novikov et al. (2025) AlphaEvolve: A coding agent for scientific and algorithmic discovery (arXiv)
- Romera-Paredes et al. (2023) Mathematical discoveries from program search with LLMs (Nature - FunSearch)
- Real et al. (2019) Regularized Evolution for Image Classifier Architecture Search (AAAI)
"""

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hypostases.alphaevolve.evaluator import GameTheoreticOracleEvaluator
from hypostases.alphaevolve.mutator import ASTMutator, FECEvaluator
from hypostases.alphaevolve.qd_archive import MAPElitesArchive, NoveltyArchive
from hypostases.alphaevolve.reservoir import MorphologicalReservoir
from hypostases.engine.memory import SkillArtifact


class AlphaEvolveEngine:
    """AlphaEvolve Engine Orchestrator (Wave 5 Front 13).

    Maintains island demes, MAP-Elites QD archives, Novelty archives, Morphological Reservoirs,
    and regularized aging population queues over state substrate sigma = (c, w, g, rho_ext).
    """

    DEFAULT_SKELETON = """def candidate_policy(state: np.ndarray) -> float:
    # State vector: [c_cognition, w_world, g_goal, rho_ext_scarcity]
    c_val, w_val, g_val, rho_val = state[0], state[1], state[2], state[3]
    return float(0.5 * c_val + 0.2 * w_val + 0.3 * g_val - 0.1 * rho_val)
"""

    def __init__(
        self,
        config_path: str | Path | None = None,
        skeleton_code: str | None = None,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.skeleton_code = skeleton_code or self.DEFAULT_SKELETON

        # Load YAML configuration (Rule 006)
        self.config = self._load_config(config_path)
        evo_cfg = self.config.get("evolutionary_search", {})
        qd_cfg = self.config.get("quality_diversity", {})
        fec_cfg = self.config.get("functional_equivalence", {})
        eval_cfg = self.config.get("evaluator_oracle", {})
        res_cfg = self.config.get("morphological_coevolution", {})

        # Sub-modules
        self.mutator = ASTMutator(
            mutation_rate=float(evo_cfg.get("mutation_rate", 0.15)), seed=seed
        )
        self.fec_evaluator = FECEvaluator(
            num_probes=int(fec_cfg.get("num_synthetic_probes_K", 16)),
            tolerance=float(fec_cfg.get("fec_tolerance_epsilon", 1e-6)),
            seed=seed,
        )
        self.oracle_evaluator = GameTheoreticOracleEvaluator(
            simulation_ticks=int(eval_cfg.get("simulation_duration_ticks", 20)),
            efe_mode=bool(self.config.get("efe_mode", True)),
            efe_beta=0.2,
            lambda_ic=float(eval_cfg.get("penalty_weights", {}).get("lambda_ic", 10.0)),
            lambda_scarcity=float(eval_cfg.get("penalty_weights", {}).get("lambda_scarcity", 5.0)),
        )
        self.qd_archive = MAPElitesArchive(grid_dims=(10, 10, 10))
        self.novelty_archive = NoveltyArchive(
            k_neighbors=int(qd_cfg.get("novelty_search", {}).get("k_nearest_neighbors", 15)),
            threshold=float(
                qd_cfg.get("novelty_search", {}).get("archive_novelty_threshold", 0.25)
            ),
        )
        self.reservoir = MorphologicalReservoir(
            reservoir_dim=int(res_cfg.get("reservoir_dim", 16)),
            coupling_strength=float(res_cfg.get("coupling_strength", 0.5)),
            seed=seed,
        )

        # Island demes & Regularized Aging Population Queue (Real et al. 2019)
        self.num_islands = int(evo_cfg.get("island_count", 4))
        self.population_capacity = int(evo_cfg.get("population_size", 32))
        self.population_queue: list[dict[str, Any]] = []
        self.islands: list[list[dict[str, Any]]] = [[] for _ in range(self.num_islands)]

        # Initialize seed candidate
        self._initialize_seed_population()

    def _load_config(self, config_path: str | Path | None) -> dict[str, Any]:
        if config_path is None:
            try:
                from hypostases.schemas.loader import load_alphaevolve_config

                data = load_alphaevolve_config()
                return data.get("alphaevolve", {}) if isinstance(data, dict) else {}
            except Exception:
                return {}

        if Path(config_path).exists():
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data.get("alphaevolve", {}) if isinstance(data, dict) else {}
        return {}

    def _initialize_seed_population(self) -> None:
        """Seed the population and island demes with the baseline skeleton candidate."""
        fn = self.mutator.compile_executable_function(self.skeleton_code)
        parsimony_score = self.mutator.compute_parsimony_score(self.skeleton_code)
        eval_res = self.oracle_evaluator.evaluate_candidate(fn, parsimony_score)
        signature = self.fec_evaluator.evaluate_signature(fn)

        seed_entry = {
            "code": self.skeleton_code,
            "score": eval_res["fitness"],
            "behavior": eval_res["behavior"],
            "signature": signature,
            "age": 0,
        }

        self.population_queue.append(seed_entry)
        for i in range(self.num_islands):
            self.islands[i].append(seed_entry.copy())

        self.qd_archive.add_candidate(self.skeleton_code, eval_res["fitness"], eval_res["behavior"])

    def step_generation(self) -> dict[str, Any]:
        """Execute 1 evolutionary generation step across island demes and QD archives."""
        # 1. Select active island
        island_idx = int(self.rng.integers(0, self.num_islands))
        active_island = self.islands[island_idx]

        # 2. Select parent from island queue using regularized tournament (Real et al. 2019)
        sample_size = min(4, len(active_island))
        sampled_indices = self.rng.choice(len(active_island), size=sample_size, replace=False)
        tournament_parents = [active_island[idx] for idx in sampled_indices]
        parent = max(tournament_parents, key=lambda p: p["score"])

        # 3. Apply AST mutations
        mutated_code = self.mutator.mutate_ast_code(parent["code"])
        compiled_fn = self.mutator.compile_executable_function(mutated_code)
        code_length = self.mutator.compute_parsimony_score(mutated_code)

        # 4. Functional Equivalence Checking (FEC)
        new_signature = self.fec_evaluator.evaluate_signature(compiled_fn)
        is_equivalent = any(
            self.fec_evaluator.are_equivalent(new_signature, candidate["signature"])
            for candidate in self.population_queue
        )

        if is_equivalent:
            # Skip redundant simulation ticks for equivalent ASTs
            return {"status": "fec_pruned", "code": mutated_code}

        # 5. Multi-agent oracle evaluation under EFE mode
        eval_res = self.oracle_evaluator.evaluate_candidate(compiled_fn, code_length)
        child_entry = {
            "code": mutated_code,
            "score": eval_res["fitness"],
            "behavior": eval_res["behavior"],
            "signature": new_signature,
            "age": 0,
        }

        # 6. Update QD MAP-Elites and Novelty Archives
        self.qd_archive.add_candidate(mutated_code, eval_res["fitness"], eval_res["behavior"])
        self.novelty_archive.add_if_novel(
            eval_res["behavior"], [cand["behavior"] for cand in self.population_queue]
        )

        # 7. Regularized Aging Population Queue Update (FIFO Purge)
        self.population_queue.append(child_entry)
        active_island.append(child_entry)

        if len(self.population_queue) > self.population_capacity:
            self.population_queue.pop(0)
        if len(active_island) > max(4, self.population_capacity // self.num_islands):
            active_island.pop(0)

        # 8. Advance physical reservoir dynamics in rho_ext
        self.reservoir.step_reservoir(eval_res["behavior"][:2])

        return {
            "status": "success",
            "fitness": eval_res["fitness"],
            "coverage": self.qd_archive.coverage(),
            "code": mutated_code,
        }

    def compile_elite_to_skill_artifact(self) -> SkillArtifact:
        """Compile top-scoring QD elite policy into a persistent SkillArtifact in c.m_procedural."""
        if not self.qd_archive.solutions:
            best_code = self.skeleton_code
            best_score = 0.0
        else:
            best_entry = max(self.qd_archive.solutions.values(), key=lambda sol: sol["score"])
            best_code = best_entry["code"]
            best_score = best_entry["score"]

        return SkillArtifact(
            skill_id="alphaevolve_elite_policy",
            description=f"Evolved AST policy candidate (fitness={best_score:.4f})",
            preconditions={"min_reserve": 0.1, "min_world_mu": 0.0},
            macro_policy=[
                {
                    "step": 0,
                    "action": "EXECUTE_AST_POLICY",
                    "code_str": best_code,
                }
            ],
            confidence=float(np.clip(0.5 + 0.1 * best_score, 0.1, 0.99)),
            gaussian_weights=np.ones(4),
            stiffness_wave_speed=1.0,
        )

    def export_dual_persistence_snapshot(self, output_path: str | Path) -> None:
        """Export dual persistence YAML snapshot (Rule 011)."""
        snapshot = {
            "version": "1.0.0",
            "front": "Wave 5 Front 13 — Evolutionary Algorithm Discovery",
            "qd_coverage": self.qd_archive.coverage(),
            "qd_precision": self.qd_archive.precision(),
            "population_size": len(self.population_queue),
            "novelty_archive_size": len(self.novelty_archive.archive),
            "top_elite_score": (
                max([sol["score"] for sol in self.qd_archive.solutions.values()])
                if self.qd_archive.solutions
                else 0.0
            ),
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(snapshot, f, sort_keys=False)
