"""AST Mutator & Functional Equivalence Checking (FEC) for AlphaEvolve Engine.

References:
- Real et al. (2020) AutoML-Zero: Evolving Machine Learning Algorithms From Scratch (ICML)
- Romera-Paredes et al. (2023) Mathematical discoveries from program search with LLMs (Nature - FunSearch)
- Novikov et al. (2025) AlphaEvolve: A coding agent for scientific and algorithmic discovery (arXiv)
"""

import ast
import random
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np


class FECEvaluator:
    """Functional Equivalence Checking (FEC) Evaluator (Real et al. 2020).

    Evaluates candidate AST algorithms on a fixed set of K synthetic probe state vectors
    to identify and prune semantically redundant program mutations prior to costly simulation.
    """

    def __init__(self, num_probes: int = 16, tolerance: float = 1e-6, seed: int = 42) -> None:
        self.num_probes = int(num_probes)
        self.tolerance = float(tolerance)
        self.rng = np.random.default_rng(seed)
        # Synthetic state probes in R^4 representing (c, w, g, rho_ext) feature vectors
        self.synthetic_probes = self.rng.uniform(-2.0, 2.0, size=(self.num_probes, 4))

    def evaluate_signature(self, program_fn: Callable[[np.ndarray], float]) -> np.ndarray:
        """Compute the evaluation signature vector s(x) across synthetic probes."""
        signature = np.zeros(self.num_probes)
        for i in range(self.num_probes):
            try:
                signature[i] = float(program_fn(self.synthetic_probes[i]))
            except Exception:
                signature[i] = np.nan
        return signature

    def are_equivalent(self, signature_a: np.ndarray, signature_b: np.ndarray) -> bool:
        """Check if two program signatures are functionally equivalent within tolerance."""
        if np.any(np.isnan(signature_a)) or np.any(np.isnan(signature_b)):
            return False
        diff = np.mean((signature_a - signature_b) ** 2)
        return bool(diff < self.tolerance)


class ASTMutator:
    """AST Code Mutator & Program Synthesizer (Real et al. 2020, Romera-Paredes et al. 2023).

    Performs AST structural mutations (operator swapping, constant tuning, node substitution)
    and constructs best-shot prompts for generative model candidate creation.
    """

    OPERATOR_MAP: ClassVar[dict[type[ast.AST], list[ast.AST]]] = {
        ast.Add: [ast.Sub(), ast.Mult()],
        ast.Sub: [ast.Add(), ast.Mult()],
        ast.Mult: [ast.Add(), ast.Sub()],
    }

    def __init__(self, mutation_rate: float = 0.15, seed: int | None = 42) -> None:
        self.mutation_rate = mutation_rate
        self.rng = random.Random(seed)

    def compute_parsimony_score(self, code_str: str) -> int:
        """Compute Kolmogorov complexity proxy (character length L(x))."""
        return len(code_str.strip())

    def mutate_ast_code(self, code_str: str) -> str:
        """Apply AST structural mutations to python code string."""
        try:
            tree = ast.parse(code_str)
        except SyntaxError:
            return code_str

        transformer = _ASTMutationVisitor(self.mutation_rate, self.rng)
        mutated_tree = transformer.visit(tree)
        ast.fix_missing_locations(mutated_tree)

        try:
            return ast.unparse(mutated_tree)
        except Exception:
            return code_str

    def compile_executable_function(
        self, code_str: str, function_name: str = "candidate_policy"
    ) -> Callable[[np.ndarray], float]:
        """Compile AST code string into an executable Python function."""
        local_scope: dict[str, Any] = {}
        # Safe execution namespace with math and numpy
        global_scope: dict[str, Any] = {
            "np": np,
            "abs": abs,
            "min": min,
            "max": max,
        }
        exec(code_str, global_scope, local_scope)
        if function_name in local_scope:
            return local_scope[function_name]

        # Fallback if function name differs
        for obj in local_scope.values():
            if callable(obj):
                return obj

        def fallback_fn(state: np.ndarray) -> float:
            return float(np.sum(state))

        return fallback_fn

    def construct_best_shot_prompt(
        self, sampled_programs: list[dict[str, Any]], skeleton_code: str
    ) -> str:
        """Construct structured LLM prompt for candidate code generation (Romera-Paredes et al. 2023)."""
        prompt_lines = [
            "# FunSearch / AlphaEvolve Best-Shot Generative Program Synthesis Prompt",
            "# Given the skeleton and elite parent programs, generate an improved candidate function.",
            "",
            "## Base Skeleton:",
            "```python",
            skeleton_code,
            "```",
            "",
        ]

        # Sort parent programs ascending by performance score
        sorted_parents = sorted(sampled_programs, key=lambda p: p.get("score", 0.0))
        for idx, parent in enumerate(sorted_parents):
            prompt_lines.extend(
                [
                    f"## Parent Candidate {idx + 1} (Score: {parent.get('score', 0.0):.4f}):",
                    "```python",
                    parent.get("code", ""),
                    "```",
                    "",
                ]
            )

        prompt_lines.extend(
            [
                "## Instructions:",
                "Write an improved Python function `candidate_policy(state)` that optimizes multi-agent utility,",
                "reduces computational complexity, and respects state invariant boundaries sigma = (c, w, g, rho_ext).",
                "",
                "```python",
                "def candidate_policy(state: np.ndarray) -> float:",
            ]
        )

        return "\n".join(prompt_lines)


class _ASTMutationVisitor(ast.NodeTransformer):
    """Internal AST transformer for operator and constant mutation."""

    def __init__(self, mutation_rate: float, rng: random.Random) -> None:
        self.mutation_rate = mutation_rate
        self.rng = rng

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        if self.rng.random() < self.mutation_rate:
            op_type = type(node.op)
            if op_type in ASTMutator.OPERATOR_MAP:
                new_op = self.rng.choice(ASTMutator.OPERATOR_MAP[op_type])
                node.op = new_op
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        if isinstance(node.value, int | float) and self.rng.random() < self.mutation_rate:
            delta = self.rng.uniform(-0.5, 0.5)
            if isinstance(node.value, int):
                node.value = max(0, node.value + int(np.sign(delta)))
            else:
                node.value = float(node.value + delta)
        return node
