"""Game-Theoretic Oracle Evaluator for AlphaEvolve Engine.

Evaluates candidate AST programs against multi-agent equilibrium, incentive compatibility regret (R_IC),
resource scarcity (kappa), and Friston Expected Free Energy (EFE) active sensing (Rule 009).
"""

from collections.abc import Callable
from typing import Any

import numpy as np


class GameTheoreticOracleEvaluator:
    """Multi-Agent Simulation Harness Evaluation Oracle (AlphaEvolve Evaluator).

    Treats the HYPOSTASES simulation harness sigma = (c, w, g, rho_ext) as an evaluation oracle,
    computing multi-criteria game-theoretic fitness and behavioral feature vectors.
    """

    def __init__(
        self,
        simulation_ticks: int = 20,
        efe_mode: bool = True,
        efe_beta: float = 0.2,
        lambda_ic: float = 10.0,
        lambda_scarcity: float = 5.0,
    ) -> None:
        self.simulation_ticks = simulation_ticks
        self.efe_mode = efe_mode
        self.efe_beta = efe_beta
        self.lambda_ic = lambda_ic
        self.lambda_scarcity = lambda_scarcity

    def evaluate_candidate(
        self,
        program_fn: Callable[[np.ndarray], float],
        code_length: int,
        initial_state: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Evaluate candidate program against simulation harness oracle.

        Returns dictionary containing multi-criteria fitness score F(pi) and behavioral vector b(pi).
        """
        if initial_state is None:
            initial_state = np.array([1.0, 0.5, 2.0, 0.8])

        state = initial_state.copy()
        pragmatic_utility = 0.0
        epistemic_utility = 0.0
        ic_regret = 0.0
        scarcity_depletion = 0.0

        for _t in range(self.simulation_ticks):
            try:
                # Execute candidate policy function
                action_val = float(program_fn(state))
            except Exception:
                # Execution error penalty
                return {
                    "fitness": -1000.0,
                    "behavior": np.array([float(code_length), 1.0, 5.0]),
                    "r_ic": 1.0,
                    "delta_kappa": 5.0,
                    "pragmatic_utility": 0.0,
                    "epistemic_utility": 0.0,
                }

            # State transition update under candidate action
            state[0] += 0.1 * action_val  # c state
            state[1] = np.clip(state[1] + 0.05 * np.sin(action_val), -3.0, 3.0)  # w world model
            state[2] += 0.02 * action_val  # g goal state
            state[3] = max(0.01, state[3] - 0.01 * (action_val**2))  # rho_ext scarcity kappa

            # Compute step utilities
            u_prag = float(state[2] + 0.5 * action_val)
            u_epist = float(1.0 / (np.var(state) + 1e-4))

            # Unilateral deviation incentive compatibility regret check
            unilateral_dev_utility = float(state[2] + 0.5 * (action_val + 0.5))
            r_ic_step = max(0.0, unilateral_dev_utility - u_prag)

            # Scarcity depletion delta kappa
            scarcity_step = float(0.01 * (action_val**2))

            pragmatic_utility += u_prag
            epistemic_utility += u_epist
            ic_regret += r_ic_step
            scarcity_depletion += scarcity_step

        # Average utilities across simulation ticks
        avg_pragmatic = pragmatic_utility / self.simulation_ticks
        avg_epistemic = epistemic_utility / self.simulation_ticks
        avg_ic_regret = ic_regret / self.simulation_ticks
        avg_scarcity = scarcity_depletion / self.simulation_ticks

        # Multi-criteria fitness formulation (Rule 009 EFE integration)
        if self.efe_mode:
            combined_utility = (1.0 - self.efe_beta) * avg_pragmatic + self.efe_beta * avg_epistemic
        else:
            combined_utility = avg_pragmatic

        fitness_score = (
            combined_utility - self.lambda_ic * avg_ic_regret - self.lambda_scarcity * avg_scarcity
        )

        # Behavioral feature vector b(pi) = [C_time, R_IC, Delta_kappa]
        behavior_vector = np.array([float(code_length), avg_ic_regret, avg_scarcity])

        return {
            "fitness": float(fitness_score),
            "behavior": behavior_vector,
            "r_ic": float(avg_ic_regret),
            "delta_kappa": float(avg_scarcity),
            "pragmatic_utility": float(avg_pragmatic),
            "epistemic_utility": float(avg_epistemic),
        }
