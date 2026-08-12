"""HYPOSTASES — Front 08 Causal World Model Data Types & Enums.

Spec Ref: docs/WAVE_2_FRONT_08/front_08_causal_world_models_spec.md
Defines data structures for Structural Causal Models (SCMs), Pearl's 3-rung hierarchy,
interventions (do-calculus), counterfactual queries, NOTEARS adjacency matrices, and Relational SCMs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum

import numpy as np


class CausalRung(IntEnum):
    """Pearl's 3-Rung Causal Hierarchy."""

    RUNG_1_ASSOCIATION = 1  # P(Y | X) - Seeing / Observation
    RUNG_2_INTERVENTION = 2  # P(Y | do(X)) - Doing / Interventional Graph Surgery
    RUNG_3_COUNTERFACTUAL = 3  # P(Y_x' | X=x, Y=y) - Imagining / Counterfactual Abduction


class VariableType(Enum):
    """Classification of SCM Variables."""

    ENDOGENOUS = "endogenous"  # Variable determined by internal mechanism function f_i
    EXOGENOUS = "exogenous"  # Background noise U_i governing mechanism variation


@dataclass
class CausalNode:
    """Represents a variable node in an SCM DAG."""

    name: str
    var_type: VariableType = VariableType.ENDOGENOUS
    domain_min: float = -np.inf
    domain_max: float = np.inf
    description: str = ""


@dataclass
class CausalEdge:
    """Represents a directed causal edge between a parent variable and a child variable."""

    source: str  # Cause node name
    target: str  # Effect node name
    weight: float = 1.0  # Structural coupling weight / coefficient
    mechanism_label: str = "linear"


@dataclass
class StructuralEquation:
    """Represents functional assignment V_i = f_i(PA_i, U_i).

    Allows functional evaluation via a parameterized callable or coefficient vector.
    """

    target_var: str
    parent_vars: list[str] = field(default_factory=list)
    coefficients: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    custom_func: Callable[[dict[str, float], float], float] | None = None
    is_intervention: bool = False

    def evaluate(self, parent_values: dict[str, float], noise: float = 0.0) -> float:
        """Evaluates the mechanism function for target variable given parent values and noise U_i."""
        if self.is_intervention:
            return float(self.intercept)
        if self.custom_func is not None:
            return self.custom_func(parent_values, noise)

        val = self.intercept + noise
        for parent, weight in self.coefficients.items():
            val += weight * parent_values.get(parent, 0.0)
        return float(val)


@dataclass
class Intervention:
    """Represents do(X = x) interventional graph surgery."""

    target_values: dict[str, float] = field(default_factory=dict)  # Variable -> target value
    cost: float = 0.0  # Direct execution cost of interventional override


@dataclass
class CounterfactualQuery:
    """Represents Rung 3 counterfactual query P(Y_{X=x'} | X=x, Y=y)."""

    observed_evidence: dict[str, float] = field(default_factory=dict)  # Fact observed: X=x, Y=y
    intervention_override: dict[str, float] = field(
        default_factory=dict
    )  # Hypothetical action: X=x'
    target_variables: list[str] = field(default_factory=list)  # Outcome variables Y to evaluate


@dataclass
class RSCMSchema:
    """Relational Structural Causal Model (RSCM) Schema definition.

    Synthesizes Ejaz & Bareinboim (2026):
    Entities E, Relations R, and Attributes A with template mechanisms O.A = f_{O.A}(...).
    """

    entities: list[str] = field(default_factory=list)
    relations: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (E_source, R_name, E_target)
    attribute_templates: dict[str, list[str]] = field(
        default_factory=dict
    )  # Entity/Relation -> [Attributes]
    template_equations: dict[str, StructuralEquation] = field(default_factory=dict)
