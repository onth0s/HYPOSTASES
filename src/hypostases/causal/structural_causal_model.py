"""HYPOSTASES — Structural Causal Model (SCM) Engine.

Spec Ref: docs/WAVE_2_FRONT_08/front_08_causal_world_models_spec.md
Synthesizes Pearl (2016, 2019) and Bareinboim et al. (2015, 2016).
Implements SCM DAG topologies, structural equations, graph surgery for do-calculus,
d-separation testing, and the 3-step Abduction-Action-Prediction counterfactual engine.
"""

from __future__ import annotations

import copy

import numpy as np

from hypostases.causal.causal_types import (
    CausalEdge,
    CausalNode,
    CounterfactualQuery,
    Intervention,
    StructuralEquation,
    VariableType,
)


class StructuralCausalModel:
    """Core Structural Causal Model (SCM) representing M = <U, V, F, P(U)>."""

    def __init__(self, name: str = "default_scm") -> None:
        self.name = name
        self.nodes: dict[str, CausalNode] = {}
        self.edges: list[CausalEdge] = []
        self.equations: dict[str, StructuralEquation] = {}
        self.exogenous_priors: dict[str, tuple[float, float]] = {}  # var -> (mean, std)

    def add_node(
        self,
        name: str,
        var_type: VariableType = VariableType.ENDOGENOUS,
        domain_min: float = -np.inf,
        domain_max: float = np.inf,
        exogenous_mean: float = 0.0,
        exogenous_std: float = 1.0,
    ) -> CausalNode:
        """Adds a variable node to the SCM DAG."""
        node = CausalNode(
            name=name, var_type=var_type, domain_min=domain_min, domain_max=domain_max
        )
        self.nodes[name] = node
        self.exogenous_priors[name] = (exogenous_mean, exogenous_std)
        return node

    def add_edge(
        self, source: str, target: str, weight: float = 1.0, label: str = "linear"
    ) -> None:
        """Adds a directed causal edge (cause -> effect) to the SCM DAG."""
        if source not in self.nodes:
            self.add_node(source)
        if target not in self.nodes:
            self.add_node(target)
        edge = CausalEdge(source=source, target=target, weight=weight, mechanism_label=label)
        self.edges.append(edge)

        # Update structural equation parent list & coefficients
        if target not in self.equations:
            self.equations[target] = StructuralEquation(target_var=target)
        eq = self.equations[target]
        if source not in eq.parent_vars:
            eq.parent_vars.append(source)
        eq.coefficients[source] = weight

    def set_equation(self, target: str, equation: StructuralEquation) -> None:
        """Registers a explicit structural equation V_i = f_i(PA_i, U_i)."""
        self.equations[target] = equation

    def get_parents(self, node_name: str) -> list[str]:
        """Returns immediate parent causes for a given variable node."""
        if node_name in self.equations:
            return self.equations[node_name].parent_vars
        return [e.source for e in self.edges if e.target == node_name]

    def get_children(self, node_name: str) -> list[str]:
        """Returns immediate child effects for a given variable node."""
        return [e.target for e in self.edges if e.source == node_name]

    def topological_sort(self) -> list[str]:
        """Returns topological ordering of DAG nodes."""
        in_degree: dict[str, int] = {n: 0 for n in self.nodes}
        for edge in self.edges:
            in_degree[edge.target] += 1

        queue = [n for n, deg in in_degree.items() if deg == 0]
        order: list[str] = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self.get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            # Fallback for cyclic graphs / robust ordering
            return list(self.nodes.keys())
        return order

    def evaluate_observational(
        self, exogenous_noise: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Rung 1 (Association): Computes observational state prediction P(V)."""
        if exogenous_noise is None:
            exogenous_noise = {
                n: float(np.random.normal(m, s)) for n, (m, s) in self.exogenous_priors.items()
            }

        state: dict[str, float] = {}
        for node in self.topological_sort():
            u = exogenous_noise.get(node, 0.0)
            if node in self.equations:
                state[node] = self.equations[node].evaluate(state, u)
            else:
                state[node] = u
            # Clamp to domain bounds
            node_obj = self.nodes[node]
            state[node] = float(np.clip(state[node], node_obj.domain_min, node_obj.domain_max))

        return state

    def surgerize(self, intervention: Intervention) -> StructuralCausalModel:
        """Rung 2 (Intervention): Performs graph surgery for do(X = x).

        Severs incoming edges to manipulated variables X and replaces their equations with constant X = x.
        """
        sub_scm = copy.deepcopy(self)
        for var, val in intervention.target_values.items():
            # Remove incoming edges to var
            sub_scm.edges = [e for e in sub_scm.edges if e.target != var]
            # Override equation to constant value
            sub_scm.equations[var] = StructuralEquation(
                target_var=var, parent_vars=[], intercept=val, is_intervention=True
            )
        return sub_scm

    def evaluate_intervention(
        self, intervention: Intervention, exogenous_noise: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Rung 2 (Intervention): Computes interventional state distribution P(V | do(X = x))."""
        sub_scm = self.surgerize(intervention)
        return sub_scm.evaluate_observational(exogenous_noise=exogenous_noise)

    def evaluate_counterfactual(self, query: CounterfactualQuery) -> dict[str, float]:
        """Rung 3 (Counterfactual): Executes 3-step Abduction-Action-Prediction cycle.

        1. Abduction: Estimate posterior noise U given observed evidence (X=x, Y=y).
        2. Action: Surgerize model under hypothetical intervention do(X = x').
        3. Prediction: Evaluate target variables Y in submodel using updated noise U.
        """
        # Step 1: Abduction - estimate exogenous noise given evidence
        inferred_noise: dict[str, float] = {}
        for n, (m, _) in self.exogenous_priors.items():
            inferred_noise[n] = m

        for n, obs_val in query.observed_evidence.items():
            if n in self.equations:
                eq = self.equations[n]
                pred_val_no_noise = eq.evaluate(query.observed_evidence, noise=0.0)
                inferred_noise[n] = float(obs_val - pred_val_no_noise)

        # Step 2: Action - graph surgery under hypothetical override
        interv = Intervention(target_values=query.intervention_override)
        sub_scm = self.surgerize(interv)

        # Step 3: Prediction - forward evaluate using abduced noise
        counterfactual_state = sub_scm.evaluate_observational(exogenous_noise=inferred_noise)

        if query.target_variables:
            return {
                k: counterfactual_state[k]
                for k in query.target_variables
                if k in counterfactual_state
            }
        return counterfactual_state

    def compute_ett(
        self,
        treatment_var: str,
        treatment_val: float,
        baseline_val: float,
        outcome_var: str,
        evidence: dict[str, float],
    ) -> float:
        """Computes Effect of Treatment on the Treated (ETT): E(Y_{X=treatment_val} - Y_{X=baseline_val} | X=baseline_val)."""
        query_interv = CounterfactualQuery(
            observed_evidence=evidence,
            intervention_override={treatment_var: treatment_val},
            target_variables=[outcome_var],
        )
        cf_res = self.evaluate_counterfactual(query_interv)
        obs_res = evidence.get(outcome_var, 0.0)
        return float(cf_res.get(outcome_var, 0.0) - obs_res)

    def is_d_separated(self, x: set[str], y: set[str], z: set[str]) -> bool:
        """Checks if sets X and Y are d-separated by set Z in the DAG (Pearl 2000)."""
        # Simple graph reachability / Bayes-ball implementation
        ancestors_z = set(z)
        added = True
        while added:
            new_ancestors = set()
            for n in ancestors_z:
                new_ancestors.update(self.get_parents(n))
            added = not new_ancestors.issubset(ancestors_z)
            ancestors_z.update(new_ancestors)

        visited = set()
        queue: list[tuple[str, str]] = [(n, "up") for n in x]

        while queue:
            node, direction = queue.pop(0)
            if (node, direction) in visited:
                continue
            visited.add((node, direction))

            if node in y:
                return False

            if direction == "up":
                if node not in z:
                    for p in self.get_parents(node):
                        queue.append((p, "up"))
                    for c in self.get_children(node):
                        queue.append((c, "down"))
            elif direction == "down":
                if node not in z:
                    for c in self.get_children(node):
                        queue.append((c, "down"))
                if node in ancestors_z:
                    for p in self.get_parents(node):
                        queue.append((p, "up"))

        return True
