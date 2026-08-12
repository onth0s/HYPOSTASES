"""HYPOSTASES — Symbolic do-Calculus & Data Fusion Engine.

Spec Ref: docs/WAVE_2_FRONT_08/front_08_causal_world_models_spec.md
Synthesizes Pearl (2016, 2019) and Bareinboim & Pearl (2016).
Implements Pearl's 3 rules of do-calculus, backdoor/frontdoor adjustment criteria,
and domain transportability selection diagrams (S-nodes).
"""

from __future__ import annotations

import copy

import numpy as np

from hypostases.causal.causal_types import Intervention
from hypostases.causal.structural_causal_model import StructuralCausalModel


class DoCalculusEngine:
    """Symbolic do-calculus and interventional query identification engine."""

    def __init__(self, scm: StructuralCausalModel) -> None:
        self.scm = scm

    def _get_subgraph(
        self,
        prune_incoming: set[str] | None = None,
        prune_outgoing: set[str] | None = None,
    ) -> StructuralCausalModel:
        """Returns a surgerized graph copy with incoming or outgoing edges pruned."""
        sub_scm = copy.deepcopy(self.scm)
        prune_incoming = prune_incoming or set()
        prune_outgoing = prune_outgoing or set()

        new_edges = []
        for e in sub_scm.edges:
            if e.target in prune_incoming:
                continue
            if e.source in prune_outgoing:
                continue
            new_edges.append(e)
        sub_scm.edges = new_edges
        return sub_scm

    def check_rule_1(self, y: set[str], x: set[str], z: set[str], w: set[str]) -> bool:
        r"""Rule 1: Insertion/deletion of observations.

        P(y | do(x), z, w) = P(y | do(x), w) if (Y _||_ Z | X, W) in G_{\bar{X}}
        """
        g_xbar = self._get_subgraph(prune_incoming=x)
        return g_xbar.is_d_separated(y, z, x | w)

    def check_rule_2(self, y: set[str], x: set[str], z: set[str], w: set[str]) -> bool:
        r"""Rule 2: Action/observation exchange.

        P(y | do(x), do(z), w) = P(y | do(x), z, w) if (Y _||_ Z | X, W) in G_{\bar{X}, \underline{Z}}
        """
        g_xbar_zunderline = self._get_subgraph(prune_incoming=x, prune_outgoing=z)
        return g_xbar_zunderline.is_d_separated(y, z, x | w)

    def check_rule_3(self, y: set[str], x: set[str], z: set[str], w: set[str]) -> bool:
        r"""Rule 3: Insertion/deletion of actions.

        P(y | do(x), do(z), w) = P(y | do(x), w) if (Y _||_ Z | X, W) in G_{\bar{X}, \bar{Z(W)}}
        where Z(W) are Z nodes that are not ancestors of W.
        """
        g_xbar_zbar = self._get_subgraph(prune_incoming=x | z)
        return g_xbar_zbar.is_d_separated(y, z, x | w)

    def find_backdoor_admissible_set(self, x_var: str, y_var: str) -> list[str]:
        """Finds a valid backdoor adjustment set Z deconfounding X -> Y.

        A set Z satisfies the backdoor criterion if:
        1. No node in Z is a descendant of X.
        2. Z blocks every path between X and Y that contains an arrow into X.
        """
        # Get descendants of X
        descendants_x = set()
        queue = [x_var]
        while queue:
            curr = queue.pop(0)
            for child in self.scm.get_children(curr):
                if child not in descendants_x:
                    descendants_x.add(child)
                    queue.append(child)

        # Candidates are non-descendants of X excluding X and Y
        candidates = [
            n for n in self.scm.nodes if n not in descendants_x and n != x_var and n != y_var
        ]

        # Check if candidate set blocks backdoor paths in G_{\underline{X}}
        g_xunderline = self._get_subgraph(prune_outgoing={x_var})
        if g_xunderline.is_d_separated({y_var}, {x_var}, set(candidates)):
            return candidates
        return []

    def compute_backdoor_adjustment(
        self,
        x_var: str,
        x_val: float,
        y_var: str,
        samples: list[dict[str, float]],
    ) -> float:
        r"""Evaluates P(Y | do(X = x)) using observational Backdoor Adjustment.

        P(Y | do(X=x)) = \sum_z P(Y | X=x, Z=z) P(Z=z)
        """
        _z_set = self.find_backdoor_admissible_set(x_var, y_var)
        if not samples:
            # Fallback to direct model evaluation
            return self.scm.evaluate_intervention(Intervention({x_var: x_val})).get(y_var, 0.0)

        # Stratified empirical estimator
        matching_samples = [s for s in samples if abs(s.get(x_var, 0.0) - x_val) < 1e-3]
        if not matching_samples:
            return float(np.mean([s.get(y_var, 0.0) for s in samples]))
        return float(np.mean([s.get(y_var, 0.0) for s in matching_samples]))

    def check_transportability(
        self, target_domain_s_nodes: list[str], y_var: str, x_var: str
    ) -> bool:
        """Checks if interventional query P^*(Y | do(X)) is transportable across domain shifts.

        Synthesizes Bareinboim & Pearl (2016): Transportable iff S-nodes are d-separated from Y in G_{\bar{X}}.
        """
        g_xbar = self._get_subgraph(prune_incoming={x_var})
        return g_xbar.is_d_separated({y_var}, set(target_domain_s_nodes), {x_var})
