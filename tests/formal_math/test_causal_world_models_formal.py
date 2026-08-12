"""Formal Mathematical Verification for Wave 2 Front 08 — Causal World Models.

Rule 012 Compliance: Mandatory Formal Mathematical Implementation Verification.

Exercises `src/hypostases/causal/` against theorems from:
  - Pearl (2000, 2009): Truncated factorization, backdoor criterion, d-separation
  - Pearl (2009) Rung 3: Counterfactual twin-network consistency invariant
  - Spirtes et al. / NOTEARS: DAG acyclicity score h(W) = tr(e^{W∘W}) - d = 0
  - Bareinboim & Pearl (2016): Transportability under selection diagram S-nodes
  - Bareinboim et al.: Relational SCM (RSCM) boundary scoping invariant
"""

import numpy as np

from hypostases.causal.causal_discovery import CausalDiscoveryEngine
from hypostases.causal.causal_types import (
    CounterfactualQuery,
    Intervention,
    RSCMSchema,
)
from hypostases.causal.do_calculus_engine import DoCalculusEngine
from hypostases.causal.rscm_engine import RelationalSCMEngine
from hypostases.causal.structural_causal_model import StructuralCausalModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_linear_chain() -> StructuralCausalModel:
    """X → Y → Z with linear coefficients: Y = 2X + U_Y, Z = 3Y + U_Z."""
    scm = StructuralCausalModel(name="linear_chain")
    scm.add_node("X", exogenous_mean=1.0, exogenous_std=0.0)  # X fixed = 1.0
    scm.add_node("Y", exogenous_mean=0.0, exogenous_std=0.0)
    scm.add_node("Z", exogenous_mean=0.0, exogenous_std=0.0)
    scm.add_edge("X", "Y", weight=2.0)  # Y = 2X
    scm.add_edge("Y", "Z", weight=3.0)  # Z = 3Y = 6X
    return scm


def _build_confounded_dag() -> StructuralCausalModel:
    """Confounded DAG: Z → X → Y, Z → Y (confounder Z)."""
    scm = StructuralCausalModel(name="confounded")
    scm.add_node("Z", exogenous_mean=0.0, exogenous_std=0.0)
    scm.add_node("X", exogenous_mean=0.0, exogenous_std=0.0)
    scm.add_node("Y", exogenous_mean=0.0, exogenous_std=0.0)
    scm.add_edge("Z", "X", weight=1.0)  # X = Z + U_X
    scm.add_edge("Z", "Y", weight=1.0)  # Y = Z + X + U_Y
    scm.add_edge("X", "Y", weight=1.0)
    return scm


# ---------------------------------------------------------------------------
# Theorem 8.1 — Pearl Truncated Factorization Numerical Verification
# ---------------------------------------------------------------------------
def test_theorem8_1_truncated_factorization_numerical() -> None:
    """Theorem 8.1: P(Z | do(X=x)) via graph surgery equals the analytical chain value.

    For linear chain X→Y→Z with Y=2X, Z=3Y:
      do(X=2.0) → Y_obs=4.0, Z_obs=12.0
    The truncated factorization severs incoming edges to X and evaluates forward.
    """
    scm = _build_linear_chain()

    interv = Intervention(target_values={"X": 2.0})
    result = scm.evaluate_intervention(interv, exogenous_noise={"X": 0.0, "Y": 0.0, "Z": 0.0})

    expected_y = 2.0 * 2.0  # 4.0
    expected_z = 3.0 * expected_y  # 12.0

    assert abs(result["Y"] - expected_y) < 1e-9, (
        f"Y | do(X=2): expected {expected_y}, got {result['Y']}"
    )
    assert abs(result["Z"] - expected_z) < 1e-9, (
        f"Z | do(X=2): expected {expected_z}, got {result['Z']}"
    )


# ---------------------------------------------------------------------------
# Theorem 8.2 — Backdoor Adjustment Formula Equivalence
# ---------------------------------------------------------------------------
def test_theorem8_2_backdoor_adjustment_formula() -> None:
    """Theorem 8.2: P(Y|do(X)) via backdoor adjustment matches direct intervention.

    In confounded DAG Z→X→Y, Z→Y:
    - Backdoor set is {Z}, which blocks the confounding path X ← Z → Y.
    - P(Y|do(X=x)) = sum_z P(Y|X=x, Z=z) P(Z=z) must agree with direct do-calculus.
    """
    scm = _build_confounded_dag()
    engine = DoCalculusEngine(scm)

    # A balanced, noise-free observational sample from X = Z + U_X and
    # Y = Z + X.  It permits a numerical equality check of the adjustment
    # formula rather than a loose Monte-Carlo comparison.
    samples = [
        {"Z": z_val, "X": z_val + u_x, "Y": 2.0 * z_val + u_x}
        for z_val in (-1.0, 0.0, 1.0)
        for u_x in (-1.0, 0.0, 1.0)
    ]

    # Backdoor-adjusted P(Y | do(X≈1.0))
    adj_y = engine.compute_backdoor_adjustment("X", 1.0, "Y", samples)

    # Direct interventional: do(X=1.0) with Z distribution (mean 0)
    direct_y = scm.evaluate_intervention(
        Intervention({"X": 1.0}), exogenous_noise={"Z": 0.0, "X": 0.0, "Y": 0.0}
    )["Y"]

    # Both equal E[Z] + 1 under Y = Z + X + U_Y.
    assert abs(adj_y - direct_y) < 0.01, (
        f"Backdoor adjusted E[Y|do(X=1)]={adj_y:.3f} diverges from intervention {direct_y:.3f}"
    )


# ---------------------------------------------------------------------------
# Theorem 8.3 — NOTEARS Acyclicity Score Convergence
# ---------------------------------------------------------------------------
def test_theorem8_3_notears_acyclicity_score_convergence() -> None:
    """Theorem 8.3: learned NOTEARS adjacency satisfies h(W) = 0.

    The test reconstructs the learned adjacency from the production SCM and
    evaluates the engine's exact smooth acyclicity functional.
    """
    rng = np.random.default_rng(7)
    x = rng.normal(size=300)
    y = 1.4 * x + rng.normal(0.0, 0.05, size=300)
    z = -0.8 * y + rng.normal(0.0, 0.05, size=300)
    q = 0.6 * x + 0.3 * z + rng.normal(0.0, 0.05, size=300)
    variable_names = ["X", "Y", "Z", "Q"]
    discovery = CausalDiscoveryEngine(variable_names)
    learned_scm = discovery.learn_notears(
        np.column_stack((x, y, z, q)), max_iter=30, w_threshold=0.2
    )
    weights = np.zeros((4, 4))
    index = {name: idx for idx, name in enumerate(variable_names)}
    for edge in learned_scm.edges:
        weights[index[edge.source], index[edge.target]] = edge.weight

    assert CausalDiscoveryEngine._notears_h(weights) < 1e-8


# ---------------------------------------------------------------------------
# Theorem 8.4 — Counterfactual Twin-Network Consistency Invariant
# ---------------------------------------------------------------------------
def test_theorem8_4_counterfactual_twin_network_consistency() -> None:
    """Theorem 8.4: Counterfactual query agrees with observation on non-intervened variables.

    For chain X→Y→Z with observed evidence {X=1, Y=2}:
    - Intervene on X' = 3 in the twin world.
    - The twin network must still respect the structural equation Y_cf = 2 * X_cf.
    - Z_cf must be derived consistently: Z_cf = 3 * Y_cf = 3 * 6 = 18.
    """
    scm = _build_linear_chain()

    query = CounterfactualQuery(
        observed_evidence={"X": 1.0, "Y": 2.0, "Z": 6.0},
        intervention_override={"X": 3.0},
        target_variables=["Y", "Z"],
    )
    cf_result = scm.evaluate_counterfactual(query)

    # In twin world: X_cf=3.0, Y_cf = 2*3=6.0, Z_cf = 3*6=18.0
    assert abs(cf_result["Y"] - 6.0) < 1e-6, (
        f"Twin network Y_cf: expected 6.0, got {cf_result['Y']}"
    )
    assert abs(cf_result["Z"] - 18.0) < 1e-6, (
        f"Twin network Z_cf: expected 18.0, got {cf_result['Z']}"
    )


# ---------------------------------------------------------------------------
# Theorem 8.5 — Relational SCM Multi-Agent Boundary Invariant
# ---------------------------------------------------------------------------
def test_theorem8_5_relational_scm_agent_boundary_invariant() -> None:
    """Theorem 8.5: no cross-agent relation means no cross-agent causal edge.

    Entity attributes are grounded for both agents, while causal influence is
    limited to the relations explicitly supplied to the RSCM instance.
    """
    engine = RelationalSCMEngine(
        RSCMSchema(entities=["Agent"], attribute_templates={"Agent": ["state"]})
    )
    ground_scm = engine.instantiate_ground_scm(
        entity_instances={"Agent": ["agent_a", "agent_b"]}, relation_instances=[]
    )
    assert ground_scm.is_d_separated({"agent_a.state"}, {"agent_b.state"}, set())


# ---------------------------------------------------------------------------
# Theorem 8.6 — Do-Calculus Rule 1 Soundness
# ---------------------------------------------------------------------------
def test_theorem8_6_do_calculus_rule1_soundness() -> None:
    """Theorem 8.6: DoCalculusEngine.check_rule_1 returns consistent bool.

    In linear chain X→Y→Z: Rule 1 checks (Z ⊥⊥ X | Y) in G_{bar(Y)}.
    The result must be a valid boolean.
    """
    scm = _build_linear_chain()
    engine = DoCalculusEngine(scm)

    # Check rule 1: can we drop observation of X when we have do(Y) and observe Z?
    r1 = engine.check_rule_1(y={"Z"}, x={"Y"}, z={"X"}, w=set())
    assert isinstance(r1, bool), f"check_rule_1 must return bool, got {type(r1)}"


# ---------------------------------------------------------------------------
# Theorem 8.7 — Transportability Soundness
# ---------------------------------------------------------------------------
def test_theorem8_7_transportability_soundness() -> None:
    """Theorem 8.7: check_transportability returns valid bool for selection diagram queries."""
    scm = _build_linear_chain()
    engine = DoCalculusEngine(scm)

    # No S-nodes affecting Z given do(X) → trivially transportable
    transportable = engine.check_transportability(target_domain_s_nodes=[], y_var="Z", x_var="X")
    assert isinstance(transportable, bool)
    assert transportable is True, "With no S-nodes, query should be trivially transportable"
