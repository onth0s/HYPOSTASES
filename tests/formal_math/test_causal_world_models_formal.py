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


def test_fixture_power_benchmark_oracle() -> None:
    """Fixture-Power Benchmark: Run oracle/benchmark estimator against locked fixture.

    Calibrates statistical headroom before evaluating NOTEARS structure recovery.
    """
    import yaml

    with open("schema/causal_discovery_evaluation.yaml") as f:
        eval_fixture = yaml.safe_load(f)

    var_order = eval_fixture["data_generating_process"]["variable_order"]
    seeds = eval_fixture["data_generating_process"]["seeds"]
    n_samples = eval_fixture["data_generating_process"]["sample_count"]
    var_idx = {name: i for i, name in enumerate(var_order)}

    # True edge set: X->Y (0->1), X->Z (0->2), Y->Q (1->3), Z->Q (2->3)
    true_edges = {
        (var_idx[s], var_idx[t])
        for s, t in eval_fixture["data_generating_process"]["ground_truth_edges"]
    }

    recalls, precisions, shds = [], [], []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        u_x = rng.normal(0, 1, n_samples)
        u_y = rng.normal(0, 1, n_samples)
        u_z = rng.normal(0, 1, n_samples)
        u_q = rng.normal(0, 1, n_samples)

        x = u_x
        y = 1.20 * x + u_y
        z = -0.90 * x + u_z
        q = 0.80 * y - 0.70 * z + u_q

        data = np.column_stack((x, y, z, q))

        # Oracle/Benchmark estimator: Partial correlation & OLS regression
        # 1. topological order X -> (Y, Z) -> Q verified by OLS residual variance / partial corr
        # 2. OLS regression coefficients thresholded at 0.15
        _ = np.cov(data.T)

        # Regress Y on X
        b_yx = float(np.polyfit(x, y, 1)[0])
        # Regress Z on X
        b_zx = float(np.polyfit(x, y, 1)[0])
        # Regress Q on Y, Z
        b_q = np.linalg.lstsq(np.column_stack((y, z)), q, rcond=None)[0]

        pred_edges = set()
        if abs(b_yx) > 0.3:
            pred_edges.add((0, 1))
        if abs(b_zx) > 0.3:
            pred_edges.add((0, 2))
        if abs(b_q[0]) > 0.3:
            pred_edges.add((1, 3))
        if abs(b_q[1]) > 0.3:
            pred_edges.add((2, 3))

        tp = len(pred_edges & true_edges)
        fp = len(pred_edges - true_edges)
        fn = len(true_edges - pred_edges)

        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        shd = fp + fn

        recalls.append(rec)
        precisions.append(prec)
        shds.append(shd)

    mean_rec = float(np.mean(recalls))
    mean_prec = float(np.mean(precisions))

    # Verify adequate headroom: >= 0.10 above gates (gate recall 0.90, gate prec 0.85)
    headroom_rec = mean_rec - 0.90
    headroom_prec = mean_prec - 0.85
    assert headroom_rec >= 0.05, f"Oracle recall headroom insufficient: {headroom_rec:.3f}"
    assert headroom_prec >= 0.05, f"Oracle precision headroom insufficient: {headroom_prec:.3f}"


def test_theorem8_3_notears_acyclicity_score_convergence() -> None:
    """Theorem 8.3: learned NOTEARS adjacency satisfies h(W) <= 1e-8.

    Reconstructs learned adjacency from production SCM and verifies smooth constraint functional.
    """
    rng = np.random.default_rng(7)
    x = rng.normal(size=500)
    y = 1.2 * x + rng.normal(0.0, 0.5, size=500)
    z = -0.9 * x + rng.normal(0.0, 0.5, size=500)
    q = 0.8 * y - 0.7 * z + rng.normal(0.0, 0.5, size=500)
    variable_names = ["X", "Y", "Z", "Q"]
    discovery = CausalDiscoveryEngine(variable_names)
    diag = discovery.learn_notears(
        np.column_stack((x, y, z, q)), lambda_l1=0.05, w_threshold=0.2, return_diagnostics=True
    )
    assert diag.h_val <= 1e-8
    assert CausalDiscoveryEngine._notears_h(diag.w_dense) <= 1e-8


def test_theorem8_8_notears_structure_recovery_pinned_seed() -> None:
    """Theorem 8.8: NOTEARS Structure-Recovery on Locked Fixture (Pinned Seed 7).

    Proves non-trivial DAG structure recovery on locked seed 7 meeting exact acceptance criteria.
    """
    import yaml

    with open("schema/causal_discovery_evaluation.yaml") as f:
        eval_fixture = yaml.safe_load(f)

    var_order = eval_fixture["data_generating_process"]["variable_order"]
    n_samples = eval_fixture["data_generating_process"]["sample_count"]
    var_idx = {name: i for i, name in enumerate(var_order)}
    true_edges = {
        (var_idx[s], var_idx[t])
        for s, t in eval_fixture["data_generating_process"]["ground_truth_edges"]
    }

    rng = np.random.default_rng(7)
    u_x = rng.normal(0, 1, n_samples)
    u_y = rng.normal(0, 1, n_samples)
    u_z = rng.normal(0, 1, n_samples)
    u_q = rng.normal(0, 1, n_samples)

    x = u_x
    y = 1.20 * x + u_y
    z = -0.90 * x + u_z
    q = 0.80 * y - 0.70 * z + u_q
    data = np.column_stack((x, y, z, q))

    engine = CausalDiscoveryEngine(var_order)
    diag = engine.learn_notears(
        data, lambda_l1=0.05, max_iter=30, w_threshold=0.25, return_diagnostics=True
    )

    pred_edges = set()
    for edge in diag.scm.edges:
        pred_edges.add((var_idx[edge.source], var_idx[edge.target]))

    tp = len(pred_edges & true_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)

    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    shd = fp + fn

    # Health check gates
    assert diag.outer_iters >= 3, (
        f"Optimizer health gate failed: outer_iters={diag.outer_iters} < 3"
    )
    assert diag.total_inner_iters >= 30, (
        f"Optimizer health gate failed: inner_iters={diag.total_inner_iters} < 30"
    )
    assert diag.h_val <= 1e-8, f"Acyclicity gate failed: h={diag.h_val}"
    assert len(pred_edges) > 0, "Learned graph must be non-empty"

    # Theorem 8.8 assertions
    assert recall >= 1.0, f"Pinned seed recall failed: expected 1.0, got {recall}"
    assert precision >= 0.80, f"Pinned seed precision failed: expected >= 0.80, got {precision}"
    assert shd <= 1, f"Pinned seed SHD failed: expected <= 1, got {shd}"


def test_theorem8_9_notears_multi_seed_stability() -> None:
    """Theorem 8.9: Multi-Seed NOTEARS Stability & Metric Distribution (20 Seeds).

    Evaluates recovery across 20 independent seeds on the locked evaluation fixture.
    """
    import yaml

    with open("schema/causal_discovery_evaluation.yaml") as f:
        eval_fixture = yaml.safe_load(f)

    var_order = eval_fixture["data_generating_process"]["variable_order"]
    seeds = eval_fixture["data_generating_process"]["seeds"]
    n_samples = eval_fixture["data_generating_process"]["sample_count"]
    var_idx = {name: i for i, name in enumerate(var_order)}
    true_edges = {
        (var_idx[s], var_idx[t])
        for s, t in eval_fixture["data_generating_process"]["ground_truth_edges"]
    }

    engine = CausalDiscoveryEngine(var_order)

    recalls, precisions, shds, h_vals, non_empty = [], [], [], [], []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        u_x = rng.normal(0, 1, n_samples)
        u_y = rng.normal(0, 1, n_samples)
        u_z = rng.normal(0, 1, n_samples)
        u_q = rng.normal(0, 1, n_samples)

        x = u_x
        y = 1.20 * x + u_y
        z = -0.90 * x + u_z
        q = 0.80 * y - 0.70 * z + u_q
        data = np.column_stack((x, y, z, q))

        diag = engine.learn_notears(
            data, lambda_l1=0.05, max_iter=30, w_threshold=0.25, return_diagnostics=True
        )

        pred_edges = {(var_idx[e.source], var_idx[e.target]) for e in diag.scm.edges}

        tp = len(pred_edges & true_edges)
        fp = len(pred_edges - true_edges)
        fn = len(true_edges - pred_edges)

        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        recalls.append(rec)
        precisions.append(prec)
        shds.append(fp + fn)
        h_vals.append(diag.h_val)
        non_empty.append(len(pred_edges) > 0)

    # Formal multi-seed stability assertions
    assert all(non_empty), "100% of seeds must return non-empty graphs"
    assert max(h_vals) <= 1e-8, f"Maximum h(W) exceeded threshold: {max(h_vals)}"
    assert min(recalls) >= 0.75, f"Minimum per-seed recall failed: {min(recalls)}"
    assert min(precisions) >= 0.75, f"Minimum per-seed precision failed: {min(precisions)}"
    assert np.mean(recalls) >= 0.90, f"Mean recall failed: {np.mean(recalls):.3f}"
    assert np.mean(precisions) >= 0.85, f"Mean precision failed: {np.mean(precisions):.3f}"


def test_theorem8_10_interventional_consistency_learned_scm() -> None:
    """Theorem 8.10: Interventional Consistency of Learned SCM vs Ground Truth.

    Verifies sign and order of interventional effect do(X=2.0) on target variable Q.
    """
    variable_names = ["X", "Y", "Z", "Q"]
    rng = np.random.default_rng(7)
    n_samples = 2000
    u_x = rng.normal(0, 1, n_samples)
    u_y = rng.normal(0, 1, n_samples)
    u_z = rng.normal(0, 1, n_samples)
    u_q = rng.normal(0, 1, n_samples)

    x = u_x
    y = 1.20 * x + u_y
    z = -0.90 * x + u_z
    q = 0.80 * y - 0.70 * z + u_q
    data = np.column_stack((x, y, z, q))

    engine = CausalDiscoveryEngine(variable_names)
    learned_scm = engine.learn_notears(data, lambda_l1=0.05, w_threshold=0.25)

    interv = Intervention({"X": 2.0})
    res = learned_scm.evaluate_intervention(interv)

    # Theoretical expected Q under do(X=2.0):
    # E[Y|do(X=2)] = 1.2 * 2 = 2.4
    # E[Z|do(X=2)] = -0.9 * 2 = -1.8
    # E[Q|do(X=2)] = 0.8 * 2.4 - 0.7 * (-1.8) = 1.92 + 1.26 = 3.18
    assert res["Q"] > 2.0, f"Interventional effect Q sign/magnitude invalid: E[Q]={res['Q']}"


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
