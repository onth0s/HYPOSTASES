"""Regression Baseline Test — Part VII §12.7 Formal WITHDRAW Gap Test.

Pins current performance targets from spec §12.
"""

from hypostases.simulation import evaluate_config, single_trial


def test_single_trial_evaluation():
    res = single_trial(n_steps=10, n_particles=50, seed=1)
    assert "z" in res
    assert "mu_low" in res
    assert "mu_high" in res
    assert "var_low" in res
    assert "var_high" in res


def test_evaluate_config_structure():
    res = evaluate_config(n_steps=10, n_particles=50, seeds=[1, 2, 3])
    assert "n_steps" in res
    assert "n_particles" in res
    assert "passed" in res
    assert "cond1_count" in res
    assert "cond2_median_z" in res
    assert len(res["all_z"]) == 3
