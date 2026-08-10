import numpy as np

from hypostases.engine._math import compute_omega, compute_temperature, softmax


def test_softmax_all_zeros():
    x = np.zeros(4)
    res = softmax(x)
    assert np.allclose(res, 0.25)


def test_softmax_large_values():
    # stable softmax doesn't overflow
    x = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    res = softmax(x)
    assert np.allclose(res, 0.25)


def test_softmax_single_element():
    x = np.array([5.0])
    res = softmax(x)
    assert np.allclose(res, 1.0)


def test_compute_omega_zero_reserve():
    u = np.array([1.0, 1.0, 1.0, 1.0])
    omega = compute_omega(u, reserve=0.0)
    # reserve of 0.0 means affordability is min(1, 0 / cost) = 0.0
    assert np.allclose(omega, 0.0)


def test_compute_omega_high_reserve():
    u = np.array([1.0, 1.0, 1.0, 1.0])
    omega = compute_omega(u, reserve=100.0)
    # reserve 100.0 means affordability is 1.0 everywhere, so it's just softmax(u/temp)
    expected = softmax(u)
    assert np.allclose(omega, expected)


def test_compute_omega_with_xi():
    u = np.array([1.0, 1.0, 1.0, 1.0])
    xi = np.array([0.5, 0.5, 0.5, 0.5])
    omega = compute_omega(u, reserve=100.0, xi=xi)
    expected = softmax(u / 0.5)
    assert np.allclose(omega, expected)


def test_compute_temperature_defaults():
    # None should default to 1.0 base
    assert compute_temperature(None, offset=0.0) == 1.0
    assert compute_temperature(None, offset=0.15) == 1.15
