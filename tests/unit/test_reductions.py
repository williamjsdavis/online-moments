import numpy as np

from online_moments.reductions import (
    kramers_moyal_regression,
    m1_per_tau,
    m2_per_tau,
)


def test_m1_per_tau_basic():
    M1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    tau = np.array([1, 2], dtype=np.int64)
    out = m1_per_tau(M1, tau, dt=0.5)
    expected = M1 / np.array([[0.5], [1.0]])
    np.testing.assert_allclose(out, expected)


def test_m2_per_tau_basic():
    M2 = np.array([[2.0, 4.0], [6.0, 8.0]])
    tau = np.array([1, 2], dtype=np.int64)
    out = m2_per_tau(M2, tau, dt=0.5)
    expected = M2 / np.array([[0.5], [1.0]])
    np.testing.assert_allclose(out, expected)


def test_km_regression_recovers_linear_slope():
    """If M^(1)(τ) = D · τ exactly, the regression returns D."""
    rng = np.random.default_rng(0)
    n_x = 5
    n_tau = 8
    dt = 1e-3
    tau_indices = np.arange(1, n_tau + 1, dtype=np.int64)
    D1_true = rng.standard_normal(n_x)
    D2_true = np.abs(rng.standard_normal(n_x)) + 0.1
    tau = tau_indices.astype(np.float64) * dt
    M1 = D1_true[None, :] * tau[:, None]            # M^(1) = D · τ
    M2 = 2.0 * D2_true[None, :] * tau[:, None]       # M^(2) = 2 · D · τ
    D1, D2 = kramers_moyal_regression(M1, M2, tau_indices, dt)
    np.testing.assert_allclose(D1, D1_true, rtol=1e-12)
    np.testing.assert_allclose(D2, D2_true, rtol=1e-12)


def test_km_regression_single_tau_equals_direct():
    M1 = np.array([[1.0, -2.0]])  # shape (1, 2)
    M2 = np.array([[3.0, 4.0]])
    tau = np.array([1], dtype=np.int64)
    dt = 0.1
    D1, D2 = kramers_moyal_regression(M1, M2, tau, dt)
    np.testing.assert_allclose(D1, M1[0] / dt)
    np.testing.assert_allclose(D2, M2[0] / (2.0 * dt))
