import numpy as np
import pytest

from online_moments.autocorr import OnlineAutoCov, offline_autocorr


def test_offline_autocorr_lag0_equals_variance_biased():
    rng = np.random.default_rng(0)
    X = rng.standard_normal(2000)
    out = offline_autocorr(X, lags=5)
    assert out[0] == pytest.approx(X.var(ddof=0), rel=1e-12)


def test_offline_autocorr_white_noise_decays_to_zero():
    rng = np.random.default_rng(1)
    X = rng.standard_normal(10_000)
    out = offline_autocorr(X, lags=10)
    # Lag 0 ≈ 1 (unit variance); lag 1+ should be small.
    assert abs(out[0] - 1.0) < 0.05
    assert (np.abs(out[1:]) < 0.1).all()


def test_online_autocov_matches_offline_lag0_exactly():
    """Lag-0 (variance) is computed via Welford and must match exactly."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal(1000)
    o = OnlineAutoCov(lags=5)
    o.update_batch(X)
    off = offline_autocorr(X, lags=5)
    assert o.autocov[0] == pytest.approx(off[0], rel=1e-12)


def test_online_autocov_lag_k_approx_offline():
    """Lag k>=1 is a streaming approximation (running-mean bias of O(1/N))."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal(20_000)
    o = OnlineAutoCov(lags=5)
    o.update_batch(X)
    off = offline_autocorr(X, lags=5)
    np.testing.assert_allclose(o.autocov, off, atol=0.02, rtol=0.05)


def test_online_autocov_white_noise():
    rng = np.random.default_rng(3)
    X = rng.standard_normal(10_000)
    o = OnlineAutoCov(lags=5)
    o.update_batch(X)
    ac = o.autocorr
    assert abs(ac[0] - 1.0) < 1e-12  # autocorr is c_k / c_0; lag 0 always 1
    assert (np.abs(ac[1:]) < 0.1).all()


def test_online_autocov_ar1_recovers_decay():
    """AR(1): X_t = ϕ X_{t-1} + ε_t. Autocorr at lag k is ϕ^k."""
    phi = 0.8
    rng = np.random.default_rng(4)
    n = 50_000
    eps = rng.standard_normal(n)
    X = np.empty(n)
    X[0] = 0.0
    for t in range(1, n):
        X[t] = phi * X[t - 1] + eps[t]

    o = OnlineAutoCov(lags=5)
    o.update_batch(X)
    ac = o.autocorr
    expected = phi ** np.arange(6)
    np.testing.assert_allclose(ac, expected, atol=0.05)
