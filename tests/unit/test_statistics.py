import numpy as np
import pytest

from online_moments.statistics import (
    update_mean,
    update_ss,
    update_var,
    update_var_welford,
    update_wmean,
    update_wss,
    update_wvar,
)


def test_update_mean_matches_numpy_mean():
    rng = np.random.default_rng(123)
    x = rng.standard_normal(1000)
    m = 0.0
    for n, xi in enumerate(x, start=1):
        m = update_mean(m, xi, n)
    assert abs(m - x.mean()) < 1e-12


def test_update_var_matches_numpy_var():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(1000)
    m_old = 0.0
    s2 = 0.0
    for n, xi in enumerate(x, start=1):
        m_new = update_mean(m_old, xi, n)
        s2 = update_var(s2, m_new, m_old, xi, n)
        m_old = m_new
    # Population variance (ddof=0) since update_var divides by n.
    assert abs(s2 - x.var(ddof=0)) < 1e-10


def test_update_var_matches_welford_form():
    rng = np.random.default_rng(11)
    x = rng.standard_normal(1000)
    m_old = 0.0
    s2 = 0.0
    S = 0.0
    for n, xi in enumerate(x, start=1):
        m_new = update_mean(m_old, xi, n)
        s2 = update_var(s2, m_new, m_old, xi, n)
        S = update_var_welford(S, m_new, m_old, xi)
        m_old = m_new
    # s2 should equal S/n by construction
    n = len(x)
    assert abs(s2 - S / n) < 1e-12


def test_update_ss_matches_mean_of_squares():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(500)
    ss = 0.0
    for n, xi in enumerate(x, start=1):
        ss = update_ss(ss, xi, n)
    assert abs(ss - (x * x).mean()) < 1e-12


def test_update_wmean_matches_weighted_average():
    rng = np.random.default_rng(13)
    x = rng.standard_normal(500)
    w = rng.uniform(0.1, 2.0, size=500)
    m = 0.0
    W = 0.0
    for xi, wi in zip(x, w, strict=True):
        m = update_wmean(m, W, xi, wi)
        W += wi
    assert abs(m - np.average(x, weights=w)) < 1e-10


def test_update_wvar_matches_weighted_variance():
    rng = np.random.default_rng(17)
    x = rng.standard_normal(500)
    w = rng.uniform(0.1, 2.0, size=500)
    m_old = 0.0
    W = 0.0
    s2 = 0.0
    for xi, wi in zip(x, w, strict=True):
        m_new = update_wmean(m_old, W, xi, wi)
        s2 = update_wvar(s2, m_old, W, xi, m_new, wi)
        W += wi
        m_old = m_new
    expected_mean = np.average(x, weights=w)
    expected_var = np.average((x - expected_mean) ** 2, weights=w)
    assert abs(s2 - expected_var) < 1e-9


def test_update_wss_matches_weighted_mean_of_squares():
    rng = np.random.default_rng(31)
    x = rng.standard_normal(500)
    w = rng.uniform(0.1, 2.0, size=500)
    ss = 0.0
    W = 0.0
    for xi, wi in zip(x, w, strict=True):
        ss = update_wss(ss, W, xi, wi)
        W += wi
    expected = np.average(x * x, weights=w)
    assert abs(ss - expected) < 1e-10


def test_update_var_constant_input():
    """Constant input has zero variance regardless of n."""
    m_old = 0.0
    s2 = 0.0
    for n in range(1, 100):
        m_new = update_mean(m_old, 7.0, n)
        s2 = update_var(s2, m_new, m_old, 7.0, n)
        m_old = m_new
    assert abs(s2) < 1e-15
