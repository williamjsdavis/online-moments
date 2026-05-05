"""Modulo OHBR/OKBR results must be invariant under shifts by k*period.

Mirrors the Julia ``test_OHBR.jl`` modulo translational-invariance test.
"""
import numpy as np

from online_moments import Epanechnikov
from online_moments.modulo import OHBRMod, OKBRMod


def test_ohbr_mod_translational_invariance(X_small):
    period = 1.0
    edges = np.linspace(0.0, period, 5)
    tau = np.array([1, 2], dtype=np.int64)

    a = OHBRMod(edges, tau, period)
    a.update_batch(X_small % period)

    shift = 1000.0 * period
    b = OHBRMod(edges, tau, period)
    b.update_batch((X_small - shift) % period)

    np.testing.assert_allclose(a.M1, b.M1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a.M2, b.M2, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(a.counts, b.counts)


def test_okbr_mod_translational_invariance(X_small):
    period = 1.0
    x_eval = np.linspace(0.05, 0.95, 10)
    tau = np.array([1, 2], dtype=np.int64)

    a = OKBRMod(x_eval, tau, period, kernel=Epanechnikov(), bandwidth=0.1)
    a.update_batch(X_small % period)

    shift = 1000.0 * period
    b = OKBRMod(x_eval, tau, period, kernel=Epanechnikov(), bandwidth=0.1)
    b.update_batch((X_small - shift) % period)

    np.testing.assert_allclose(a.W, b.W, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a.M1, b.M1, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(a.M2, b.M2, rtol=1e-9, atol=1e-11)


def test_ohbr_mod_streaming_matches_batch(X_small):
    period = 1.0
    edges = np.linspace(0.0, period, 5)
    tau = np.array([1, 2], dtype=np.int64)

    a = OHBRMod(edges, tau, period)
    a.update_batch(X_small % period)

    b = OHBRMod(edges, tau, period)
    for x in X_small % period:
        b.update(float(x))

    np.testing.assert_array_equal(a.M1, b.M1)
    np.testing.assert_array_equal(a.M2, b.M2)
    np.testing.assert_array_equal(a.counts, b.counts)
