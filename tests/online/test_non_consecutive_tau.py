"""Regression test for the Julia bug: non-consecutive tau_indices must work.

The Julia ``OHBR_multiple`` and ``OKBR_multiple`` types store ``tau_i::UnitRange{Int}``
and use ``mem[i_tau]`` directly, which silently misbehaves for arbitrary lags.
The Python implementation sizes the ring buffer to ``max(tau_indices)`` and looks
up ``ring[tau - 1]``, so this works.
"""
import numpy as np

from online_moments import OHBR, OKBR, Epanechnikov, hbr_moments, kbr_moments


def test_ohbr_non_consecutive(X_small):
    edges = np.linspace(0.0, 0.5, 6)
    tau = np.array([1, 3, 5], dtype=np.int64)

    online = OHBR(edges, tau)
    online.update_batch(X_small)

    M1_off, M2_off = hbr_moments(X_small, tau_indices=tau, edges=edges)
    np.testing.assert_array_equal(online.M1, M1_off)
    np.testing.assert_array_equal(online.M2, M2_off)


def test_okbr_non_consecutive(X_small):
    x_eval = np.linspace(0.0, 0.5, 20)
    tau = np.array([2, 4, 7], dtype=np.int64)

    online = OKBR(x_eval, tau, kernel=Epanechnikov(), bandwidth=0.05)
    online.update_batch(X_small)

    M1_off, M2_off = kbr_moments(
        X_small, tau_indices=tau, x_eval=x_eval,
        kernel=Epanechnikov(), bandwidth=0.05,
    )
    np.testing.assert_allclose(online.M1, M1_off, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(online.M2, M2_off, rtol=1e-10, atol=1e-12)
