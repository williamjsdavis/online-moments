"""Numba-jitted streaming kernels for periodic-state-space estimators."""
from __future__ import annotations

import numpy as np
from numba import njit

from ..online._inner_loop import _kernel_value


@njit(cache=True, fastmath=False)
def _d_plus(a: float, b: float, period: float) -> float:
    return (b - a) % period


@njit(cache=True, fastmath=False)
def _is_in_interval_mod(a: float, b: float, period: float, x: float) -> bool:
    return _d_plus(a, x, period) < _d_plus(a, b, period)


@njit(cache=True, fastmath=False)
def _find_mod_bin(edges: np.ndarray, period: float, x: float) -> int:
    n_bins = edges.shape[0] - 1
    for i in range(n_bins):
        if _is_in_interval_mod(edges[i], edges[i + 1], period, x):
            return i
    return -1


@njit(cache=True, fastmath=False)
def _d_mod(x: float, period: float) -> float:
    a = x % period
    b = (-x) % period
    return a if a < b else b


@njit(cache=True, fastmath=False)
def ohbr_mod_update(
    x_new, edges, tau_indices, period,
    ring, n_pushed,
    counts, M1, M2, use_variance,
):
    n_tau = tau_indices.shape[0]
    for i_tau in range(n_tau):
        tau = tau_indices[i_tau]
        if n_pushed < tau:
            continue
        x_left = ring[tau - 1]
        j_bin = _find_mod_bin(edges, period, x_left)
        if j_bin < 0:
            continue
        # Increment in modulo space: shortest signed difference.
        dx = x_new - x_left
        # Wrap into (-period/2, period/2] for a signed shortest distance.
        dx = ((dx + period / 2.0) % period) - period / 2.0
        counts[i_tau, j_bin] += 1
        n_now = counts[i_tau, j_bin]
        m1_old = M1[i_tau, j_bin]
        m1_new = m1_old + (dx - m1_old) / n_now
        M1[i_tau, j_bin] = m1_new
        if use_variance:
            s2 = M2[i_tau, j_bin]
            M2[i_tau, j_bin] = s2 + ((dx - m1_new) * (dx - m1_old) - s2) / n_now
        else:
            ss = M2[i_tau, j_bin]
            M2[i_tau, j_bin] = ss + (dx * dx - ss) / n_now

    L = ring.shape[0]
    if L > 0:
        for k in range(L - 1, 0, -1):
            ring[k] = ring[k - 1]
        ring[0] = x_new
    return n_pushed + 1


@njit(cache=True, fastmath=False)
def ohbr_mod_update_batch(
    X, edges, tau_indices, period, ring, n_pushed,
    counts, M1, M2, use_variance,
):
    for k in range(X.shape[0]):
        n_pushed = ohbr_mod_update(
            X[k], edges, tau_indices, period, ring, n_pushed,
            counts, M1, M2, use_variance,
        )
    return n_pushed


@njit(cache=True, fastmath=False)
def okbr_mod_update(
    x_new, x_eval, tau_indices, period, kernel_id, hinv,
    ring, n_pushed,
    W, M1, M2, use_variance,
):
    n_tau = tau_indices.shape[0]
    n_x = x_eval.shape[0]
    for i_tau in range(n_tau):
        tau = tau_indices[i_tau]
        if n_pushed < tau:
            continue
        x_left = ring[tau - 1]
        dx = x_new - x_left
        dx = ((dx + period / 2.0) % period) - period / 2.0
        for j in range(n_x):
            d = _d_mod(x_eval[j] - x_left, period)
            u = d * hinv
            k_unscaled = _kernel_value(u, kernel_id)
            if k_unscaled <= 0.0:
                continue
            k_weight = hinv * k_unscaled
            w_old = W[i_tau, j]
            w_new = w_old + k_weight
            ratio = k_weight / w_new
            m1_old = M1[i_tau, j]
            m1 = m1_old + (dx - m1_old) * ratio
            M1[i_tau, j] = m1
            if use_variance:
                s_old = M2[i_tau, j] * w_old
                M2[i_tau, j] = (s_old + k_weight * (dx - m1_old) * (dx - m1)) / w_new
            else:
                M2[i_tau, j] = M2[i_tau, j] + (dx * dx - M2[i_tau, j]) * ratio
            W[i_tau, j] = w_new

    L = ring.shape[0]
    if L > 0:
        for k in range(L - 1, 0, -1):
            ring[k] = ring[k - 1]
        ring[0] = x_new
    return n_pushed + 1


@njit(cache=True, fastmath=False)
def okbr_mod_update_batch(
    X, x_eval, tau_indices, period, kernel_id, hinv,
    ring, n_pushed,
    W, M1, M2, use_variance,
):
    for k in range(X.shape[0]):
        n_pushed = okbr_mod_update(
            X[k], x_eval, tau_indices, period, kernel_id, hinv,
            ring, n_pushed, W, M1, M2, use_variance,
        )
    return n_pushed
