"""Numba-jitted streaming kernels for OHBR and OKBR.

These functions are the hot path. They consume one new sample at a time and
update the accumulator arrays in place. The same arithmetic is also expressed
in the offline reference implementations and (for HBR) is bit-identical to
``hbr_moments``.
"""
from __future__ import annotations

import numpy as np
from numba import njit


# Kernel ID constants — duplicated from kernels.py because Numba cannot import
# Python module attributes inside @njit code.
_KERNEL_BOXCAR = 0
_KERNEL_EPANECHNIKOV = 1


@njit(cache=True, fastmath=False)
def _kernel_value(u: float, kernel_id: int) -> float:
    """Unscaled kernel: K(u). Bandwidth scaling K_h(x) = K(x/h)/h is applied
    by the caller (multiply by hinv after this returns)."""
    if kernel_id == _KERNEL_BOXCAR:
        if -1.0 < u < 1.0:
            return 0.5
        return 0.0
    elif kernel_id == _KERNEL_EPANECHNIKOV:
        if -1.0 < u < 1.0:
            return 0.75 * (1.0 - u * u)
        return 0.0
    return 0.0


@njit(cache=True, fastmath=False)
def _find_bin(edges: np.ndarray, x: float) -> int:
    """Half-open bin lookup; returns -1 if x is out of range."""
    n_bins = edges.shape[0] - 1
    if x < edges[0] or x > edges[-1]:
        return -1
    # Linear scan is fine for small n_bins; for large n_bins switch to bisection.
    # The paper's examples use n_bins ~25..100 so linear is competitive.
    if n_bins <= 32:
        for i in range(n_bins):
            if x < edges[i + 1]:
                return i
        return n_bins - 1  # x == edges[-1]
    # Bisection
    lo = 0
    hi = n_bins
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if x < edges[mid]:
            hi = mid
        else:
            lo = mid
    return lo


@njit(cache=True, fastmath=False)
def ohbr_update(
    x_new: float,
    edges: np.ndarray,
    tau_indices: np.ndarray,
    ring: np.ndarray,
    n_pushed: int,
    counts: np.ndarray,
    M1: np.ndarray,
    M2: np.ndarray,
    use_variance: bool,
) -> int:
    """Process one new sample for the online HBR estimator.

    Mutates ``ring``, ``counts``, ``M1``, ``M2`` in place. Returns the new
    ``n_pushed`` value (caller stores it on the Python-level estimator).
    """
    n_tau = tau_indices.shape[0]
    for i_tau in range(n_tau):
        tau = tau_indices[i_tau]
        if n_pushed < tau:
            continue
        x_left = ring[tau - 1]
        if x_left < edges[0] or x_left > edges[-1]:
            continue
        j_bin = _find_bin(edges, x_left)
        if j_bin < 0:
            continue
        dx = x_new - x_left
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

    # Push x_new to the front of the ring (right-shift, write index 0).
    L = ring.shape[0]
    if L > 0:
        for k in range(L - 1, 0, -1):
            ring[k] = ring[k - 1]
        ring[0] = x_new
    return n_pushed + 1


@njit(cache=True, fastmath=False)
def ohbr_update_batch(
    X_batch: np.ndarray,
    edges: np.ndarray,
    tau_indices: np.ndarray,
    ring: np.ndarray,
    n_pushed: int,
    counts: np.ndarray,
    M1: np.ndarray,
    M2: np.ndarray,
    use_variance: bool,
) -> int:
    for k in range(X_batch.shape[0]):
        n_pushed = ohbr_update(
            X_batch[k], edges, tau_indices, ring, n_pushed,
            counts, M1, M2, use_variance,
        )
    return n_pushed


@njit(cache=True, fastmath=False)
def okbr_update(
    x_new: float,
    x_eval: np.ndarray,
    tau_indices: np.ndarray,
    kernel_id: int,
    hinv: float,
    ring: np.ndarray,
    n_pushed: int,
    W: np.ndarray,
    M1: np.ndarray,
    M2: np.ndarray,
    use_variance: bool,
) -> int:
    """Process one new sample for the online KBR estimator.

    Mutates ``ring``, ``W``, ``M1``, ``M2`` in place. Returns the new
    ``n_pushed``.
    """
    n_tau = tau_indices.shape[0]
    n_x = x_eval.shape[0]
    for i_tau in range(n_tau):
        tau = tau_indices[i_tau]
        if n_pushed < tau:
            continue
        x_left = ring[tau - 1]
        dx = x_new - x_left
        for j in range(n_x):
            u = (x_eval[j] - x_left) * hinv
            k_unscaled = _kernel_value(u, kernel_id)
            if k_unscaled <= 0.0:
                continue
            k_weight = hinv * k_unscaled
            w_old = W[i_tau, j]
            w_new = w_old + k_weight
            m1_old = M1[i_tau, j]
            ratio = k_weight / w_new  # = w_new / (w_old + w_new) up to algebraic identity
            m1 = m1_old + (dx - m1_old) * ratio
            M1[i_tau, j] = m1
            if use_variance:
                # Weighted Welford: S_n = S_{n-1} + k * (dx - m1_old)*(dx - m1_new),
                # Var = S / W. Equivalent to update_wvar with re-derivation.
                s_old = M2[i_tau, j] * w_old
                s_new = s_old + k_weight * (dx - m1_old) * (dx - m1)
                M2[i_tau, j] = s_new / w_new
            else:
                # Weighted raw second moment.
                M2[i_tau, j] = M2[i_tau, j] + (dx * dx - M2[i_tau, j]) * ratio
            W[i_tau, j] = w_new

    L = ring.shape[0]
    if L > 0:
        for k in range(L - 1, 0, -1):
            ring[k] = ring[k - 1]
        ring[0] = x_new
    return n_pushed + 1


@njit(cache=True, fastmath=False)
def okbr_update_batch(
    X_batch: np.ndarray,
    x_eval: np.ndarray,
    tau_indices: np.ndarray,
    kernel_id: int,
    hinv: float,
    ring: np.ndarray,
    n_pushed: int,
    W: np.ndarray,
    M1: np.ndarray,
    M2: np.ndarray,
    use_variance: bool,
) -> int:
    for k in range(X_batch.shape[0]):
        n_pushed = okbr_update(
            X_batch[k], x_eval, tau_indices, kernel_id, hinv,
            ring, n_pushed, W, M1, M2, use_variance,
        )
    return n_pushed


__all__ = [
    "_KERNEL_BOXCAR",
    "_KERNEL_EPANECHNIKOV",
    "ohbr_update",
    "ohbr_update_batch",
    "okbr_update",
    "okbr_update_batch",
]
