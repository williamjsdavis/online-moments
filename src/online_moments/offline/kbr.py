"""Offline kernel-based regression (KBR) for conditional moments.

Single reference implementation: accumulate kernel-weighted ΔX and ΔX^2 on a
single pass over X, then divide by W. The two-pass form is mathematically
equivalent to the streaming Welford recurrence used in :class:`OKBR`, so the
online-vs-offline equivalence test holds to ``rtol=1e-10`` (the residual
difference is from accumulator-order floating-point noise).
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .._typing import FloatArray, IntArray
from ..kernels import Kernel


def kbr_moments(
    X: FloatArray,
    *,
    tau_indices: IntArray,
    x_eval: FloatArray,
    kernel: Kernel,
    bandwidth: float,
    moment_form: Literal["variance", "raw"] = "variance",
) -> tuple[FloatArray, FloatArray]:
    """Conditional moments via kernel-based regression.

    Parameters
    ----------
    X
        Time-series data, shape (N,).
    tau_indices
        Strictly positive integer lags, ascending.
    x_eval
        Evaluation points in state space, shape (N_x,).
    kernel
        A :class:`Kernel` instance (e.g., :class:`Boxcar`, :class:`Epanechnikov`).
    bandwidth
        Kernel bandwidth ``h``, positive.
    moment_form
        ``"variance"`` or ``"raw"`` (see :func:`hbr_moments`).

    Returns
    -------
    M1, M2 : tuple of ndarray, both shape (N_tau, N_x).
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    tau_indices = np.ascontiguousarray(tau_indices, dtype=np.int64)
    x_eval = np.ascontiguousarray(x_eval, dtype=np.float64)
    _validate(tau_indices, bandwidth, moment_form)

    n_x = len(x_eval)
    n_tau = len(tau_indices)
    n = len(X)
    hinv = 1.0 / bandwidth

    W = np.zeros((n_tau, n_x), dtype=np.float64)
    sum_dx = np.zeros((n_tau, n_x), dtype=np.float64)
    sum_dx2 = np.zeros((n_tau, n_x), dtype=np.float64)

    for i_left in range(n - 1):
        x_left = X[i_left]
        # Evaluate kernel weights for all evaluation points (vectorized over j).
        u = (x_eval - x_left) * hinv
        # Build kernel weights without a per-j Python call: handle our two
        # supported kernels directly. Falls back to scalar for custom kernels.
        kid = getattr(kernel, "_numba_id", None)
        if kid == 0:  # Boxcar
            mask = np.abs(u) < 1.0
            kvals = np.where(mask, 0.5 * hinv, 0.0)
        elif kid == 1:  # Epanechnikov
            u2 = u * u
            mask = u2 < 1.0
            kvals = np.where(mask, 0.75 * (1.0 - u2) * hinv, 0.0)
        else:
            kvals = np.array([hinv * kernel(ui) for ui in u])

        for i_tau, tau in enumerate(tau_indices):
            i_right = i_left + int(tau)
            if i_right >= n:
                continue
            dx = X[i_right] - x_left
            W[i_tau] += kvals
            sum_dx[i_tau] += kvals * dx
            sum_dx2[i_tau] += kvals * dx * dx

    M1 = np.zeros_like(W)
    M2 = np.zeros_like(W)
    nz = W > 0
    M1[nz] = sum_dx[nz] / W[nz]
    if moment_form == "variance":
        # Var[ΔX] = E[ΔX^2] - E[ΔX]^2
        M2[nz] = sum_dx2[nz] / W[nz] - M1[nz] * M1[nz]
    else:  # "raw"
        M2[nz] = sum_dx2[nz] / W[nz]
    return M1, M2


def _validate(tau_indices, bandwidth, moment_form):
    if tau_indices.ndim != 1 or tau_indices.size == 0:
        raise ValueError("tau_indices must be a non-empty 1-D array")
    if (tau_indices <= 0).any():
        raise ValueError("tau_indices must be strictly positive")
    if (np.diff(tau_indices) <= 0).any():
        raise ValueError("tau_indices must be strictly increasing")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if moment_form not in ("variance", "raw"):
        raise ValueError(f"moment_form must be 'variance' or 'raw', got {moment_form!r}")


__all__ = ["kbr_moments"]
