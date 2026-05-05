"""Offline histogram-based regression (HBR) for conditional moments.

Single reference implementation (Algorithm C from the Julia code: a single pass
over X using the streaming-mean / streaming-variance recurrences). This is the
*same* arithmetic the online :class:`OHBR` class uses, applied in the same
order — the equivalence test asserts exact bit equality between the two.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .._typing import FloatArray, IntArray
from ..binning import find_bin, in_range
from ..statistics import update_mean, update_ss, update_var


def hbr_moments(
    X: FloatArray,
    *,
    tau_indices: IntArray,
    edges: FloatArray,
    moment_form: Literal["variance", "raw"] = "variance",
) -> tuple[FloatArray, FloatArray]:
    """Conditional moments via histogram-based regression.

    Parameters
    ----------
    X
        Time-series data, shape (N,).
    tau_indices
        Strictly positive integer lags, ascending. Need not be consecutive.
    edges
        Bin edges, shape (N_x + 1,). Half-open `[e_i, e_{i+1})` except the last
        bin which is closed.
    moment_form
        ``"variance"`` returns the conditional variance ``Var[ΔX|x]``;
        ``"raw"`` returns the conditional raw second moment ``E[ΔX^2|x]``.

    Returns
    -------
    M1, M2
        Both shape (N_tau, N_x). M1 is the conditional mean of ΔX = X_{n+τ} − X_n.
        M2 is the conditional variance or raw second moment depending on
        ``moment_form``.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    tau_indices = np.ascontiguousarray(tau_indices, dtype=np.int64)
    edges = np.ascontiguousarray(edges, dtype=np.float64)
    _validate_tau_indices(tau_indices)

    n_x = len(edges) - 1
    n_tau = len(tau_indices)
    n = len(X)

    counts = np.zeros((n_tau, n_x), dtype=np.int64)
    M1 = np.zeros((n_tau, n_x), dtype=np.float64)
    M2 = np.zeros((n_tau, n_x), dtype=np.float64)

    use_variance = moment_form == "variance"
    if moment_form not in ("variance", "raw"):
        raise ValueError(f"moment_form must be 'variance' or 'raw', got {moment_form!r}")

    for i_left in range(n - 1):
        x_left = X[i_left]
        if not in_range(edges, x_left):
            continue
        j_bin = find_bin(edges, x_left)
        for i_tau, tau in enumerate(tau_indices):
            i_right = i_left + int(tau)
            if i_right >= n:
                continue
            dx = X[i_right] - x_left
            counts[i_tau, j_bin] += 1
            n_now = int(counts[i_tau, j_bin])
            m1_old = M1[i_tau, j_bin]
            m1_new = update_mean(m1_old, dx, n_now)
            M1[i_tau, j_bin] = m1_new
            if use_variance:
                M2[i_tau, j_bin] = update_var(
                    M2[i_tau, j_bin], m1_new, m1_old, dx, n_now
                )
            else:
                M2[i_tau, j_bin] = update_ss(M2[i_tau, j_bin], dx, n_now)

    return M1, M2


def _validate_tau_indices(tau_indices: IntArray) -> None:
    if tau_indices.ndim != 1 or tau_indices.size == 0:
        raise ValueError("tau_indices must be a non-empty 1-D array")
    if (tau_indices <= 0).any():
        raise ValueError("tau_indices must be strictly positive")
    if (np.diff(tau_indices) <= 0).any():
        raise ValueError("tau_indices must be strictly increasing")


__all__ = ["hbr_moments"]
