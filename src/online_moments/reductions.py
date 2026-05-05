"""Reductions from accumulated conditional moments to drift/diffusion functions."""
from __future__ import annotations

import numpy as np

from ._typing import FloatArray, IntArray


def m1_per_tau(M1: FloatArray, tau_indices: IntArray, dt: float) -> FloatArray:
    return M1 / (dt * np.asarray(tau_indices, dtype=np.float64)[:, None])


def m2_per_tau(M2: FloatArray, tau_indices: IntArray, dt: float) -> FloatArray:
    return M2 / (dt * np.asarray(tau_indices, dtype=np.float64)[:, None])


def kramers_moyal_regression(
    M1: FloatArray,
    M2: FloatArray,
    tau_indices: IntArray,
    dt: float,
) -> tuple[FloatArray, FloatArray]:
    """OLS τ→0 regression for drift and diffusion.

    Fits ``M^(k)(τ) ≈ k! · D^(k) · τ`` with zero intercept across all τ values
    in ``tau_indices``. Returns ``(D1, D2)`` as 1-D arrays of length ``N_x``.

    For a single τ this reduces to direct estimation:
    ``D^(1) = M^(1)/(Δt τ)``, ``D^(2) = M^(2)/(2 Δt τ)``.
    """
    M1 = np.asarray(M1, dtype=np.float64)
    M2 = np.asarray(M2, dtype=np.float64)
    tau = np.asarray(tau_indices, dtype=np.float64) * dt
    n_tau = len(tau)
    if M1.shape[0] != n_tau or M2.shape[0] != n_tau:
        raise ValueError("First axis of M1/M2 must equal len(tau_indices)")

    # Zero-intercept OLS slope per evaluation point: D = sum(τ M) / sum(τ²)
    # for M^(1), and similarly D^(2) = sum(τ M^(2)) / (2 sum(τ²)).
    denom = float((tau * tau).sum())
    if denom == 0:
        raise ValueError("tau values must be non-zero")
    D1 = (tau[:, None] * M1).sum(axis=0) / denom
    D2 = (tau[:, None] * M2).sum(axis=0) / (2.0 * denom)
    return D1, D2


__all__ = ["m1_per_tau", "m2_per_tau", "kramers_moyal_regression"]
