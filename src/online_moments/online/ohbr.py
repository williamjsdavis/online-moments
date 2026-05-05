from __future__ import annotations

from typing import Literal

import numpy as np

from .._typing import FloatArray, IntArray
from . import _inner_loop


class OHBR:
    """Online histogram-based regression for conditional moments.

    Streaming, ``O(N_τ * N_x)``-space estimator. ``update(x)`` consumes one
    sample at a time; ``update_batch(X)`` is the same recurrence in a single
    Numba-jitted loop.

    The accumulator state is bit-identical to what
    :func:`online_moments.offline.hbr_moments` produces on the same input —
    both apply the same Welford-style recurrences in the same order.
    """

    def __init__(
        self,
        edges: FloatArray,
        tau_indices: IntArray,
        *,
        moment_form: Literal["variance", "raw"] = "variance",
    ) -> None:
        edges = np.ascontiguousarray(edges, dtype=np.float64)
        tau_indices = np.ascontiguousarray(tau_indices, dtype=np.int64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("edges must be a 1-D array with length >= 2")
        if (np.diff(edges) <= 0).any():
            raise ValueError("edges must be strictly increasing")
        if tau_indices.ndim != 1 or tau_indices.size == 0:
            raise ValueError("tau_indices must be a non-empty 1-D array")
        if (tau_indices <= 0).any():
            raise ValueError("tau_indices must be strictly positive")
        if (np.diff(tau_indices) <= 0).any():
            raise ValueError("tau_indices must be strictly increasing")
        if moment_form not in ("variance", "raw"):
            raise ValueError(
                f"moment_form must be 'variance' or 'raw', got {moment_form!r}"
            )

        n_tau = len(tau_indices)
        n_x = len(edges) - 1
        max_lag = int(tau_indices[-1])

        self.edges = edges
        self.tau_indices = tau_indices
        self.moment_form = moment_form
        self.counts: np.ndarray = np.zeros((n_tau, n_x), dtype=np.int64)
        self.M1: FloatArray = np.zeros((n_tau, n_x), dtype=np.float64)
        self.M2: FloatArray = np.zeros((n_tau, n_x), dtype=np.float64)
        self._ring: FloatArray = np.zeros(max_lag, dtype=np.float64)
        self._n_pushed: int = 0
        self._use_variance = moment_form == "variance"

    @property
    def n_processed(self) -> int:
        return self._n_pushed

    @property
    def N(self) -> np.ndarray:
        """Alias for ``counts`` to match the Julia field name."""
        return self.counts

    def update(self, x: float) -> None:
        self._n_pushed = _inner_loop.ohbr_update(
            float(x),
            self.edges,
            self.tau_indices,
            self._ring,
            self._n_pushed,
            self.counts,
            self.M1,
            self.M2,
            self._use_variance,
        )

    def update_batch(self, X: FloatArray) -> None:
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 1:
            raise ValueError("X must be 1-D")
        self._n_pushed = _inner_loop.ohbr_update_batch(
            X,
            self.edges,
            self.tau_indices,
            self._ring,
            self._n_pushed,
            self.counts,
            self.M1,
            self.M2,
            self._use_variance,
        )

    def m1_per_tau(self, dt: float) -> FloatArray:
        """M1 / (Δt · τ_i). Used in direct estimation D^(1) ≈ M^(1)/(Δt τ)."""
        return self.M1 / (dt * self.tau_indices[:, None])

    def m2_per_tau(self, dt: float) -> FloatArray:
        """M2 / (Δt · τ_i). Note: factor 1/2 for D^(2) is *not* applied here."""
        return self.M2 / (dt * self.tau_indices[:, None])

    def drift_diffusion(
        self, dt: float, *, fit: Literal["direct", "regression"] = "regression",
    ) -> tuple[FloatArray, FloatArray]:
        """Recover D^(1)(x), D^(2)(x) from accumulated M^(k).

        ``"direct"`` uses only the smallest τ. ``"regression"`` fits an OLS
        line ``M^(k) ≈ k! · D^(k) · τ`` (zero intercept) across all τ values
        — preferred when N_τ > 1.
        """
        from ..reductions import kramers_moyal_regression

        if fit == "direct":
            i = 0
            tau_dt = float(self.tau_indices[i]) * dt
            return self.M1[i] / tau_dt, self.M2[i] / (2.0 * tau_dt)
        if fit == "regression":
            return kramers_moyal_regression(self.M1, self.M2, self.tau_indices, dt)
        raise ValueError(f"fit must be 'direct' or 'regression', got {fit!r}")


__all__ = ["OHBR"]
