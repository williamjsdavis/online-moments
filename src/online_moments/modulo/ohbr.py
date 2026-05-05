from __future__ import annotations

from typing import Literal

import numpy as np

from .._typing import FloatArray, IntArray
from . import _inner_loop


class OHBRMod:
    """Online HBR with periodic state space (modulo ``period``).

    Use for angular variables (e.g., geomagnetic dipole tilt). The increment
    ``ΔX = X_n - X_{n-τ}`` is wrapped into ``(-period/2, period/2]`` so the
    estimator measures shortest signed jumps on the circle.
    """

    def __init__(
        self,
        edges: FloatArray,
        tau_indices: IntArray,
        period: float,
        *,
        moment_form: Literal["variance", "raw"] = "variance",
    ) -> None:
        edges = np.ascontiguousarray(edges, dtype=np.float64)
        tau_indices = np.ascontiguousarray(tau_indices, dtype=np.int64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("edges must be a 1-D array with length >= 2")
        if tau_indices.ndim != 1 or tau_indices.size == 0:
            raise ValueError("tau_indices must be a non-empty 1-D array")
        if (tau_indices <= 0).any():
            raise ValueError("tau_indices must be strictly positive")
        if (np.diff(tau_indices) <= 0).any():
            raise ValueError("tau_indices must be strictly increasing")
        if period <= 0:
            raise ValueError("period must be positive")
        if moment_form not in ("variance", "raw"):
            raise ValueError(f"moment_form must be 'variance' or 'raw'")

        n_tau = len(tau_indices)
        n_x = len(edges) - 1
        max_lag = int(tau_indices[-1])

        self.edges = edges
        self.tau_indices = tau_indices
        self.period = float(period)
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

    def update(self, x: float) -> None:
        self._n_pushed = _inner_loop.ohbr_mod_update(
            float(x), self.edges, self.tau_indices, self.period,
            self._ring, self._n_pushed,
            self.counts, self.M1, self.M2, self._use_variance,
        )

    def update_batch(self, X: FloatArray) -> None:
        X = np.ascontiguousarray(X, dtype=np.float64)
        self._n_pushed = _inner_loop.ohbr_mod_update_batch(
            X, self.edges, self.tau_indices, self.period,
            self._ring, self._n_pushed,
            self.counts, self.M1, self.M2, self._use_variance,
        )
