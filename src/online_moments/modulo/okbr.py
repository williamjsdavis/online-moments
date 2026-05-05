from __future__ import annotations

from typing import Literal

import numpy as np

from .._typing import FloatArray, IntArray
from ..kernels import KERNEL_BOXCAR, KERNEL_EPANECHNIKOV, Boxcar, Epanechnikov, Kernel
from . import _inner_loop


_SUPPORTED_KERNELS = {Boxcar: KERNEL_BOXCAR, Epanechnikov: KERNEL_EPANECHNIKOV}


class OKBRMod:
    """Online KBR with periodic state space (modulo ``period``).

    The kernel is evaluated against the shortest distance ``d_mod(x_eval - x_left, period)``,
    so the estimator uses kernels on the circle.
    """

    def __init__(
        self,
        x_eval: FloatArray,
        tau_indices: IntArray,
        period: float,
        *,
        kernel: Kernel,
        bandwidth: float,
        moment_form: Literal["variance", "raw"] = "variance",
    ) -> None:
        x_eval = np.ascontiguousarray(x_eval, dtype=np.float64)
        tau_indices = np.ascontiguousarray(tau_indices, dtype=np.int64)
        if x_eval.ndim != 1 or x_eval.size == 0:
            raise ValueError("x_eval must be a non-empty 1-D array")
        if tau_indices.ndim != 1 or tau_indices.size == 0:
            raise ValueError("tau_indices must be a non-empty 1-D array")
        if (tau_indices <= 0).any():
            raise ValueError("tau_indices must be strictly positive")
        if (np.diff(tau_indices) <= 0).any():
            raise ValueError("tau_indices must be strictly increasing")
        if period <= 0:
            raise ValueError("period must be positive")
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if moment_form not in ("variance", "raw"):
            raise ValueError(f"moment_form must be 'variance' or 'raw'")
        kernel_id = _SUPPORTED_KERNELS.get(type(kernel))
        if kernel_id is None:
            raise ValueError(f"Unsupported kernel {type(kernel).__name__}")

        n_tau = len(tau_indices)
        n_x = len(x_eval)
        max_lag = int(tau_indices[-1])

        self.x_eval = x_eval
        self.tau_indices = tau_indices
        self.period = float(period)
        self.kernel = kernel
        self.bandwidth = float(bandwidth)
        self.moment_form = moment_form
        self.W: FloatArray = np.zeros((n_tau, n_x), dtype=np.float64)
        self.M1: FloatArray = np.zeros((n_tau, n_x), dtype=np.float64)
        self.M2: FloatArray = np.zeros((n_tau, n_x), dtype=np.float64)
        self._ring: FloatArray = np.zeros(max_lag, dtype=np.float64)
        self._n_pushed: int = 0
        self._kernel_id = int(kernel_id)
        self._hinv = 1.0 / float(bandwidth)
        self._use_variance = moment_form == "variance"

    @property
    def n_processed(self) -> int:
        return self._n_pushed

    def update(self, x: float) -> None:
        self._n_pushed = _inner_loop.okbr_mod_update(
            float(x), self.x_eval, self.tau_indices, self.period,
            self._kernel_id, self._hinv,
            self._ring, self._n_pushed,
            self.W, self.M1, self.M2, self._use_variance,
        )

    def update_batch(self, X: FloatArray) -> None:
        X = np.ascontiguousarray(X, dtype=np.float64)
        self._n_pushed = _inner_loop.okbr_mod_update_batch(
            X, self.x_eval, self.tau_indices, self.period,
            self._kernel_id, self._hinv,
            self._ring, self._n_pushed,
            self.W, self.M1, self.M2, self._use_variance,
        )
