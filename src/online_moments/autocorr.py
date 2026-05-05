"""Autocorrelation: offline and a Welford-style online estimator."""
from __future__ import annotations

import numpy as np

from ._typing import FloatArray


def offline_autocorr(X: FloatArray, lags: int) -> FloatArray:
    """Sample autocovariance at lags 0..lags inclusive.

    Mirrors the Julia ``offline_autocorr``: divides each lag-k sum by ``N - k``,
    not by N. Returns a 1-D array of length ``lags + 1``.
    """
    X = np.asarray(X, dtype=np.float64)
    N = len(X)
    mean = X.mean()
    Xd = X - mean
    out = np.zeros(lags + 1, dtype=np.float64)
    for j in range(lags + 1):
        out[j] = (Xd[: N - j] * Xd[j:]).sum() / (N - j)
    return out


class OnlineAutoCov:
    """Streaming autocovariance via Welford-style updates.

    Maintains a small ring buffer of the last ``lags + 1`` samples and an
    accumulator per lag. Memory is ``O(lags)`` independent of N.

    Returns autocovariance ``c_k = E[(X_t - μ)(X_{t-k} - μ)]`` at lags 0..lags.
    """

    def __init__(self, lags: int) -> None:
        if lags < 0:
            raise ValueError("lags must be >= 0")
        self.lags = int(lags)
        self._ring = np.zeros(self.lags + 1, dtype=np.float64)
        self._mean = 0.0
        # M2 in Welford-S form per lag (sum of products of deviations from
        # *current* mean — recovered to autocovariance by dividing by N - k).
        self._M2 = np.zeros(self.lags + 1, dtype=np.float64)
        self._N = 0

    def update(self, x: float) -> None:
        self._N += 1
        n = self._N
        old_mean = self._mean
        new_mean = old_mean + (x - old_mean) / n
        self._mean = new_mean
        # Lag 0: Welford recurrence — S_n = S_{n-1} + (x - μ_old)(x - μ_new),
        # which is algebraically identical to Σ (x_t - μ_n)^2. Exact match
        # to ``np.var(X) * N`` after any number of samples.
        self._M2[0] += (x - old_mean) * (x - new_mean)
        # Lag k >= 1: streaming approximation using running mean.
        # Σ (x_t - μ_t)(x_{t-k} - μ_t) is biased at finite n but consistent.
        for k in range(1, self.lags + 1):
            if n - 1 < k:
                continue
            lagged = self._ring[k - 1]
            self._M2[k] += (x - new_mean) * (lagged - new_mean)

        # Push x onto the ring (so ring[0] is the previous sample, ring[1] two
        # steps ago, etc.). The k=1 lookup above uses ring[0] before this push.
        L = self.lags
        if L > 0:
            for j in range(L - 1, 0, -1):
                self._ring[j] = self._ring[j - 1]
            self._ring[0] = x

    def update_batch(self, X: FloatArray) -> None:
        for x in np.asarray(X, dtype=np.float64):
            self.update(float(x))

    @property
    def autocov(self) -> FloatArray:
        out = np.zeros(self.lags + 1, dtype=np.float64)
        for k in range(self.lags + 1):
            denom = self._N - k
            if denom > 0:
                out[k] = self._M2[k] / denom
        return out

    @property
    def autocorr(self) -> FloatArray:
        c = self.autocov
        if c[0] == 0:
            return np.zeros_like(c)
        return c / c[0]
