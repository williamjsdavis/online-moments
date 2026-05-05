from __future__ import annotations

import numpy as np

from ._typing import FloatArray


def in_range(edges: FloatArray, x: float) -> bool:
    """True iff x lies in the closed interval [edges[0], edges[-1]]."""
    return edges[0] <= x <= edges[-1]


def find_bin(edges: FloatArray, x: float) -> int:
    """Index of the bin containing x, assuming x is in range.

    Half-open `[edges[i], edges[i+1])` for all bins except the last, which is
    closed `[edges[-2], edges[-1]]`. Returns an index in ``[0, len(edges) - 2]``.
    """
    n_bins = len(edges) - 1
    idx = int(np.searchsorted(edges, x, side="right")) - 1
    if idx >= n_bins:
        idx = n_bins - 1
    if idx < 0:
        idx = 0
    return idx


def get_bin(edges: FloatArray, x: float) -> int:
    """Bin index, or -1 if out of range. -1 is the sentinel "not in any bin"."""
    if not in_range(edges, x):
        return -1
    return find_bin(edges, x)


def d_mod(x: float, period: float) -> float:
    """Shortest non-negative distance to 0 modulo ``period``.

    ``min(x mod period, (-x) mod period)`` — Julia's ``d_mod``.
    """
    a = x % period
    b = (-x) % period
    return a if a < b else b


def _d_plus(a: float, b: float, period: float) -> float:
    return (b - a) % period


def is_in_interval_mod(a: float, b: float, period: float, x: float) -> bool:
    """Half-open mod-period interval: x in [a, b) on the circle of circumference period."""
    return _d_plus(a, x, period) < _d_plus(a, b, period)


def find_mod_bin(edges: FloatArray, period: float, x: float) -> int:
    """Bin index for a point on a periodic state space.

    Returns -1 if x is not in any bin (the bin set need not cover the full
    period). Mirrors Julia's ``find_mod_bin`` but returns -1 (not 0) for the
    "no bin" sentinel.
    """
    n_bins = len(edges) - 1
    for i in range(n_bins):
        a = edges[i]
        b = edges[i + 1]
        if is_in_interval_mod(a, b, period, x):
            return i
    return -1


__all__ = [
    "in_range",
    "find_bin",
    "get_bin",
    "d_mod",
    "is_in_interval_mod",
    "find_mod_bin",
]
