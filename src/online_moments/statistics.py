"""Streaming statistics primitives.

These match the textbook-style updates used in ``OnlineMoments.jl``. They are
intentionally written as small pure functions so the offline reference paths
and the streaming code paths can share the same arithmetic operations in the
same order — that's what makes the HBR online-vs-offline equivalence test
hold to bit precision.
"""
from __future__ import annotations


def update_mean(x_bar: float, x_new: float, n: int) -> float:
    """Welford-style mean update: x_bar_n = x_bar_{n-1} + (x_n - x_bar_{n-1}) / n."""
    return x_bar + (x_new - x_bar) / n


def update_var(s2: float, x_bar_new: float, x_bar_old: float, x_new: float, n: int) -> float:
    """Variance update parallel to ``update_mean`` (population variance, ddof=0).

    s2_n = s2_{n-1} + ((x - x_bar_new)*(x - x_bar_old) - s2_{n-1}) / n.
    Requires the *new* mean ``x_bar_new`` to have been computed first.
    """
    return s2 + ((x_new - x_bar_new) * (x_new - x_bar_old) - s2) / n


def update_ss(ss: float, x_new: float, n: int) -> float:
    """Streaming raw second moment: ss_n = mean of x_k^2 over k=1..n."""
    return ss + (x_new * x_new - ss) / n


def update_var_welford(S: float, x_bar_new: float, x_bar_old: float, x_new: float) -> float:
    """Welford S accumulator (sum of squared deviations from running mean).

    S_n = S_{n-1} + (x - x_bar_old)*(x - x_bar_new). Variance = S / n.
    More numerically stable than ``update_var`` for very long streams.
    """
    return S + (x_new - x_bar_old) * (x_new - x_bar_new)


def update_wmean(x_bar: float, w: float, x_new: float, w_new: float) -> float:
    """Weighted streaming mean with non-negative scalar weights."""
    return x_bar + (x_new - x_bar) * (w_new / (w + w_new))


def update_wvar(
    s2: float,
    x_bar_old: float,
    w: float,
    x_new: float,
    x_bar_new: float,
    w_new: float,
) -> float:
    """Weighted streaming variance (population, weight-normalised).

    Equivalent to (w*s2 + w_new*(x - x_bar_old)*(x - x_bar_new)) / (w + w_new).
    """
    return (s2 * w + w_new * (x_new - x_bar_old) * (x_new - x_bar_new)) / (w + w_new)


def update_wss(ss: float, w: float, x_new: float, w_new: float) -> float:
    """Weighted streaming raw second moment."""
    return ss + (x_new * x_new - ss) * (w_new / (w + w_new))


__all__ = [
    "update_mean",
    "update_var",
    "update_ss",
    "update_var_welford",
    "update_wmean",
    "update_wvar",
    "update_wss",
]
