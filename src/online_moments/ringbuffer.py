from __future__ import annotations

import numpy as np

from ._typing import FloatArray


def push(buf: FloatArray, x: float) -> None:
    """Push x to the front of a fixed-size ring buffer in-place.

    After ``push``, ``buf[0]`` is the most recent value and ``buf[k]`` is the
    sample pushed k steps ago. Older entries roll off the end.
    """
    if len(buf) == 0:
        return
    buf[1:] = buf[:-1]
    buf[0] = x


def push_int(buf: np.ndarray, x: int) -> None:
    """Integer-valued analog of ``push`` (e.g., for cached bin indices)."""
    if len(buf) == 0:
        return
    buf[1:] = buf[:-1]
    buf[0] = x


__all__ = ["push", "push_int"]
