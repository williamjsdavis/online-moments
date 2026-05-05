from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


KERNEL_BOXCAR = 0
KERNEL_EPANECHNIKOV = 1


class Kernel(ABC):
    """Symmetric kernel with finite support on |x| < 1.

    Subclasses set the integer ``_numba_id`` so the streaming inner loop can
    dispatch without a Python call. The integer constants ``KERNEL_BOXCAR``
    and ``KERNEL_EPANECHNIKOV`` are reused inside the jitted code.
    """

    _numba_id: int

    @abstractmethod
    def __call__(self, x: float) -> float: ...


@dataclass(frozen=True)
class Boxcar(Kernel):
    _numba_id: int = KERNEL_BOXCAR

    def __call__(self, x: float) -> float:
        return 0.5 if abs(x) < 1.0 else 0.0


@dataclass(frozen=True)
class Epanechnikov(Kernel):
    """Standard textbook Epanechnikov: K(x) = (3/4)(1 - x^2) for |x| < 1.

    Note: the Julia ``OnlineMoments.jl`` reference uses a non-standard variance-1
    rescaling (3*sqrt(5)/100 * (5 - x^2) for x^2 < 5). Outputs at the same ``h``
    therefore differ; see ``docs/compared_to_julia.md``.
    """

    _numba_id: int = KERNEL_EPANECHNIKOV

    def __call__(self, x: float) -> float:
        if x * x < 1.0:
            return 0.75 * (1.0 - x * x)
        return 0.0


def apply_kernel(x: float, kernel: Kernel, hinv: float) -> float:
    """Bandwidth-scaled kernel: K_h(x) = K(x/h) / h."""
    return hinv * kernel(hinv * x)


__all__ = [
    "Kernel",
    "Boxcar",
    "Epanechnikov",
    "apply_kernel",
    "KERNEL_BOXCAR",
    "KERNEL_EPANECHNIKOV",
]
