"""online_moments — streaming estimation of conditional moments and Kramers-Moyal coefficients."""
from __future__ import annotations

from .kernels import Boxcar, Epanechnikov, Kernel, apply_kernel
from .offline.hbr import hbr_moments
from .offline.kbr import kbr_moments
from .online.ohbr import OHBR
from .online.okbr import OKBR
from .reductions import kramers_moyal_regression, m1_per_tau, m2_per_tau

__version__ = "0.1.0"

__all__ = [
    "OKBR",
    "OHBR",
    "Kernel",
    "Boxcar",
    "Epanechnikov",
    "apply_kernel",
    "hbr_moments",
    "kbr_moments",
    "m1_per_tau",
    "m2_per_tau",
    "kramers_moyal_regression",
    "__version__",
]
